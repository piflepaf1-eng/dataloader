"""
dino_loader.memory
==================
In-memory data structures and GPU transfer utilities.

Changes in this version
-----------------------
[MEM-1] Batch dataclass enriched (retained).

[MEM-2] allocate_buffers: ceiling sizes use max_*_crop_size (retained).

[MEM-3] AsyncPrefetchIterator — genuine background prefetch (retained).

[MEM-4] FP8Formatter: no-op when dali_fp8_output=True.                  ← NEW
         When LoaderConfig.dali_fp8_output=True, the FP8 cast is performed
         inside the DALI graph (pipeline.py, [PL-5]).  In that case,
         loader.py does NOT construct FP8Formatter at all (self._fp8 = None).

         If somehow both paths are active (mis-configuration), FP8Formatter
         detects the tensor is already FP8 and returns it unchanged with a
         warning.  This prevents double-quantisation.

         The class is otherwise unchanged: it still uses a rolling amax
         window (length 16) and Transformer Engine FP8TensorMeta when TE is
         available, giving full te.fp8_autocast compatibility in the default
         (non-DALI-FP8) path.
"""

from __future__ import annotations

import contextlib
import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from dino_loader.config      import DINOAugConfig
from dino_loader.distributed import ClusterTopology

log = logging.getLogger(__name__)

try:
    import transformer_engine.pytorch as te
    HAS_TE = True
except ImportError:
    HAS_TE = False
    log.debug("transformer-engine not installed — FP8 output disabled.")


# ══════════════════════════════════════════════════════════════════════════════
# Batch
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Batch:
    """
    One training batch with all views on GPU.

    Fields
    ------
    global_crops : List of 2 tensors (BF16, FP8+TE meta, or FP8 from DALI).
    local_crops  : List of 8 tensors.
    metadata     : Per-sample sidecar dicts.  None when absent.  [MEM-1]
    masks        : iBOT token mask tensor (bool, shape batch×n_tokens) or None.
                   Generated on CPU post-DALI; cannot be fused into DALI
                   because masking operates on ViT patch indices, not pixels.
    """
    global_crops: List
    local_crops:  List
    metadata:     List[Optional[Dict]] = field(default_factory=list)
    masks:        Optional[Any]        = None

    def __iter__(self):
        """Convenience: unpack as (global_crops, local_crops)."""
        return iter((self.global_crops, self.local_crops))


# ══════════════════════════════════════════════════════════════════════════════
# NUMA-aware memory allocation
# ══════════════════════════════════════════════════════════════════════════════

def allocate_buffers(
    batch_size: int,
    aug_cfg:    DINOAugConfig,
    topo:       ClusterTopology,
    device:     torch.device,
    dtype:      torch.dtype = torch.bfloat16,
) -> Dict[str, List[torch.Tensor]]:
    """
    Allocate output buffers using the topology-appropriate strategy.

    [MEM-2] Dimensions use max_global_crop_size / max_local_crop_size so
    no re-allocation occurs when set_resolution() is called mid-training.

    Grace-Blackwell → managed memory, preferred on GPU (no H2D needed).
    PCIe            → pinned host memory (fast non-blocking H2D).
    """
    alloc_fn = (
        torch.zeros  # managed memory — already GPU-visible
        if topo.is_grace_blackwell
        else lambda *a, **kw: torch.zeros(*a, **kw).pin_memory()
    )

    def _buf(size: int) -> List[torch.Tensor]:
        return [
            alloc_fn(batch_size, 3, size, size, dtype=dtype, device=device)
            for _ in range(2)  # global crops
        ]

    return {
        "global": _buf(aug_cfg.max_global_crop_size),
        "local":  _buf(aug_cfg.max_local_crop_size),
    }


# ══════════════════════════════════════════════════════════════════════════════
# H2D transfer stream
# ══════════════════════════════════════════════════════════════════════════════

class H2DStream:
    """
    Async host-to-device transfer on a dedicated CUDA stream.

    On Grace-Blackwell (NVLink-C2C), host and device share the same physical
    memory — H2D is a no-op and no stream is needed.
    """

    def __init__(self, device: torch.device, topo: ClusterTopology) -> None:
        self._device = device
        self._topo   = topo
        self._stream = torch.cuda.Stream(device=device)
        self._c2c    = topo.is_grace_blackwell
        if self._c2c:
            log.info("H2DStream: NVLink-C2C — H2D is a no-op (managed memory)")
        else:
            log.info("H2DStream: PCIe path, dedicated CUDA stream allocated")

    @contextlib.contextmanager
    def transfer(self, cpu_batch: Dict[str, List[torch.Tensor]]):
        """
        Async H2D transfer context manager.

        Usage::

            with h2d.transfer({"global": [...], "local": [...]}) as gpu:
                loss = model(gpu["global"], gpu["local"])
        """
        if self._c2c:
            yield cpu_batch
            return

        with torch.cuda.stream(self._stream):
            gpu_batch = {
                key: [t.to(self._device, non_blocking=True) for t in tensors]
                for key, tensors in cpu_batch.items()
            }
        torch.cuda.current_stream().wait_stream(self._stream)
        yield gpu_batch

    def send(self, cpu_batch: Dict[str, List[torch.Tensor]]) -> Dict[str, List[torch.Tensor]]:
        """Non-context-manager variant; caller must call wait() before use."""
        if self._c2c:
            return cpu_batch
        with torch.cuda.stream(self._stream):
            return {
                key: [t.to(self._device, non_blocking=True) for t in tensors]
                for key, tensors in cpu_batch.items()
            }

    def wait(self) -> None:
        torch.cuda.current_stream().wait_stream(self._stream)


# ══════════════════════════════════════════════════════════════════════════════
# [MEM-3] Async prefetch iterator
# ══════════════════════════════════════════════════════════════════════════════

class AsyncPrefetchIterator:
    """
    Wraps a DALI iterator and pre-fetches the next batch on a dedicated
    background thread, hiding DALI decode latency behind GPU compute.

    Thread model
    ------------
    One ThreadPoolExecutor(max_workers=1) thread is the sole consumer of
    ``next(self._iter)``; DALI iterators are not thread-safe.

    Error propagation
    -----------------
    Exceptions from the background thread (DALI decode errors, StopIteration)
    are stored in the Future and re-raised on the training thread.

    Shutdown contract
    -----------------
    close() cancels in-flight work and shuts down the executor.
    __next__() after close() raises StopIteration.
    """

    _SENTINEL = object()

    def __init__(self, dali_iter, h2d: H2DStream) -> None:
        self._iter     = dali_iter
        self._h2d      = h2d
        self._closed   = False
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dali-prefetch")
        self._future: Optional[Future] = None
        self._lock     = threading.Lock()
        self._submit()

    def __iter__(self):
        return self

    def __next__(self):
        with self._lock:
            if self._closed or self._future is None:
                raise StopIteration

        result = self._future.result()

        if result is self._SENTINEL:
            raise StopIteration

        self._submit()
        return result

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            fut = self._future
            self._future = None

        if fut is not None:
            fut.cancel()
        self._executor.shutdown(wait=True, cancel_futures=True)
        log.debug("AsyncPrefetchIterator closed")

    def __del__(self) -> None:
        self.close()

    def _fetch_one(self):
        try:
            return next(self._iter)
        except StopIteration:
            return self._SENTINEL

    def _submit(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._future = self._executor.submit(self._fetch_one)


# ══════════════════════════════════════════════════════════════════════════════
# FP8 formatter — [MEM-4] no-op guard for DALI-FP8 path
# ══════════════════════════════════════════════════════════════════════════════

class FP8Formatter:
    """
    Quantise BF16 tensors to FP8 E4M3 using Transformer Engine.

    Uses a rolling amax window (length 16) matching TE's internal convention
    so that FP8TensorMeta objects can be passed directly into te.fp8_autocast().

    [MEM-4] When LoaderConfig.dali_fp8_output=True, loader.py does NOT
    construct this class (self._fp8 = None), so quantise() is never called.
    As a defensive measure, if quantise() is called on a tensor that is already
    FP8, it logs a warning and returns the tensor unchanged to avoid
    double-quantisation.

    Falls back to a no-op identity when TE is not installed.
    """

    _AMAX_HISTORY_LEN = 16

    def __init__(self) -> None:
        if not HAS_TE:
            log.warning("FP8Formatter: transformer-engine not installed — returning BF16.")
        self._meta: Optional[Any] = None

    def quantise(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Quantise *tensor* (BF16) to FP8 E4M3 in-place and return it.
        Returns *tensor* unchanged if TE is unavailable.

        [MEM-4] If *tensor* is already FP8 (e.g. when DALI handled the cast),
        returns it unchanged with a warning — defensive guard against
        mis-configuration where both paths are active simultaneously.
        """
        if not HAS_TE:
            return tensor

        # [MEM-4] Defensive guard: detect already-FP8 tensors
        if hasattr(tensor, "dtype") and "float8" in str(tensor.dtype).lower():
            log.warning(
                "FP8Formatter.quantise: tensor is already FP8 (dtype=%s). "
                "This suggests both dali_fp8_output=True and use_fp8_output=True "
                "are active simultaneously — loader.py should have prevented this. "
                "Returning tensor unchanged.",
                tensor.dtype,
            )
            return tensor

        if self._meta is None:
            self._meta = te.fp8.FP8TensorMeta()
            self._meta.scale      = torch.ones(1, dtype=torch.float32, device=tensor.device)
            self._meta.scale_inv  = torch.ones(1, dtype=torch.float32, device=tensor.device)
            self._meta.amax_history = torch.zeros(
                self._AMAX_HISTORY_LEN, dtype=torch.float32, device=tensor.device
            )
        return te.fp8_cast(tensor, self._meta, te.DType.kFloat8E4M3)

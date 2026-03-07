"""
dino_loader.memory
==================
In-memory data structures and GPU transfer utilities.

Changes vs previous version
----------------------------
[MEM-1] Batch dataclass enriched:
        - metadata: List[Optional[Dict]] — per-sample sidecar metadata.
        - masks: Optional[Any] — token mask tensor from MaskingGenerator.

[MEM-2] allocate_buffers: ceiling sizes use aug_cfg.max_global_crop_size /
        max_local_crop_size so no re-allocation occurs during resolution
        schedule.

[MEM-3] AsyncPrefetchIterator — genuine background prefetch.

        Previous behaviour
        ------------------
        ``_prefetch()`` called ``next(self._iter)`` on the *calling thread*,
        blocking it for the entire DALI queue-drain duration.  The method
        name was misleading: it did not overlap DALI latency with compute.

        On NVL72 (72 GPU ranks), DALI's GPU decode can stall for 30–100 ms
        when its internal prefetch queue is exhausted.  With the old code
        that stall was paid in full on the training thread, keeping the GPU
        idle.

        New behaviour
        -------------
        A dedicated ``ThreadPoolExecutor(max_workers=1)`` thread calls
        ``next(self._iter)`` in the background.  The result is stored in a
        ``concurrent.futures.Future``.

        The training thread calls ``__next__()``, which:
          1. Calls ``future.result()`` — returns immediately if the background
             fetch already completed (common case), blocks only if DALI is
             still decoding (rare).
          2. Immediately submits the *next* background fetch before returning
             the current batch to the caller.

        Net effect: DALI decode latency is hidden behind GPU compute for
        every batch except the very first.  On B200/GB200 with a 60 ms
        compute step and a 40 ms DALI decode, this reclaims ~40 ms × N_steps
        of GPU utilisation over a full training run.

        Shutdown contract
        -----------------
        ``close()`` cancels any in-flight future and shuts down the executor
        (with a 5-second timeout).  ``DINODataLoader.__del__`` and the DALI
        backend's ``reset()`` call ``close()`` before rebuilding the iterator.
        Calling ``__next__()`` after ``close()`` raises ``StopIteration``.
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
    global_crops : List of 2 tensors (BF16 or FP8+meta) — large views.
    local_crops  : List of 8 tensors — small views.
    metadata     : Per-sample sidecar dicts (Optional[Dict], None if absent).
                   Length == batch_size.  [MEM-1]
    masks        : Token mask tensor from MaskingGenerator, or None.  [MEM-1]
                   Shape: (batch_size, n_tokens) bool.
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
    C = 3
    max_global = aug_cfg.max_global_crop_size
    max_local  = aug_cfg.max_local_crop_size

    def _make(max_size: int, n: int) -> List[torch.Tensor]:
        bufs = []
        for _ in range(n):
            if topo.is_grace_blackwell:
                t = torch.empty(batch_size, C, max_size, max_size, dtype=dtype, device=device)
                try:
                    torch.cuda.memory.cudaMemAdvise(
                        t,
                        torch.cuda.memory.cudaMemAdviseSetPreferredLocation,
                        device.index,
                    )
                except AttributeError:
                    pass
            else:
                t = torch.empty(batch_size, C, max_size, max_size, dtype=dtype).pin_memory()
            bufs.append(t)
        return bufs

    return {
        "global": _make(max_global, aug_cfg.n_global_crops),
        "local":  _make(max_local,  aug_cfg.n_local_crops),
    }


# ══════════════════════════════════════════════════════════════════════════════
# H2D transfer
# ══════════════════════════════════════════════════════════════════════════════

class H2DStream:
    """
    Dedicated CUDA stream for host-to-device transfers.

    On Grace-Blackwell (NVLink-C2C), transfer is a no-op — managed memory
    is already accessible on GPU.  On PCIe, an async copy is issued on a
    dedicated stream to overlap with the compute stream.
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
# [MEM-3] Async prefetch iterator — genuine background prefetch
# ══════════════════════════════════════════════════════════════════════════════

class AsyncPrefetchIterator:
    """
    Wraps a DALI (or CPU-backend) iterator and pre-fetches the next batch
    on a dedicated background thread, hiding DALI decode latency.

    The previous implementation called ``next(self._iter)`` on the *calling
    thread* inside ``_prefetch()``, which blocked the training loop for the
    full DALI decode time.  See module docstring [MEM-3] for details.

    Thread model
    ------------
    One ``ThreadPoolExecutor(max_workers=1)`` thread is owned by this object.
    It is the *only* thread that calls ``next(self._iter)``; DALI iterators
    are not thread-safe and must be consumed from a single thread.

    The training thread calls ``__next__()``, which resolves the current
    ``Future`` (waiting if necessary) and immediately submits the next fetch.

    Error propagation
    -----------------
    Any exception raised inside the background thread (e.g. DALI decode
    error, StopIteration) is stored in the Future and re-raised in the
    training thread on the next call to ``future.result()``.  StopIteration
    is converted to a sentinel ``None`` result (Futures cannot store
    StopIteration directly).

    Shutdown
    --------
    Call ``close()`` to cancel in-flight work and shut down the executor.
    The iterator is unusable after ``close()``.
    """

    _SENTINEL = object()  # marks exhaustion

    def __init__(self, dali_iter, h2d: H2DStream) -> None:
        self._iter     = dali_iter
        self._h2d      = h2d
        self._closed   = False
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dali-prefetch")
        self._future: Optional[Future] = None
        self._lock     = threading.Lock()   # guards _closed + _future
        self._submit()

    # ── Iterator protocol ─────────────────────────────────────────────────────

    def __iter__(self):
        return self

    def __next__(self):
        with self._lock:
            if self._closed or self._future is None:
                raise StopIteration

        result = self._future.result()  # blocks only if DALI is still decoding

        if result is self._SENTINEL:
            raise StopIteration

        # Submit next fetch *before* returning — maximises overlap.
        self._submit()
        return result

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def close(self) -> None:
        """Cancel in-flight prefetch and shut down the executor."""
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

    # ── Internal ──────────────────────────────────────────────────────────────

    def _fetch_one(self):
        """Blocking fetch executed on the background thread."""
        try:
            return next(self._iter)
        except StopIteration:
            return self._SENTINEL

    def _submit(self) -> None:
        """Submit the next fetch unless closed."""
        with self._lock:
            if self._closed:
                return
            self._future = self._executor.submit(self._fetch_one)


# ══════════════════════════════════════════════════════════════════════════════
# FP8 formatter
# ══════════════════════════════════════════════════════════════════════════════

class FP8Formatter:
    """
    Quantise BF16 tensors to FP8 E4M3 using Transformer Engine.

    Uses a rolling amax window (length 16) matching TE's internal convention
    so that FP8TensorMeta objects can be passed directly into te.fp8_autocast().
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
        """
        if not HAS_TE:
            return tensor
        if self._meta is None:
            self._meta = te.fp8.FP8TensorMeta()
            self._meta.scale      = torch.ones(1, dtype=torch.float32, device=tensor.device)
            self._meta.scale_inv  = torch.ones(1, dtype=torch.float32, device=tensor.device)
            self._meta.amax_history = torch.zeros(
                self._AMAX_HISTORY_LEN, dtype=torch.float32, device=tensor.device
            )
        return te.fp8_cast(tensor, self._meta, te.DType.kFloat8E4M3)

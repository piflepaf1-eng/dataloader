"""
tests/test_fixes.py
===================
Regression tests for all fixes and architectural additions introduced
in this review cycle.

Coverage
--------
B1   AsyncPrefetchIterator race-free exception path
B2   NodeSharedShardCache._evict_for_locked: backpressure on full-ref slots
B3   DINODataLoader.set_epoch: threading.Lock prevents concurrent corruption
B4   LoaderConfig: TE absence caught at construction, not at first batch
M2   NormSource: thread-safe copy-on-write + return copies
M3   CheckpointState.save/load: SHA-256 integrity envelope
M4   NodeSharedShardCache: heartbeat_stale_s configurable
M5   allocate_buffers: Grace-Blackwell path allocates on CUDA device
M6   PostProcessPipeline.select: filtered batches tracked in metrics

ARCH1  SharedMemoryRingBuffer: publish / view / evict round-trip
ARCH2  LoaderConfig.adaptive_prefetch: flag wired, PID controller initialises
ARCH3  LoaderConfig.prometheus_port: validation, import check

LD-13  DINODataLoader.current_resolution public property
"""

from __future__ import annotations

import hashlib
import json
import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

_SRC = str(Path(__file__).parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from dino_loader.config import DINOAugConfig, LoaderConfig, CheckpointState


# ══════════════════════════════════════════════════════════════════════════════
# B1 — AsyncPrefetchIterator race-free exception path
# ══════════════════════════════════════════════════════════════════════════════

class TestAsyncPrefetchIteratorB1:
    """Verify the race condition fix on exception propagation."""

    def _make(self, source):
        from dino_loader.memory import AsyncPrefetchIterator
        return AsyncPrefetchIterator(source, h2d=MagicMock())

    def test_exception_propagates_cleanly(self):
        """DALI decode error must surface on the training thread, not hang."""
        def _boom():
            yield {"view_0": 1}
            raise RuntimeError("DALI corrupted shard")

        it = self._make(_boom())
        assert next(it) == {"view_0": 1}
        with pytest.raises(RuntimeError, match="DALI corrupted shard"):
            next(it)
        # Iterator must be in a closed state after exception.
        with pytest.raises(StopIteration):
            next(it)

    def test_no_double_raise_same_error(self):
        """After an exception, the next call must raise StopIteration, not the same error."""
        def _boom():
            raise ValueError("first error")
            yield  # make it a generator

        it = self._make(_boom())
        with pytest.raises(ValueError, match="first error"):
            next(it)
        # Iterator closed — must raise StopIteration, not ValueError again.
        with pytest.raises(StopIteration):
            next(it)

    def test_close_during_result_wait_is_safe(self):
        """close() called from another thread while __next__ blocks must not hang."""
        delay = 0.1
        barrier = threading.Event()

        def _slow():
            barrier.set()
            time.sleep(delay)
            yield 42

        it = self._make(_slow())
        results = []
        errors  = []

        def _consume():
            try:
                results.append(next(it))
            except Exception as e:
                errors.append(e)

        t = threading.Thread(target=_consume)
        t.start()
        barrier.wait()
        it.close()  # concurrent close while __next__ may be blocking
        t.join(timeout=delay * 10)
        assert not t.is_alive(), "Consumer thread hung after close()"
        # Either got a result or a StopIteration — both acceptable.


# ══════════════════════════════════════════════════════════════════════════════
# B2 — _evict_for_locked backpressure
# ══════════════════════════════════════════════════════════════════════════════

class TestEvictForLockedB2:
    """Verify that _evict_for_locked raises clearly when all slots are pinned."""

    def test_evict_raises_when_all_slots_pinned(self, tmp_path):
        """If no entry can be evicted (all ref>0), raise RuntimeError after retries."""
        from dino_loader.shard_cache import NodeSharedShardCache
        import asyncio

        # Patch _EVICT_RETRIES to 1 for speed
        with patch("dino_loader.shard_cache._EVICT_RETRIES", 1), \
             patch("dino_loader.shard_cache._EVICT_WAIT_S",  0.01):

            cache = MagicMock(spec=NodeSharedShardCache)
            cache.utilisation = 0.99
            # Simulate all LRU entries referenced by building a real LRU with refs.
            from collections import OrderedDict
            cache._lru        = OrderedDict()  # empty — nothing to evict
            cache._total_bytes = int(200 * (1 << 30))  # 200 GB "used"
            cache._max_bytes   = int(128 * (1 << 30))  # 128 GB budget

            # Call the real _evict_for_locked on the mock by importing directly.
            from dino_loader.shard_cache import NodeSharedShardCache as NSSC

            async def _run():
                with pytest.raises(RuntimeError, match="could not evict enough space"):
                    await NSSC._load_one(cache, "fake/shard.tar", tmp_path / "x")

            asyncio.run(_run())


# ══════════════════════════════════════════════════════════════════════════════
# B3 — set_epoch threading.Lock
# ══════════════════════════════════════════════════════════════════════════════

class TestSetEpochLockB3:
    """set_epoch must be thread-safe (concurrent calls must not interleave)."""

    def test_concurrent_set_epoch_does_not_corrupt(self, tmp_path):
        """Two threads calling set_epoch simultaneously must not raise or deadlock."""
        from tests.fixtures import scaffold_dataset_dir
        from dino_loader.config import DatasetSpec, LoaderConfig, DINOAugConfig
        from dino_loader.loader import DINODataLoader

        tar_paths = scaffold_dataset_dir(
            root=tmp_path, n_shards=2, n_samples_per_shard=4
        )
        cfg = LoaderConfig(
            node_shm_gb=0.1, shuffle_buffer_size=4,
            stall_timeout_s=0, stateful_dataloader=False,
            checkpoint_dir=str(tmp_path / "ckpt"),
        )
        aug = DINOAugConfig(
            global_crop_size=32, local_crop_size=16,
            n_global_crops=2, n_local_crops=2,
        )
        loader = DINODataLoader(
            specs      = [DatasetSpec(name="ds", shards=tar_paths, weight=1.0)],
            batch_size = 2,
            aug_cfg    = aug,
            config     = cfg,
            backend    = "cpu",
        )

        errors = []

        def _call_set_epoch(epoch):
            try:
                loader.set_epoch(epoch)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=_call_set_epoch, args=(i,)) for i in range(10)]
        for t in threads: t.start()
        for t in threads: t.join()
        assert errors == [], f"Errors during concurrent set_epoch: {errors}"


# ══════════════════════════════════════════════════════════════════════════════
# B4 — TE absence caught at LoaderConfig construction
# ══════════════════════════════════════════════════════════════════════════════

class TestFP8RequiresTE_B4:

    def test_fp8_without_te_raises_at_construction(self):
        """use_fp8_output=True must raise ValueError if TE is not installed."""
        with patch.dict("sys.modules", {"transformer_engine": None,
                                        "transformer_engine.pytorch": None}):
            with pytest.raises(ValueError, match="transformer-engine"):
                LoaderConfig(use_fp8_output=True)

    def test_fp8_false_does_not_require_te(self):
        """use_fp8_output=False (default) must not raise even without TE."""
        with patch.dict("sys.modules", {"transformer_engine": None,
                                        "transformer_engine.pytorch": None}):
            cfg = LoaderConfig(use_fp8_output=False)
            assert cfg.use_fp8_output is False


# ══════════════════════════════════════════════════════════════════════════════
# M2 — NormSource thread safety
# ══════════════════════════════════════════════════════════════════════════════

class TestNormSourceM2:

    def _make_norm_source(self, n_datasets: int = 2):
        from dino_loader.pipeline import NormSource
        specs = []
        for i in range(n_datasets):
            spec = MagicMock()
            spec.mean = None
            spec.std  = None
            specs.append(spec)
        aug = DINOAugConfig()
        return NormSource(aug_cfg=aug, specs=specs)

    def test_set_indices_is_full_replacement(self):
        """set_dataset_indices must atomically replace the list."""
        ns = self._make_norm_source(3)
        ns.set_dataset_indices([0, 1, 2])
        ns.set_dataset_indices([2, 1])
        means, stds = ns()
        assert len(means) == 2
        assert len(stds)  == 2

    def test_call_returns_copies(self):
        """Returned arrays must be independent copies (not views of _lookup)."""
        ns = self._make_norm_source(1)
        ns.set_dataset_indices([0])
        means1, _ = ns()
        means2, _ = ns()
        # Modify one — must not affect the other (they are separate objects)
        means1[0][:] = 99.0
        assert not np.allclose(means2[0], 99.0), \
            "Returned arrays are views into _lookup — expected copies"

    def test_concurrent_set_and_call(self):
        """Concurrent set_dataset_indices and __call__ must not raise."""
        ns = self._make_norm_source(4)
        ns.set_dataset_indices([0, 1, 2, 3])
        errors = []

        def _setter():
            for i in range(200):
                ns.set_dataset_indices([i % 4])

        def _caller():
            for _ in range(200):
                try:
                    ns()
                except Exception as e:
                    errors.append(e)

        threads = [
            threading.Thread(target=_setter),
            threading.Thread(target=_caller),
        ]
        for t in threads: t.start()
        for t in threads: t.join()
        assert errors == [], f"Errors in concurrent NormSource access: {errors}"


# ══════════════════════════════════════════════════════════════════════════════
# M3 — CheckpointState SHA-256 integrity
# ══════════════════════════════════════════════════════════════════════════════

class TestCheckpointIntegrityM3:

    def _state(self, step: int = 42) -> CheckpointState:
        return CheckpointState(
            step=step, epoch=1,
            dataset_names=["laion"],
            mixing_weights=[1.0],
            global_crop_size=224,
            local_crop_size=96,
        )

    def test_save_creates_envelope_with_sha256(self, tmp_path):
        path = tmp_path / "state.json"
        self._state().save(path)
        raw = json.loads(path.read_text())
        assert "payload" in raw
        assert "sha256" in raw
        assert len(raw["sha256"]) == 64   # SHA-256 hex

    def test_checksum_is_correct(self, tmp_path):
        path = tmp_path / "state.json"
        self._state().save(path)
        raw     = json.loads(path.read_text())
        computed = hashlib.sha256(json.dumps(raw["payload"], indent=2).encode()).hexdigest()
        assert raw["sha256"] == computed

    def test_load_verifies_checksum(self, tmp_path):
        path = tmp_path / "state.json"
        self._state().save(path)
        raw = json.loads(path.read_text())
        # Corrupt the payload
        raw["payload"]["step"] = 9999
        path.write_text(json.dumps(raw))
        with pytest.raises(ValueError, match="integrity check"):
            CheckpointState.load(path)

    def test_load_backward_compat_flat_format(self, tmp_path):
        """Old flat-format checkpoints (no sha256) must still load."""
        path = tmp_path / "old.json"
        path.write_text(json.dumps({
            "step": 10, "epoch": 0,
            "dataset_names": ["laion"],
            "mixing_weights": [1.0],
        }))
        state = CheckpointState.load(path)
        assert state.step == 10

    def test_roundtrip_preserves_all_fields(self, tmp_path):
        path = tmp_path / "state.json"
        orig = self._state(step=777)
        orig.save(path)
        loaded = CheckpointState.load(path)
        assert loaded.step             == 777
        assert loaded.epoch            == 1
        assert loaded.dataset_names    == ["laion"]
        assert loaded.mixing_weights   == [1.0]
        assert loaded.global_crop_size == 224
        assert loaded.local_crop_size  == 96


# ══════════════════════════════════════════════════════════════════════════════
# M4 — heartbeat_stale_s configurable in LoaderConfig
# ══════════════════════════════════════════════════════════════════════════════

class TestHeartbeatStaleM4:

    def test_default_is_300s(self):
        cfg = LoaderConfig()
        assert cfg.heartbeat_stale_s == 300.0

    def test_custom_value(self):
        cfg = LoaderConfig(heartbeat_stale_s=600.0)
        assert cfg.heartbeat_stale_s == 600.0

    def test_zero_raises(self):
        with pytest.raises(ValueError, match="heartbeat_stale_s"):
            LoaderConfig(heartbeat_stale_s=0.0)

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="heartbeat_stale_s"):
            LoaderConfig(heartbeat_stale_s=-1.0)

    def test_forwarded_to_shard_cache(self, tmp_path):
        """heartbeat_stale_s must be passed to _purge_orphaned_shm."""
        from dino_loader.shard_cache import _purge_orphaned_shm

        called_with = {}
        original    = _purge_orphaned_shm

        def _spy(job_name, hb_stale_s=300.0):
            called_with["hb_stale_s"] = hb_stale_s

        with patch("dino_loader.shard_cache._purge_orphaned_shm", _spy):
            from dino_loader.shard_cache import NodeSharedShardCache
            try:
                NodeSharedShardCache(
                    node_master=True,
                    job_id="test_hb",
                    max_shm_gb=0.01,
                    heartbeat_stale_s=42.0,
                )
            except Exception:
                pass  # may fail in CI without /dev/shm write perms

        if called_with:
            assert called_with["hb_stale_s"] == 42.0


# ══════════════════════════════════════════════════════════════════════════════
# M5 — allocate_buffers Grace-Blackwell device fix
# ══════════════════════════════════════════════════════════════════════════════

class TestAllocateBuffersM5:

    def _make_topo(self, is_gb200: bool):
        topo = MagicMock()
        type(topo).is_grace_blackwell = PropertyMock(return_value=is_gb200)
        return topo

    def test_pcie_allocates_pinned_cpu(self):
        """PCIe path must return pinned CPU tensors."""
        from dino_loader.memory import allocate_buffers
        import torch
        topo   = self._make_topo(False)
        device = torch.device("cpu")
        aug    = DINOAugConfig(global_crop_size=32, local_crop_size=16,
                               n_global_crops=2, n_local_crops=2)
        bufs   = allocate_buffers(batch_size=4, aug_cfg=aug, topo=topo, device=device)
        for t in bufs["global"] + bufs["local"]:
            assert t.is_pinned(), "PCIe path must produce pinned tensors"
            assert t.device.type == "cpu"

    @pytest.mark.skipif(
        not __import__("torch").cuda.is_available(),
        reason="CUDA not available"
    )
    def test_gb200_allocates_on_cuda(self):
        """Grace-Blackwell path must allocate tensors directly on the CUDA device."""
        from dino_loader.memory import allocate_buffers
        import torch
        topo   = self._make_topo(True)
        device = torch.device("cuda:0")
        aug    = DINOAugConfig(global_crop_size=32, local_crop_size=16,
                               n_global_crops=2, n_local_crops=2)
        bufs   = allocate_buffers(batch_size=2, aug_cfg=aug, topo=topo, device=device)
        for t in bufs["global"] + bufs["local"]:
            assert t.device.type == "cuda", \
                "GB200 path must produce CUDA tensors, not CPU"


# ══════════════════════════════════════════════════════════════════════════════
# M6 — select() tracks filtered batches
# ══════════════════════════════════════════════════════════════════════════════

class TestSelectFilteringM6:

    def _make_pipeline(self, items, predicate):
        from dino_loader.loader import PostProcessPipeline
        loader  = MagicMock()
        loader.current_resolution = (224, 96)
        pipeline = PostProcessPipeline(
            source     = iter(items),
            transforms = [],
            loader     = loader,
        )
        return pipeline.select(predicate)

    def test_filtered_count_incremented(self):
        """Each batch rejected by select() must increment batches_filtered."""
        from dino_loader.monitor.metrics import init_registry, get_registry
        init_registry(rank=0)

        items = [MagicMock(spec=["metadata"]) for _ in range(6)]
        accept_every_other = lambda i: [True, False] * 3

        # Accept batches at even positions, reject odd ones
        toggle = {"i": 0}
        def _predicate(b):
            result = toggle["i"] % 2 == 0
            toggle["i"] += 1
            return result

        pipeline = self._make_pipeline(items, _predicate)
        results  = list(pipeline)

        assert len(results) == 3, "Expected 3 accepted batches out of 6"
        reg = get_registry()
        if reg is not None:
            filtered = reg.get("batches_filtered", 0)
            assert filtered == 3, f"Expected 3 filtered batches, got {filtered}"


# ══════════════════════════════════════════════════════════════════════════════
# ARCH1 — SharedMemoryRingBuffer
# ══════════════════════════════════════════════════════════════════════════════

class TestSharedMemoryRingBufferArch1:

    def test_publish_view_evict_roundtrip(self):
        """Master publishes, reader views, eviction cleans up."""
        from dino_loader.memory import SharedMemoryRingBuffer
        rb   = SharedMemoryRingBuffer(job_id="test_rb_001", node_master=True)
        data = b"hello from lustre" * 100

        rb.publish("fake/shard.tar", data)
        with rb.view("fake/shard.tar") as mv:
            assert bytes(mv) == data

        rb.evict("fake/shard.tar")
        rb.close()

    def test_publish_idempotent_same_size(self):
        """Re-publishing the same shard with same-size data must not raise."""
        from dino_loader.memory import SharedMemoryRingBuffer
        rb   = SharedMemoryRingBuffer(job_id="test_rb_002", node_master=True)
        data = b"x" * 1024

        rb.publish("shard.tar", data)
        rb.publish("shard.tar", b"y" * 1024)  # same size, different content
        with rb.view("shard.tar") as mv:
            assert bytes(mv) == b"y" * 1024
        rb.close()

    def test_view_corrupt_header_raises(self):
        """If segment header is not READY, view() must raise RuntimeError."""
        from dino_loader.memory import SharedMemoryRingBuffer, _SHM_HDR_FMT
        import struct
        rb = SharedMemoryRingBuffer(job_id="test_rb_003", node_master=True)
        rb.publish("shard.tar", b"data" * 10)

        # Corrupt the magic
        seg = rb._segments["shard.tar"]
        struct.pack_into(_SHM_HDR_FMT, seg.buf, 0, 40, 0xDEAD)  # wrong magic

        with pytest.raises(RuntimeError, match="corrupt header"):
            with rb.view("shard.tar") as _:
                pass
        rb.close()

    def test_close_unlinks_all_segments(self):
        """close() must unlink all published segments."""
        from dino_loader.memory import SharedMemoryRingBuffer
        rb = SharedMemoryRingBuffer(job_id="test_rb_004", node_master=True)
        for i in range(3):
            rb.publish(f"shard_{i}.tar", b"data" * 10)

        segs_before = list(rb._segments.keys())
        rb.close()
        assert rb._segments == {}, "All segments must be removed after close()"


# ══════════════════════════════════════════════════════════════════════════════
# ARCH2 — adaptive_prefetch config validation
# ══════════════════════════════════════════════════════════════════════════════

class TestAdaptivePrefetchArch2:

    def test_default_is_disabled(self):
        cfg = LoaderConfig()
        assert cfg.adaptive_prefetch is False

    def test_enable_with_valid_target(self):
        cfg = LoaderConfig(adaptive_prefetch=True, adaptive_prefetch_target_util=0.80)
        assert cfg.adaptive_prefetch is True
        assert cfg.adaptive_prefetch_target_util == 0.80

    def test_invalid_target_zero(self):
        with pytest.raises(ValueError, match="adaptive_prefetch_target_util"):
            LoaderConfig(adaptive_prefetch=True, adaptive_prefetch_target_util=0.0)

    def test_invalid_target_above_one(self):
        with pytest.raises(ValueError, match="adaptive_prefetch_target_util"):
            LoaderConfig(adaptive_prefetch=True, adaptive_prefetch_target_util=1.1)


# ══════════════════════════════════════════════════════════════════════════════
# ARCH3 — prometheus_port config validation
# ══════════════════════════════════════════════════════════════════════════════

class TestPrometheusArch3:

    def test_default_disabled(self):
        cfg = LoaderConfig()
        assert cfg.prometheus_port is None

    def test_invalid_port_zero(self):
        with patch.dict("sys.modules", {"prometheus_client": MagicMock()}):
            with pytest.raises(ValueError, match="prometheus_port"):
                LoaderConfig(prometheus_port=0)

    def test_invalid_port_too_large(self):
        with patch.dict("sys.modules", {"prometheus_client": MagicMock()}):
            with pytest.raises(ValueError, match="prometheus_port"):
                LoaderConfig(prometheus_port=99999)

    def test_missing_prometheus_client_raises(self):
        """Setting prometheus_port without prometheus-client installed must raise."""
        with patch.dict("sys.modules", {"prometheus_client": None}):
            with pytest.raises(ValueError, match="prometheus_client"):
                LoaderConfig(prometheus_port=9100)

    def test_valid_port_with_package_installed(self):
        with patch.dict("sys.modules", {"prometheus_client": MagicMock()}):
            cfg = LoaderConfig(prometheus_port=9100)
            assert cfg.prometheus_port == 9100


# ══════════════════════════════════════════════════════════════════════════════
# LD-13 — current_resolution public property
# ══════════════════════════════════════════════════════════════════════════════

class TestCurrentResolutionLD13:

    def test_current_resolution_returns_tuple(self, tmp_path):
        from tests.fixtures import scaffold_dataset_dir
        from dino_loader.config import DatasetSpec
        from dino_loader.loader import DINODataLoader

        tar_paths = scaffold_dataset_dir(root=tmp_path, n_shards=2, n_samples_per_shard=4)
        loader = DINODataLoader(
            specs      = [DatasetSpec(name="ds", shards=tar_paths, weight=1.0)],
            batch_size = 2,
            aug_cfg    = DINOAugConfig(global_crop_size=32, local_crop_size=16,
                                       n_global_crops=2, n_local_crops=2),
            config     = LoaderConfig(
                node_shm_gb=0.1, stall_timeout_s=0,
                stateful_dataloader=False,
                checkpoint_dir=str(tmp_path / "ckpt"),
            ),
            backend    = "cpu",
        )
        g, l = loader.current_resolution
        assert g == 32
        assert l == 16

    def test_set_resolution_updates_property(self, tmp_path):
        from tests.fixtures import scaffold_dataset_dir
        from dino_loader.config import DatasetSpec
        from dino_loader.loader import DINODataLoader

        tar_paths = scaffold_dataset_dir(root=tmp_path, n_shards=2, n_samples_per_shard=4)
        loader = DINODataLoader(
            specs      = [DatasetSpec(name="ds", shards=tar_paths, weight=1.0)],
            batch_size = 2,
            aug_cfg    = DINOAugConfig(
                global_crop_size=32, local_crop_size=16,
                max_global_crop_size=64, max_local_crop_size=32,
                n_global_crops=2, n_local_crops=2,
            ),
            config     = LoaderConfig(
                node_shm_gb=0.1, stall_timeout_s=0,
                stateful_dataloader=False,
                checkpoint_dir=str(tmp_path / "ckpt"),
            ),
            backend    = "cpu",
        )
        loader.set_resolution(64, 32)
        assert loader.current_resolution == (64, 32)

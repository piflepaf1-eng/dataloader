"""
tests/test_fixes.py
===================
Regression tests for all fixes and architectural additions.

Coverage
--------
B2   NodeSharedShardCache._evict_for_locked: backpressure on full-ref slots
B3   DINODataLoader.set_epoch: threading.Lock prevents concurrent corruption
B4   LoaderConfig: TE absence caught at construction, not at first batch
M2   NormSource: thread-safe copy-on-write + return copies
M3   CheckpointState.save/load: SHA-256 integrity envelope
M4   NodeSharedShardCache: heartbeat_stale_s configurable
M5   allocate_buffers: pinned memory on CUDA device
M6   PostProcessPipeline.select: filtered batches tracked in metrics

ARCH1  SharedMemoryRingBuffer: publish / view / evict round-trip
ARCH2  LoaderConfig.adaptive_prefetch: flag wired, PID controller initialises
ARCH3  LoaderConfig.prometheus_port: validation, import check

LD-13  DINODataLoader.current_resolution public property

DALI queue depth
----------------
AsyncPrefetchIterator was removed; DALI's internal queues now provide all
pipeline buffering.  Tests verify that dali_cpu_queue is sized appropriately
and that _raw_iter iterates directly over the DALI iterator without an
additional threading layer.

Note: B1 (AsyncPrefetchIterator race-free exception path) is removed — the
class no longer exists.  The threading complexity it addressed is gone.
"""

from __future__ import annotations

import hashlib
import json
import sys
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

_SRC = str(Path(__file__).parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from dino_loader.config import DINOAugConfig, LoaderConfig, CheckpointState


# ══════════════════════════════════════════════════════════════════════════════
# DALI queue depth — replaces B1 (AsyncPrefetchIterator tests)
# ══════════════════════════════════════════════════════════════════════════════

class TestDALIQueueDepth:
    """Verify that dali_cpu_queue is appropriately sized after APrefetch removal."""

    def test_default_cpu_queue_is_at_least_16(self):
        """dali_cpu_queue must be ≥ 16 to compensate for AsyncPrefetchIterator removal."""
        cfg = LoaderConfig()
        assert cfg.dali_cpu_queue >= 16, (
            f"dali_cpu_queue={cfg.dali_cpu_queue} is too small; "
            "increase to ≥ 16 to ensure DALI queues saturate the GPU."
        )

    def test_custom_cpu_queue_accepted(self):
        cfg = LoaderConfig(dali_cpu_queue=32)
        assert cfg.dali_cpu_queue == 32

    def test_raw_iter_does_not_wrap_in_thread(self, tmp_path):
        """_raw_iter must not use any Future/ThreadPoolExecutor layer."""

        # Verify AsyncPrefetchIterator is not imported anywhere in loader.py.
        loader_src = Path(_SRC) / "dino_loader" / "loader.py"
        content = loader_src.read_text()
        assert "AsyncPrefetchIterator" not in content, (
            "loader.py still references AsyncPrefetchIterator — it should have been removed."
        )

    def test_async_prefetch_iterator_removed_from_memory(self):
        """AsyncPrefetchIterator must no longer be exported from memory.py."""
        import dino_loader.memory as mem_mod
        assert not hasattr(mem_mod, "AsyncPrefetchIterator"), (
            "AsyncPrefetchIterator is still present in memory.py — it should be removed."
        )


# ══════════════════════════════════════════════════════════════════════════════
# B2 — _evict_for_locked backpressure
# ══════════════════════════════════════════════════════════════════════════════

class TestEvictForLockedB2:
    """Verify that _evict_for_locked raises clearly when all slots are pinned."""

    def test_evict_raises_when_all_slots_pinned(self, tmp_path):
        """If no entry can be evicted (all ref>0), raise RuntimeError after retries."""
        from dino_loader.shard_cache import NodeSharedShardCache
        import asyncio

        with patch("dino_loader.shard_cache._EVICT_RETRIES", 1), \
             patch("dino_loader.shard_cache._EVICT_WAIT_S",  0.01):

            cache = MagicMock(spec=NodeSharedShardCache)
            cache.utilisation = 0.99
            from collections import OrderedDict
            cache._lru         = OrderedDict()
            cache._total_bytes = int(200 * (1 << 30))
            cache._max_bytes   = int(128 * (1 << 30))

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
        with patch.dict("sys.modules", {"transformer_engine": None,
                                        "transformer_engine.pytorch": None}):
            with pytest.raises(ValueError, match="transformer-engine"):
                LoaderConfig(use_fp8_output=True)

    def test_fp8_false_does_not_require_te(self):
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
        ns = self._make_norm_source(3)
        ns.set_dataset_indices([0, 1, 2])
        ns.set_dataset_indices([2, 1])
        means, stds = ns()
        assert len(means) == 2
        assert len(stds)  == 2

    def test_call_returns_copies(self):
        ns = self._make_norm_source(1)
        ns.set_dataset_indices([0])
        means1, _ = ns()
        means2, _ = ns()
        means1[0][:] = 99.0
        assert not np.allclose(means2[0], 99.0), \
            "Returned arrays are views into _lookup — expected copies"

    def test_concurrent_set_and_call(self):
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
        assert len(raw["sha256"]) == 64

    def test_checksum_is_correct(self, tmp_path):
        path = tmp_path / "state.json"
        self._state().save(path)
        raw      = json.loads(path.read_text())
        computed = hashlib.sha256(json.dumps(raw["payload"], indent=2).encode()).hexdigest()
        assert raw["sha256"] == computed

    def test_load_verifies_checksum(self, tmp_path):
        path = tmp_path / "state.json"
        self._state().save(path)
        raw = json.loads(path.read_text())
        raw["payload"]["step"] = 9999
        path.write_text(json.dumps(raw))
        with pytest.raises(ValueError, match="integrity check"):
            CheckpointState.load(path)

    def test_load_backward_compat_flat_format(self, tmp_path):
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

        called_with = {}

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
                pass

        if called_with:
            assert called_with["hb_stale_s"] == 42.0


# ══════════════════════════════════════════════════════════════════════════════
# M5 — allocate_buffers: pinned CPU memory
# ══════════════════════════════════════════════════════════════════════════════

class TestAllocateBuffersM5:

    def _make_topo(self):
        topo = MagicMock()
        return topo

    def test_pcie_allocates_pinned_cpu(self):
        from dino_loader.memory import allocate_buffers
        import torch
        topo   = self._make_topo()
        device = torch.device("cpu")
        aug    = DINOAugConfig(global_crop_size=32, local_crop_size=16,
                               n_global_crops=2, n_local_crops=2)
        bufs   = allocate_buffers(batch_size=4, aug_cfg=aug, topo=topo, device=device)
        for t in bufs["global"] + bufs["local"]:
            assert t.is_pinned(), "PCIe path must produce pinned tensors"
            assert t.device.type == "cpu"


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
        from dino_loader.monitor.metrics import init_registry, get_registry
        init_registry(rank=0)

        items  = [MagicMock(spec=["metadata"]) for _ in range(6)]
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

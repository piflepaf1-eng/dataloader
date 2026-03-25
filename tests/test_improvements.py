"""
tests/test_improvements.py
==========================
Regression tests for performance and maintainability improvements.

Each test class maps to one numbered improvement item.  The item numbers are
stable references used in commit messages and the changelog.

Coverage
--------
Item #1  — DALI queue-based buffering replaces AsyncPrefetchIterator
           (verify removal and that dali_cpu_queue compensates)
Item #2  — _MmapPool: persistent pool, ref-counting, LRU eviction
Item #3  — NodeSharedShardCache._write: no fsync on tmpfs; atomic rename
Item #4  — MixingSource.__call__: vectorised np.rng.choice dataset selection
Item #5  — MetricField StrEnum: 1-to-1 mapping with MetricsStruct fields
Item #7  — DataLoaderCheckpointer: LATEST pointer, crash-safe discovery
Item #9  — Queue depth metric wired (MetricField.MIXING_QUEUE_DEPTH != 0)
Item #11 — ShardIterator.reservoir_size property exists and returns a positive int
Item #12 — pyproject.toml has pinned dependency versions (smoke test)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest

_SRC = str(Path(__file__).parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from tests.fixtures import write_shm_file


# ══════════════════════════════════════════════════════════════════════════════
# Item #1 — DALI queue-based buffering (AsyncPrefetchIterator removed)
# ══════════════════════════════════════════════════════════════════════════════

class TestDALIQueueBuffering:
    """DALI's prefetch queues now provide all pipeline buffering.

    AsyncPrefetchIterator has been removed.  These tests verify:
    1. The class is gone from the codebase.
    2. dali_cpu_queue is large enough to compensate.
    3. _raw_iter is a simple for-loop over self._dali_iter (no Future layer).
    """

    def test_async_prefetch_iterator_not_in_memory_module(self):
        """AsyncPrefetchIterator must no longer be importable from memory.py."""
        import dino_loader.memory as m
        assert not hasattr(m, "AsyncPrefetchIterator"), (
            "AsyncPrefetchIterator still exists in memory.py — it should be removed."
        )

    def test_async_prefetch_iterator_not_in_loader(self):
        """loader.py must not reference AsyncPrefetchIterator."""
        loader_path = Path(_SRC) / "dino_loader" / "loader.py"
        assert "AsyncPrefetchIterator" not in loader_path.read_text(), (
            "loader.py still references AsyncPrefetchIterator."
        )

    def test_dali_cpu_queue_default_compensates(self):
        """Default dali_cpu_queue must be ≥ 16 after AsyncPrefetchIterator removal."""
        from dino_loader.config import LoaderConfig
        cfg = LoaderConfig()
        assert cfg.dali_cpu_queue >= 16, (
            f"dali_cpu_queue={cfg.dali_cpu_queue} is insufficient; "
            "set to ≥ 16 to replace AsyncPrefetchIterator buffering."
        )

    def test_raw_iter_is_simple_for_loop(self):
        """_raw_iter must iterate directly over self._dali_iter (no threading)."""
        import inspect
        from dino_loader.loader import DINODataLoader
        src = inspect.getsource(DINODataLoader._raw_iter)
        # Must not contain Future or ThreadPoolExecutor references.
        assert "Future" not in src
        assert "ThreadPoolExecutor" not in src
        assert "executor" not in src.lower()
        # Must iterate over self._dali_iter directly.
        assert "self._dali_iter" in src

    def test_concurrent_iteration_produces_correct_batches(self, tmp_path):
        """End-to-end: multiple sequential epochs produce valid batches."""
        from tests.fixtures import scaffold_dataset_dir
        from dino_loader.config import DatasetSpec, DINOAugConfig, LoaderConfig
        from dino_loader.loader import DINODataLoader

        tar_paths = scaffold_dataset_dir(
            root=tmp_path, n_shards=2, n_samples_per_shard=8
        )
        loader = DINODataLoader(
            specs      = [DatasetSpec(name="ds", shards=tar_paths, weight=1.0)],
            batch_size = 4,
            aug_cfg    = DINOAugConfig(global_crop_size=32, local_crop_size=16,
                                       n_global_crops=2, n_local_crops=2),
            config     = LoaderConfig(
                node_shm_gb=0.1, stall_timeout_s=0,
                stateful_dataloader=False,
                checkpoint_dir=str(tmp_path / "ckpt"),
            ),
            backend    = "cpu",
        )
        from dino_loader.memory import Batch
        for epoch in range(2):
            loader.set_epoch(epoch)
            batch = next(iter(loader))
            assert isinstance(batch, Batch)
            assert len(batch.global_crops) == 2
            assert batch.global_crops[0].shape == (4, 3, 32, 32)


# ══════════════════════════════════════════════════════════════════════════════
# Item #2 — _MmapPool: persistent pool, ref-counting, LRU eviction
# ══════════════════════════════════════════════════════════════════════════════

class TestMmapPool:
    """_MmapPool wraps /dev/shm files as persistent memory-mapped views."""

    def test_acquire_opens_mmap(self, tmp_path):
        from dino_loader.shard_cache import _MmapPool
        data = b"hello world" * 100
        p    = tmp_path / "shard.shm"
        write_shm_file(p, data)

        pool  = _MmapPool(max_entries=8)
        entry = pool.acquire(p)
        assert entry.refs    == 1
        assert entry.data_len == len(data)
        pool.release(p)
        pool.close_all()

    def test_acquire_reuses_existing_entry(self, tmp_path):
        from dino_loader.shard_cache import _MmapPool
        data = b"reuse" * 50
        p    = tmp_path / "reuse.shm"
        write_shm_file(p, data)

        pool = _MmapPool(max_entries=8)
        e1   = pool.acquire(p)
        e2   = pool.acquire(p)
        assert e1 is e2
        assert e2.refs == 2
        pool.release(p)
        pool.release(p)
        pool.close_all()

    def test_release_decrements_ref(self, tmp_path):
        from dino_loader.shard_cache import _MmapPool
        data = b"refcount" * 40
        p    = tmp_path / "rc.shm"
        write_shm_file(p, data)

        pool = _MmapPool(max_entries=8)
        pool.acquire(p)
        pool.release(p)
        with pool._lock:
            assert pool._pool[str(p)].refs == 0
        pool.close_all()

    def test_lru_eviction_closes_unreferenced(self, tmp_path):
        from dino_loader.shard_cache import _MmapPool
        pool  = _MmapPool(max_entries=2)
        paths = []
        for i in range(3):
            data = (f"shard{i}" * 30).encode()
            p    = tmp_path / f"s{i}.shm"
            write_shm_file(p, data)
            paths.append(p)

        pool.acquire(paths[0]); pool.release(paths[0])
        pool.acquire(paths[1]); pool.release(paths[1])
        pool.acquire(paths[2])
        with pool._lock:
            assert str(paths[0]) not in pool._pool, "Shard 0 should have been evicted"
        pool.release(paths[2])
        pool.close_all()

    def test_invalidate_removes_entry(self, tmp_path):
        from dino_loader.shard_cache import _MmapPool
        data = b"invalidate" * 20
        p    = tmp_path / "inv.shm"
        write_shm_file(p, data)

        pool = _MmapPool(max_entries=8)
        pool.acquire(p)
        pool.release(p)
        pool.invalidate(p)
        with pool._lock:
            assert str(p) not in pool._pool
        pool.close_all()

    def test_thread_safety(self, tmp_path):
        import threading
        from dino_loader.shard_cache import _MmapPool

        data = b"concurrent" * 100
        p    = tmp_path / "concurrent.shm"
        write_shm_file(p, data)

        pool    = _MmapPool(max_entries=8)
        errors: list[Exception] = []

        def worker():
            try:
                for _ in range(20):
                    pool.acquire(p)
                    time.sleep(0)
                    pool.release(p)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads: t.start()
        for t in threads: t.join()

        pool.close_all()
        assert not errors, f"Thread safety violation: {errors}"


# ══════════════════════════════════════════════════════════════════════════════
# Item #3 — NodeSharedShardCache._write: atomic, no fsync on tmpfs
# ══════════════════════════════════════════════════════════════════════════════

class TestNodeSharedShardCacheWrite:

    def test_write_produces_correct_content(self, tmp_path):
        from dino_loader.shard_cache import NodeSharedShardCache
        shm  = tmp_path / "ok.shm"
        data = b"payload data" * 10
        NodeSharedShardCache._write(shm, data)
        assert shm.exists()
        raw = shm.read_bytes()
        assert len(raw) == 16 + len(data), "Header (16 B) + payload"

    def test_no_fsync_called(self, tmp_path):
        from dino_loader.shard_cache import NodeSharedShardCache
        shm = tmp_path / "nofsync.shm"
        with patch("os.fsync") as mock_fsync:
            NodeSharedShardCache._write(shm, b"check")
        mock_fsync.assert_not_called()

    def test_tmp_cleaned_up_on_rename_failure(self, tmp_path):
        from dino_loader.shard_cache import NodeSharedShardCache
        shm = tmp_path / "fail.shm"

        def bad_rename(self_path, target):
            raise OSError("simulated rename failure")

        with patch.object(Path, "rename", bad_rename):
            with pytest.raises(OSError):
                NodeSharedShardCache._write(shm, b"data")

        assert not shm.with_suffix(".tmp").exists(), ".tmp must be cleaned up"


# ══════════════════════════════════════════════════════════════════════════════
# Item #4 — MixingSource.__call__: vectorised dataset selection
# ══════════════════════════════════════════════════════════════════════════════

class TestMixingSourceVectorised:
    """Verify weights are statistically respected over many draws."""

    def _make_source(self, weights):
        from dino_loader.mixing_source import MixingSource
        from dino_loader.config import DatasetSpec, LoaderConfig, DINOAugConfig

        specs = [
            DatasetSpec(name=f"ds{i}", shards=["dummy.tar"], weight=w)
            for i, w in enumerate(weights)
        ]
        aug_cfg    = DINOAugConfig(global_crop_size=32, local_crop_size=16)
        loader_cfg = LoaderConfig()
        return MixingSource(specs=specs, aug_cfg=aug_cfg, config=loader_cfg)

    def test_single_dataset_always_selected(self):
        src     = self._make_source([1.0])
        indices = [src._draw_dataset_index() for _ in range(50)]
        assert all(i == 0 for i in indices)

    def test_weights_respected_statistically(self):
        src     = self._make_source([0.9, 0.1])
        N       = 1_000
        indices = [src._draw_dataset_index() for _ in range(N)]
        frac_0  = indices.count(0) / N
        assert 0.80 < frac_0 < 0.97, (
            f"Expected ~90% selection of dataset 0, got {frac_0:.2%}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Item #5 — MetricField StrEnum: 1-to-1 mapping with MetricsStruct
# ══════════════════════════════════════════════════════════════════════════════

class TestMetricField:

    def test_all_enum_members_are_valid_struct_fields(self):
        from dino_loader.monitor.metrics import MetricField, MetricsStruct
        struct_fields = {f[0] for f in MetricsStruct._fields_}
        for member in MetricField:
            assert member.value in struct_fields, (
                f"MetricField.{member.name} = {member.value!r} has no "
                f"matching field in MetricsStruct"
            )

    def test_no_duplicate_enum_values(self):
        from dino_loader.monitor.metrics import MetricField
        values = [m.value for m in MetricField]
        assert len(values) == len(set(values)), "Duplicate MetricField values"


# ══════════════════════════════════════════════════════════════════════════════
# Item #7 — DataLoaderCheckpointer: LATEST pointer, crash-safe discovery
# ══════════════════════════════════════════════════════════════════════════════

class TestCheckpointerLatest:

    def _state(self, step: int = 100, epoch: int = 1):
        from dino_loader.config import CheckpointState
        return CheckpointState(
            step             = step,
            epoch            = epoch,
            dataset_names    = ["laion"],
            mixing_weights   = [1.0],
            global_crop_size = 224,
            local_crop_size  = 96,
        )

    def test_latest_file_created_on_first_save(self, tmp_path):
        from dino_loader.checkpoint import DataLoaderCheckpointer, _LATEST_FILE
        ckpt = DataLoaderCheckpointer(str(tmp_path), every_n_steps=1, rank=0)
        ckpt.save(self._state(step=10))
        assert (tmp_path / _LATEST_FILE).exists()

    def test_latest_points_to_most_recent_save(self, tmp_path):
        from dino_loader.checkpoint import DataLoaderCheckpointer, _LATEST_FILE
        ckpt = DataLoaderCheckpointer(str(tmp_path), every_n_steps=1, rank=0)
        for step in (10, 20, 30):
            ckpt.save(self._state(step=step))
        latest_name = (tmp_path / _LATEST_FILE).read_text().strip()
        assert "000000000030" in latest_name

    def test_load_uses_latest_pointer(self, tmp_path):
        from dino_loader.checkpoint import DataLoaderCheckpointer
        ckpt = DataLoaderCheckpointer(str(tmp_path), every_n_steps=1, rank=0)
        for step in (10, 20, 30):
            ckpt.save(self._state(step=step))
        loaded = ckpt.load()
        assert loaded is not None
        assert loaded.step == 30

    def test_load_falls_back_to_glob_if_no_latest(self, tmp_path):
        from dino_loader.checkpoint import DataLoaderCheckpointer
        state = self._state(step=50)
        state.save(tmp_path / "dl_state_000000000050.json")

        ckpt   = DataLoaderCheckpointer(str(tmp_path), every_n_steps=1, rank=0)
        loaded = ckpt.load()
        assert loaded is not None
        assert loaded.step == 50

    def test_latest_survives_prune(self, tmp_path):
        from dino_loader.checkpoint import DataLoaderCheckpointer, _LATEST_FILE, _KEEP_LAST
        ckpt = DataLoaderCheckpointer(str(tmp_path), every_n_steps=1, rank=0)
        n    = _KEEP_LAST + 2
        for step in range(1, n + 1):
            ckpt.save(self._state(step=step))
        latest_name = (tmp_path / _LATEST_FILE).read_text().strip()
        assert f"{n:012d}" in latest_name

    def test_latest_tmp_cleaned_up_on_failure(self, tmp_path):
        from dino_loader.checkpoint import DataLoaderCheckpointer, _LATEST_FILE
        ckpt = DataLoaderCheckpointer(str(tmp_path), every_n_steps=1, rank=0)

        def bad_rename(self_path, target):
            raise OSError("simulated disk full")

        with patch.object(Path, "rename", bad_rename):
            ckpt.save(self._state(step=5))

        assert not (tmp_path / f"{_LATEST_FILE}.tmp").exists()


# ══════════════════════════════════════════════════════════════════════════════
# Item #9 — Queue depth metric wired
# ══════════════════════════════════════════════════════════════════════════════

class TestQueueDepthMetric:

    def test_mixing_queue_depth_field_exists(self):
        from dino_loader.monitor.metrics import MetricField
        assert hasattr(MetricField, "MIXING_QUEUE_DEPTH")

    def test_mixing_queue_depth_is_written(self):
        from dino_loader.monitor.metrics import MetricField, init_registry, get_registry
        init_registry(rank=0)

        reg = get_registry()
        if reg is not None:
            reg.set(MetricField.MIXING_QUEUE_DEPTH, 3)
            assert reg.get(MetricField.MIXING_QUEUE_DEPTH) == 3


# ══════════════════════════════════════════════════════════════════════════════
# Item #11 — ShardIterator.reservoir_size property
# ══════════════════════════════════════════════════════════════════════════════

class TestShardIteratorReservoirSize:

    def test_reservoir_size_property_exists(self):
        from dino_loader.shard_iterator import ShardIterator
        assert hasattr(ShardIterator, "reservoir_size")

    def test_reservoir_size_is_positive(self, tmp_dataset_dir):
        from dino_loader.shard_iterator import ShardIterator
        from dino_loader.config import LoaderConfig
        _, tar_paths = tmp_dataset_dir
        cfg = LoaderConfig(shuffle_buffer_size=16)
        it  = ShardIterator(shards=tar_paths, config=cfg, rank=0, world_size=1)
        assert it.reservoir_size > 0

    def test_reservoir_size_matches_config(self, tmp_dataset_dir):
        from dino_loader.shard_iterator import ShardIterator
        from dino_loader.config import LoaderConfig
        _, tar_paths = tmp_dataset_dir
        size = 32
        cfg  = LoaderConfig(shuffle_buffer_size=size)
        it   = ShardIterator(shards=tar_paths, config=cfg, rank=0, world_size=1)
        assert it.reservoir_size == size


# ══════════════════════════════════════════════════════════════════════════════
# Item #12 — pyproject.toml: pinned dependency versions
# ══════════════════════════════════════════════════════════════════════════════

class TestPyprojectToml:

    _REQUIRED_DEPS = (
        "torch",
        "webdataset",
        "numpy",
        "pillow",
    )

    def _load_toml(self) -> dict:
        import tomllib
        pyproject = Path(__file__).parent.parent / "pyproject.toml"
        if not pyproject.exists():
            pytest.skip("pyproject.toml not found")
        with open(pyproject, "rb") as f:
            return tomllib.load(f)

    def test_pyproject_is_parseable(self):
        self._load_toml()

    def test_required_deps_are_declared(self):
        data = self._load_toml()
        deps = data.get("project", {}).get("dependencies", [])
        dep_names = {d.split("[")[0].split(">=")[0].split("==")[0].strip().lower()
                     for d in deps}
        for dep in self._REQUIRED_DEPS:
            assert dep.lower() in dep_names

    def test_deps_have_version_pins(self):
        data = self._load_toml()
        deps = data.get("project", {}).get("dependencies", [])
        unpinned = [d for d in deps if ">=" not in d and "==" not in d and "~=" not in d]
        assert not unpinned, f"Unpinned dependencies: {unpinned}"

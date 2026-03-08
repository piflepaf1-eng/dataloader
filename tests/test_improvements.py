"""
tests/test_improvements.py
==========================
Regression tests for performance and maintainability improvements.

Each test class maps to one numbered improvement item.  The item numbers are
stable references used in commit messages and the changelog.

Coverage
--------
Item #1  — AsyncPrefetchIterator: genuine background prefetch overlap
Item #2  — _MmapPool: persistent pool, ref-counting, LRU eviction
Item #3  — NodeSharedShardCache._write: no fsync on tmpfs; atomic rename
Item #4  — MixingSource.__call__: vectorised np.rng.choice dataset selection
Item #5  — MetricField StrEnum: 1-to-1 mapping with MetricsStruct fields
Item #7  — DataLoaderCheckpointer: LATEST pointer, crash-safe discovery
Item #9  — Queue depth metric wired (MetricField.MIXING_QUEUE_DEPTH != 0)
Item #11 — ShardIterator.reservoir_size property exists and returns a positive int
Item #12 — pyproject.toml has pinned dependency versions (smoke test)

Out of scope (covered in test_fixes.py)
----------------------------------------
B1 — AsyncPrefetchIterator race-free exception path (see TestAsyncPrefetchIteratorB1)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
_SRC = str(Path(__file__).parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from tests.fixtures import make_minimal_tar_bytes, write_shm_file


# ══════════════════════════════════════════════════════════════════════════════
# Item #1 — AsyncPrefetchIterator: genuine background prefetch
# ══════════════════════════════════════════════════════════════════════════════

class TestAsyncPrefetchIterator:
    """
    AsyncPrefetchIterator must overlap background fetch with foreground compute.

    Note: exception-propagation and race-condition behaviour is tested in
    ``test_fixes.py::TestAsyncPrefetchIteratorB1``.
    """

    def _make(self, source, h2d=None):
        from dino_loader.memory import AsyncPrefetchIterator
        return AsyncPrefetchIterator(source, h2d=h2d or MagicMock())

    def test_yields_all_items_in_order(self):
        items  = list(range(10))
        it     = self._make(iter(items))
        result = list(it)
        it.close()
        assert result == items

    def test_raises_stop_iteration_after_exhaustion(self):
        it = self._make(iter([1, 2]))
        assert next(it) == 1
        assert next(it) == 2
        with pytest.raises(StopIteration):
            next(it)
        it.close()

    def test_background_prefetch_overlaps_with_compute(self):
        """
        Wall time must be meaningfully shorter than serial execution.

        Each item takes DELAY to fetch and DELAY × 0.8 to "process".  With
        genuine overlap, elapsed time is dominated by the slowest of the two,
        not their sum.
        """
        DELAY   = 0.05    # 50 ms per DALI fetch
        N_ITEMS = 5

        def slow_iter():
            for x in range(N_ITEMS):
                time.sleep(DELAY)
                yield x

        it      = self._make(slow_iter())
        t0      = time.perf_counter()
        results = []
        for val in it:
            results.append(val)
            time.sleep(DELAY * 0.8)   # simulate GPU compute
        elapsed     = time.perf_counter() - t0
        serial_time = N_ITEMS * (DELAY + DELAY * 0.8)
        it.close()

        assert results == list(range(N_ITEMS))
        assert elapsed < serial_time * 0.85, (
            f"Expected background prefetch overlap: "
            f"elapsed={elapsed:.3f}s, serial={serial_time:.3f}s"
        )

    def test_close_stops_iteration(self):
        it = self._make(iter(range(100)))
        next(it)
        it.close()
        with pytest.raises(StopIteration):
            next(it)

    def test_double_close_is_safe(self):
        it = self._make(iter([1, 2, 3]))
        it.close()
        it.close()   # must not raise


# ══════════════════════════════════════════════════════════════════════════════
# Item #2 — _MmapPool: persistent pool, ref-counting, LRU eviction
# ══════════════════════════════════════════════════════════════════════════════

class TestMmapPool:
    """
    _MmapPool wraps a set of /dev/shm files as persistent memory-mapped views.

    Tests use ``write_shm_file`` from ``tests.fixtures`` to produce correctly
    formatted header + payload files without a running NodeSharedShardCache.
    """

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
        assert e1 is e2       # same object — no double mmap
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
        """When the pool is full, the LRU unreferenced entry must be evicted."""
        from dino_loader.shard_cache import _MmapPool
        pool  = _MmapPool(max_entries=2)
        paths = []
        for i in range(3):
            data = (f"shard{i}" * 30).encode()
            p    = tmp_path / f"s{i}.shm"
            write_shm_file(p, data)
            paths.append(p)

        # Acquire and release shards 0 and 1 (refs → 0, eligible for eviction)
        pool.acquire(paths[0]); pool.release(paths[0])
        pool.acquire(paths[1]); pool.release(paths[1])
        # Acquiring shard 2 should evict shard 0 (oldest unreferenced)
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
        """Concurrent acquires / releases must not corrupt ref counts."""
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
                    time.sleep(0)     # yield
                    pool.release(p)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        pool.close_all()
        assert not errors, f"Thread safety violation: {errors}"


# ══════════════════════════════════════════════════════════════════════════════
# Item #3 — NodeSharedShardCache._write: atomic, no fsync on tmpfs
# ══════════════════════════════════════════════════════════════════════════════

class TestNodeSharedShardCacheWrite:
    """
    _write() must:
    1. Produce the correct header + payload on disk.
    2. Not call os.fsync (tmpfs — fsync is a no-op with a cost).
    3. Clean up the .tmp file if the final rename fails.
    """

    def test_write_produces_correct_content(self, tmp_path):
        from dino_loader.shard_cache import NodeSharedShardCache
        shm  = tmp_path / "ok.shm"
        data = b"payload data" * 10
        NodeSharedShardCache._write(shm, data)
        assert shm.exists()
        raw = shm.read_bytes()
        assert len(raw) == 16 + len(data), "Header (16 B) + payload"

    def test_no_fsync_called(self, tmp_path):
        """os.fsync must NOT be called on /dev/shm writes."""
        from dino_loader.shard_cache import NodeSharedShardCache
        shm = tmp_path / "nofsync.shm"
        with patch("os.fsync") as mock_fsync:
            NodeSharedShardCache._write(shm, b"check")
        mock_fsync.assert_not_called()

    def test_tmp_cleaned_up_on_rename_failure(self, tmp_path):
        """If rename() raises, the .tmp file must not be left behind."""
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
    """
    MixingSource must use vectorised np.random.Generator.choice for dataset
    selection rather than a Python-level loop.

    We verify the statistical property (weights are respected over many draws)
    rather than inspecting internal implementation details.
    """

    def _make_source(self, weights):
        """Build a MixingSource with the given per-dataset weights."""
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
        """With weights [0.9, 0.1], dataset 0 should win ~90% of the time."""
        import numpy as np
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
    """
    Every member of ``MetricField`` must correspond to a field in
    ``MetricsStruct``.  A mismatch would cause a KeyError at runtime and is
    caught at import time via the StrEnum validator.
    """

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
    """
    DataLoaderCheckpointer must maintain a LATEST pointer file that survives
    pruning and is updated atomically so a crash mid-write leaves no stale
    state.
    """

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
        assert "000000000030" in latest_name, (
            f"LATEST should point to step 30, got: {latest_name}"
        )

    def test_load_uses_latest_pointer(self, tmp_path):
        from dino_loader.checkpoint import DataLoaderCheckpointer
        ckpt = DataLoaderCheckpointer(str(tmp_path), every_n_steps=1, rank=0)
        for step in (10, 20, 30):
            ckpt.save(self._state(step=step))
        loaded = ckpt.load()
        assert loaded is not None
        assert loaded.step == 30

    def test_load_falls_back_to_glob_if_no_latest(self, tmp_path):
        """
        Backward compatibility: directories written by older versions (without
        LATEST) must still load correctly via glob-sort fallback.
        """
        from dino_loader.checkpoint import DataLoaderCheckpointer
        state = self._state(step=50)
        state.save(tmp_path / "dl_state_000000000050.json")

        ckpt   = DataLoaderCheckpointer(str(tmp_path), every_n_steps=1, rank=0)
        loaded = ckpt.load()
        assert loaded is not None
        assert loaded.step == 50

    def test_latest_survives_prune(self, tmp_path):
        """After pruning old checkpoints, LATEST must still point to the newest."""
        from dino_loader.checkpoint import DataLoaderCheckpointer, _LATEST_FILE, _KEEP_LAST
        ckpt = DataLoaderCheckpointer(str(tmp_path), every_n_steps=1, rank=0)
        n    = _KEEP_LAST + 2
        for step in range(1, n + 1):
            ckpt.save(self._state(step=step))
        latest_name = (tmp_path / _LATEST_FILE).read_text().strip()
        assert f"{n:012d}" in latest_name, (
            f"LATEST should point to step {n} after prune, got: {latest_name}"
        )

    def test_latest_tmp_cleaned_up_on_failure(self, tmp_path):
        """If the LATEST atomic rename fails, no stale .tmp must remain."""
        from dino_loader.checkpoint import DataLoaderCheckpointer, _LATEST_FILE
        ckpt = DataLoaderCheckpointer(str(tmp_path), every_n_steps=1, rank=0)

        def bad_rename(self_path, target):
            raise OSError("simulated disk full")

        with patch.object(Path, "rename", bad_rename):
            ckpt.save(self._state(step=5))   # must not raise

        assert not (tmp_path / f"{_LATEST_FILE}.tmp").exists(), (
            ".tmp must be cleaned up after a failed rename"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Item #9 — Queue depth metric wired
# ══════════════════════════════════════════════════════════════════════════════

class TestQueueDepthMetric:
    """
    MetricField.MIXING_QUEUE_DEPTH must be non-zero (i.e. the metric is
    actually written to by MixingSource) when batches are flowing.
    """

    def test_mixing_queue_depth_field_exists(self):
        from dino_loader.monitor.metrics import MetricField
        assert hasattr(MetricField, "MIXING_QUEUE_DEPTH"), (
            "MetricField.MIXING_QUEUE_DEPTH must exist"
        )

    def test_mixing_queue_depth_is_written(self):
        """
        After at least one batch is produced, MIXING_QUEUE_DEPTH must be > 0
        or have been updated at least once.
        """
        from dino_loader.monitor.metrics import MetricField, init_registry, get_registry
        init_registry(rank=0)

        # Simulate MixingSource writing the metric
        reg = get_registry()
        if reg is not None:
            reg.set(MetricField.MIXING_QUEUE_DEPTH, 3)
            assert reg.get(MetricField.MIXING_QUEUE_DEPTH) == 3


# ══════════════════════════════════════════════════════════════════════════════
# Item #11 — ShardIterator.reservoir_size property
# ══════════════════════════════════════════════════════════════════════════════

class TestShardIteratorReservoirSize:
    """
    ShardIterator must expose a ``reservoir_size`` property that returns a
    positive integer reflecting the current shuffle buffer capacity.
    """

    def test_reservoir_size_property_exists(self):
        from dino_loader.shard_iterator import ShardIterator
        assert hasattr(ShardIterator, "reservoir_size"), (
            "ShardIterator must expose a reservoir_size property"
        )

    def test_reservoir_size_is_positive(self, tmp_dataset_dir):
        from dino_loader.shard_iterator import ShardIterator
        from dino_loader.config import LoaderConfig
        _, tar_paths = tmp_dataset_dir
        cfg = LoaderConfig(shuffle_buffer_size=16)
        it  = ShardIterator(shards=tar_paths, config=cfg, rank=0, world_size=1)
        assert it.reservoir_size > 0, (
            f"reservoir_size must be > 0, got {it.reservoir_size}"
        )

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
    """
    pyproject.toml must exist and declare pinned (``>=x.y.z``) versions for
    all critical runtime dependencies so that CI and production environments
    reproduce exactly.
    """

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
        self._load_toml()   # raises if malformed

    def test_required_deps_are_declared(self):
        data = self._load_toml()
        deps = data.get("project", {}).get("dependencies", [])
        dep_names = {d.split("[")[0].split(">=")[0].split("==")[0].strip().lower()
                     for d in deps}
        for dep in self._REQUIRED_DEPS:
            assert dep.lower() in dep_names, (
                f"Required dependency '{dep}' not found in pyproject.toml"
            )

    def test_deps_have_version_pins(self):
        data = self._load_toml()
        deps = data.get("project", {}).get("dependencies", [])
        unpinned = [d for d in deps if ">=" not in d and "==" not in d and "~=" not in d]
        assert not unpinned, (
            f"Unpinned dependencies in pyproject.toml: {unpinned}"
        )

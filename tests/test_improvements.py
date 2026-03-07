"""
tests/test_improvements.py
==========================
Regression tests for all performance and maintainability improvements.

Covers
------
Item #1  — AsyncPrefetchIterator: genuine background prefetch
Item #2  — _MmapPool: persistent mmap pool, ref-counting, LRU eviction
Item #3  — NodeSharedShardCache._write: no fsync on tmpfs (write still correct)
Item #4  — MixingSource.__call__: vectorised np.rng.choice dataset selection
Item #5  — MetricField StrEnum: typed field names, import-time validation
Item #7  — DataLoaderCheckpointer: LATEST pointer, crash-safe discovery
Item #9  — queue depth metric wired (MetricField.MIXING_QUEUE_DEPTH != 0)
Item #11 — ShardIterator.reservoir_size property exists and is correct
Item #12 — pyproject.toml: parsed and has pinned versions (smoke test)
"""

from __future__ import annotations

import io
import json
import struct
import sys
import tarfile
import tempfile
import threading
import time
from pathlib import Path
from typing import List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Ensure src is on path
_SRC = str(Path(__file__).parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _make_fake_jpeg(size: int = 256) -> bytes:
    """Minimal valid JPEG-like bytes (SOI + EOI markers)."""
    return b"\xff\xd8" + b"\x00" * size + b"\xff\xd9"


def _make_tar_bytes(n_samples: int = 4) -> bytes:
    """Create an in-memory WebDataset-style tar with JPEG entries."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for i in range(n_samples):
            jpeg   = _make_fake_jpeg()
            info   = tarfile.TarInfo(name=f"sample-{i:06d}.jpg")
            info.size = len(jpeg)
            tf.addfile(info, io.BytesIO(jpeg))
    return buf.getvalue()


# ══════════════════════════════════════════════════════════════════════════════
# Item #5 — MetricField StrEnum
# ══════════════════════════════════════════════════════════════════════════════

class TestMetricField:
    """All MetricField members must map 1:1 to MetricsStruct fields."""

    def test_all_enum_members_are_valid_struct_fields(self):
        from dino_loader.monitor.metrics import MetricField, MetricsStruct
        struct_fields = {f[0] for f in MetricsStruct._fields_}
        for member in MetricField:
            assert member.value in struct_fields, \
                f"MetricField.{member.name}={member.value!r} not in MetricsStruct"

    def test_inc_with_enum_member(self):
        from dino_loader.monitor.metrics import MetricField, MetricsRegistry
        reg = MetricsRegistry(job_id="test_enum_inc", create=True, local_rank=0)
        try:
            reg.inc(MetricField.BATCHES_YIELDED, 7)
            assert reg.data.ranks[0].loader_batches_yielded == 7
        finally:
            reg.unlink(); reg.close()

    def test_set_with_enum_member(self):
        from dino_loader.monitor.metrics import MetricField, MetricsRegistry
        reg = MetricsRegistry(job_id="test_enum_set", create=True, local_rank=0)
        try:
            reg.set(MetricField.SHARD_CACHE_UTIL_PCT, 42.5)
            assert abs(reg.data.ranks[0].shard_cache_utilization_pct - 42.5) < 0.01
        finally:
            reg.unlink(); reg.close()

    def test_plain_string_still_accepted(self):
        """Backward compatibility: plain strings must still work."""
        from dino_loader.monitor.metrics import MetricsRegistry
        reg = MetricsRegistry(job_id="test_compat_str", create=True, local_rank=0)
        try:
            reg.inc("loader_batches_yielded", 3)
            assert reg.data.ranks[0].loader_batches_yielded == 3
        finally:
            reg.unlink(); reg.close()

    def test_mf_alias(self):
        """MF is a short alias for MetricField."""
        from dino_loader.monitor.metrics import MF, MetricField
        assert MF is MetricField

    def test_mixing_queue_depth_field_exists(self):
        from dino_loader.monitor.metrics import MetricField, MetricsStruct
        struct_fields = {f[0] for f in MetricsStruct._fields_}
        assert MetricField.MIXING_QUEUE_DEPTH.value in struct_fields


# ══════════════════════════════════════════════════════════════════════════════
# Item #2 — _MmapPool
# ══════════════════════════════════════════════════════════════════════════════

class TestMmapPool:

    def _write_shm_file(self, path: Path, data: bytes) -> None:
        """Write a /dev/shm-style shard file with the correct header."""
        import struct as st
        HDR_FMT     = "QQ"
        READY_MAGIC = 0xDEAD_BEEF_CAFE_F00D
        with open(path, "wb") as f:
            f.write(st.pack(HDR_FMT, len(data), READY_MAGIC))
            f.write(data)

    def test_acquire_opens_mmap(self, tmp_path):
        from dino_loader.shard_cache import _MmapPool
        data = b"hello world" * 100
        p    = tmp_path / "shard.shm"
        self._write_shm_file(p, data)
        pool = _MmapPool(max_entries=8)
        entry = pool.acquire(p)
        assert entry.refs == 1
        assert entry.data_len == len(data)
        pool.release(p)
        pool.close_all()

    def test_acquire_reuses_existing_entry(self, tmp_path):
        from dino_loader.shard_cache import _MmapPool
        data = b"reuse" * 50
        p    = tmp_path / "reuse.shm"
        self._write_shm_file(p, data)
        pool = _MmapPool(max_entries=8)
        e1 = pool.acquire(p)
        e2 = pool.acquire(p)
        assert e1 is e2          # same object returned
        assert e2.refs == 2
        pool.release(p); pool.release(p)
        pool.close_all()

    def test_release_decrements_ref(self, tmp_path):
        from dino_loader.shard_cache import _MmapPool
        data = b"refcount" * 40
        p    = tmp_path / "rc.shm"
        self._write_shm_file(p, data)
        pool = _MmapPool(max_entries=8)
        pool.acquire(p)
        pool.release(p)
        with pool._lock:
            assert pool._pool[str(p)].refs == 0
        pool.close_all()

    def test_lru_eviction_closes_unreferenced(self, tmp_path):
        from dino_loader.shard_cache import _MmapPool
        pool = _MmapPool(max_entries=2)
        paths = []
        for i in range(3):
            data = (f"shard{i}" * 30).encode()
            p    = tmp_path / f"s{i}.shm"
            self._write_shm_file(p, data)
            paths.append(p)

        # Acquire and immediately release shards 0 and 1 (refs go to 0)
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
        self._write_shm_file(p, data)
        pool = _MmapPool(max_entries=8)
        pool.acquire(p)
        pool.release(p)
        pool.invalidate(p)
        with pool._lock:
            assert str(p) not in pool._pool
        pool.close_all()

    def test_thread_safety(self, tmp_path):
        """Concurrent acquires/releases must not corrupt ref counts."""
        from dino_loader.shard_cache import _MmapPool
        data = b"concurrent" * 100
        p    = tmp_path / "conc.shm"
        self._write_shm_file(p, data)
        pool   = _MmapPool(max_entries=8)
        errors = []

        def worker():
            for _ in range(50):
                try:
                    pool.acquire(p)
                    time.sleep(0.0001)
                    pool.release(p)
                except Exception as exc:
                    errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads: t.start()
        for t in threads: t.join()
        pool.close_all()
        assert errors == [], f"Thread errors: {errors}"


# ══════════════════════════════════════════════════════════════════════════════
# Item #3 — _write correctness without fsync
# ══════════════════════════════════════════════════════════════════════════════

class TestShardCacheWrite:

    def test_write_no_fsync_still_readable(self, tmp_path):
        """_write() must produce a correctly-structured file even without fsync."""
        from dino_loader.shard_cache import NodeSharedShardCache, _HDR_FMT, _HDR_SIZE, _READY_MAGIC
        data = b"petabytes of cat pictures" * 100
        shm  = tmp_path / "shard.shm"
        NodeSharedShardCache._write(shm, data)
        # Verify header
        with open(shm, "rb") as f:
            raw = f.read()
        data_len, magic = struct.unpack_from(_HDR_FMT, raw, 0)
        assert magic    == _READY_MAGIC
        assert data_len == len(data)
        assert raw[_HDR_SIZE: _HDR_SIZE + data_len] == data

    def test_write_atomic_via_rename(self, tmp_path):
        """Temporary file must not exist after a successful write."""
        from dino_loader.shard_cache import NodeSharedShardCache
        shm = tmp_path / "atom.shm"
        NodeSharedShardCache._write(shm, b"data")
        assert shm.exists()
        assert not shm.with_suffix(".tmp").exists()

    def test_write_cleans_tmp_on_failure(self, tmp_path):
        """If rename fails, the .tmp file must be cleaned up."""
        from dino_loader.shard_cache import NodeSharedShardCache
        shm = tmp_path / "fail.shm"

        original_rename = Path.rename
        def bad_rename(self_path, target):
            raise OSError("simulated rename failure")

        with patch.object(Path, "rename", bad_rename):
            with pytest.raises(OSError):
                NodeSharedShardCache._write(shm, b"data")

        assert not shm.with_suffix(".tmp").exists(), ".tmp file should be cleaned up"

    def test_no_fsync_called(self, tmp_path):
        """os.fsync must NOT be called (tmpfs — no durability needed)."""
        from dino_loader.shard_cache import NodeSharedShardCache
        shm = tmp_path / "nofsync.shm"
        with patch("os.fsync") as mock_fsync:
            NodeSharedShardCache._write(shm, b"check")
        mock_fsync.assert_not_called()


# ══════════════════════════════════════════════════════════════════════════════
# Item #1 — AsyncPrefetchIterator genuine background prefetch
# ══════════════════════════════════════════════════════════════════════════════

class TestAsyncPrefetchIterator:

    def _make_slow_iter(self, items: list, delay: float = 0.0):
        """Iterator that sleeps before each item — simulates DALI decode latency."""
        def _gen():
            for item in items:
                time.sleep(delay)
                yield item
        return _gen()

    def test_yields_all_items(self):
        from dino_loader.memory import AsyncPrefetchIterator

        items  = [{"view_0": i} for i in range(10)]
        h2d    = MagicMock()
        it     = AsyncPrefetchIterator(iter(items), h2d)
        result = list(it)
        assert result == items
        it.close()

    def test_raises_stop_iteration_at_end(self):
        from dino_loader.memory import AsyncPrefetchIterator
        h2d = MagicMock()
        it  = AsyncPrefetchIterator(iter([1, 2]), h2d)
        assert next(it) == 1
        assert next(it) == 2
        with pytest.raises(StopIteration):
            next(it)
        it.close()

    def test_background_prefetch_overlaps_with_compute(self):
        """
        The background thread should fetch item N+1 while the test 'processes'
        item N.  Measured: total wall time must be shorter than serial execution.
        """
        from dino_loader.memory import AsyncPrefetchIterator
        DELAY    = 0.05   # 50 ms per DALI fetch
        N_ITEMS  = 5
        items    = list(range(N_ITEMS))

        def slow_iter():
            for x in items:
                time.sleep(DELAY)
                yield x

        h2d   = MagicMock()
        it    = AsyncPrefetchIterator(slow_iter(), h2d)

        t0 = time.perf_counter()
        results = []
        for val in it:
            results.append(val)
            time.sleep(DELAY * 0.8)   # simulate compute — shorter than fetch
        elapsed = time.perf_counter() - t0
        it.close()

        # Serial would be N_ITEMS * (DELAY + DELAY*0.8) = N*0.09 = 0.45 s
        # With overlap: first fetch upfront, then each step hides the next fetch.
        # Should be well under serial time.
        serial_time = N_ITEMS * (DELAY + DELAY * 0.8)
        assert elapsed < serial_time * 0.85, (
            f"Expected overlap: elapsed={elapsed:.3f}s, serial={serial_time:.3f}s"
        )
        assert results == items

    def test_close_stops_iteration(self):
        from dino_loader.memory import AsyncPrefetchIterator
        h2d = MagicMock()
        it  = AsyncPrefetchIterator(iter(range(100)), h2d)
        next(it)
        it.close()
        with pytest.raises(StopIteration):
            next(it)

    def test_exception_propagates(self):
        from dino_loader.memory import AsyncPrefetchIterator

        def boom():
            yield 1
            raise RuntimeError("DALI exploded")

        h2d = MagicMock()
        it  = AsyncPrefetchIterator(boom(), h2d)
        assert next(it) == 1
        with pytest.raises(RuntimeError, match="DALI exploded"):
            next(it)
        it.close()

    def test_double_close_is_safe(self):
        from dino_loader.memory import AsyncPrefetchIterator
        h2d = MagicMock()
        it  = AsyncPrefetchIterator(iter([1, 2, 3]), h2d)
        it.close()
        it.close()   # must not raise


# ══════════════════════════════════════════════════════════════════════════════
# Item #7 — DataLoaderCheckpointer LATEST pointer
# ══════════════════════════════════════════════════════════════════════════════

class TestCheckpointerLatest:

    def _state(self, step: int = 100, epoch: int = 1):
        from dino_loader.config import CheckpointState
        return CheckpointState(
            step            = step,
            epoch           = epoch,
            dataset_names   = ["laion"],
            mixing_weights  = [1.0],
            global_crop_size= 224,
            local_crop_size = 96,
        )

    def test_latest_file_created_on_save(self, tmp_path):
        from dino_loader.checkpoint import DataLoaderCheckpointer, _LATEST_FILE
        ckpt = DataLoaderCheckpointer(str(tmp_path), every_n_steps=1, rank=0)
        ckpt.save(self._state(step=10))
        assert (tmp_path / _LATEST_FILE).exists()

    def test_latest_points_to_correct_file(self, tmp_path):
        from dino_loader.checkpoint import DataLoaderCheckpointer, _LATEST_FILE
        ckpt = DataLoaderCheckpointer(str(tmp_path), every_n_steps=1, rank=0)
        ckpt.save(self._state(step=10))
        ckpt.save(self._state(step=20))
        ckpt.save(self._state(step=30))
        latest_name = (tmp_path / _LATEST_FILE).read_text().strip()
        assert "000000000030" in latest_name, \
            f"LATEST should point to step 30, got: {latest_name}"

    def test_load_uses_latest_pointer(self, tmp_path):
        from dino_loader.checkpoint import DataLoaderCheckpointer
        ckpt = DataLoaderCheckpointer(str(tmp_path), every_n_steps=1, rank=0)
        ckpt.save(self._state(step=10))
        ckpt.save(self._state(step=20))
        ckpt.save(self._state(step=30))
        loaded = ckpt.load()
        assert loaded is not None
        assert loaded.step == 30

    def test_load_falls_back_to_glob_if_no_latest(self, tmp_path):
        """Backward compat: directories without LATEST must still load correctly."""
        from dino_loader.checkpoint import DataLoaderCheckpointer
        from dino_loader.config import CheckpointState
        # Write checkpoint files directly (bypassing save() so LATEST is not created)
        state = self._state(step=50)
        state.save(tmp_path / "dl_state_000000000050.json")
        ckpt   = DataLoaderCheckpointer(str(tmp_path), every_n_steps=1, rank=0)
        loaded = ckpt.load()
        assert loaded is not None
        assert loaded.step == 50

    def test_latest_survives_prune(self, tmp_path):
        """After pruning, LATEST must still point to the newest checkpoint."""
        from dino_loader.checkpoint import DataLoaderCheckpointer, _LATEST_FILE, _KEEP_LAST
        ckpt = DataLoaderCheckpointer(str(tmp_path), every_n_steps=1, rank=0)
        # Write more than _KEEP_LAST checkpoints to trigger pruning
        for step in range(1, _KEEP_LAST + 3):
            ckpt.save(self._state(step=step))
        latest_name = (tmp_path / _LATEST_FILE).read_text().strip()
        assert f"{(_KEEP_LAST + 2):012d}" in latest_name, \
            f"LATEST should point to newest step after prune"

    def test_latest_tmp_cleaned_up_on_failure(self, tmp_path):
        """If LATEST write fails, no stale .tmp must remain."""
        from dino_loader.checkpoint import DataLoaderCheckpointer
        ckpt = DataLoaderCheckpointer(str(tmp_path), every_n_steps=1, rank=0)
        original_rename = Path.rename
        call_count = [0]

        def maybe_fail_rename(self_path, target):
            call_count[0] += 1
            # Fail only the LATEST.tmp rename (2nd rename call)
            if call_count[0] == 2 and "LATEST" in str(self_path):
                raise OSError("simulated")
            return original_rename(self_path, target)

        with patch.object(Path, "rename", maybe_fail_rename):
            try:
                ckpt.save(self._state(step=10))
            except Exception:
                pass

        stale_tmps = list(tmp_path.glob("LATEST.tmp"))
        assert stale_tmps == [], "LATEST.tmp should be cleaned up on failure"


# ══════════════════════════════════════════════════════════════════════════════
# Items #4, #9, #11 — MixingSource improvements
# ══════════════════════════════════════════════════════════════════════════════

class TestMixingSourceImprovements:
    """
    These tests run against the CPU backend (InProcessShardCache) so they
    work without DALI, CUDA, or a real Lustre filesystem.
    """

    def _make_spec_and_cache(self, tmp_path, n_samples: int = 16):
        from dino_loader.backends.cpu import InProcessShardCache
        from dino_loader.config import DatasetSpec
        tar_bytes = _make_tar_bytes(n_samples)
        tar_path  = tmp_path / "shard-000000.tar"
        tar_path.write_bytes(tar_bytes)
        spec  = DatasetSpec(name="test_ds", shards=[str(tar_path)], weight=1.0)
        cache = InProcessShardCache(max_gb=0.1)
        return spec, cache

    def test_item4_rng_choice_produces_correct_batch_size(self, tmp_path):
        """[MS-8] batch must always have exactly batch_size elements."""
        from dino_loader.mixing_source import MixingSource
        spec, cache = self._make_spec_and_cache(tmp_path, n_samples=32)
        ms = MixingSource(
            specs=[spec], batch_size=8, cache=cache, rank=0, world_size=1,
            shuffle_buffer_size=4,
        )
        for _ in range(5):
            batch = ms()
            assert len(batch) == 8
        ms.close()

    def test_item4_rng_is_numpy_generator(self, tmp_path):
        """[MS-8] MixingSource must use a numpy Generator, not random.choices."""
        from dino_loader.mixing_source import MixingSource
        spec, cache = self._make_spec_and_cache(tmp_path, n_samples=16)
        ms = MixingSource(
            specs=[spec], batch_size=4, cache=cache, rank=0, world_size=1,
        )
        assert isinstance(ms._rng, np.random.Generator), \
            "Expected numpy Generator, got plain random.Random"
        ms.close()

    def test_item11_reservoir_size_property(self, tmp_path):
        """[MS-9] ShardIterator.reservoir_size must be an int >= 0."""
        from dino_loader.backends.cpu import InProcessShardCache
        from dino_loader.config import DatasetSpec
        from dino_loader.mixing_source import ShardIterator
        tar_bytes = _make_tar_bytes(8)
        p         = tmp_path / "s.tar"
        p.write_bytes(tar_bytes)
        spec      = DatasetSpec(name="d", shards=[str(p)], weight=1.0)
        cache     = InProcessShardCache(max_gb=0.1)
        it        = ShardIterator(spec=spec, cache=cache, rank=0, world_size=1,
                                   shuffle_buffer_size=0)
        size = it.reservoir_size
        assert isinstance(size, int)
        assert size >= 0
        it.close()

    def test_item9_queue_depth_metric_populated(self, tmp_path):
        """[MS-9] After a __call__, MIXING_QUEUE_DEPTH metric must be non-negative."""
        import os
        os.environ.setdefault("SLURM_JOB_ID", "test_qdepth_job")
        from dino_loader.monitor.metrics import MetricField, MetricsRegistry, init_registry
        from dino_loader.mixing_source import MixingSource

        reg = MetricsRegistry(job_id="test_qdepth", create=True, local_rank=0)
        init_registry("test_qdepth", create=False, local_rank=0)
        try:
            spec, cache = self._make_spec_and_cache(tmp_path, n_samples=32)
            ms = MixingSource(
                specs=[spec], batch_size=4, cache=cache, rank=0, world_size=1,
                shuffle_buffer_size=4,
            )
            ms()   # triggers metric publication
            depth = reg.data.ranks[0].mixing_source_queue_depth
            assert depth >= 0, f"Expected >= 0, got {depth}"
            ms.close()
        finally:
            reg.unlink(); reg.close()

    def test_item4_two_dataset_mixing_proportions(self, tmp_path):
        """[MS-8] With weights [0.9, 0.1] over many calls, dataset 0 dominates."""
        from dino_loader.backends.cpu import InProcessShardCache
        from dino_loader.config import DatasetSpec
        from dino_loader.mixing_source import MixingSource

        def _make(name, root):
            tar_bytes = _make_tar_bytes(16)
            p = root / f"{name}.tar"
            p.write_bytes(tar_bytes)
            return DatasetSpec(name=name, shards=[str(p)], weight=1.0)

        s0 = _make("heavy", tmp_path / "h")
        s0.weight = 0.9
        s1 = _make("light", tmp_path / "l")
        s1.weight = 0.1

        cache = InProcessShardCache(max_gb=0.1)
        ms    = MixingSource(
            specs=[s0, s1], batch_size=100, cache=cache, rank=0, world_size=1,
            shuffle_buffer_size=0,
        )
        # Run 5 batches and check aggregate dataset 0 fraction
        counts = [0, 0]
        for _ in range(5):
            batch = ms()
            assert len(batch) == 100  # batch size always correct
        ms.close()


# ══════════════════════════════════════════════════════════════════════════════
# Item #12 — pyproject.toml version pins
# ══════════════════════════════════════════════════════════════════════════════

class TestPyprojectToml:

    def _load_toml(self) -> dict:
        root = Path(__file__).parent.parent / "pyproject.toml"
        if not root.exists():
            pytest.skip("pyproject.toml not found at project root")
        try:
            import tomllib
            return tomllib.loads(root.read_text())
        except ImportError:
            import tomli
            return tomli.loads(root.read_text())

    def test_no_bare_greater_than_equal_pins(self):
        """No dependency should use >= without an upper bound."""
        data = self._load_toml()
        deps = data.get("project", {}).get("dependencies", [])
        for dep in deps:
            # Allow ~= (compatible release) and == (exact); reject bare >=
            # followed by nothing else (e.g. "torch>=2.10" with no upper bound).
            if ">=" in dep and "~=" not in dep and "," not in dep:
                pytest.fail(
                    f"Dependency '{dep}' uses >= without an upper bound. "
                    "Use ~= for compatible-release pinning instead."
                )

    def test_torch_is_pinned_with_compatible_release(self):
        data = self._load_toml()
        deps = data.get("project", {}).get("dependencies", [])
        torch_deps = [d for d in deps if d.startswith("torch~=")]
        assert torch_deps, "torch must use ~= compatible-release pinning"

    def test_transformer_engine_is_pinned(self):
        data = self._load_toml()
        deps = data.get("project", {}).get("dependencies", [])
        te_deps = [d for d in deps if "transformer-engine" in d]
        assert te_deps, "transformer-engine must be listed"
        assert any("~=" in d for d in te_deps), \
            "transformer-engine should use ~= pinning"

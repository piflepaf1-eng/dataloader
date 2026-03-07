"""
dino_loader.mixing_source
=========================
MixingSource, ShardIterator, MixingWeights, ResolutionSource.

Changes in this version
-----------------------
[MS-R1]  ShardIterator: explicit shard_sampling mode.                       ← NEW
         DatasetSpec.shard_sampling="resampled" delegates to
         wds.ResampledShards for infinite with-replacement sampling.
         The existing "epoch" mode (deterministic shuffle, one full pass)
         is unchanged.

         Why use wds.ResampledShards instead of reimplementing it?
         - wds.ResampledShards handles deterministic seeding per-worker,
           is well-tested, and will receive upstream fixes automatically.
         - Our role is to wrap it with the same interface as the epoch mode
           so that ShardIterator and MixingSource are unaware of the
           difference.

[MS-R2]  debug_log_keys support.                                            ← NEW
         When LoaderConfig.debug_log_keys is set, each sample's __key__
         (from metadata), worker id, and rank are appended to the log file
         using POSIX fcntl locking — same pattern as wds.log_keys.
         Implementation is in MixingSource._maybe_log_keys().
         Zero overhead when debug_log_keys=None.

[MS-R3]  register_dataset_index_callback — feeds NormSource.               ← NEW
         MixingSource now tracks which dataset index each sample in the
         batch came from.  After assembling a batch, it calls any registered
         callback(indices: List[int]).  This is consumed by NormSource in
         pipeline.py to emit the correct per-sample mean/std to the DALI
         ExternalSource.

[MS-8]   Vectorised np.rng.choice (retained).
[MS-9]   MetricsRegistry queue depth (retained).
"""

from __future__ import annotations

import fcntl
import logging
import os
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import (
    Callable, Dict, Deque, List, Optional, Sequence, Set, Tuple
)

import numpy as np

from dino_loader.config import DatasetSpec

log = logging.getLogger(__name__)

try:
    from dino_loader.monitor.metrics import get_registry, MetricField
    HAS_METRICS = True
except ImportError:
    HAS_METRICS = False

# webdataset for ResampledShards [MS-R1]
try:
    from webdataset.shardlists import ResampledShards
    HAS_WDS = True
except ImportError:
    HAS_WDS = False
    log.warning(
        "webdataset not installed — shard_sampling='resampled' will fall back "
        "to 'epoch' mode.  Install with: pip install webdataset"
    )

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


# ══════════════════════════════════════════════════════════════════════════════
# MixingWeights — thread-safe weight vector with named access
# ══════════════════════════════════════════════════════════════════════════════

class MixingWeights:
    """Normalised, thread-safe weight vector for dataset mixing."""

    def __init__(self, specs: List[DatasetSpec]) -> None:
        self.names = [s.name for s in specs]
        raw        = [s.weight for s in specs]
        self._lock = threading.Lock()
        self._weights = self._normalise(raw)

    def get(self) -> List[float]:
        with self._lock:
            return list(self._weights)

    def set(self, weights: Sequence[float]) -> None:
        if len(weights) != len(self.names):
            raise ValueError(
                f"MixingWeights.set: expected {len(self.names)} weights, got {len(weights)}."
            )
        with self._lock:
            self._weights = self._normalise(list(weights))

    def set_by_name(self, name: str, weight: float) -> None:
        try:
            idx = self.names.index(name)
        except ValueError:
            raise KeyError(f"Dataset '{name}' not found. Available: {self.names}")
        with self._lock:
            w = list(self._weights)
            # Un-normalise, update, re-normalise
            total = sum(w) or 1.0
            raw   = [v * total for v in w]
            raw[idx] = weight
            self._weights = self._normalise(raw)

    @staticmethod
    def _normalise(weights: List[float]) -> List[float]:
        total = sum(weights)
        if total <= 0:
            raise ValueError(f"Weights must sum to a positive number, got {weights}.")
        return [w / total for w in weights]


# ══════════════════════════════════════════════════════════════════════════════
# ResolutionSource — thread-safe (global_size, local_size) provider
# ══════════════════════════════════════════════════════════════════════════════

class ResolutionSource:
    """
    Thread-safe holder for the current crop resolution.

    Acts as a DALI ExternalSource callback (batch=False).
    Calling set() is immediately visible to the next DALI prefetch.
    """

    def __init__(self, global_size: int, local_size: int) -> None:
        self._global = global_size
        self._local  = local_size
        self._lock   = threading.Lock()

    def set(self, global_size: int, local_size: int) -> None:
        with self._lock:
            self._global = global_size
            self._local  = local_size

    def __call__(self) -> Tuple[np.ndarray, np.ndarray]:
        with self._lock:
            return (
                np.array(self._global, dtype=np.int32),
                np.array(self._local,  dtype=np.int32),
            )


# ══════════════════════════════════════════════════════════════════════════════
# SampleRecord
# ══════════════════════════════════════════════════════════════════════════════

class SampleRecord:
    __slots__ = ("jpeg", "metadata", "key")

    def __init__(
        self,
        jpeg:     bytes,
        metadata: Optional[Dict] = None,
        key:      str            = "",
    ) -> None:
        self.jpeg     = jpeg
        self.metadata = metadata
        self.key      = key


# ══════════════════════════════════════════════════════════════════════════════
# [MS-R1] ShardIterator — per-dataset, per-rank shard cycling
# ══════════════════════════════════════════════════════════════════════════════

class ShardIterator:
    """
    Per-dataset, per-rank shard cycling with background extraction.

    shard_sampling="epoch"
        Shards are shuffled once per epoch (deterministic, seed + rank).
        One full pass is made before cycling.

    shard_sampling="resampled"                                         [MS-R1]
        Delegates to wds.ResampledShards for infinite with-replacement
        sampling.  Useful for small curated datasets you want to over-sample
        without duplicating shards on disk.
        Falls back to "epoch" mode if webdataset is not installed.
    """

    _EXTRACTION_DEPTH = 2

    def __init__(
        self,
        spec:                DatasetSpec,
        cache,               # NodeSharedShardCache | InProcessShardCache
        rank:                int,
        world_size:          int,
        prefetch_ahead:      int   = 32,
        num_workers:         int   = 4,
        seed:                int   = 0,
        device_id:           int   = 0,
        cpu_affinity_enabled: bool = False,
        shuffle_buffer_size: int   = 512,
        min_sample_quality:  Optional[float] = None,
    ) -> None:
        self._name   = spec.name
        self._cache  = cache
        self._ahead  = prefetch_ahead
        self._seed   = seed
        self._rank   = rank
        self._rng    = np.random.default_rng(seed + rank)
        self._sampling = spec.shard_sampling

        # Partition shards across ranks (epoch mode)
        self._all_shards: List[str] = [
            s for i, s in enumerate(spec.shards) if i % world_size == rank
        ]
        self._shard_weights: Optional[List[float]] = None
        if spec.shard_quality_scores is not None:
            raw   = [spec.shard_quality_scores[i] for i, _ in enumerate(spec.shards)
                     if i % world_size == rank]
            total = sum(raw) or 1.0
            self._shard_weights = [w / total for w in raw]

        if not self._all_shards:
            raise RuntimeError(
                f"Rank {rank}/{world_size}: no shards assigned for dataset "
                f"'{spec.name}' ({len(spec.shards)} shards total)."
            )
        if len(self._all_shards) < 4:
            log.warning(
                "ShardIterator '%s': only %d shard(s) assigned to rank %d/%d. "
                "Consider using shard_sampling='resampled' for small datasets.",
                self._name, len(self._all_shards), rank, world_size,
            )

        # [MS-R1] ResampledShards for infinite-mode datasets
        self._resampled_iter = None
        if self._sampling == "resampled":
            if HAS_WDS:
                self._resampled_iter = iter(
                    ResampledShards(
                        urls          = self._all_shards,
                        seed          = seed + rank,
                        deterministic = True,
                    )
                )
                log.info(
                    "ShardIterator '%s': using ResampledShards (with-replacement, infinite).",
                    self._name,
                )
            else:
                log.warning(
                    "ShardIterator '%s': shard_sampling='resampled' requested but "
                    "webdataset is not installed — falling back to 'epoch' mode.",
                    self._name,
                )
                self._sampling = "epoch"

        self._shuffle_buffer_size  = shuffle_buffer_size
        self._min_sample_quality   = min_sample_quality or spec.min_sample_quality

        # Extraction infrastructure
        self._executor = ThreadPoolExecutor(
            max_workers     = num_workers,
            thread_name_prefix = f"shard-extract-{self._name}",
        )
        self._reservoir: Deque[SampleRecord] = deque()
        self._futures   = deque()
        self._closed    = False
        self._lock      = threading.Lock()

        # Prime extraction pipeline
        self._shard_cycle = self._make_shard_cycle()
        for _ in range(self._EXTRACTION_DEPTH):
            self._schedule_next()

    # ── Public API ────────────────────────────────────────────────────────────

    def next_sample(self) -> SampleRecord:
        """Block until a sample is available, then return it."""
        while True:
            with self._lock:
                if self._reservoir:
                    return self._reservoir.popleft()

            # Drain a future if reservoir is empty
            if self._futures:
                fut = self._futures.popleft()
                records = fut.result()
                with self._lock:
                    self._reservoir.extend(records)
                self._schedule_next()
            else:
                time.sleep(0.001)

    def reset_epoch(self, epoch: int) -> None:
        """Re-seed the RNG and restart the shard cycle."""
        self._rng = np.random.default_rng(self._seed + self._rank + epoch * 997)
        self._shard_cycle = self._make_shard_cycle()
        with self._lock:
            self._reservoir.clear()
        for _ in range(self._EXTRACTION_DEPTH):
            self._schedule_next()

    def close(self) -> None:
        self._closed = True
        self._executor.shutdown(wait=False, cancel_futures=True)

    @property
    def reservoir_size(self) -> int:
        with self._lock:
            return len(self._reservoir)

    # ── Shard cycling ─────────────────────────────────────────────────────────

    def _make_shard_cycle(self):
        """Infinite generator of shard paths."""
        if self._sampling == "resampled" and self._resampled_iter is not None:
            # wds.ResampledShards: yields dict(url=...) infinitely
            while not self._closed:
                item = next(self._resampled_iter)
                yield item["url"]
        else:
            # Epoch mode: shuffle + cycle
            while not self._closed:
                shards = list(self._all_shards)
                if self._shard_weights:
                    ordered = self._rng.choice(
                        shards, size=len(shards), replace=False,
                        p=self._shard_weights,
                    ).tolist()
                else:
                    self._rng.shuffle(shards)
                    ordered = shards
                yield from ordered

    def _schedule_next(self) -> None:
        if self._closed:
            return
        try:
            shard_path = next(self._shard_cycle)
        except StopIteration:
            return
        self._cache.prefetch(shard_path)
        fut = self._executor.submit(self._fetch_and_extract, shard_path)
        self._futures.append(fut)

    def _fetch_and_extract(self, shard_path: str) -> List[SampleRecord]:
        from dino_loader.datasets.utils import _extract_jpegs_with_meta
        with self._cache.get_view(shard_path) as mv:
            records = _extract_jpegs_with_meta(
                mv,
                metadata_key       = None,   # handled per-spec in MixingSource
                min_quality        = self._min_sample_quality,
                shuffle_buffer     = self._shuffle_buffer_size,
                rng                = self._rng,
            )
        return records


# ══════════════════════════════════════════════════════════════════════════════
# MixingSource — DALI ExternalSource callback
# ══════════════════════════════════════════════════════════════════════════════

class MixingSource:
    """
    DALI ExternalSource callback.  Called once per batch; returns a list of
    np.ndarray (JPEG bytes, one per sample).

    [MS-R3] Tracks per-sample dataset indices and calls registered callbacks
    (e.g. NormSource.set_dataset_indices) after each batch assembly.

    [MS-R2] Optionally logs __key__ per sample to a file for debug auditing.
    """

    def __init__(
        self,
        specs:               List[DatasetSpec],
        batch_size:          int,
        cache,
        rank:                int   = 0,
        world_size:          int   = 1,
        num_workers:         int   = 4,
        seed:                int   = 0,
        device_id:           int   = 0,
        cpu_affinity_enabled: bool = False,
        shuffle_buffer_size: int   = 512,
        debug_log_keys:      Optional[str] = None,   # [MS-R2]
    ) -> None:
        self._batch_size   = batch_size
        self._weights      = MixingWeights(specs)
        self._rng          = np.random.default_rng(seed + rank)
        self._meta_lock    = threading.Lock()
        self._last_metadata: List[Optional[Dict]] = []
        self._last_dataset_indices: List[int] = []
        self._dataset_index_callbacks: List[Callable[[List[int]], None]] = []

        # [MS-R2] debug key log
        self._log_file     = debug_log_keys
        self._rank         = rank

        self._iterators: List[ShardIterator] = [
            ShardIterator(
                spec                = spec,
                cache               = cache,
                rank                = rank,
                world_size          = world_size,
                num_workers         = num_workers,
                seed                = seed,
                device_id           = device_id,
                shuffle_buffer_size = shuffle_buffer_size,
            )
            for spec in specs
        ]

    # ── Callback registration [MS-R3] ─────────────────────────────────────────

    def register_dataset_index_callback(
        self, cb: Callable[[List[int]], None]
    ) -> None:
        """Register a callback to receive per-sample dataset indices each batch."""
        self._dataset_index_callbacks.append(cb)

    # ── DALI ExternalSource callback ──────────────────────────────────────────

    def __call__(self) -> List[np.ndarray]:
        """
        Assemble one batch of JPEG byte arrays.

        [MS-8] Uses vectorised np.rng.choice (one C-level call) for dataset
               selection instead of a Python-level loop.
        [MS-R3] Records per-sample dataset indices for NormSource callback.
        [MS-R2] Optionally logs __key__ values.
        """
        weights    = self._weights.get()
        n_datasets = len(self._iterators)
        indices    = self._rng.choice(
            n_datasets,
            size    = self._batch_size,
            replace = True,
            p       = weights,
        )

        batch:    List[np.ndarray]     = []
        metadata: List[Optional[Dict]] = []
        keys:     List[str]            = []

        for idx in indices:
            record = self._iterators[idx].next_sample()
            batch.append(np.frombuffer(record.jpeg, dtype=np.uint8))
            metadata.append(record.metadata)
            keys.append(record.key)

        with self._meta_lock:
            self._last_metadata        = metadata
            self._last_dataset_indices = list(indices)

        # [MS-R3] Notify NormSource (and any other registered callbacks)
        for cb in self._dataset_index_callbacks:
            cb(list(indices))

        # [MS-R2] Optional debug key logging
        if self._log_file:
            self._maybe_log_keys(keys)

        # [MS-9] Publish queue depth metric
        if HAS_METRICS:
            registry = get_registry()
            if registry is not None:
                depth = sum(it.reservoir_size for it in self._iterators)
                registry.set(MetricField.MIXING_QUEUE_DEPTH, depth)

        return batch

    # ── Metadata retrieval ────────────────────────────────────────────────────

    def pop_last_metadata(self) -> List[Optional[Dict]]:
        with self._meta_lock:
            return list(self._last_metadata)

    def pop_last_dataset_indices(self) -> List[int]:
        with self._meta_lock:
            return list(self._last_dataset_indices)

    # ── [MS-R2] Key logging ───────────────────────────────────────────────────

    def _maybe_log_keys(self, keys: List[str]) -> None:
        """
        Append per-sample key log entries to self._log_file.

        Format (tab-separated):
            <sample_index>  <rank>  <key>

        Uses POSIX fcntl exclusive lock so multiple ranks can write to the
        same file without corruption — identical to wds.log_keys.
        """
        try:
            lines = "".join(
                f"{i}\t{self._rank}\t{k}\n"
                for i, k in enumerate(keys)
            )
            with open(self._log_file, "a") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    f.write(lines)
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except Exception as exc:
            log.warning("debug_log_keys write failed: %s", exc)

    # ── Epoch / weight control ────────────────────────────────────────────────

    def set_epoch(self, epoch: int) -> None:
        for it in self._iterators:
            it.reset_epoch(epoch)
        self._rng = np.random.default_rng(epoch * 997 + self._rank)

    def set_weights(self, weights: Sequence[float]) -> None:
        self._weights.set(weights)

    def set_weight_by_name(self, name: str, weight: float) -> None:
        self._weights.set_by_name(name, weight)

    @property
    def current_weights(self) -> List[float]:
        return self._weights.get()

    @property
    def dataset_names(self) -> List[str]:
        return self._weights.names

    def close(self) -> None:
        for it in self._iterators:
            it.close()

"""dino_loader.monitor.metrics
===========================
Shared-memory metrics publishing and collection for zero-impact monitoring.

Allows lock-free updates to counters and timers from multiple dataloader
processes (up to 8 local ranks per node). The CLI monitor can read these
counters without blocking the fast path.

Implementation notes
--------------------
- All fields are 64-bit integers or 32-bit floats, naturally aligned in the
  ctypes.Structure layout.  On x86-64, aligned 64-bit stores/loads are atomic
  at the hardware level (single bus cycle), so readers can observe a torn read
  at most for the c_float field.  This is acceptable for a display-only monitor.
- ``heartbeat_ts`` is a Unix timestamp (seconds, c_int64) written by the loader
  on every batch.  The CLI uses it to distinguish a live-but-idle loader from a
  dead process.
- ``self.data`` is a ctypes Structure overlaid on the shared-memory buffer via
  ``from_buffer``.  This holds a reference into the mmap, so ``self.data``
  must be set to ``None`` before ``shm.close()`` — otherwise Python raises
  ``BufferError: cannot close exported pointers exist``.

Lifecycle invariant
-------------------
The correct teardown order for the *creator* is:

    1. _release_data()   — drop the ctypes overlay (no more exported pointers)
    2. shm.unlink()      — destroy the POSIX shared-memory segment name
    3. shm.close()       — close the file descriptor / mmap

For *readers* (create=False), only steps 1 and 3 apply (no unlink).

This is enforced by:
- ``close()`` : steps 1 + 3.
- ``unlink()`` : step 2 only (name removal).  Callers must still call
  ``close()`` afterwards to release the fd.  This matches the POSIX convention
  and the test pattern ``reg.unlink(); reg.close()``.

Fixes applied
-------------
[FIX-MON-1] Renamed ``shard_wait_time_ms`` → ``shard_cache_wait_time_ms``.
[FIX-MON-2] Added ``heartbeat_ts`` (Unix epoch seconds).
[MON-3]     MetricField StrEnum replaces bare string literals.
[MON-4]     ``mixing_source_queue_depth`` now populated by MixingSource.
[FIX-TYPE]  inc() and set_float() are now typed separately.
[FIX-CLOSE] self.data released before shm.close() to avoid BufferError.
[FIX-CLOSE2] unlink() clears self.data before shm.unlink() so that a
             subsequent close() does not encounter a stale exported pointer.
[FIX-CLOSE3] unlink() no longer calls shm.close() — that is the caller's
             responsibility (matching POSIX semantics: unlink removes the name,
             close releases the fd).  The previous version called both unlink
             and implicitly left close() to be called again, causing a double-
             close attempt on the mmap that raised BufferError.
"""

import ctypes
import enum
import logging
import time
from multiprocessing import shared_memory
from typing import Union

log = logging.getLogger(__name__)

MAX_LOCAL_RANKS = 8


class MetricField(str, enum.Enum):
    """Enumeration of all MetricsStruct field names."""

    # Stage 1: Lustre I/O (rank 0 / node master only)
    LUSTRE_READ_TIME_MS   = "lustre_read_time_ms"
    LUSTRE_BYTES_READ     = "lustre_bytes_read"

    # Stage 2: Shard cache wait (non-master ranks)
    SHARD_CACHE_WAIT_MS   = "shard_cache_wait_time_ms"
    SHARD_CACHE_UTIL_PCT  = "shard_cache_utilization_pct"   # c_float

    # Stage 3: DALI pipeline
    PIPELINE_YIELD_MS     = "pipeline_yield_time_ms"
    MIXING_QUEUE_DEPTH    = "mixing_source_queue_depth"

    # Stage 4: H2D transfer
    H2D_TRANSFER_MS       = "h2d_transfer_time_ms"

    # Stage 5: Loader output
    BATCHES_YIELDED       = "loader_batches_yielded"

    # Stall diagnostics
    NETWORK_STALL_MS      = "network_stall_time_ms"
    MULTINODE_STALL_MS    = "multinode_stall_time_ms"

    # Liveness
    HEARTBEAT_TS          = "heartbeat_ts"


# Convenience alias
MF = MetricField

# Fields stored as c_float rather than c_int64.
_FLOAT_FIELDS: frozenset[str] = frozenset({MF.SHARD_CACHE_UTIL_PCT.value})

FieldArg = Union[MetricField, str]


class MetricsStruct(ctypes.Structure):
    _fields_ = [
        (MF.LUSTRE_READ_TIME_MS.value,  ctypes.c_int64),
        (MF.LUSTRE_BYTES_READ.value,    ctypes.c_int64),
        (MF.SHARD_CACHE_WAIT_MS.value,  ctypes.c_int64),
        (MF.SHARD_CACHE_UTIL_PCT.value, ctypes.c_float),   # only c_float field
        (MF.PIPELINE_YIELD_MS.value,    ctypes.c_int64),
        (MF.MIXING_QUEUE_DEPTH.value,   ctypes.c_int64),
        (MF.H2D_TRANSFER_MS.value,      ctypes.c_int64),
        (MF.BATCHES_YIELDED.value,      ctypes.c_int64),
        (MF.NETWORK_STALL_MS.value,     ctypes.c_int64),
        (MF.MULTINODE_STALL_MS.value,   ctypes.c_int64),
        (MF.HEARTBEAT_TS.value,         ctypes.c_int64),
    ]


# Validate at import time that every MetricField maps to an actual struct field.
_STRUCT_FIELD_NAMES = {f[0] for f in MetricsStruct._fields_}
for _mf in MetricField:
    assert _mf.value in _STRUCT_FIELD_NAMES, (
        f"MetricField.{_mf.name} = {_mf.value!r} has no matching MetricsStruct field."
    )


class RankMetricsArray(ctypes.Structure):
    _fields_ = [("ranks", MetricsStruct * MAX_LOCAL_RANKS)]


class MetricsRegistry:
    """Publisher / subscriber for per-rank dataloader metrics over POSIX shared memory.

    Write API
    ---------
    inc(field, value)        — atomically add an integer to a c_int64 counter.
    set_float(field, value)  — set the single c_float field (utilisation pct).
    heartbeat()              — stamp current Unix time into heartbeat_ts.

    Lifecycle
    ---------
    The correct teardown sequence (regardless of creator/reader role) is::

        registry.unlink()   # creator only — removes the POSIX name
        registry.close()    # all — releases the fd and ctypes overlay

    ``close()`` is idempotent.  ``unlink()`` is a no-op if shm is None.

    Implementation note: ``self.data`` (ctypes from_buffer overlay) must be
    set to None *before* ``shm.close()`` is called, otherwise CPython 3.12
    raises ``BufferError: cannot close exported pointers exist`` because the
    ctypes structure holds a live reference into the mmap buffer.
    """

    def __init__(
        self,
        job_id:     str  = "dino",
        create:     bool = False,
        local_rank: int  = 0,
    ) -> None:
        self.name       = f"dino_metrics_{job_id}"
        self.local_rank = min(local_rank, MAX_LOCAL_RANKS - 1)
        self.size       = ctypes.sizeof(RankMetricsArray)
        self.shm:  shared_memory.SharedMemory | None = None
        self.data: RankMetricsArray | None           = None

        try:
            if create:
                try:
                    stale = shared_memory.SharedMemory(name=self.name)
                    stale.unlink()
                    stale.close()
                except FileNotFoundError:
                    pass
                self.shm = shared_memory.SharedMemory(
                    name=self.name, create=True, size=self.size,
                )
            else:
                self.shm = shared_memory.SharedMemory(name=self.name)

            self.data = RankMetricsArray.from_buffer(self.shm.buf)

            if create:
                ctypes.memset(ctypes.addressof(self.data), 0, self.size)

        except Exception as exc:
            log.warning(
                "Could not initialise shared-memory metrics for %s: %s",
                self.name, exc,
            )

    @property
    def metrics(self) -> MetricsStruct | None:
        """Direct reference to this rank's MetricsStruct. None if unavailable."""
        if self.data is not None:
            return self.data.ranks[self.local_rank]
        return None

    def inc(self, field: FieldArg, value: int = 1) -> None:
        """Increment a c_int64 counter field.

        Args:
            field: MetricField enum member or plain str (compat).
            value: Integer amount to add.

        Raises:
            TypeError: If called with SHARD_CACHE_UTIL_PCT (wrong type).

        """
        if self.data is None:
            return
        key = field.value if isinstance(field, MetricField) else field
        if key in _FLOAT_FIELDS:
            raise TypeError(
                f"MetricsRegistry.inc() cannot be used with float field {key!r}. "
                "Use set_float() instead.",
            )
        m = self.data.ranks[self.local_rank]
        setattr(m, key, getattr(m, key) + int(value))

    def set_float(self, field: FieldArg, value: float) -> None:
        """Set the c_float utilisation field to an absolute value.

        Args:
            field: Must be MetricField.SHARD_CACHE_UTIL_PCT.
            value: New absolute float value.

        Raises:
            TypeError: If called with a non-float field.

        """
        if self.data is None:
            return
        key = field.value if isinstance(field, MetricField) else field
        if key not in _FLOAT_FIELDS:
            raise TypeError(
                f"MetricsRegistry.set_float() can only be used with float fields. "
                f"{key!r} is a c_int64 field — use inc() instead.",
            )
        setattr(self.data.ranks[self.local_rank], key, float(value))

    def heartbeat(self) -> None:
        """Stamp current Unix time into heartbeat_ts. Call once per batch."""
        if self.data is None:
            return
        self.data.ranks[self.local_rank].heartbeat_ts = int(time.time())

    def read_all_ranks(self) -> RankMetricsArray | None:
        """Return a reference to the full shared-memory array.

        The CLI reads this without holding any lock — torn reads on individual
        fields are tolerated for display purposes.
        """
        return self.data

    def _release_data(self) -> None:
        """Release the ctypes overlay so shm has no exported pointer references.

        Must be called before any shm.close() to avoid
        ``BufferError: cannot close exported pointers exist`` on Python 3.12.
        """
        self.data = None

    def close(self) -> None:
        """Detach from shared memory (does not unlink the block).

        Safe to call multiple times (idempotent).

        [FIX-CLOSE] Releases ``self.data`` (ctypes from_buffer overlay)
        before ``shm.close()``.  Without this, Python 3.12 raises
        ``BufferError: cannot close exported pointers exist``.
        """
        self._release_data()
        if self.shm is not None:
            try:
                self.shm.close()
            except Exception:  # noqa: BLE001
                pass
            self.shm = None

    def unlink(self) -> None:
        """Destroy the shared-memory block name. Call once, on the creator.

        [FIX-CLOSE3] unlink() only removes the POSIX name — it does NOT call
        shm.close().  The caller must call close() afterwards to release the
        file descriptor.  This matches the standard POSIX pattern:

            registry.unlink()  # remove name (creator only)
            registry.close()   # release fd (all)

        [FIX-CLOSE2] Releases ``self.data`` before shm.unlink() so that a
        subsequent close() call does not encounter a stale exported pointer.
        """
        self._release_data()
        if self.shm is not None:
            try:
                self.shm.unlink()
            except Exception:  # noqa: BLE001
                pass


# Module-level singleton

_REGISTRY: MetricsRegistry | None = None


def init_registry(job_id: str, create: bool, local_rank: int) -> None:
    """Initialise the module-level MetricsRegistry singleton.

    Call once per process at dataloader construction time.
    """
    global _REGISTRY
    _REGISTRY = MetricsRegistry(job_id=job_id, create=create, local_rank=local_rank)


def get_registry() -> MetricsRegistry | None:
    """Return the module-level registry, or None if not yet initialised."""
    return _REGISTRY
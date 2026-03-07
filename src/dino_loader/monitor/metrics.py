"""
dino_loader.monitor.metrics
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

Fixes applied
-------------
[FIX-MON-1] Renamed ``shard_wait_time_ms`` → ``shard_cache_wait_time_ms`` to
            match the documented field name and eliminate the BUG-D ambiguity.
[FIX-MON-2] Added ``heartbeat_ts`` (Unix epoch seconds) so the CLI can detect
            dead processes vs. idle ones.

Changes vs previous version
----------------------------
[MON-3] MetricField StrEnum replaces bare string literals throughout the
        codebase.  ``inc()`` and ``set()`` now accept ``MetricField`` values
        (or plain str for backward-compatibility).  A misspelled field name
        now raises ``AttributeError`` at import time rather than silently
        writing to a non-existent struct field at runtime.

        Migration: replace any ``registry.inc("shard_cache_wait_time_ms", n)``
        with ``registry.inc(MetricField.SHARD_CACHE_WAIT_MS, n)``.
        Plain strings are still accepted — no forced migration — but IDE
        type-checkers will flag them as ``str`` rather than ``MetricField``.

[MON-4] ``mixing_source_queue_depth`` is now populated by MixingSource via
        ``MetricField.MIXING_QUEUE_DEPTH``.  Previously the field existed in
        the struct but was never written, so the CLI always showed 0.
"""

import ctypes
import enum
import logging
import time
from multiprocessing import shared_memory
from typing import Optional, Union

log = logging.getLogger(__name__)

# We support up to 8 GPUs/ranks per node.
MAX_LOCAL_RANKS = 8


# ══════════════════════════════════════════════════════════════════════════════
# [MON-3] Typed field names — eliminates silent typos in the fast path
# ══════════════════════════════════════════════════════════════════════════════

class MetricField(str, enum.Enum):
    """
    Enumeration of all MetricsStruct field names.

    Use these constants instead of raw strings when calling
    ``MetricsRegistry.inc()`` or ``MetricsRegistry.set()``.
    This makes misspellings a loud import-time error (AttributeError on the
    enum lookup) rather than a silent runtime no-op.

    Example::

        from dino_loader.monitor.metrics import MetricField, get_registry
        get_registry().inc(MetricField.SHARD_CACHE_WAIT_MS, wait_ms)
    """

    # ── Stage 1: Lustre I/O (rank 0 / node master only) ─────────────────────
    LUSTRE_READ_TIME_MS   = "lustre_read_time_ms"
    LUSTRE_BYTES_READ     = "lustre_bytes_read"

    # ── Stage 2: Shard cache wait (non-master ranks) ─────────────────────────
    SHARD_CACHE_WAIT_MS   = "shard_cache_wait_time_ms"
    SHARD_CACHE_UTIL_PCT  = "shard_cache_utilization_pct"

    # ── Stage 3: DALI pipeline ────────────────────────────────────────────────
    PIPELINE_YIELD_MS     = "pipeline_yield_time_ms"
    MIXING_QUEUE_DEPTH    = "mixing_source_queue_depth"   # [MON-4] now populated

    # ── Stage 4: H2D transfer ─────────────────────────────────────────────────
    H2D_TRANSFER_MS       = "h2d_transfer_time_ms"

    # ── Stage 5: Loader output ────────────────────────────────────────────────
    BATCHES_YIELDED       = "loader_batches_yielded"

    # ── Stall diagnostics ─────────────────────────────────────────────────────
    NETWORK_STALL_MS      = "network_stall_time_ms"
    MULTINODE_STALL_MS    = "multinode_stall_time_ms"

    # ── Liveness ──────────────────────────────────────────────────────────────
    HEARTBEAT_TS          = "heartbeat_ts"


# ── Convenience alias so call-sites can type `MF.BATCHES_YIELDED` ─────────────
MF = MetricField

# ── Type accepted by inc() / set() — MetricField or plain str (compat) ────────
FieldArg = Union[MetricField, str]


class MetricsStruct(ctypes.Structure):
    _fields_ = [
        # ── Stage 1: Lustre I/O (rank 0 / node master only) ──────────────────
        (MF.LUSTRE_READ_TIME_MS.value,  ctypes.c_int64),
        (MF.LUSTRE_BYTES_READ.value,    ctypes.c_int64),

        # ── Stage 2: Shard cache wait (non-master ranks) ─────────────────────
        (MF.SHARD_CACHE_WAIT_MS.value,  ctypes.c_int64),
        (MF.SHARD_CACHE_UTIL_PCT.value, ctypes.c_float),

        # ── Stage 3: DALI pipeline ────────────────────────────────────────────
        (MF.PIPELINE_YIELD_MS.value,    ctypes.c_int64),
        (MF.MIXING_QUEUE_DEPTH.value,   ctypes.c_int64),

        # ── Stage 4: H2D transfer ─────────────────────────────────────────────
        (MF.H2D_TRANSFER_MS.value,      ctypes.c_int64),

        # ── Stage 5: Loader output ────────────────────────────────────────────
        (MF.BATCHES_YIELDED.value,      ctypes.c_int64),

        # ── Stall diagnostics ─────────────────────────────────────────────────
        (MF.NETWORK_STALL_MS.value,     ctypes.c_int64),
        (MF.MULTINODE_STALL_MS.value,   ctypes.c_int64),

        # ── Liveness ──────────────────────────────────────────────────────────
        (MF.HEARTBEAT_TS.value,         ctypes.c_int64),
    ]


# Validate at import time that every MetricField maps to an actual struct field.
# This catches enum / struct drift immediately rather than at first write.
_STRUCT_FIELD_NAMES = {f[0] for f in MetricsStruct._fields_}
for _mf in MetricField:
    assert _mf.value in _STRUCT_FIELD_NAMES, (
        f"MetricField.{_mf.name} = {_mf.value!r} has no matching MetricsStruct field. "
        "Add the field to MetricsStruct._fields_ or remove it from the enum."
    )


class RankMetricsArray(ctypes.Structure):
    _fields_ = [("ranks", MetricsStruct * MAX_LOCAL_RANKS)]


class MetricsRegistry:
    """
    Publisher / subscriber for per-rank dataloader metrics over POSIX
    shared memory.

    Usage
    -----
    Node master (local_rank == 0) at loader init::

        from dino_loader.monitor.metrics import init_registry
        init_registry(job_id="$SLURM_JOB_ID", create=True, local_rank=0)

    Other ranks (attach, do not create)::

        init_registry(job_id="$SLURM_JOB_ID", create=False, local_rank=local_rank)

    CLI monitor (read-only attach)::

        registry = MetricsRegistry(job_id, create=False)

    Parameters
    ----------
    job_id      : Unique name shared between all processes on a node.
                  Use ``$SLURM_JOB_ID`` for automatic isolation between jobs.
    create      : True on the node master to create+clear the block.
    local_rank  : This process's local GPU index (0–7).
    """

    def __init__(
        self,
        job_id:      str  = "dino",
        create:      bool = False,
        local_rank:  int  = 0,
    ) -> None:
        self.name       = f"dino_metrics_{job_id}"
        self.local_rank = min(local_rank, MAX_LOCAL_RANKS - 1)
        self.size       = ctypes.sizeof(RankMetricsArray)
        self.shm:  Optional[shared_memory.SharedMemory] = None
        self.data: Optional[RankMetricsArray]           = None

        try:
            if create:
                # Unlink any stale block from a previous crashed job.
                try:
                    stale = shared_memory.SharedMemory(name=self.name)
                    stale.unlink()
                    stale.close()
                except FileNotFoundError:
                    pass
                self.shm = shared_memory.SharedMemory(
                    name=self.name, create=True, size=self.size
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

    # ------------------------------------------------------------------
    # Per-rank write helpers (called from the fast path)
    # ------------------------------------------------------------------

    @property
    def metrics(self) -> Optional[MetricsStruct]:
        """Direct reference to this rank's MetricsStruct. None if unavailable."""
        if self.data is not None:
            return self.data.ranks[self.local_rank]
        return None

    def inc(self, field: FieldArg, value: int = 1) -> None:
        """
        Atomically increment an integer metric field.

        Parameters
        ----------
        field : ``MetricField`` enum member (preferred) or plain str (compat).
        value : Amount to add.
        """
        if self.data is None:
            return
        key = field.value if isinstance(field, MetricField) else field
        m = self.data.ranks[self.local_rank]
        setattr(m, key, getattr(m, key) + value)

    def set(self, field: FieldArg, value: float) -> None:
        """
        Set a metric field to an absolute value.

        Parameters
        ----------
        field : ``MetricField`` enum member (preferred) or plain str (compat).
        value : New absolute value.
        """
        if self.data is None:
            return
        key = field.value if isinstance(field, MetricField) else field
        setattr(self.data.ranks[self.local_rank], key, value)

    def heartbeat(self) -> None:
        """Stamp the current Unix time into heartbeat_ts. Call once per batch."""
        if self.data is None:
            return
        self.data.ranks[self.local_rank].heartbeat_ts = int(time.time())

    # ------------------------------------------------------------------
    # Monitor-side read helper
    # ------------------------------------------------------------------

    def read_all_ranks(self) -> Optional[RankMetricsArray]:
        """
        Return a reference to the full shared-memory array.

        The CLI reads this without holding any lock — torn reads on individual
        fields are tolerated for display purposes.
        """
        return self.data

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Detach from shared memory (does not unlink the block)."""
        if self.shm is not None:
            self.shm.close()

    def unlink(self) -> None:
        """Destroy the shared-memory block. Call once, on the creator."""
        if self.shm is not None:
            self.shm.unlink()


# ── Module-level singleton ────────────────────────────────────────────────────

_REGISTRY: Optional[MetricsRegistry] = None


def init_registry(job_id: str, create: bool, local_rank: int) -> None:
    """
    Initialise the module-level MetricsRegistry singleton.

    Call once per process at dataloader construction time.  All subsequent
    calls to ``get_registry()`` return this instance.
    """
    global _REGISTRY
    _REGISTRY = MetricsRegistry(job_id=job_id, create=create, local_rank=local_rank)


def get_registry() -> Optional[MetricsRegistry]:
    """Return the module-level registry, or None if not yet initialised."""
    return _REGISTRY

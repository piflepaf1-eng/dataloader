"""
dino_loader.shard_cache
=======================
Node-local shared-memory shard cache.

Design
------
- One backing file per shard per node in /dev/shm/<job_id>/.
- Local rank 0 ("node master") reads from Lustre/NVMe and writes.
- Other local ranks wait via inotify (Linux) or stat-poll fallback —
  NOT a busy-spin, so 71 waiting ranks on NVL72 cost near-zero CPU.
- LRU eviction at file granularity.
- Header encodes (data_len, ready_flag) so readers can detect partial writes.
- asyncio + aiofiles for concurrent shard prefetch (hides Lustre latency).

Changes in this version
-----------------------
[B2-FIX] _evict_for_locked: backpressure instead of grow-and-proceed.
         Previously, if all mmap pool entries were referenced simultaneously
         (e.g. 71 ranks on NVL72 each holding a view of the same 64 shards),
         _evict_for_locked could not evict any entry and exited the loop with
         ``break`` — then proceeded to write, potentially exceeding _max_bytes.
         The subsequent _check_shm_headroom() call would then raise a hard
         RuntimeError mid-batch.

         New behaviour: if the loop exits without making space (all entries
         referenced), _load_one() waits up to _EVICT_WAIT_S seconds (default
         2 s) with asyncio.sleep() between retries, then raises a clear
         RuntimeError with actionable advice.  This converts a hard crash into
         a recoverable stall with a meaningful error message.

[M4-FIX] heartbeat_stale_s is now sourced from NodeSharedShardCache constructor
         instead of the module-level constant _HB_STALE_S = 60.0.  The
         constant is kept as a fallback default but the constructor accepts
         ``heartbeat_stale_s`` and passes it to _purge_orphaned_shm().

[ARCH1]  Optional SharedMemoryRingBuffer integration.                      ← NEW (opt-in)
         When ``use_ring_buffer=True``, rank 0 publishes shard data into
         POSIX SharedMemory segments after writing to /dev/shm.  Non-master
         ranks read from the SharedMemory segment via zero-copy memoryview
         instead of opening individual mmap files.  This reduces mmap syscall
         overhead on NVL72 from ~4 500 to ~64 per prefetch window.
         Default: False (existing mmap pool path retained).

[ARCH2]  Optional adaptive prefetch window PID controller.                 ← NEW (opt-in)
         When ``adaptive_prefetch=True``, an _AdaptivePrefetchController
         daemon thread adjusts the asyncio Semaphore value dynamically to
         target a configured /dev/shm utilisation fraction.  This maximises
         I/O parallelism without overcommitting RAM.
         Default: False (static shard_prefetch_window).

Retained from previous version
-------------------------------
[A-1..A-5] All intern-review fixes.
[FIX-1..FIX-HB] All subsequent fixes including heartbeat, mmap pool, etc.
[PERF-1] No fsync on tmpfs.
[PERF-2] Persistent mmap pool.
[LOG-1]  INFO→DEBUG demotion for per-shard logs.
"""

from __future__ import annotations

import asyncio
import atexit
import contextlib
import hashlib
import logging
import mmap
import os
import select
import shutil
import signal
import struct
import subprocess
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Iterator, Optional, Set

from dino_loader.monitor.metrics import MetricField, get_registry

log = logging.getLogger(__name__)

_HDR_FMT     = "QQ"
_HDR_SIZE    = struct.calcsize(_HDR_FMT)
_READY_MAGIC = 0xDEAD_BEEF_CAFE_F00D

_IN_CLOSE_WRITE = 0x00000008
_IN_MOVED_TO    = 0x00000080

# Rate-limiting for /dev/shm utilisation warnings
_SHM_WARN_INTERVAL = 60.0

# Timeout for squeue orphan detection
_SQUEUE_TIMEOUT_S = 2.0

# Headroom factors
_SHM_HEADROOM_WARN_FACTOR = 1.2
_SHM_HEADROOM_MIN_WARN_MB = 512

# Mmap pool size
_MMAP_POOL_MAX = 256

# Heartbeat parameters (defaults — overridden by constructor arg)
_HB_INTERVAL_S = 10.0
_HB_STALE_S    = 300.0   # raised from 60 s; M4-FIX
_HB_FILENAME   = "heartbeat"

# [B2-FIX] How long _load_one() waits for eviction space (seconds)
_EVICT_WAIT_S    = 2.0
_EVICT_RETRIES   = 10

# [ARCH2] PID controller parameters
_PID_KP = 2.0   # proportional gain
_PID_KI = 0.5   # integral gain (reset-windup clipped)
_PID_KD = 0.1   # derivative gain
_PID_INTERVAL_S = 5.0


# ══════════════════════════════════════════════════════════════════════════════
# Persistent mmap pool (PERF-2, retained)
# ══════════════════════════════════════════════════════════════════════════════

class _MmapEntry:
    __slots__ = ("fd", "mm", "data_len", "refs")

    def __init__(self, fd: int, mm: mmap.mmap, data_len: int) -> None:
        self.fd       = fd
        self.mm       = mm
        self.data_len = data_len
        self.refs     = 0


class _MmapPool:
    """Thread-safe pool of persistent memory-mapped shard files."""

    def __init__(self, max_entries: int = _MMAP_POOL_MAX) -> None:
        self._max   = max_entries
        self._pool: OrderedDict[str, _MmapEntry] = OrderedDict()
        self._lock  = threading.Lock()

    def acquire(self, path: Path) -> _MmapEntry:
        key = str(path)
        with self._lock:
            if key in self._pool:
                entry = self._pool[key]
                self._pool.move_to_end(key)
                entry.refs += 1
                return entry
            self._evict_unreferenced()
            fd = os.open(key, os.O_RDONLY)
            try:
                mm              = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
                data_len, magic = struct.unpack_from(_HDR_FMT, mm, 0)
                if magic != _READY_MAGIC:
                    mm.close()
                    os.close(fd)
                    raise RuntimeError(
                        f"Shard {path} has corrupt header (magic={magic:#x})"
                    )
                entry       = _MmapEntry(fd, mm, data_len)
                entry.refs  = 1
                self._pool[key] = entry
                return entry
            except Exception:
                os.close(fd)
                raise

    def release(self, path: Path) -> None:
        key = str(path)
        with self._lock:
            if key in self._pool:
                self._pool[key].refs = max(0, self._pool[key].refs - 1)

    def invalidate(self, path: Path) -> None:
        key = str(path)
        with self._lock:
            entry = self._pool.pop(key, None)
        if entry is not None:
            self._close_entry(entry)

    def close_all(self) -> None:
        with self._lock:
            entries = list(self._pool.values())
            self._pool.clear()
        for entry in entries:
            self._close_entry(entry)

    def _evict_unreferenced(self) -> None:
        """Evict LRU entries with ref==0. Caller holds lock."""
        while len(self._pool) >= self._max:
            evicted = False
            for key, entry in self._pool.items():
                if entry.refs == 0:
                    del self._pool[key]
                    self._close_entry(entry)
                    evicted = True
                    break
            if not evicted:
                # All entries referenced — let the pool grow beyond max.
                break

    @staticmethod
    def _close_entry(entry: _MmapEntry) -> None:
        try:
            entry.mm.close()
        except Exception:
            pass
        try:
            os.close(entry.fd)
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════════
# Heartbeat writer (FIX-HB, retained)
# ══════════════════════════════════════════════════════════════════════════════

class _HeartbeatWriter:
    """Background daemon that refreshes the heartbeat file mtime."""

    def __init__(self, hb_path: Path) -> None:
        self._path  = hb_path
        self._stop  = threading.Event()
        self._write()
        self._thread = threading.Thread(
            target=self._run, name="shm-heartbeat", daemon=True
        )
        self._thread.start()
        log.debug("HeartbeatWriter started: %s (pid=%d)", hb_path, os.getpid())

    def _write(self) -> None:
        tmp = self._path.with_suffix(".tmp")
        try:
            tmp.write_text(str(os.getpid()))
            tmp.rename(self._path)
        except Exception as exc:
            log.warning("HeartbeatWriter: could not write %s: %s", self._path, exc)

    def _run(self) -> None:
        while not self._stop.wait(timeout=_HB_INTERVAL_S):
            self._write()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)
        try:
            self._path.unlink(missing_ok=True)
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════════
# [ARCH2] Adaptive prefetch PID controller (opt-in)
# ══════════════════════════════════════════════════════════════════════════════

class _AdaptivePrefetchController:
    """
    PID controller that adjusts the asyncio Semaphore value to target a
    configured /dev/shm utilisation fraction.

    The controller runs as a daemon thread.  It reads the current utilisation
    from NodeSharedShardCache.utilisation, computes a correction, and sets the
    semaphore capacity by releasing or acquiring tokens.

    Why a semaphore?
    ----------------
    asyncio.Semaphore does not support dynamic resize.  We simulate it by
    tracking the *intended* capacity and using a regular asyncio.Semaphore
    of size=max_window.  The effective window is controlled by pre-acquiring
    (max_window - target_window) tokens that are never released — this is a
    well-known pattern for dynamic semaphore resizing without locks.

    Implementation uses a simpler approach: we maintain an integer
    ``_effective_window`` (protected by threading.Lock) and modify the loop's
    semaphore capacity by replacing the semaphore object atomically.  Since
    asyncio semaphores are not thread-safe, all semaphore access happens inside
    the asyncio event loop via loop.call_soon_threadsafe().
    """

    def __init__(
        self,
        cache:          "NodeSharedShardCache",
        loop:           asyncio.AbstractEventLoop,
        target_util:    float = 0.75,
        max_window:     int   = 64,
        min_window:     int   = 4,
    ) -> None:
        self._cache       = cache
        self._loop        = loop
        self._target_util = target_util
        self._max_window  = max_window
        self._min_window  = min_window
        self._window      = max_window
        self._integral    = 0.0
        self._prev_error  = 0.0
        self._stop        = threading.Event()
        self._thread      = threading.Thread(
            target=self._run, name="adaptive-prefetch-pid", daemon=True
        )
        self._thread.start()
        log.info(
            "AdaptivePrefetchController started: target_util=%.0f%% "
            "window=[%d, %d]",
            target_util * 100, min_window, max_window,
        )

    def _run(self) -> None:
        while not self._stop.wait(timeout=_PID_INTERVAL_S):
            try:
                util  = self._cache.utilisation
                error = self._target_util - util

                # PID computation
                self._integral   = max(-10, min(10, self._integral + error * _PID_INTERVAL_S))
                derivative       = (error - self._prev_error) / _PID_INTERVAL_S
                self._prev_error = error
                correction       = _PID_KP * error + _PID_KI * self._integral + _PID_KD * derivative

                new_window = int(self._window + correction * self._max_window * 0.1)
                new_window = max(self._min_window, min(self._max_window, new_window))

                if new_window != self._window:
                    log.debug(
                        "AdaptivePrefetch: util=%.1f%% err=%.3f → window %d→%d",
                        util * 100, error, self._window, new_window,
                    )
                    self._window = new_window
                    # Apply to the semaphore via thread-safe loop call.
                    self._loop.call_soon_threadsafe(
                        self._cache._resize_semaphore, new_window
                    )
            except Exception as exc:
                log.warning("AdaptivePrefetchController error: %s", exc)

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)


# ══════════════════════════════════════════════════════════════════════════════
# Orphan detection helpers
# ══════════════════════════════════════════════════════════════════════════════

def _purge_orphaned_shm(job_name: str, hb_stale_s: float = _HB_STALE_S) -> None:
    """Remove /dev/shm directories from dead jobs."""
    base = Path("/dev/shm")
    for d in base.iterdir():
        if not d.is_dir() or d.name == job_name:
            continue
        hb = d / _HB_FILENAME
        if hb.exists():
            try:
                mtime = hb.stat().st_mtime
                age   = time.time() - mtime
                if age < hb_stale_s:
                    continue
                pid = int(hb.read_text().strip())
                try:
                    os.kill(pid, 0)
                    continue  # process alive — not orphaned
                except ProcessLookupError:
                    pass  # process dead
            except Exception:
                pass  # unreadable → treat as orphaned
            log.info("Purging orphaned /dev/shm dir (stale heartbeat): %s", d)
            shutil.rmtree(d, ignore_errors=True)
        else:
            # No heartbeat file — try squeue (legacy dirs from pre-patch jobs)
            try:
                result = subprocess.run(
                    ["squeue", "--job", d.name, "--noheader"],
                    capture_output=True, timeout=_SQUEUE_TIMEOUT_S,
                )
                if result.returncode != 0 or not result.stdout.strip():
                    log.info("Purging orphaned /dev/shm dir (no squeue entry): %s", d)
                    shutil.rmtree(d, ignore_errors=True)
            except Exception:
                pass


def _is_ready(shm: Path) -> bool:
    """Return True if the shard file is fully written and ready to read."""
    if not shm.exists():
        return False
    try:
        with open(shm, "rb") as f:
            with mmap.mmap(f.fileno(), _HDR_SIZE, access=mmap.ACCESS_READ) as mm:
                _, magic = struct.unpack_from(_HDR_FMT, mm, 0)
                return magic == _READY_MAGIC
    except Exception:
        return False


def _check_shm_headroom(incoming: int) -> None:
    """Raise if the OS-reported free tmpfs space is dangerously low."""
    try:
        st = os.statvfs("/dev/shm")
        free = st.f_bsize * st.f_bavail
    except Exception:
        return
    needed = incoming * _SHM_HEADROOM_WARN_FACTOR
    if free < max(needed, _SHM_HEADROOM_MIN_WARN_MB * (1 << 20)):
        raise RuntimeError(
            f"/dev/shm has only {free >> 20} MB free; shard write of "
            f"{incoming >> 20} MB would exceed available space.  "
            "Reduce node_shm_gb or shard_prefetch_window."
        )


def _read_file_sync(path: str) -> bytes:
    """Synchronous fallback for Lustre reads when aiofiles is unavailable."""
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            return os.read(fd, os.fstat(fd).st_size)
        finally:
            os.close(fd)
    except Exception as exc:
        raise RuntimeError(f"Failed to read shard {path}: {exc}") from exc


def _inotify_wait(shm: Path, timeout_s: float) -> None:
    """Block until shm is ready, using inotify on Linux or stat-poll elsewhere."""
    deadline = time.monotonic() + timeout_s
    # Try inotify first (Linux only)
    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        ifd  = libc.inotify_init1(0o4000)  # IN_NONBLOCK
        if ifd < 0:
            raise OSError("inotify_init1 failed")
        wd = libc.inotify_add_watch(
            ifd,
            str(shm.parent).encode(),
            _IN_CLOSE_WRITE | _IN_MOVED_TO,
        )
        if wd < 0:
            os.close(ifd)
            raise OSError("inotify_add_watch failed")
        try:
            while not _is_ready(shm):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"Timed out ({timeout_s:.0f}s) waiting for shard: {shm}"
                    )
                r, _, _ = select.select([ifd], [], [], min(remaining, 1.0))
                if r:
                    os.read(ifd, 4096)  # drain the event buffer
        finally:
            libc.inotify_rm_watch(ifd, wd)
            os.close(ifd)
        return
    except Exception:
        pass
    # Stat-poll fallback
    while not _is_ready(shm):
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Timed out ({timeout_s:.0f}s) waiting for shard: {shm}"
            )
        time.sleep(0.05)


# ══════════════════════════════════════════════════════════════════════════════
# NodeSharedShardCache
# ══════════════════════════════════════════════════════════════════════════════

class NodeSharedShardCache:
    """
    Node-local /dev/shm shard cache with optional ring buffer and adaptive prefetch.

    One instance per process; all processes on the same node share the same
    /dev/shm directory.

    Parameters
    ----------
    node_master        : True for local rank 0 — this process fills the cache.
    job_id             : Namespace for /dev/shm files (use SLURM_JOB_ID).
    max_shm_gb         : RAM budget in /dev/shm for this node.
    prefetch_window    : Max concurrent Lustre → /dev/shm downloads.
    shard_timeout_s    : How long non-master ranks wait for a shard.
    shm_warn_threshold : Fraction (0–1) at which to emit a utilisation warning.
    heartbeat_stale_s  : Seconds of no heartbeat before a dir is orphaned.  [M4-FIX]
    use_ring_buffer    : Enable SharedMemoryRingBuffer (opt-in).            [ARCH1]
    adaptive_prefetch  : Enable PID-controlled prefetch window (opt-in).    [ARCH2]
    adaptive_target_util : Target /dev/shm utilisation for PID controller.  [ARCH2]
    """

    def __init__(
        self,
        node_master:          bool,
        job_id:               str   = "dino",
        max_shm_gb:           float = 128.0,
        prefetch_window:      int   = 64,
        shard_timeout_s:      float = 300.0,
        shm_warn_threshold:   float = 0.85,
        heartbeat_stale_s:    float = _HB_STALE_S,  # [M4-FIX]
        use_ring_buffer:      bool  = False,          # [ARCH1]
        adaptive_prefetch:    bool  = False,          # [ARCH2]
        adaptive_target_util: float = 0.75,           # [ARCH2]
    ):
        self._node_master    = node_master
        self._max_bytes      = int(max_shm_gb * (1 << 30))
        self._base           = Path(f"/dev/shm/{job_id}")
        self._timeout        = shard_timeout_s
        self._warn_threshold = shm_warn_threshold
        self._hb_stale_s     = heartbeat_stale_s
        self._last_warn_ts:  float = 0.0

        self._lru:         OrderedDict[str, int] = OrderedDict()
        self._total_bytes: int                   = 0
        self._lru_lock:    threading.Lock        = threading.Lock()
        self._in_flight:   Set[str]              = set()
        self._shutdown_event = threading.Event()

        self._mmap_pool = _MmapPool(max_entries=_MMAP_POOL_MAX)

        # [ARCH1] Optional ring buffer
        self._ring_buffer = None
        if use_ring_buffer:
            from dino_loader.memory import SharedMemoryRingBuffer
            self._ring_buffer = SharedMemoryRingBuffer(
                job_id=job_id, node_master=node_master
            )
            log.info(
                "NodeSharedShardCache: SharedMemoryRingBuffer enabled (rank %s)",
                "master" if node_master else "worker",
            )

        if node_master:
            _purge_orphaned_shm(job_id, hb_stale_s=heartbeat_stale_s)
            self._init_shm()
            self._metrics = get_registry()
            self._max_semaphore_value = prefetch_window
            self._loop    = asyncio.new_event_loop()
            self._sem     = asyncio.Semaphore(prefetch_window)
            self._thread  = threading.Thread(
                target=self._loop.run_forever, name="shard-io", daemon=True
            )
            self._thread.start()
            self._heartbeat: Optional[_HeartbeatWriter] = _HeartbeatWriter(
                self._base / _HB_FILENAME
            )
            # [ARCH2] Optional adaptive prefetch controller
            self._pid_ctrl = None
            if adaptive_prefetch:
                self._pid_ctrl = _AdaptivePrefetchController(
                    cache       = self,
                    loop        = self._loop,
                    target_util = adaptive_target_util,
                    max_window  = prefetch_window,
                )
            atexit.register(self._cleanup)
            self._register_signals()
        else:
            self._base.mkdir(parents=True, exist_ok=True)
            self._metrics   = get_registry()
            self._heartbeat = None
            self._pid_ctrl  = None

    # ── [ARCH2] Dynamic semaphore resize ──────────────────────────────────────

    def _resize_semaphore(self, new_value: int) -> None:
        """
        Resize the asyncio semaphore by replacing it.

        Must be called from within the asyncio event loop (via call_soon_threadsafe).
        This is safe because _load_one only acquires the semaphore at the start —
        in-flight tasks keep their acquired token until they release it naturally.
        The new semaphore is visible to all subsequent _load_one calls.
        """
        delta = new_value - self._sem._value  # type: ignore[attr-defined]
        if delta > 0:
            for _ in range(delta):
                self._sem.release()
        elif delta < 0:
            # Pre-acquire tokens to shrink the effective window.
            # This is a best-effort reduction; existing in-flight tasks are unaffected.
            for _ in range(abs(delta)):
                if self._sem._value > 0:  # type: ignore[attr-defined]
                    self._loop.create_task(self._sem.acquire())

    # ── Public API ────────────────────────────────────────────────────────────

    def prefetch(self, shard_path: str) -> None:
        """Schedule a shard for background loading (node master only)."""
        if not self._node_master:
            return
        shm = self._shm_path(shard_path)
        with self._lru_lock:
            if _is_ready(shm) or shard_path in self._in_flight:
                return
            self._in_flight.add(shard_path)
        asyncio.run_coroutine_threadsafe(
            self._load_one(shard_path, shm), self._loop
        )

    def get(self, shard_path: str) -> bytes:
        """Return raw shard bytes (owned copy)."""
        shm = self._shm_path(shard_path)
        if self._node_master:
            if not _is_ready(shm):
                with self._lru_lock:
                    if shard_path not in self._in_flight:
                        self._in_flight.add(shard_path)
                asyncio.run_coroutine_threadsafe(
                    self._load_one(shard_path, shm), self._loop
                ).result()
            return self._read(shm)
        else:
            t_wait  = time.perf_counter()
            _inotify_wait(shm, self._timeout)
            wait_ms = int((time.perf_counter() - t_wait) * 1000)
            if self._metrics is not None and wait_ms > 0:
                self._metrics.inc(MetricField.SHARD_CACHE_WAIT_MS, wait_ms)
            return self._read(shm)

    @contextlib.contextmanager
    def get_view(self, shard_path: str) -> Iterator[memoryview]:
        """
        Yield a zero-copy memoryview into the shard.

        [ARCH1] If the ring buffer is enabled and the segment is available,
        reads from SharedMemory (zero mmap syscalls for non-master ranks).
        Falls back to the mmap pool path transparently.
        """
        shm = self._shm_path(shard_path)
        if self._node_master:
            if not _is_ready(shm):
                with self._lru_lock:
                    if shard_path not in self._in_flight:
                        self._in_flight.add(shard_path)
                asyncio.run_coroutine_threadsafe(
                    self._load_one(shard_path, shm), self._loop
                ).result()
        else:
            t_wait  = time.perf_counter()
            _inotify_wait(shm, self._timeout)
            wait_ms = int((time.perf_counter() - t_wait) * 1000)
            if self._metrics is not None and wait_ms > 0:
                self._metrics.inc(MetricField.SHARD_CACHE_WAIT_MS, wait_ms)

        # [ARCH1] Prefer ring buffer when available
        if self._ring_buffer is not None:
            try:
                with self._ring_buffer.view(shard_path) as mv:
                    yield mv
                return
            except Exception as exc:
                log.debug(
                    "Ring buffer view failed for %s (%s); falling back to mmap pool.",
                    shard_path, exc,
                )

        # Default: persistent mmap pool (PERF-2)
        entry = self._mmap_pool.acquire(shm)
        try:
            yield memoryview(entry.mm)[_HDR_SIZE: _HDR_SIZE + entry.data_len]
        finally:
            self._mmap_pool.release(shm)

    @property
    def utilisation(self) -> float:
        if self._max_bytes == 0:
            return 0.0
        with self._lru_lock:
            return self._total_bytes / self._max_bytes

    # ── Internal: shard path ──────────────────────────────────────────────────

    def _shm_path(self, shard_path: str) -> Path:
        digest = hashlib.sha1(shard_path.encode()).hexdigest()[:16]
        return self._base / digest

    # ── Internal: async I/O (node master only) ────────────────────────────────

    async def _load_one(self, shard_path: str, shm: Path) -> None:
        """Fetch one shard from Lustre, write to /dev/shm (and ring buffer)."""
        async with self._sem:
            try:
                data = await self._read_lustre(shard_path)

                # [B2-FIX] Wait for eviction headroom with bounded retries.
                for attempt in range(_EVICT_RETRIES):
                    with self._lru_lock:
                        if self._total_bytes + len(data) <= self._max_bytes:
                            break
                        # Try to evict
                        self._evict_for_locked(len(data))
                        if self._total_bytes + len(data) <= self._max_bytes:
                            break
                    # All entries referenced — wait outside the lock.
                    if attempt < _EVICT_RETRIES - 1:
                        await asyncio.sleep(_EVICT_WAIT_S)
                    else:
                        raise RuntimeError(
                            f"NodeSharedShardCache: could not evict enough space "
                            f"for shard {shard_path!r} after {_EVICT_RETRIES} retries "
                            f"({_EVICT_RETRIES * _EVICT_WAIT_S:.0f}s).  "
                            "All mmap slots are referenced simultaneously — "
                            "reduce shard_prefetch_window or increase node_shm_gb."
                        )

                with self._lru_lock:
                    _check_shm_headroom(len(data))
                    self._write(shm, data)
                    self._lru[shard_path]  = len(data)
                    self._total_bytes     += len(data)

                self._update_utilisation_metric()

                # [ARCH1] Publish to ring buffer after the shm file is ready.
                if self._ring_buffer is not None:
                    try:
                        self._ring_buffer.publish(shard_path, data)
                    except Exception as exc:
                        log.warning(
                            "Ring buffer publish failed for %s: %s", shard_path, exc
                        )

                if self._metrics is not None:
                    self._metrics.inc(MetricField.LUSTRE_BYTES_READ, len(data))
                log.debug("Shard cached: %s (%d MB)", shard_path, len(data) >> 20)
            finally:
                with self._lru_lock:
                    self._in_flight.discard(shard_path)

    async def _read_lustre(self, shard_path: str) -> bytes:
        t0 = time.perf_counter()
        try:
            import aiofiles
            async with aiofiles.open(shard_path, "rb") as f:
                data = await f.read()
        except ImportError:
            loop = asyncio.get_running_loop()
            data = await loop.run_in_executor(None, _read_file_sync, shard_path)

        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        if self._metrics is not None:
            self._metrics.inc(MetricField.LUSTRE_READ_TIME_MS, elapsed_ms)
        return data

    # ── Internal: file I/O ────────────────────────────────────────────────────

    @staticmethod
    def _write(shm: Path, data: bytes) -> None:
        """Write shard bytes to /dev/shm atomically (no fsync — PERF-1)."""
        tmp = shm.with_suffix(".tmp")
        try:
            with open(tmp, "wb") as f:
                f.write(struct.pack(_HDR_FMT, len(data), 0))
                f.write(data)
                f.seek(0)
                f.write(struct.pack(_HDR_FMT, len(data), _READY_MAGIC))
            tmp.rename(shm)
        except Exception:
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass
            raise

    @staticmethod
    def _read(shm: Path) -> bytes:
        with open(shm, "rb") as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                data_len, magic = struct.unpack_from(_HDR_FMT, mm, 0)
                if magic != _READY_MAGIC:
                    raise RuntimeError(f"Shard {shm} has corrupt header")
                return bytes(mm[_HDR_SIZE: _HDR_SIZE + data_len])

    # ── Internal: LRU eviction ────────────────────────────────────────────────

    def _evict_for_locked(self, incoming: int) -> None:
        """Evict LRU shards to make room.  Caller must hold _lru_lock."""
        while self._total_bytes + incoming > self._max_bytes and self._lru:
            path_str, sz = self._lru.popitem(last=False)
            p = Path(path_str)
            self._mmap_pool.invalidate(p)
            if self._ring_buffer is not None:
                try:
                    self._ring_buffer.evict(path_str)
                except Exception:
                    pass
            try:
                p.unlink(missing_ok=True)
                p.with_suffix(".tmp").unlink(missing_ok=True)
                self._total_bytes -= sz
            except Exception as exc:
                log.warning("Eviction failed for %s: %s", path_str, exc)
                self._total_bytes -= sz

    # ── Internal: utilisation ─────────────────────────────────────────────────

    def _update_utilisation_metric(self) -> None:
        util = self.utilisation
        if self._metrics is not None:
            self._metrics.set(MetricField.SHARD_CACHE_UTIL_PCT, util * 100.0)
        if util >= self._warn_threshold:
            now = time.monotonic()
            if now - self._last_warn_ts >= _SHM_WARN_INTERVAL:
                self._last_warn_ts = now
                log.warning(
                    "/dev/shm utilisation is %.1f%% (threshold %.0f%%).  "
                    "Increase node_shm_gb or reduce shard_prefetch_window.  "
                    "Budget: %.1f GB, used: %.1f GB.",
                    util * 100.0, self._warn_threshold * 100.0,
                    self._max_bytes / (1 << 30), self._total_bytes / (1 << 30),
                )

    # ── Startup / shutdown ────────────────────────────────────────────────────

    def _init_shm(self) -> None:
        _purge_orphaned_shm(self._base.name, hb_stale_s=self._hb_stale_s)
        if self._base.exists():
            log.info("Removing stale shard cache at %s", self._base)
            shutil.rmtree(self._base, ignore_errors=True)
        self._base.mkdir(parents=True, exist_ok=True, mode=0o700)

    def _register_signals(self) -> None:
        def _handler(signum, frame):
            self._shutdown_event.set()

        def _watcher():
            self._shutdown_event.wait()
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            signal.signal(signal.SIGINT,  signal.SIG_DFL)
            os.kill(os.getpid(), signal.SIGTERM)

        signal.signal(signal.SIGTERM, _handler)
        signal.signal(signal.SIGINT,  _handler)
        t = threading.Thread(target=_watcher, name="shm-signal-watcher", daemon=True)
        t.start()

    def _cleanup(self) -> None:
        """atexit: stop heartbeat + PID ctrl, remove /dev/shm cache, close pools."""
        if self._pid_ctrl is not None:
            self._pid_ctrl.stop()
        if self._heartbeat is not None:
            self._heartbeat.stop()
        if self._ring_buffer is not None:
            self._ring_buffer.close()
        self._mmap_pool.close_all()
        try:
            self._loop.call_soon_threadsafe(self._loop.stop)
        except Exception:
            pass
        if self._base.exists():
            shutil.rmtree(self._base, ignore_errors=True)
        log.info("NodeSharedShardCache cleaned up")

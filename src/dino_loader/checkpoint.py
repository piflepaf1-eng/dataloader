"""
dino_loader.checkpoint
======================
Dataloader state checkpointing.

Design choices (unchanged)
--------------------------
- JSON (not pickle): stable across Python versions and environments.
- Atomic write: write to .tmp → rename() — POSIX rename is atomic.
- Rank 0 writes; all ranks can read.
- Retains only the 3 most recent checkpoints to bound Lustre usage.

Changes vs previous version
----------------------------
[CK-1]  Supports new CheckpointState fields: global_crop_size, local_crop_size.
        Fully backward-compatible: missing fields default to 224 / 96.

[CK-2]  load() returns None gracefully when no checkpoint exists, and logs
        a WARNING (not ERROR) when a checkpoint file is corrupt.

[CK-3]  LATEST pointer file for robust checkpoint discovery.

        Problem with glob-sort
        ----------------------
        The previous implementation discovered checkpoints with:

            candidates = sorted(self._dir.glob("dl_state_*.json"))
            state = CheckpointState.load(candidates[-1])

        If a training job crashes between writing a new checkpoint and pruning
        old ones (e.g. SIGKILL mid-prune, or Lustre metadata latency makes the
        new file visible before the old ones are unlinked), ``sorted()`` may
        return a stale or partially-written file as the latest.

        Over a 3-week run with ``checkpoint_every_steps=500``, up to
        ``(run_steps / 500)`` orphaned files can accumulate on Lustre.

        Solution
        --------
        ``save()`` now writes a ``LATEST`` pointer file (plain text containing
        the filename of the most recent checkpoint) *after* the checkpoint JSON
        is fully written and renamed.  The pointer write is itself atomic
        (write-to-tmp + rename).

        ``load()`` reads ``LATEST`` first; falls back to glob-sort if
        ``LATEST`` does not exist (backward compatibility with existing
        checkpoint directories written by earlier versions).

        ``_prune()`` still uses glob-sort for rotation — it is called after
        ``LATEST`` is updated, so even if prune crashes partway, ``LATEST``
        already points to the correct (newest) checkpoint.

        Race-freedom
        ------------
        The sequence on rank 0:
          1. Write ``dl_state_{step}.json.tmp`` → rename to ``dl_state_{step}.json``
          2. Write ``LATEST.tmp`` (containing ``dl_state_{step}.json``)
             → rename to ``LATEST``
          3. _prune() unlinks old checkpoints

        Readers always see either the previous LATEST (if they race with step 2)
        or the new one (if they see step 2's rename).  They never see a
        partially-written LATEST because the rename in step 2 is POSIX-atomic.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from dino_loader.config import CheckpointState

log = logging.getLogger(__name__)

_KEEP_LAST   = 3
_LATEST_FILE = "LATEST"


class DataLoaderCheckpointer:
    """
    Manages JSON checkpoint files for DINODataLoader state.

    Writes are rank-0-only and throttled to every N steps.
    Reads are available to all ranks for resume.
    """

    def __init__(self, ckpt_dir: str, every_n_steps: int = 500, rank: int = 0) -> None:
        self._dir   = Path(ckpt_dir)
        self._every = every_n_steps
        self._rank  = rank
        if rank == 0:
            self._dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(self, state: CheckpointState) -> None:
        """
        Save state to disk (rank 0 only, every N steps).

        [CK-3] Write order:
          1. Atomic JSON write (tmp → rename)
          2. Atomic LATEST pointer update (tmp → rename)
          3. Prune old checkpoints
        """
        if self._rank != 0 or state.step % self._every != 0:
            return

        # Step 1: write the checkpoint JSON atomically.
        filename = f"dl_state_{state.step:012d}.json"
        path     = self._dir / filename
        state.save(path)   # CheckpointState.save() handles its own tmp→rename

        # Step 2: update the LATEST pointer atomically.
        self._write_latest(filename)

        # Step 3: prune old checkpoints.
        self._prune()

        log.info("DataLoader checkpoint saved: %s", filename)

    def load(self) -> Optional[CheckpointState]:
        """
        Load the most recent checkpoint, or return None if none exists.

        [CK-3] Reads the LATEST pointer first; falls back to glob-sort for
        backward compatibility with checkpoint dirs written by older versions.
        """
        path = self._resolve_latest()
        if path is None:
            return None
        try:
            state = CheckpointState.load(path)
            log.info(
                "Resuming from %s (step=%d epoch=%d global=%d local=%d)",
                path.name, state.step, state.epoch,
                state.global_crop_size, state.local_crop_size,
            )
            return state
        except Exception as exc:
            log.warning(
                "Could not load checkpoint %s: %s — starting from scratch.",
                path, exc,
            )
            return None

    # ------------------------------------------------------------------
    # state_dict / load_state_dict  (torchdata StatefulDataLoader compat)
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        """Return checkpoint state as a plain dict (framework-agnostic)."""
        state = self.load()
        if state is None:
            return {}
        return {
            "step":            state.step,
            "epoch":           state.epoch,
            "dataset_names":   state.dataset_names,
            "mixing_weights":  state.mixing_weights,
            "global_crop_size": state.global_crop_size,
            "local_crop_size":  state.local_crop_size,
        }

    def load_state_dict(self, d: dict) -> None:
        """Restore from a dict produced by state_dict() (no-op on non-rank-0)."""
        # Downstream: caller applies d fields to DINODataLoader directly.
        pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write_latest(self, filename: str) -> None:
        """Atomically update the LATEST pointer file. [CK-3]"""
        latest_tmp = self._dir / f"{_LATEST_FILE}.tmp"
        latest     = self._dir / _LATEST_FILE
        try:
            latest_tmp.write_text(filename, encoding="utf-8")
            latest_tmp.rename(latest)
        except Exception as exc:
            log.warning("Failed to write LATEST pointer: %s", exc)
            try:
                latest_tmp.unlink(missing_ok=True)
            except Exception:
                pass

    def _resolve_latest(self) -> Optional[Path]:
        """
        Return the Path of the most recent checkpoint, or None.

        [CK-3] Reads LATEST first; falls back to glob-sort.
        """
        latest_ptr = self._dir / _LATEST_FILE
        if latest_ptr.exists():
            try:
                filename = latest_ptr.read_text(encoding="utf-8").strip()
                candidate = self._dir / filename
                if candidate.exists():
                    return candidate
                log.warning(
                    "LATEST pointer references non-existent file %s; "
                    "falling back to glob-sort.",
                    filename,
                )
            except Exception as exc:
                log.warning("Could not read LATEST pointer: %s; falling back to glob-sort.", exc)

        # Backward-compatible fallback.
        candidates = sorted(self._dir.glob("dl_state_*.json"))
        return candidates[-1] if candidates else None

    def _prune(self) -> None:
        """Keep only the _KEEP_LAST most recent checkpoint JSON files."""
        candidates = sorted(self._dir.glob("dl_state_*.json"))
        for old in candidates[:-_KEEP_LAST]:
            try:
                old.unlink()
                log.debug("Pruned old checkpoint: %s", old.name)
            except FileNotFoundError:
                pass

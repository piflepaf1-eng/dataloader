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

Changes in this version
-----------------------
[CK-1]  Supports CheckpointState fields: global_crop_size, local_crop_size (retained).
[CK-2]  load() returns None gracefully; warns on corrupt files (retained).
[CK-3]  LATEST pointer file for robust checkpoint discovery (retained).

[M3-FIX] SHA-256 integrity envelope.                                   ← FIX M3
         Previously a checkpoint file truncated by a node crash (SIGKILL
         during the .tmp write, or Lustre write-back timeout) would produce
         a JSONDecodeError at resume time with no actionable message.

         New format: CheckpointState.save() wraps the payload in a JSON
         envelope:

             {"payload": { ... state fields ... }, "sha256": "<hex>"}

         The sha256 field is the SHA-256 hex digest of
         ``json.dumps(payload, indent=2)``.  CheckpointState.load() verifies
         the checksum before deserialising.  A mismatch raises ValueError with
         the stored and computed hashes so operators can identify corruption.

         Backward compatibility: CheckpointState.load() detects the legacy
         flat format (no "payload" key) and reads it without checksum
         verification — existing checkpoints continue to work.  A WARNING
         is logged so operators know they're on the old format.

         Note: the checksum lives in config.py alongside CheckpointState
         (the class that owns save/load).  This file (checkpoint.py) only
         manages the checkpointer lifecycle and does not need to change for
         the integrity feature itself.  This comment is here for traceability.
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

    File format (new — envelope with SHA-256)
    -----------------------------------------
    ::

        {
          "payload": {
            "step": 1000,
            "epoch": 5,
            "dataset_names": ["laion2b", "imagenet22k"],
            "mixing_weights": [0.7, 0.3],
            "global_crop_size": 224,
            "local_crop_size": 96
          },
          "sha256": "a3f2..."
        }

    Legacy format (flat, no checksum): still supported for backward compat.
    """

    def __init__(self, ckpt_dir: str, every_n_steps: int = 500, rank: int = 0) -> None:
        self._dir   = Path(ckpt_dir)
        self._every = every_n_steps
        self._rank  = rank
        if rank == 0:
            self._dir.mkdir(parents=True, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def save(self, state: CheckpointState) -> None:
        """
        Save state to disk (rank 0 only, every N steps).

        Write order (race-free):
          1. Atomic JSON write with SHA-256 envelope (tmp → rename).
          2. Atomic LATEST pointer update (tmp → rename).
          3. Prune old checkpoints (best-effort; does not affect LATEST).
        """
        if self._rank != 0 or state.step % self._every != 0:
            return

        filename = f"dl_state_{state.step:012d}.json"
        path     = self._dir / filename
        state.save(path)   # CheckpointState.save() handles tmp→rename + checksum

        self._write_latest(filename)
        self._prune()

        log.info("DataLoader checkpoint saved: %s", filename)

    def load(self) -> Optional[CheckpointState]:
        """
        Load the most recent checkpoint, or return None if none exists.

        Reads the LATEST pointer first; falls back to glob-sort for backward
        compatibility with checkpoint dirs written by older versions.

        Returns None (with a WARNING) if the checkpoint file is corrupt or
        fails the SHA-256 integrity check.
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
        except ValueError as exc:
            # Includes SHA-256 mismatch (M3) and other validation errors.
            log.warning(
                "Checkpoint %s failed integrity check: %s — starting from scratch.",
                path, exc,
            )
            return None
        except Exception as exc:
            log.warning(
                "Could not load checkpoint %s: %s — starting from scratch.",
                path, exc,
            )
            return None

    # ── state_dict / load_state_dict (torchdata StatefulDataLoader compat) ────

    def state_dict(self) -> dict:
        """Return checkpoint state as a plain dict."""
        state = self.load()
        if state is None:
            return {}
        return {
            "step":             state.step,
            "epoch":            state.epoch,
            "dataset_names":    state.dataset_names,
            "mixing_weights":   state.mixing_weights,
            "global_crop_size": state.global_crop_size,
            "local_crop_size":  state.local_crop_size,
        }

    def load_state_dict(self, d: dict) -> None:
        """Restore from a dict produced by state_dict(). Caller applies fields."""
        pass

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _write_latest(self, filename: str) -> None:
        """Atomically update the LATEST pointer file."""
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
        """Return the Path of the most recent checkpoint, or None."""
        latest_ptr = self._dir / _LATEST_FILE
        if latest_ptr.exists():
            try:
                filename  = latest_ptr.read_text(encoding="utf-8").strip()
                candidate = self._dir / filename
                if candidate.exists():
                    return candidate
                log.warning(
                    "LATEST pointer references non-existent file %s; "
                    "falling back to glob-sort.",
                    filename,
                )
            except Exception as exc:
                log.warning(
                    "Could not read LATEST pointer: %s; falling back to glob-sort.",
                    exc,
                )

        # Backward-compatible fallback
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

"""
dino_loader.datasets.dataset
============================
Dataset discovery and shard resolution for the new filesystem hierarchy:

    <root>/
      <confidentiality>/
        <modality>/
          <dataset_name>/
            raw/
            pivot/
            outputs/
              <strategy>/          ← e.g. "default"
                <split>/           ← e.g. "train", "val"
                  shard-000000.tar
                  shard-000000.idx
            metadonnees/
            subset_selection/

Key design decisions
--------------------
- The dataloader **always** resolves shards under ``outputs/<strategy>/<split>/``.
  It never touches ``raw/``, ``pivot/``, ``metadonnees/``, or ``subset_selection/``.
- ``strategy`` is a single string (default: ``"default"``).  Only one strategy
  is active per training run; selecting multiple strategies simultaneously is
  intentionally not supported.
- ``allowed_splits`` filters the split sub-directories (train / val / test / …)
  independently from the strategy.
- A dataset is considered **invalid** (ignored by stub generation and listing)
  if it has no valid ``.tar`` + ``.idx`` pair anywhere under ``outputs/``.
- A dataset with the same name can exist under several
  confidentiality / modality combinations; ``resolve()`` aggregates shards
  from all matching (conf, mod) pairs.
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from dino_loader.config import DatasetSpec
from dino_loader.datasets.utils import validate_webdataset_shard
from dino_loader.datasets.settings import resolve_datasets_root

log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_STRATEGY = "default"

#: Sub-directories inside a dataset root that are NOT strategy folders.
#: They are always skipped when scanning ``outputs/``.
_OUTPUTS_NON_STRATEGY_DIRS: frozenset[str] = frozenset()

#: Top-level sub-directories inside a dataset root that are not
#: confidentiality/modality hierarchies (used only by the CLI scaffolder).
DATASET_RESERVED_DIRS = frozenset(
    {"raw", "pivot", "outputs", "metadonnees", "subset_selection"}
)


# ── Filter dataclasses ────────────────────────────────────────────────────────

@dataclass
class GlobalDatasetFilter:
    """
    Filters applied globally to every dataset unless overridden by a
    per-dataset ``DatasetConfig``.

    Parameters
    ----------
    allowed_confidentialities
        If set, only confidentiality directories in this list are visited.
        ``None`` means "all confidentialities".
    allowed_modalities
        If set, only modality directories in this list are visited.
        ``None`` means "all modalities".
    allowed_datasets
        If set, only datasets whose name is in this list are resolved.
        ``None`` means "all datasets".
    allowed_splits
        If set, only split sub-directories in this list are included.
        ``None`` means "all splits".
    strategy
        Strategy folder to look in under ``outputs/``.
        Defaults to ``DEFAULT_STRATEGY`` (``"default"``).
    """

    allowed_confidentialities: Optional[List[str]] = None
    allowed_modalities: Optional[List[str]] = None
    allowed_datasets: Optional[List[str]] = None
    allowed_splits: Optional[List[str]] = None
    strategy: str = DEFAULT_STRATEGY


@dataclass
class DatasetConfig:
    """
    Per-dataset overrides; take precedence over ``GlobalDatasetFilter``.

    Parameters
    ----------
    allowed_confidentialities
        Restricts which confidentiality directories are visited for *this*
        dataset.  Overrides the global filter when set.
    allowed_modalities
        Restricts which modality directories are visited for *this* dataset.
        Overrides the global filter when set.
    allowed_splits
        Restricts which splits are included for *this* dataset.
        Overrides the global filter when set.
    strategy
        Strategy to use for *this* dataset.  Overrides the global filter
        when set.  ``None`` means "fall back to the global strategy".
    weight
        Mixing weight for this dataset (re-normalised automatically).
    """

    allowed_confidentialities: Optional[List[str]] = None
    allowed_modalities: Optional[List[str]] = None
    allowed_splits: Optional[List[str]] = None
    strategy: Optional[str] = None          # None → inherit from global
    weight: float = 1.0


# ── Dataset class ─────────────────────────────────────────────────────────────

class Dataset:
    """
    Represents a named dataset that may exist under multiple
    (confidentiality, modality) pairs in the filesystem.

    Usage
    -----
    ::

        from dino_loader.datasets.dataset import Dataset, GlobalDatasetFilter

        ds = Dataset("imagenet")
        shards = ds.resolve(
            global_filter=GlobalDatasetFilter(
                allowed_confidentialities=["public"],
                allowed_splits=["train"],
                strategy="default",
            )
        )
        spec = ds.to_spec(global_filter=...)
    """

    def __init__(self, name: str, root_path: Optional[str] = None) -> None:
        self.name = name
        self.root_path = resolve_datasets_root(root_path)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _effective(
        self,
        global_filter: Optional[GlobalDatasetFilter],
        config: Optional[DatasetConfig],
        attr: str,
    ) -> Optional[Set[str]]:
        """Return the effective allowed-set for *attr*, as a set or None."""
        local_val = getattr(config, attr, None) if config else None
        if local_val is not None:
            return set(local_val)
        global_val = getattr(global_filter, attr, None) if global_filter else None
        if global_val is not None:
            return set(global_val)
        return None

    def _effective_strategy(
        self,
        global_filter: Optional[GlobalDatasetFilter],
        config: Optional[DatasetConfig],
    ) -> str:
        """Return the active strategy string (never None)."""
        if config and config.strategy is not None:
            return config.strategy
        if global_filter:
            return global_filter.strategy
        return DEFAULT_STRATEGY

    # ── Public API ────────────────────────────────────────────────────────────

    def resolve(
        self,
        global_filter: Optional[GlobalDatasetFilter] = None,
        config: Optional[DatasetConfig] = None,
    ) -> List[str]:
        """
        Discover and return all valid shard paths for this dataset.

        The method walks::

            <root>/<conf>/<mod>/<name>/outputs/<strategy>/<split>/*.tar

        filtering at each level according to the active ``GlobalDatasetFilter``
        and ``DatasetConfig``.  A shard is included only when its companion
        ``.idx`` file exists alongside it.

        Parameters
        ----------
        global_filter
            Broad filters (confidentialities, modalities, splits, strategy).
        config
            Per-dataset overrides; take precedence over *global_filter*.

        Returns
        -------
        list of str
            Sorted list of absolute ``.tar`` paths.  Empty if nothing matches
            or the dataset root does not exist.
        """
        # ── Global dataset name filter ────────────────────────────────────────
        if global_filter and global_filter.allowed_datasets is not None:
            if self.name not in global_filter.allowed_datasets:
                return []

        if not os.path.exists(self.root_path):
            log.warning("Dataset root path does not exist: %s", self.root_path)
            return []

        allowed_confs   = self._effective(global_filter, config, "allowed_confidentialities")
        allowed_mods    = self._effective(global_filter, config, "allowed_modalities")
        allowed_splits  = self._effective(global_filter, config, "allowed_splits")
        strategy        = self._effective_strategy(global_filter, config)

        valid_shards: List[str] = []

        # ── Walk conf / mod ───────────────────────────────────────────────────
        for conf in _listdirs(self.root_path):
            if allowed_confs is not None and conf not in allowed_confs:
                continue

            conf_path = os.path.join(self.root_path, conf)

            for mod in _listdirs(conf_path):
                if allowed_mods is not None and mod not in allowed_mods:
                    continue

                dataset_path = os.path.join(conf_path, mod, self.name)
                if not os.path.isdir(dataset_path):
                    continue

                outputs_path = os.path.join(dataset_path, "outputs")
                if not os.path.isdir(outputs_path):
                    log.debug(
                        "Dataset '%s' has no outputs/ directory at %s",
                        self.name, dataset_path,
                    )
                    continue

                strategy_path = os.path.join(outputs_path, strategy)
                if not os.path.isdir(strategy_path):
                    log.debug(
                        "Dataset '%s': strategy '%s' not found at %s",
                        self.name, strategy, outputs_path,
                    )
                    continue

                # ── Walk splits ───────────────────────────────────────────────
                for split in _listdirs(strategy_path):
                    if allowed_splits is not None and split not in allowed_splits:
                        continue

                    split_path = os.path.join(strategy_path, split)
                    valid_shards.extend(
                        _collect_shards(split_path, self.name)
                    )

        return sorted(valid_shards)

    def to_spec(
        self,
        global_filter: Optional[GlobalDatasetFilter] = None,
        config: Optional[DatasetConfig] = None,
    ) -> Optional[DatasetSpec]:
        """
        Resolve shards and build a ``DatasetSpec`` ready for ``DINODataLoader``.

        Returns ``None`` if no valid shards are found (so callers can safely
        filter with ``filter(None, [ds.to_spec(...) for ds in datasets])``).
        """
        shards = self.resolve(global_filter, config)
        if not shards:
            return None
        weight = config.weight if config else 1.0
        return DatasetSpec(name=self.name, shards=shards, weight=weight)


# ── Module-level helpers ──────────────────────────────────────────────────────

def _listdirs(path: str) -> List[str]:
    """Return sorted list of sub-directory names under *path* (no files)."""
    try:
        return sorted(
            e for e in os.listdir(path)
            if os.path.isdir(os.path.join(path, e))
        )
    except PermissionError as exc:
        log.warning("Cannot list directory %s: %s", path, exc)
        return []


def _collect_shards(split_path: str, dataset_name: str) -> List[str]:
    """
    Return all ``.tar`` paths in *split_path* that have a companion ``.idx``.

    Logs a warning for any shard missing its index file.
    """
    result: List[str] = []
    try:
        entries = os.listdir(split_path)
    except PermissionError as exc:
        log.warning("Cannot list split directory %s: %s", split_path, exc)
        return result

    for fname in sorted(entries):
        if not fname.endswith(".tar"):
            continue
        tar_path = os.path.join(split_path, fname)
        idx_path = os.path.join(split_path, fname[:-4] + ".idx")
        if os.path.exists(idx_path):
            result.append(tar_path)
        else:
            log.warning(
                "Dataset '%s': missing .idx for shard %s — shard skipped.",
                dataset_name, tar_path,
            )
    return result

"""
dino_loader.datasets.dataset
============================
Dataset discovery and shard resolution.

Filesystem hierarchy
---------------------
Each confidentiality lives in its **own** root directory, registered in the
global :class:`~dino_loader.datasets.settings.ConfidentialityRegistry`.
Within a confidentiality root the layout is::

    <conf_root>/
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

Multiple confidentialities can be active simultaneously; each has its own
path (possibly on different filesystems / Lustre mount points).

Key design decisions
--------------------
- The dataloader **always** resolves shards under ``outputs/<strategy>/<split>/``.
  It never touches ``raw/``, ``pivot/``, ``metadonnees/``, or ``subset_selection/``.
- ``strategy`` is a single string (default: ``"default"``).  Only one strategy
  is active per training run.
- A dataset is considered **invalid** if it has no valid ``.tar`` + ``.idx``
  pair anywhere under ``outputs/``.
- A dataset with the same name can exist under several
  (confidentiality, modality) combinations; :meth:`Dataset.resolve` aggregates
  shards from all matching pairs.

Discovery metadata
------------------
:meth:`Dataset.to_spec` performs a **single** filesystem pass and populates
:attr:`~dino_loader.config.DatasetSpec.confidentialities`,
:attr:`~dino_loader.config.DatasetSpec.modalities`,
:attr:`~dino_loader.config.DatasetSpec.splits`, and
:attr:`~dino_loader.config.DatasetSpec.strategies` on the returned spec.
The stub generator reads these fields to emit ``Literal``-typed
``TypedDatasetSpec`` subclasses in ``hub.py`` for IDE autocomplete.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from dino_loader.config import DatasetSpec
from dino_loader.datasets.settings import (
    ConfidentialityMount,
    get_confidentiality_mounts,
    resolve_path_for_confidentiality,
)
from dino_loader.datasets.utils import validate_webdataset_shard

log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_STRATEGY = "default"

#: Top-level sub-directories inside a dataset root that are reserved names
#: (not modality folders).  Used by the CLI scaffolder and stub generator.
DATASET_RESERVED_DIRS = frozenset(
    {"raw", "pivot", "outputs", "metadonnees", "subset_selection"}
)


# ── Private filesystem helpers ─────────────────────────────────────────────────

def _listdirs(path: str) -> List[str]:
    """
    Return sorted sub-directory names under *path*, ignoring hidden entries
    and any filesystem errors (e.g. permission denied).
    """
    try:
        return sorted(
            entry for entry in os.listdir(path)
            if not entry.startswith(".")
            and os.path.isdir(os.path.join(path, entry))
        )
    except OSError:
        return []


# ── Filter dataclasses ────────────────────────────────────────────────────────

@dataclass
class GlobalDatasetFilter:
    """
    Filters applied globally to every dataset unless overridden by a
    per-dataset :class:`DatasetConfig`.

    Parameters
    ----------
    allowed_confidentialities:
        If set, only confidentialities whose **name** appears in this list are
        visited.  ``None`` means all registered confidentialities.
    allowed_modalities:
        If set, only modality directories in this list are visited.
        ``None`` means all modalities.
    allowed_datasets:
        If set, only datasets whose name is in this list are resolved.
        ``None`` means all datasets.
    allowed_splits:
        If set, only split sub-directories in this list are included.
        ``None`` means all splits.
    strategy:
        Strategy folder to look in under ``outputs/``.
        Defaults to :data:`DEFAULT_STRATEGY` (``"default"``).
    """

    allowed_confidentialities: Optional[List[str]] = None
    allowed_modalities: Optional[List[str]] = None
    allowed_datasets: Optional[List[str]] = None
    allowed_splits: Optional[List[str]] = None
    strategy: str = DEFAULT_STRATEGY


@dataclass
class DatasetConfig:
    """
    Per-dataset overrides; take precedence over :class:`GlobalDatasetFilter`.

    Parameters
    ----------
    allowed_confidentialities:
        Restrict which confidentialities are visited for *this* dataset.
    allowed_modalities:
        Restrict which modalities are visited for *this* dataset.
    allowed_splits:
        Restrict which splits are included for *this* dataset.
    strategy:
        Strategy for *this* dataset.  ``None`` inherits from the global filter.
    weight:
        Mixing weight for this dataset (re-normalised automatically).
    """

    allowed_confidentialities: Optional[List[str]] = None
    allowed_modalities: Optional[List[str]] = None
    allowed_splits: Optional[List[str]] = None
    strategy: Optional[str] = None
    weight: float = 1.0


# ── Dataset class ─────────────────────────────────────────────────────────────

class Dataset:
    """
    Represents a named dataset that may exist under multiple
    ``(confidentiality, modality)`` pairs across potentially different
    filesystem roots.

    Usage
    -----
    ::

        from dino_loader.datasets.dataset import Dataset, GlobalDatasetFilter

        ds = Dataset("imagenet")
        shards = ds.resolve(
            global_filter=GlobalDatasetFilter(
                allowed_confidentialities=["public"],
                allowed_splits=["train"],
            )
        )
        spec = ds.to_spec(global_filter=...)
        # spec.confidentialities == ["public"]
        # spec.modalities        == ["rgb"]
        # spec.splits            == ["train"]
        # spec.strategies        == ["default"]
    """

    def __init__(self, name: str) -> None:
        """
        Parameters
        ----------
        name:
            Dataset name, e.g. ``"imagenet"``, ``"custom"``.
            The library discovers actual paths from the
            :class:`~dino_loader.datasets.settings.ConfidentialityRegistry`.
        """
        self.name = name

    # ── Private helpers ───────────────────────────────────────────────────

    @staticmethod
    def _effective_set(
        global_filter: Optional[GlobalDatasetFilter],
        config: Optional[DatasetConfig],
        attr: str,
    ) -> Optional[Set[str]]:
        """Return the effective allowed-set for *attr* (local overrides global)."""
        local_val = getattr(config, attr, None) if config else None
        if local_val is not None:
            return set(local_val)
        global_val = getattr(global_filter, attr, None) if global_filter else None
        if global_val is not None:
            return set(global_val)
        return None

    @staticmethod
    def _effective_strategy(
        global_filter: Optional[GlobalDatasetFilter],
        config: Optional[DatasetConfig],
    ) -> str:
        if config and config.strategy is not None:
            return config.strategy
        if global_filter:
            return global_filter.strategy
        return DEFAULT_STRATEGY

    def _resolve_with_metadata(
        self,
        global_filter: Optional[GlobalDatasetFilter] = None,
        config: Optional[DatasetConfig] = None,
    ) -> Tuple[List[str], Dict[str, Set[str]]]:
        """
        Walk all registered confidentiality mounts and return both the resolved
        shard paths **and** the discovery metadata collected along the way.

        This is the single canonical filesystem traversal used by both
        :meth:`resolve` and :meth:`to_spec`, avoiding a double walk.

        Parameters
        ----------
        global_filter:
            Broad filters (confidentialities, modalities, splits, strategy).
        config:
            Per-dataset overrides (take precedence over *global_filter*).

        Returns
        -------
        shards : List[str]
            Absolute paths to all valid ``.tar`` shards, sorted lexicographically.
        meta : Dict[str, Set[str]]
            Keys ``"confidentialities"``, ``"modalities"``, ``"splits"``,
            ``"strategies"`` — each value is the set of directory names that
            actually contributed at least one valid shard.  Only directories
            from which shards were accepted are recorded, so the metadata
            always reflects what is *reachable*, not what was *visited*.
        """
        allowed_confs  = self._effective_set(global_filter, config, "allowed_confidentialities")
        allowed_mods   = self._effective_set(global_filter, config, "allowed_modalities")
        allowed_splits = self._effective_set(global_filter, config, "allowed_splits")
        strategy       = self._effective_strategy(global_filter, config)

        shards: List[str] = []
        meta: Dict[str, Set[str]] = {
            "confidentialities": set(),
            "modalities":        set(),
            "splits":            set(),
            "strategies":        set(),
        }

        for mount in get_confidentiality_mounts():
            if allowed_confs is not None and mount.name not in allowed_confs:
                continue
            if not mount.path.is_dir():
                log.debug("Confidentiality path does not exist, skipping: %s", mount.path)
                continue

            conf_root = str(mount.path)

            for mod in _listdirs(conf_root):
                if allowed_mods is not None and mod not in allowed_mods:
                    continue

                dataset_path = os.path.join(conf_root, mod, self.name)
                if not os.path.isdir(dataset_path):
                    continue

                outputs_path = os.path.join(dataset_path, "outputs", strategy)
                if not os.path.isdir(outputs_path):
                    log.debug(
                        "No outputs/%s for %s/%s/%s",
                        strategy, mount.name, mod, self.name,
                    )
                    continue

                for split in _listdirs(outputs_path):
                    if allowed_splits is not None and split not in allowed_splits:
                        continue

                    split_path = os.path.join(outputs_path, split)
                    contributed = False

                    for fname in sorted(os.listdir(split_path)):
                        if not fname.endswith(".tar"):
                            continue
                        tar_path = os.path.join(split_path, fname)
                        idx_path = os.path.join(split_path, fname[:-4] + ".idx")
                        if os.path.isfile(idx_path):
                            shards.append(tar_path)
                            contributed = True
                        else:
                            log.debug("Missing .idx for shard %s, skipping.", tar_path)

                    # Only record metadata for splits that contributed shards.
                    # This keeps discovery metadata *tight*: it reflects what is
                    # actually reachable, not everything that was visited.
                    if contributed:
                        meta["confidentialities"].add(mount.name)
                        meta["modalities"].add(mod)
                        meta["splits"].add(split)
                        meta["strategies"].add(strategy)

        return sorted(shards), meta

    # ── Public API ────────────────────────────────────────────────────────

    def resolve(
        self,
        global_filter: Optional[GlobalDatasetFilter] = None,
        config: Optional[DatasetConfig] = None,
    ) -> List[str]:
        """
        Discover and return all valid shard paths for this dataset.

        The method walks all registered confidentiality mounts, filtering at
        each level.  A shard is included only when its companion ``.idx``
        file exists alongside it.

        Parameters
        ----------
        global_filter:
            Broad filters (confidentialities, modalities, splits, strategy).
        config:
            Per-dataset overrides (take precedence over *global_filter*).

        Returns
        -------
        list[str]
            Absolute paths to all valid ``.tar`` shards, sorted lexicographically.
        """
        shards, _ = self._resolve_with_metadata(
            global_filter=global_filter,
            config=config,
        )
        return shards

    def locations(self) -> List[Tuple[str, str, Path]]:
        """
        Return all ``(confidentiality_name, modality, dataset_path)`` triples
        where this dataset has been found across the registered mounts.

        Useful for introspection and diagnostics.
        """
        results: List[Tuple[str, str, Path]] = []
        for mount in get_confidentiality_mounts():
            if not mount.path.is_dir():
                continue
            for mod in _listdirs(str(mount.path)):
                dataset_path = mount.path / mod / self.name
                if dataset_path.is_dir():
                    results.append((mount.name, mod, dataset_path))
        return results

    def to_spec(
        self,
        global_filter: Optional[GlobalDatasetFilter] = None,
        config: Optional[DatasetConfig] = None,
    ) -> Optional[DatasetSpec]:
        """
        Build a :class:`~dino_loader.config.DatasetSpec` from resolved shards.

        Unlike a raw :meth:`resolve` call, this method populates the four
        *discovery metadata* fields on the returned spec:

        * ``confidentialities`` — sorted list of confidentiality names that
          contributed at least one shard.
        * ``modalities`` — sorted list of modality directories.
        * ``splits`` — sorted list of split directories.
        * ``strategies`` — sorted list of strategy directories.

        The stub generator (``stub_gen.py``) reads these fields to emit
        ``Literal``-typed ``TypedDatasetSpec`` subclasses in ``hub.py``,
        giving IDEs concrete autocomplete for per-dataset metadata.

        The filesystem is traversed **once** (via :meth:`_resolve_with_metadata`)
        regardless of whether only shards or full metadata is needed.

        Parameters
        ----------
        global_filter:
            Broad filters forwarded to :meth:`_resolve_with_metadata`.
        config:
            Per-dataset overrides forwarded to :meth:`_resolve_with_metadata`.

        Returns
        -------
        DatasetSpec
            Fully populated spec including discovery metadata.
        None
            When no valid shards are found (unknown dataset name, missing
            mount, or all shards filtered out).  Callers can use::

                specs = [s for s in (ds.to_spec() for ds in datasets) if s is not None]
        """
        shards, meta = self._resolve_with_metadata(
            global_filter=global_filter,
            config=config,
        )

        if not shards:
            return None

        weight = config.weight if config else 1.0
        return DatasetSpec(
            name              = self.name,
            shards            = shards,
            weight            = weight,
            # ── Discovery metadata ────────────────────────────────────────
            # Sorted for determinism: same filesystem → same spec, always.
            confidentialities = sorted(meta["confidentialities"]),
            modalities        = sorted(meta["modalities"]),
            splits            = sorted(meta["splits"]),
            strategies        = sorted(meta["strategies"]),
        )

    def __repr__(self) -> str:
        return f"Dataset({self.name!r})"

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
:attr:`~dino_loader.datasets.spec.DatasetSpec.confidentialities`,
:attr:`~dino_loader.datasets.spec.DatasetSpec.modalities`,
:attr:`~dino_loader.datasets.spec.DatasetSpec.splits`, and
:attr:`~dino_loader.datasets.spec.DatasetSpec.strategies` on the returned spec.
The stub generator reads these fields to emit ``Literal``-typed
``TypedDatasetSpec`` subclasses in ``hub/`` for IDE autocomplete.

_default_filter
---------------
:class:`Dataset` accepts an optional ``_default_filter`` constructor argument.
Modality sub-modules in ``hub/`` use this to pre-restrict every
``to_spec()`` / ``resolve()`` call to a single modality without requiring
the caller to pass a filter explicitly.

:func:`_merge_filters` merges two :class:`GlobalDatasetFilter` objects with
*override* winning per field over *base*.  This is the mechanism that allows
callers to further restrict a pre-filtered dataset (e.g. additionally
filtering to ``allowed_splits=["train"]``).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# [REFACTOR] DatasetSpec now lives here in the datasets sub-package.
# No upward dependency on dino_loader.config.
from dino_loader.datasets.spec import DatasetSpec  # noqa: F401
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
    allowed_modalities:        Optional[List[str]] = None
    allowed_datasets:          Optional[List[str]] = None
    allowed_splits:            Optional[List[str]] = None
    strategy:                  str                 = DEFAULT_STRATEGY


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
    allowed_modalities:        Optional[List[str]] = None
    allowed_splits:            Optional[List[str]] = None
    strategy:                  Optional[str]       = None
    weight:                    float               = 1.0


# ── Filter merge helper ───────────────────────────────────────────────────────

def _merge_filters(
    base: Optional[GlobalDatasetFilter],
    override: Optional[GlobalDatasetFilter],
) -> Optional[GlobalDatasetFilter]:
    """
    Merge two :class:`GlobalDatasetFilter` objects.

    *override* wins on every field it explicitly sets (i.e. is not ``None``
    for list fields, or not the default value for ``strategy``).  *base*
    fills anything *override* leaves unset.

    If both are ``None``, returns ``None``.

    This is used by:

    - :meth:`Dataset.resolve` and :meth:`Dataset.to_spec` to merge
      ``self._default_filter`` (set by modality hub modules) with the
      caller-supplied filter.
    - The generated typed Dataset subclasses in ``hub/<modality>.py``.

    Examples
    --------
    ::

        base     = GlobalDatasetFilter(allowed_modalities=["rgb"],
                                       allowed_splits=["train"])
        override = GlobalDatasetFilter(allowed_splits=["val"])

        result = _merge_filters(base, override)
        # result.allowed_modalities == ["rgb"]   ← from base
        # result.allowed_splits     == ["val"]   ← override wins
    """
    if base is None and override is None:
        return None
    if base is None:
        return override
    if override is None:
        return base

    # Both set: override wins per field, base fills unset fields.
    return GlobalDatasetFilter(
        allowed_confidentialities=(
            override.allowed_confidentialities
            if override.allowed_confidentialities is not None
            else base.allowed_confidentialities
        ),
        allowed_modalities=(
            override.allowed_modalities
            if override.allowed_modalities is not None
            else base.allowed_modalities
        ),
        allowed_datasets=(
            override.allowed_datasets
            if override.allowed_datasets is not None
            else base.allowed_datasets
        ),
        allowed_splits=(
            override.allowed_splits
            if override.allowed_splits is not None
            else base.allowed_splits
        ),
        # strategy: override wins only if it differs from the default,
        # so that a modality filter's default strategy doesn't silently
        # stomp a caller's explicit strategy choice.
        strategy=(
            override.strategy
            if override.strategy != DEFAULT_STRATEGY
            else base.strategy
        ),
    )


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

    Modality-scoped access (set automatically by hub modality modules)
    ------------------------------------------------------------------
    ::

        # hub/infrared.py sets _default_filter automatically:
        from dino_loader.datasets.hub import infrared
        spec = infrared.laion.to_spec()
        # spec.modalities == ["infrared"]  — no filter arg needed

        # Callers can still further restrict:
        spec = infrared.laion.to_spec(
            global_filter=GlobalDatasetFilter(allowed_splits=["train"])
        )
        # Both the modality filter and the split filter are applied.
    """

    def __init__(
        self,
        name: str,
        _default_filter: Optional[GlobalDatasetFilter] = None,
    ) -> None:
        """
        Parameters
        ----------
        name:
            Dataset name, e.g. ``"imagenet"``, ``"custom"``.
            The library discovers actual paths from the
            :class:`~dino_loader.datasets.settings.ConfidentialityRegistry`.
        _default_filter:
            Optional base filter applied before any caller-supplied filter.
            Set automatically by modality sub-modules in ``hub/`` so that
            ``infrared.laion.to_spec()`` restricts to modality ``"infrared"``
            without the caller having to pass a filter explicitly.

            Regular user code **never** sets this directly.  It is a private
            contract between ``stub_gen.py`` and the generated hub modules.
        """
        self.name = name
        self._default_filter: Optional[GlobalDatasetFilter] = _default_filter

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
            Merged with ``self._default_filter`` (default filter wins on
            fields it sets; caller filter can further restrict).
        config:
            Per-dataset overrides (take precedence over *global_filter*).

        Returns
        -------
        shards : List[str]
            Absolute paths to all valid ``.tar`` shards, sorted lexicographically.
        meta : Dict[str, Set[str]]
            Keys ``"confidentialities"``, ``"modalities"``, ``"splits"``,
            ``"strategies"`` — each value is the set of directory names that
            actually contributed at least one valid shard.
        """
        # Merge the instance-level default filter with the caller's filter.
        # The caller's filter wins per field, allowing further restriction.
        effective_filter = _merge_filters(self._default_filter, global_filter)

        allowed_confs  = self._effective_set(effective_filter, config, "allowed_confidentialities")
        allowed_mods   = self._effective_set(effective_filter, config, "allowed_modalities")
        allowed_splits = self._effective_set(effective_filter, config, "allowed_splits")
        strategy       = self._effective_strategy(effective_filter, config)

        shards: List[str] = []
        meta: Dict[str, Set[str]] = {
            "confidentialities": set(),
            "modalities":        set(),
            "splits":            set(),
            "strategies":        set(),
        }

        for mount in get_confidentiality_mounts():
            if allowed_confs and mount.name not in allowed_confs:
                continue
            if not mount.path.is_dir():
                continue

            for mod in _listdirs(str(mount.path)):
                if mod in DATASET_RESERVED_DIRS:
                    continue
                if allowed_mods and mod not in allowed_mods:
                    continue

                dataset_path  = mount.path / mod / self.name
                outputs_path  = dataset_path / "outputs" / strategy

                if not outputs_path.is_dir():
                    continue

                contributed = False
                for split in _listdirs(str(outputs_path)):
                    if allowed_splits and split not in allowed_splits:
                        continue

                    split_path = str(outputs_path / split)
                    try:
                        entries = os.listdir(split_path)
                    except OSError:
                        continue

                    for fname in sorted(entries):
                        if not fname.endswith(".tar"):
                            continue
                        tar_path = os.path.join(split_path, fname)
                        if not validate_webdataset_shard(tar_path):
                            log.debug("Skipping invalid shard: %s", tar_path)
                            continue
                        shards.append(tar_path)
                        contributed = True

                    if contributed:
                        meta["splits"].add(split)

                if contributed:
                    meta["confidentialities"].add(mount.name)
                    meta["modalities"].add(mod)
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

        If ``self._default_filter`` is set (as done by hub modality modules),
        it is merged with *global_filter* before resolution.

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
        Build a :class:`~dino_loader.datasets.spec.DatasetSpec` from resolved shards.

        Unlike a raw :meth:`resolve` call, this method populates the four
        *discovery metadata* fields on the returned spec:

        * ``confidentialities`` — sorted list of confidentiality names that
          contributed at least one shard.
        * ``modalities`` — sorted list of modality directories.
        * ``splits`` — sorted list of split directories.
        * ``strategies`` — sorted list of strategy directories.

        If ``self._default_filter`` is set (as done by hub modality modules),
        it is merged with *global_filter* before resolution — allowing callers
        to further restrict without losing the modality pre-filter.

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
            When no valid shards are found.
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
            confidentialities = sorted(meta["confidentialities"]),
            modalities        = sorted(meta["modalities"]),
            splits            = sorted(meta["splits"]),
            strategies        = sorted(meta["strategies"]),
        )

    def __repr__(self) -> str:
        if self._default_filter is not None:
            return f"Dataset({self.name!r}, _default_filter={self._default_filter!r})"
        return f"Dataset({self.name!r})"

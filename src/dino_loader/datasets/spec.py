"""
dino_loader.datasets.spec
=========================
:class:`DatasetSpec` — the canonical descriptor of one WebDataset source.

Why this module exists
----------------------
``DatasetSpec`` is the primary data contract of the ``datasets`` sub-system:
it is produced by :meth:`~dino_loader.datasets.dataset.Dataset.to_spec`,
consumed by :class:`~dino_loader.datasets.stub_gen` when generating IDE stubs,
and referenced throughout the ``hub/`` generated package.

Keeping it in ``dino_loader.config`` — alongside loader-level dataclasses such
as ``LoaderConfig`` and ``DINOAugConfig`` — created an upward dependency that
prevented ``dino_loader.datasets`` from operating as a self-contained,
independently-importable sub-package (e.g. for dataset cataloguing tools that
have no interest in DALI or CUDA).

``DatasetSpec`` is now the **sole** export of this module.
``dino_loader.config`` re-exports it transparently so all existing imports
remain valid without modification.

Backward compatibility
----------------------
::

    # Both continue to work:
    from dino_loader.datasets import DatasetSpec      # canonical (new)
    from dino_loader.datasets.spec import DatasetSpec  # canonical (new)
    from dino_loader.config import DatasetSpec          # shim (unchanged API)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple


# ── DatasetSpec ───────────────────────────────────────────────────────────────

@dataclass
class DatasetSpec:
    """
    One WebDataset source with mixing weight, optional quality metadata,
    and discovery metadata populated by :meth:`Dataset.to_spec
    <dino_loader.datasets.dataset.Dataset.to_spec>`.

    Parameters
    ----------
    name
        Human-readable identifier, used in logs and checkpoint state.
    shards
        List of absolute shard paths (``.tar`` files on Lustre).
    weight
        Initial mixing weight (re-normalised automatically; need not sum to 1).
    prob
        Alias for ``weight=`` to align with the ``wds.RandomMix`` API.  If
        both are provided, ``weight=`` takes precedence and a
        :class:`DeprecationWarning` is emitted.
    shard_sampling
        How shards are sampled within this dataset:

        ``"epoch"`` (default)
            One full, deterministic-shuffled pass per epoch.

        ``"resampled"``
            Infinite with-replacement sampling via ``wds.ResampledShards``.
            Use for small curated sets you want to over-sample, or for
            streaming datasets without epoch boundaries.

    shard_quality_scores
        Optional per-shard quality score in ``[0, 1]``.  When provided,
        :class:`~dino_loader.mixing_source.ShardIterator` samples shards
        proportionally to these scores rather than uniformly.  Scores are
        re-normalised internally.  Length must match ``len(shards)`` if
        provided.
    min_sample_quality
        Hard filter: samples whose ``.json`` sidecar ``quality_score`` field
        is below this threshold are discarded before entering the augmentation
        pipeline.  Set to ``None`` to disable (default, no filtering).
    metadata_key
        WebDataset sidecar extension to extract alongside ``.jpg`` files.
        Set to ``None`` to skip sidecar extraction (legacy behaviour, faster).
    mean
        Per-channel normalisation mean for this dataset.  When ``None``, the
        global :attr:`~dino_loader.config.DINOAugConfig.mean` is used
        (ImageNet stats).
    std
        Per-channel normalisation std for this dataset.  When ``None``, the
        global :attr:`~dino_loader.config.DINOAugConfig.std` is used
        (ImageNet stats).
    confidentialities / modalities / splits / strategies
        Discovery metadata populated by
        :meth:`~dino_loader.datasets.dataset.Dataset.to_spec`.
        Informational only — not used by the loader itself.
    """

    name:   str
    shards: List[str]
    weight: float = 1.0
    prob:   Optional[float] = None  # [CFG-S2] wds.RandomMix alias

    shard_sampling:       Literal["epoch", "resampled"] = "epoch"
    shard_quality_scores: Optional[List[float]]         = None
    min_sample_quality:   Optional[float]               = None
    metadata_key:         Optional[str]                 = "json"
    mean:                 Optional[Tuple[float, float, float]] = None
    std:                  Optional[Tuple[float, float, float]] = None

    # Discovery metadata (populated by Dataset.to_spec — informational only)
    confidentialities: List[str] = field(default_factory=list)
    modalities:        List[str] = field(default_factory=list)
    splits:            List[str] = field(default_factory=list)
    strategies:        List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        # ── [CFG-S2] prob= alias ──────────────────────────────────────────────
        if self.prob is not None and self.weight == 1.0:
            self.weight = self.prob
        elif self.prob is not None:
            warnings.warn(
                "DatasetSpec: both 'weight' and 'prob' provided; "
                "'weight' takes precedence.  'prob' is deprecated.",
                DeprecationWarning,
                stacklevel=2,
            )

        # ── Basic validation ──────────────────────────────────────────────────
        if not self.shards:
            raise ValueError(
                f"DatasetSpec '{self.name}': shards list must not be empty."
            )
        if self.weight < 0.0:
            raise ValueError(
                f"DatasetSpec '{self.name}': weight must be ≥ 0, "
                f"got {self.weight}."
            )
        if self.shard_quality_scores is not None:
            if len(self.shard_quality_scores) != len(self.shards):
                raise ValueError(
                    f"DatasetSpec '{self.name}': shard_quality_scores length "
                    f"({len(self.shard_quality_scores)}) must match shards "
                    f"length ({len(self.shards)})."
                )
            if any(s < 0.0 for s in self.shard_quality_scores):
                raise ValueError(
                    f"DatasetSpec '{self.name}': all shard_quality_scores "
                    "must be ≥ 0."
                )
        if self.min_sample_quality is not None:
            if not (0.0 <= self.min_sample_quality <= 1.0):
                raise ValueError(
                    f"DatasetSpec '{self.name}': min_sample_quality must be "
                    f"in [0, 1], got {self.min_sample_quality}."
                )

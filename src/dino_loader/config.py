"""
dino_loader.config
==================
All configuration lives here.  No logic — pure dataclasses.
Serialised to / from JSON for checkpointing (no pickle fragility).

Changes vs previous version
----------------------------
[CFG-1] DatasetSpec enriched (DinoV3 alignment):
        - shard_quality_scores: Optional[List[float]] — per-shard quality for
          weighted shard sampling (replaces uniform random.choices at shard level).
        - min_sample_quality: Optional[float] — hard filter threshold applied
          per sample via .json sidecar metadata.  Requires wds.TarIterator
          extraction (see mixing_source.py).
        - metadata_key: str — sidecar extension to extract alongside JPEGs.
          Defaults to "json" (standard WebDataset convention).
        - mean / std: Optional per-dataset normalisation statistics.  When None,
          falls back to DINOAugConfig.mean / DINOAugConfig.std (ImageNet).
          Allows LAION-specific stats without changing global aug config.

[CFG-1b] DatasetSpec discovery metadata (new):
        - confidentialities: List[str] — confidentiality labels under which
          shards were found (e.g. ["public", "private"]).  Populated by
          Dataset.to_spec(); defaults to [] so existing DatasetSpec(...)
          constructions remain valid.
        - modalities: List[str] — modality directories (e.g. ["rgb"]).
        - splits: List[str] — split directories (e.g. ["train", "val"]).
        - strategies: List[str] — strategy directories (e.g. ["default"]).
        These fields are *informational only* — the dataloader does not read
        them.  stub_gen.py reads them to emit Literal-typed TypedDatasetSpec
        subclasses in hub.py for IDE autocomplete.

[CFG-2] DINOAugConfig additions:
        - preserve_aspect_ratio: bool — use resize-then-crop (aspect-ratio-safe)
          instead of fn.resize with fixed output size.  Maps to DALI
          fn.resize(mode="not_smaller") + fn.crop in pipeline.py.
        - resolution_schedule: Optional[List[Tuple[int,int]]] — list of
          (epoch, global_crop_size) pairs for progressive resolution training.
          The loader applies these automatically via set_resolution() without
          rebuilding the DALI pipeline (zero downtime).
        - max_global_crop_size / max_local_crop_size: int — upper bounds used
          to pre-allocate DALI nvjpeg buffers and output tensors at the maximum
          planned resolution, avoiding GPU memory re-allocation during training.

[CFG-3] LoaderConfig additions:
        - shuffle_buffer_size: int — intra-shard sample shuffle buffer depth.
          Previously reserved as a comment; now wired in ShardIterator.
          Default 512: large enough to break within-shard web-crawl correlations
          without exceeding per-rank RAM budgets.
        - stateful_dataloader: bool — expose state_dict() / load_state_dict()
          on DINODataLoader, aligning with the PyTorch StatefulDataLoader
          interface (torchdata ≥ 0.8).  Enables integration with Lightning,
          torchtitan and other frameworks that call these methods automatically.

[CFG-4] CheckpointState additions:
        - global_crop_size / local_crop_size: persisted so that a resumed run
          starts at the correct resolution without re-reading the schedule.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ── Dataset ───────────────────────────────────────────────────────────────────

@dataclass
class DatasetSpec:
    """
    One WebDataset source with mixing weight, optional quality metadata,
    and discovery metadata populated by :meth:`Dataset.to_spec`.

    Parameters
    ----------
    name
        Human-readable identifier, used in logs and checkpoint state.
    shards
        List of absolute shard paths (.tar files on Lustre).
    weight
        Initial mixing weight (re-normalised automatically; need not sum to 1).
    shard_quality_scores
        Optional per-shard quality score in [0, 1].  When provided,
        ShardIterator samples shards proportionally to these scores rather
        than uniformly.  Scores are re-normalised internally.
        Length must match ``len(shards)`` if provided.
    min_sample_quality
        Hard filter: samples whose .json sidecar ``quality_score`` field is
        below this threshold are discarded before entering the DALI pipeline.
        Set to None to disable (default, no filtering).
    metadata_key
        WebDataset sidecar extension to extract alongside .jpg files.
        Set to None to skip sidecar extraction (legacy behaviour, faster).
    mean
        Per-channel normalisation mean for this dataset.  When None, the
        global DINOAugConfig.mean is used (ImageNet stats).
    std
        Per-channel normalisation std for this dataset.  When None, the
        global DINOAugConfig.std is used (ImageNet stats).
    confidentialities
        Confidentiality labels under which shards were found during filesystem
        discovery (e.g. ``["public", "private"]``).  Populated automatically
        by :meth:`Dataset.to_spec`; left empty when constructing manually.
        *Informational only* — not read by the dataloader.
    modalities
        Modality directories found during discovery (e.g. ``["rgb"]``).
        Populated by :meth:`Dataset.to_spec`.  *Informational only.*
    splits
        Split directories found during discovery (e.g. ``["train", "val"]``).
        Populated by :meth:`Dataset.to_spec`.  *Informational only.*
    strategies
        Strategy directories found during discovery (e.g. ``["default"]``).
        Populated by :meth:`Dataset.to_spec`.  *Informational only.*
    """
    name:                 str
    shards:               List[str]
    weight:               float                = 1.0
    shard_quality_scores: Optional[List[float]] = None
    min_sample_quality:   Optional[float]       = None
    metadata_key:         Optional[str]         = "json"
    mean:                 Optional[Tuple[float, float, float]] = None
    std:                  Optional[Tuple[float, float, float]] = None

    # ── Discovery metadata (populated by Dataset.to_spec()) ───────────────────
    # Defaults to [] so that DatasetSpec(name=..., shards=...) remains valid
    # without specifying these fields — no breaking change.
    confidentialities:    List[str]            = field(default_factory=list)
    modalities:           List[str]            = field(default_factory=list)
    splits:               List[str]            = field(default_factory=list)
    strategies:           List[str]            = field(default_factory=list)

    def __post_init__(self):
        if not self.shards:
            raise ValueError(f"DatasetSpec '{self.name}' has no shards.")
        if self.weight < 0:
            raise ValueError(f"DatasetSpec '{self.name}': weight must be ≥ 0.")
        if self.shard_quality_scores is not None:
            if len(self.shard_quality_scores) != len(self.shards):
                raise ValueError(
                    f"DatasetSpec '{self.name}': shard_quality_scores length "
                    f"({len(self.shard_quality_scores)}) must match shards "
                    f"({len(self.shards)})."
                )
            if any(s < 0 for s in self.shard_quality_scores):
                raise ValueError(
                    f"DatasetSpec '{self.name}': shard_quality_scores must all be ≥ 0."
                )
        if self.min_sample_quality is not None and not (
            0.0 <= self.min_sample_quality <= 1.0
        ):
            raise ValueError(
                f"DatasetSpec '{self.name}': min_sample_quality must be in [0, 1], "
                f"got {self.min_sample_quality}."
            )

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "DatasetSpec":
        return cls(**d)


# ── Augmentation ──────────────────────────────────────────────────────────────

@dataclass
class DINOAugConfig:
    """
    Augmentation hyper-parameters for the DINOv3 multi-crop pipeline.

    Parameters
    ----------
    global_crop_size
        Output resolution for global (teacher) crops in pixels.
    local_crop_size
        Output resolution for local (student) crops in pixels.
    n_global_crops
        Number of global crops per image.
    n_local_crops
        Number of local crops per image.
    global_crop_scale
        (min, max) scale range for global random-resized crop.
    local_crop_scale
        (min, max) scale range for local random-resized crop.
    hflip_prob
        Horizontal-flip probability.
    color_jitter_strength
        Strength scalar applied to brightness / contrast / saturation / hue.
    grayscale_prob
        Probability of converting to grayscale.
    gaussian_blur_prob_g1
        Gaussian-blur probability for the first global crop.
    gaussian_blur_prob_g2
        Gaussian-blur probability for the second global crop.
    gaussian_blur_prob_local
        Gaussian-blur probability for local crops.
    solarize_prob
        Solarisation probability (applied to second global crop in DINOv2).
    mean
        Global normalisation mean (ImageNet default).
    std
        Global normalisation std (ImageNet default).
    preserve_aspect_ratio
        When True, uses resize-then-crop (aspect-ratio-safe) instead of
        a direct resize to fixed output size.  Maps to DALI
        ``fn.resize(mode="not_smaller")`` followed by ``fn.crop``.
    resolution_schedule
        List of ``(epoch, global_crop_size)`` pairs.  The loader calls
        ``set_resolution()`` automatically at the appropriate epoch boundary.
        ``None`` disables progressive resolution (train at fixed size).
    max_global_crop_size
        Upper bound for the global crop dimension used to pre-allocate DALI
        nvjpeg buffers.  Defaults to ``global_crop_size`` (no headroom).
        Set higher when using a resolution schedule (e.g. 518 for ViT-g).
    max_local_crop_size
        Same as ``max_global_crop_size`` but for local crops.
    """
    global_crop_size:        int   = 224
    local_crop_size:         int   = 96
    n_global_crops:          int   = 2
    n_local_crops:           int   = 8
    global_crop_scale:       Tuple[float, float] = (0.32, 1.0)
    local_crop_scale:        Tuple[float, float] = (0.05, 0.32)
    hflip_prob:              float = 0.5
    color_jitter_strength:   float = 0.4
    grayscale_prob:          float = 0.2
    gaussian_blur_prob_g1:   float = 1.0
    gaussian_blur_prob_g2:   float = 0.1
    gaussian_blur_prob_local: float = 0.5
    solarize_prob:           float = 0.2
    mean:  Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std:   Tuple[float, float, float] = (0.229, 0.224, 0.225)
    preserve_aspect_ratio:   bool  = False
    resolution_schedule:     Optional[List[Tuple[int, int]]] = None
    max_global_crop_size:    Optional[int] = None
    max_local_crop_size:     Optional[int] = None

    def __post_init__(self):
        if self.max_global_crop_size is None:
            self.max_global_crop_size = self.global_crop_size
        if self.max_local_crop_size is None:
            self.max_local_crop_size = self.local_crop_size

    @property
    def n_views(self) -> int:
        """Total number of crop views per image."""
        return self.n_global_crops + self.n_local_crops

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "DINOAugConfig":
        return cls(**d)


# ── Loader ────────────────────────────────────────────────────────────────────

@dataclass
class LoaderConfig:
    """
    Infrastructure knobs for :class:`~dino_loader.loader.DINODataLoader`.

    Parameters
    ----------
    prefetch_factor
        Number of shards to prefetch per dataset per rank.
    num_workers
        CPU threads per shard prefetch pool.
    seed
        Base random seed; per-rank seed is derived as ``seed + rank``.
    cache_max_gb
        Maximum in-process shard cache size in GiB.
    shuffle_buffer_size
        Intra-shard sample shuffle reservoir depth.  Set to 0 to disable.
        Default 512 breaks within-shard web-crawl correlations without
        exceeding per-rank RAM budgets on typical Lustre-attached nodes.
    stateful_dataloader
        When True, expose ``state_dict()`` / ``load_state_dict()`` on
        :class:`~dino_loader.loader.DINODataLoader` for integration with
        Lightning, torchtitan and PyTorch StatefulDataLoader-aware frameworks.
    fp8_output
        When True, pipeline outputs are formatted to FP8 (requires
        TransformerEngine).
    use_ibot_mask
        When True, the loader generates iBOT token masks and attaches them
        to each :class:`~dino_loader.memory.Batch`.
    ibot_mask_ratio
        Fraction of tokens to mask in iBOT mode.
    """
    prefetch_factor:      int   = 2
    num_workers:          int   = 4
    seed:                 int   = 42
    cache_max_gb:         float = 4.0
    shuffle_buffer_size:  int   = 512
    stateful_dataloader:  bool  = False
    fp8_output:           bool  = False
    use_ibot_mask:        bool  = False
    ibot_mask_ratio:      float = 0.5

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "LoaderConfig":
        return cls(**d)


# ── Checkpoint ────────────────────────────────────────────────────────────────

@dataclass
class CheckpointState:
    """
    Serialisable snapshot of dataloader progress for mid-epoch resume.

    Parameters
    ----------
    epoch
        Current training epoch (0-indexed).
    sample_idx
        Number of samples consumed within the current epoch.
    shard_cursors
        Per-dataset shard cursor (index into the shuffled shard list).
    global_crop_size
        Active global crop size (may differ from DINOAugConfig when a
        resolution schedule is in use).
    local_crop_size
        Active local crop size.
    """
    epoch:            int              = 0
    sample_idx:       int              = 0
    shard_cursors:    Dict[str, int]   = field(default_factory=dict)
    global_crop_size: Optional[int]    = None
    local_crop_size:  Optional[int]    = None

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> "CheckpointState":
        return cls(**json.loads(path.read_text()))

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "CheckpointState":
        return cls(**d)

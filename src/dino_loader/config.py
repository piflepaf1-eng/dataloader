"""
dino_loader.config
==================
All configuration lives here.  No logic — pure dataclasses.
Serialised to / from JSON for checkpointing (no pickle fragility).

Changes in this version
-----------------------
[CFG-S1]  DatasetSpec.shard_sampling — explicit sampling mode.
          Two modes:
          - "epoch"     (default) — deterministic shuffle, one full pass per epoch.
          - "resampled" — infinite sampling with replacement via wds.ResampledShards.
            Ideal for heavily imbalanced datasets (e.g. a small curated set mixed
            with a large noisy one) where you want controlled over-sampling without
            duplicating shards on disk.

[CFG-S2]  DatasetSpec.prob — alias for weight= aligned with wds.RandomMix API.
          Both are accepted; weight= takes precedence if both provided (deprecation
          warning emitted).  Lowers barrier for users already familiar with webdataset.

[CFG-S3]  LoaderConfig.debug_log_keys — optional path for per-sample key logging.
          When set, every sample key (__key__), worker id, and rank is appended to
          this file using fcntl POSIX locking (same pattern as wds.log_keys).
          Overhead: ~1 syscall per sample — disable in production.
          Useful for: reproducibility audits, corruption debugging, distribution bias
          detection (e.g. detecting that one dataset is under-represented).

[CFG-S4]  LoaderConfig.fuse_normalization — controls whether per-dataset
          normalisation is fused into the DALI graph (True, default) or applied
          post-DALI in memory.py (False, legacy).  When True, mean/std scalars are
          emitted by an ExternalSource node in pipeline.py, enabling the DALI
          compiler to fuse normalize → cast → transpose into a single GPU kernel.

[CFG-S5]  LoaderConfig.dali_fp8_output — when True AND use_fp8_output is True,
          the FP8 cast is performed inside the DALI graph via fn.cast(FLOAT8_E4M3)
          rather than post-DALI in FP8Formatter.  Eliminates one kernel launch and
          one BF16 intermediate buffer per batch.
          Trade-off: loses Transformer Engine FP8TensorMeta (rolling amax window).
          Set to False (default) to retain TE metadata for use with te.fp8_autocast.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple


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
    prob
        Alias for weight= to align with the wds.RandomMix API.  If both are
        provided, weight= takes precedence and a DeprecationWarning is emitted.
    shard_sampling
        How shards are sampled within this dataset:
        - ``"epoch"``     : one full, deterministic-shuffled pass per epoch.
        - ``"resampled"`` : infinite with-replacement sampling via
          wds.ResampledShards — use for small curated sets you want to
          over-sample, or for streaming datasets without epoch boundaries.
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
    confidentialities / modalities / splits / strategies
        Discovery metadata populated by Dataset.to_spec().  Informational only.
    """

    name:   str
    shards: List[str]
    weight: float = 1.0

    # [CFG-S2] wds.RandomMix-compatible alias
    prob:   Optional[float] = None

    # [CFG-S1] Shard sampling mode
    shard_sampling: Literal["epoch", "resampled"] = "epoch"

    # Quality gating
    shard_quality_scores: Optional[List[float]] = None
    min_sample_quality:   Optional[float]       = None
    metadata_key:         Optional[str]         = "json"

    # Per-dataset normalisation stats (override DINOAugConfig.mean/std)
    mean: Optional[Tuple[float, float, float]] = None
    std:  Optional[Tuple[float, float, float]] = None

    # Discovery metadata (populated by Dataset.to_spec, ignored by dataloader)
    confidentialities: List[str] = field(default_factory=list)
    modalities:        List[str] = field(default_factory=list)
    splits:            List[str] = field(default_factory=list)
    strategies:        List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        # [CFG-S2] Resolve weight / prob alias
        if self.prob is not None:
            if self.weight != 1.0:
                warnings.warn(
                    f"DatasetSpec '{self.name}': both weight= and prob= provided. "
                    "weight= takes precedence.  prob= is deprecated; use weight= only.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            else:
                self.weight = self.prob
        self.prob = None  # normalise: always use weight internally

        # Validate shard_sampling
        if self.shard_sampling not in ("epoch", "resampled"):
            raise ValueError(
                f"DatasetSpec '{self.name}': shard_sampling must be 'epoch' or "
                f"'resampled', got {self.shard_sampling!r}."
            )

        # Validate shard_quality_scores
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

        # Validate min_sample_quality
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
        # Strip unknown keys for forward-compat
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})


# ── Augmentation ──────────────────────────────────────────────────────────────

@dataclass
class DINOAugConfig:
    """
    Augmentation hyper-parameters for the DINOv3 multi-crop pipeline.

    Parameters
    ----------
    global_crop_size / local_crop_size
        Starting output resolution for global / local crops.
    n_global_crops / n_local_crops
        Number of crops per image (DINOv2 paper: 2 global, 8 local).
    global_crops_scale / local_crops_scale
        Area fraction ranges for random-resized crops.
    flip_prob
        Horizontal-flip probability.
    color_jitter_prob
        Probability of applying colour jitter.
    brightness / contrast / saturation / hue
        Colour jitter magnitudes (strength = 1.0 → DINOv2 paper defaults).
    grayscale_prob
        Probability of converting to grayscale.
    blur_prob_global1 / blur_prob_global2 / blur_prob_local
        Gaussian-blur probabilities per view type.
    blur_sigma_min / blur_sigma_max
        Sigma range for Gaussian blur.
    solarize_prob
        Solarisation probability (second global crop only in DINOv2).
    mean / std
        Global normalisation statistics (ImageNet defaults).
        Per-dataset overrides live in DatasetSpec.mean / DatasetSpec.std.
    preserve_aspect_ratio
        True → resize shorter side then centre-crop (avoids distortion).
        False → direct square resize (legacy, faster by one DALI op).
    resolution_schedule
        List of (epoch, global_crop_size) pairs for progressive resolution.
        set_epoch() applies these automatically — no DALI rebuild required.
    max_global_crop_size / max_local_crop_size
        nvjpeg pre-allocation ceiling.  Must be ≥ the largest size in the
        resolution schedule.  Default = initial crop size.
    """

    # Crop sizes
    global_crop_size:     int   = 224
    local_crop_size:      int   = 96
    n_global_crops:       int   = 2
    n_local_crops:        int   = 8
    global_crops_scale:   Tuple[float, float] = (0.32, 1.0)
    local_crops_scale:    Tuple[float, float] = (0.05, 0.32)

    # Geometric
    flip_prob:            float = 0.5

    # Colour
    color_jitter_prob:    float = 0.8
    brightness:           float = 0.4
    contrast:             float = 0.4
    saturation:           float = 0.2
    hue:                  float = 0.1
    grayscale_prob:       float = 0.2

    # Blur / solarise
    blur_prob_global1:    float = 1.0
    blur_prob_global2:    float = 0.1
    blur_prob_local:      float = 0.5
    blur_sigma_min:       float = 0.1
    blur_sigma_max:       float = 2.0
    solarize_prob:        float = 0.2

    # Normalisation (ImageNet defaults)
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std:  Tuple[float, float, float] = (0.229, 0.224, 0.225)

    # Aspect ratio
    preserve_aspect_ratio: bool = True

    # Progressive resolution
    resolution_schedule:    Optional[List[Tuple[int, int]]] = None
    max_global_crop_size:   Optional[int] = None
    max_local_crop_size:    Optional[int] = None

    def __post_init__(self) -> None:
        if self.max_global_crop_size is None:
            self.max_global_crop_size = self.global_crop_size
        if self.max_local_crop_size is None:
            self.max_local_crop_size = self.local_crop_size

    @property
    def n_views(self) -> int:
        return self.n_global_crops + self.n_local_crops

    def crop_size_at_epoch(self, epoch: int) -> int:
        """Return the scheduled global crop size for a given epoch."""
        if not self.resolution_schedule:
            return self.global_crop_size
        size = self.global_crop_size
        for trigger_epoch, new_size in sorted(self.resolution_schedule):
            if epoch >= trigger_epoch:
                size = new_size
        return size


# ── Infrastructure ────────────────────────────────────────────────────────────

@dataclass
class LoaderConfig:
    """
    Infrastructure knobs for DINODataLoader.

    Parameters
    ----------
    node_shm_gb
        /dev/shm budget per node in GB.  ~50% of node RAM is a safe default.
    shard_prefetch_window
        Max concurrent Lustre → /dev/shm downloads (node master asyncio).
    shard_timeout_s
        Seconds a non-master rank waits for a shard to appear in /dev/shm.
    shard_extraction_workers
        Thread-pool workers for tar → JPEG extraction.
    dali_cpu_queue / dali_gpu_queue
        DALI prefetch queue depths (CPU and GPU sides).
    dali_num_threads
        DALI CPU worker threads for pre-decode operations.
    hw_decoder_load
        Fraction of JPEG decoding sent to nvjpeg HW ASIC (0–1).
    shuffle_buffer_size
        Intra-shard sample shuffle buffer depth.
    use_fp8_output
        Quantise global/local crop tensors to FP8 E4M3 before yielding.
    dali_fp8_output                                              [CFG-S5]
        When True (and use_fp8_output=True), perform FP8 cast inside the
        DALI graph rather than post-DALI, enabling kernel fusion with the
        final normalise + transpose.  Loses TE FP8TensorMeta.
    fuse_normalization                                           [CFG-S4]
        When True (default), per-dataset mean/std is emitted via DALI
        ExternalSource and fused with the final normalize kernel.
    output_dtype
        Intermediate computation dtype ("bf16" or "fp32").
    stateful_dataloader
        Expose state_dict() / load_state_dict() (PyTorch StatefulDataLoader).
    checkpoint_dir
        Directory for JSON dataloader checkpoint files (rank 0 only).
    checkpoint_every_steps
        Checkpoint write frequency (steps).
    force_topology
        Override topology detection: "nvl72" | "pcie" | None (auto).
    seed
        Base random seed for shard shuffling and augmentation.
    debug_log_keys                                               [CFG-S3]
        When set to a file path, every sample's __key__, worker id, and rank
        are appended using POSIX fcntl locking (wds.log_keys pattern).
        Set to None (default) to disable — zero overhead in production.
    """

    # I/O
    node_shm_gb:              float = 128.0
    shard_prefetch_window:    int   = 64
    shard_timeout_s:          float = 300.0
    shard_extraction_workers: int   = 4

    # DALI
    dali_cpu_queue:           int   = 8
    dali_gpu_queue:           int   = 6
    dali_num_threads:         int   = 8
    hw_decoder_load:          float = 0.90

    # Data
    shuffle_buffer_size:      int   = 512

    # Output precision
    use_fp8_output:           bool  = False
    dali_fp8_output:          bool  = False   # [CFG-S5] fuse FP8 into DALI graph
    fuse_normalization:       bool  = True    # [CFG-S4] fuse per-dataset norm in DALI
    output_dtype:             str   = "bf16"

    # StatefulDataLoader
    stateful_dataloader:      bool  = True
    checkpoint_dir:           str   = "/tmp/dino_loader_ckpt"
    checkpoint_every_steps:   int   = 500

    # Cluster
    force_topology:           Optional[str] = None
    seed:                     int   = 0

    # Debug                                                      [CFG-S3]
    debug_log_keys:           Optional[str] = None

    def __post_init__(self) -> None:
        if self.dali_fp8_output and not self.use_fp8_output:
            raise ValueError(
                "LoaderConfig: dali_fp8_output=True requires use_fp8_output=True."
            )
        if self.output_dtype not in ("bf16", "fp32"):
            raise ValueError(
                f"LoaderConfig: output_dtype must be 'bf16' or 'fp32', "
                f"got {self.output_dtype!r}."
            )

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "LoaderConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})


# ── Checkpoint state ──────────────────────────────────────────────────────────

@dataclass
class CheckpointState:
    """Persisted dataloader state — JSON-serialisable."""

    step:              int   = 0
    epoch:             int   = 0
    shard_cursors:     Dict  = field(default_factory=dict)
    global_crop_size:  int   = 224
    local_crop_size:   int   = 96
    dataset_weights:   List[float] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "CheckpointState":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2))
        tmp.rename(path)

    @classmethod
    def load(cls, path: Path) -> "CheckpointState":
        return cls.from_dict(json.loads(path.read_text()))

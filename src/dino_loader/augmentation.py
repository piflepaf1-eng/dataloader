"""dino_loader.augmentation.

Augmentation pipeline abstraction for dino_loader.

Architecture
------------
``AugmentationSpec`` is the single configuration object that describes *what*
augmentation strategy to apply.  ``AugmentationPipeline`` is the runtime
object built from it, consumed by ``DALIBackend`` and ``CPUBackend``.

The clean separation enables:

* **Preset strategies** — ``DinoV2AugSpec``, ``EvalAugSpec``, ``LeJEPAAugSpec``
  ship out of the box and cover the common training, evaluation, and
  self-supervised pre-training use cases.
* **User-defined strategies** — ``UserAugSpec`` accepts any callable that
  receives decoded GPU tensors and returns a dict of named views.  JPEG
  decoding always happens in DALI via the hardware nvjpeg ASIC; the user
  function never sees raw bytes.
* **Composability with the fluid loader API** — ``DINODataLoader.map()`` and
  ``DINODataLoader.select()`` continue to work unchanged; the augmentation
  choice is made at construction time, not at iteration time.

Early filtering
---------------
``SamplePredicate`` is called by ``ShardIterator`` *before* a sample enters
the DALI pipeline.  This eliminates GPU/DALI decode cost for samples that
would have been discarded anyway by a post-pipeline ``select()``.

The predicate receives only the lightweight ``SampleMeta`` (metadata dict +
shard path + key) — no image decoding is performed at this stage.
"""

from __future__ import annotations

import logging
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable

from dino_loader.config import DINOAugConfig

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Early filtering — predicate evaluated before JPEG enters DALI
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class SampleMeta:
    """Lightweight sample descriptor available before JPEG decoding.

    Passed to :class:`SamplePredicate` callables so that filtering can happen
    at zero image-decode cost — only the JSON sidecar metadata and the shard
    coordinates are available here.

    Attributes:
        key: WebDataset sample key (e.g. ``"000042"``).
        shard_path: Absolute path to the ``.tar`` shard file.
        metadata: Parsed JSON sidecar dict, or ``None`` if absent.
    """

    key:        str
    shard_path: str
    metadata:   dict[str, Any] | None


@runtime_checkable
class SamplePredicate(Protocol):
    """Callable protocol for early sample filtering.

    Return ``True`` to *keep* the sample, ``False`` to discard it.
    The sample JPEG will never be decoded if ``False`` is returned.

    The predicate is called from extraction worker threads; implementations
    must be thread-safe (read-only access to shared state is fine).

    Example — keep only samples with quality_score ≥ 0.5::

        def quality_filter(meta: SampleMeta) -> bool:
            if meta.metadata is None:
                return True  # no metadata → keep (conservative)
            return meta.metadata.get("quality_score", 1.0) >= 0.5

        loader = DINODataLoader(
            specs=[spec],
            sample_predicate=quality_filter,
            ...
        )

    Note:
        For simple quality-score thresholds, prefer
        ``DatasetSpec.min_sample_quality`` which is evaluated at the same
        stage but without a Python function call overhead.  Use
        ``SamplePredicate`` for logic that cannot be expressed as a simple
        float threshold (e.g. class-balanced sampling, domain filtering, …).
    """

    def __call__(self, meta: SampleMeta) -> bool:
        """Return True to keep, False to discard."""
        ...


# ══════════════════════════════════════════════════════════════════════════════
# Augmentation pipeline protocol
# ══════════════════════════════════════════════════════════════════════════════

@runtime_checkable
class AugmentationPipeline(Protocol):
    """Runtime augmentation pipeline consumed by loader backends.

    All implementations must be constructable by the backend's
    ``build_aug_pipeline()`` factory, which receives the spec and the
    ``MixingSource`` callback as arguments.

    The pipeline is an iterator: each call to ``__next__`` returns one batch
    as a dict mapping view names to ``Tensor[B, C, H, W]``.
    """

    @property
    def output_map(self) -> list[str]:
        """Ordered list of view names this pipeline produces."""
        ...

    def __iter__(self) -> AugmentationPipeline:
        ...

    def __next__(self) -> dict[str, Any]:
        """Return one batch dict.  Raise ``StopIteration`` at epoch end."""
        ...

    def reset(self) -> None:
        """Reset for a new epoch (called by ``DINODataLoader.set_epoch``)."""
        ...


# ══════════════════════════════════════════════════════════════════════════════
# Augmentation specifications (pure config, no runtime state)
# ══════════════════════════════════════════════════════════════════════════════

class AugmentationSpec(ABC):
    """Base class for augmentation strategy specifications.

    A spec is a **pure configuration object** — it carries no runtime state
    and can be serialised, hashed, and compared.  The backend's
    ``build_aug_pipeline()`` factory turns it into a live
    :class:`AugmentationPipeline`.
    """

    @property
    @abstractmethod
    def output_map(self) -> list[str]:
        """Ordered list of view names produced by this spec."""
        ...

    @property
    def n_views(self) -> int:
        """Total number of views per sample."""
        return len(self.output_map)

    @property
    def uses_dali(self) -> bool:
        """True if this spec can be fully fused into a DALI pipeline graph."""
        return True

    def __repr__(self) -> str:
        return f"{type(self).__name__}(n_views={self.n_views})"


# ── Preset: DINOv2 multi-crop (default) ──────────────────────────────────────

@dataclass
class DinoV2AugSpec(AugmentationSpec):
    """DINOv2-style multi-crop augmentation.

    This is the default spec used when ``DINODataLoader`` is constructed
    without an explicit ``aug_spec`` argument.  It wraps the existing
    ``DINOAugConfig`` so existing training scripts require zero changes.

    Attributes:
        aug_cfg: Full DINOv2 augmentation configuration.
        fuse_normalization: Fuse per-dataset mean/std into the DALI graph.
        fp8_output: Emit FP8-cast tensors directly from the DALI graph.
    """

    aug_cfg:            DINOAugConfig = field(default_factory=DINOAugConfig)
    fuse_normalization: bool          = True
    fp8_output:         bool          = False

    @property
    def output_map(self) -> list[str]:
        return [f"view_{i}" for i in range(self.aug_cfg.n_views)]


# ── Preset: evaluation / inference (resize + centre crop, no jitter) ─────────

@dataclass
class EvalAugSpec(AugmentationSpec):
    """Evaluation augmentation: resize-then-centre-crop, no stochastic ops.

    Suitable for val/test loops and fine-tuning phases where data augmentation
    should be deterministic and minimal.

    Attributes:
        crop_size: Output spatial resolution in pixels (default 224).
        mean: Per-channel normalisation mean.
        std: Per-channel normalisation std.
        interpolation: Resize interpolation mode (``"bicubic"`` or ``"bilinear"``).
    """

    crop_size:     int                        = 224
    mean:          tuple[float, float, float] = (0.485, 0.456, 0.406)
    std:           tuple[float, float, float] = (0.229, 0.224, 0.225)
    interpolation: str                        = "bicubic"

    @property
    def output_map(self) -> list[str]:
        return ["view_0"]

    def __post_init__(self) -> None:
        valid_interp = {"bicubic", "bilinear"}
        if self.interpolation not in valid_interp:
            msg = f"EvalAugSpec.interpolation must be one of {valid_interp}, got {self.interpolation!r}."
            raise ValueError(msg)


# ── Preset: LeJEPA ────────────────────────────────────────────────────────────

@dataclass
class LeJEPAAugSpec(AugmentationSpec):
    """LeJEPA-style augmentation: context + target patch views.

    Produces two views per sample:
    - ``"context"``: a large crop (typically ≥50% of image area) used as the
      encoder input.
    - ``"target"``: one or more smaller crops used as the predictor target.

    The patch masking required by JEPA is applied *after* the pipeline, on
    CPU, using ``MaskingGenerator`` — identical to how DINOv2 handles iBOT
    masks (DALI cannot express patch-index operations).

    Attributes:
        context_crop_size: Spatial resolution of the context view.
        target_crop_size: Spatial resolution of each target view.
        n_target_views: Number of independent target crops per sample.
        context_scale: RandomResizedCrop scale range for the context view.
        target_scale: RandomResizedCrop scale range for target views.
        mean: Per-channel normalisation mean.
        std: Per-channel normalisation std.
    """

    context_crop_size: int                        = 224
    target_crop_size:  int                        = 96
    n_target_views:    int                        = 4
    context_scale:     tuple[float, float]        = (0.85, 1.0)
    target_scale:      tuple[float, float]        = (0.15, 0.30)
    mean:              tuple[float, float, float] = (0.485, 0.456, 0.406)
    std:               tuple[float, float, float] = (0.229, 0.224, 0.225)

    @property
    def output_map(self) -> list[str]:
        return ["context", *[f"target_{i}" for i in range(self.n_target_views)]]

    def __post_init__(self) -> None:
        if self.n_target_views < 1:
            msg = f"LeJEPAAugSpec.n_target_views must be ≥ 1, got {self.n_target_views}."
            raise ValueError(msg)
        if not (0.0 < self.context_scale[0] <= self.context_scale[1] <= 1.0):
            msg = f"LeJEPAAugSpec.context_scale must be in (0, 1], got {self.context_scale}."
            raise ValueError(msg)
        if not (0.0 < self.target_scale[0] <= self.target_scale[1] <= 1.0):
            msg = f"LeJEPAAugSpec.target_scale must be in (0, 1], got {self.target_scale}."
            raise ValueError(msg)


# ── User-defined: arbitrary callable ─────────────────────────────────────────

# Type alias for user-provided augmentation functions.
# The function receives a decoded GPU tensor of shape [B, C, H, W] (BF16/FP32)
# and returns a dict mapping view names to tensors of the same batch dimension.
UserAugFn = Callable[
    ["torch.Tensor"],  # type: ignore[name-defined]  # noqa: F821
    "dict[str, torch.Tensor]",  # type: ignore[name-defined]  # noqa: F821
]


@dataclass
class UserAugSpec(AugmentationSpec):
    """User-provided augmentation function applied to decoded GPU tensors.

    JPEG decoding is always performed by DALI's hardware nvjpeg pipeline —
    the user function receives *already-decoded* ``float16`` or ``float32``
    tensors on the GPU (shape ``[B, C, H, W]``) and never sees raw bytes.

    This design means:
    - The user focuses on transform logic, not I/O or byte manipulation.
    - nvjpeg hardware decode throughput is unchanged.
    - The only overhead vs. a pure-DALI pipeline is one Python call boundary
      per batch and the inability to fuse ops into the DALI graph.

    A ``UserWarning`` is emitted at construction time to remind users that
    their function runs outside the DALI graph and therefore cannot benefit
    from DALI-level kernel fusion or prefetch pipelining.

    Attributes:
        aug_fn: Callable ``(Tensor[B,C,H,W]) → dict[str, Tensor[B,C,H,W]]``.
            The input tensor is the decoded image batch, normalised to
            ``[0, 1]`` float16, **before** any augmentation.  The function
            must return a dict whose keys match ``output_map``.
        output_map: Names of the views returned by ``aug_fn``.
            Must be consistent across all calls.
        decode_size: Spatial resolution at which DALI decodes and resizes
            images before handing them to ``aug_fn``.  Should be set to the
            largest crop size your function may produce.
        mean: Per-channel normalisation mean applied *before* calling
            ``aug_fn`` (so the user receives normalised tensors).
        std: Per-channel normalisation std (same note).
        warn_not_dali: Emit the non-DALI warning (default True; set False
            to silence it once you have acknowledged the trade-off).

    Example::

        def my_aug(images: torch.Tensor) -> dict[str, torch.Tensor]:
            # images: [B, C, H, W], float16, normalised, on GPU
            global_crop = torchvision.transforms.functional.center_crop(images, 224)
            local_crop  = torchvision.transforms.functional.random_crop(images, 96)
            return {"view_0": global_crop, "view_1": local_crop}

        spec = UserAugSpec(
            aug_fn     = my_aug,
            output_map = ["view_0", "view_1"],
            decode_size = 256,
        )
        loader = DINODataLoader(specs=[...], aug_spec=spec, ...)
    """

    aug_fn:      UserAugFn
    output_map:  list[str]
    decode_size: int                         = 256
    mean:        tuple[float, float, float]  = (0.485, 0.456, 0.406)
    std:         tuple[float, float, float]  = (0.229, 0.224, 0.225)
    warn_not_dali: bool                      = True

    # AugmentationSpec.output_map is a property; override as a plain field here.
    # The ABC check is satisfied because the field shadows the abstract property.

    @property
    def uses_dali(self) -> bool:
        """UserAugSpec uses DALI only for decoding, not for the full graph."""
        return False

    @property
    def n_views(self) -> int:
        return len(self.output_map)

    def __post_init__(self) -> None:
        if not callable(self.aug_fn):
            msg = "UserAugSpec.aug_fn must be callable."
            raise TypeError(msg)
        if not self.output_map:
            msg = "UserAugSpec.output_map must be a non-empty list of view names."
            raise ValueError(msg)
        if self.decode_size < 1:
            msg = f"UserAugSpec.decode_size must be ≥ 1, got {self.decode_size}."
            raise ValueError(msg)

        if self.warn_not_dali:
            warnings.warn(
                "UserAugSpec: the provided aug_fn runs outside the DALI computation "
                "graph.  JPEG decoding still uses the nvjpeg hardware pipeline for "
                "full throughput, but augmentation ops cannot be fused with decode.  "
                "Expect a ~10–20% throughput reduction vs. a native DALI pipeline.  "
                "Suppress this warning with warn_not_dali=False once acknowledged.",
                UserWarning,
                stacklevel=3,
            )

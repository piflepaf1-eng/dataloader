"""
dino_loader.pipeline
====================
DALI augmentation pipeline for DINOv3 multi-crop.

Changes in this version
-----------------------
[PL-1]  Dynamic resize via ResolutionSource (zero pipeline rebuild).

[PL-2]  Aspect-ratio-preserving resize.

[PL-3]  Per-dataset normalisation stats fused into DALI graph.         ← UPDATED
        Previously, the normalise step used global ImageNet stats baked
        into numpy constants at build time.  Per-dataset overrides were
        applied post-DALI in memory.py via a separate GPU kernel, keeping
        the DALI graph topology fixed but forgoing fusion.

        New behaviour (fuse_normalization=True, default):
        - A second ExternalSource node emits per-sample (mean, std) pairs
          as FLOAT32 tensors of shape (3,) each, broadcast across the batch.
        - pipeline.py's NormSource callback is driven by MixingSource, which
          sets the active dataset index per sample.
        - The DALI compiler fuses normalize → cast → transpose into a single
          GPU kernel.  This eliminates one kernel launch and one intermediate
          BF16 buffer per view per batch.

        Legacy behaviour (fuse_normalization=False):
        - Global mean/std baked into numpy constants (original code path).
        - Per-dataset correction applied in memory.py if needed.

[PL-4]  Explicit FLOAT16 cast hardening retained.

[PL-5]  Optional in-graph FP8 cast.                                    ← NEW
        When LoaderConfig.dali_fp8_output=True, the final cast in
        _augment_view switches from FLOAT16 to FLOAT8_E4M3 (requires
        DALI ≥ 1.36).  This fuses:

            normalize (FLOAT16) → cast FLOAT8_E4M3 → transpose

        into a single kernel, eliminating one extra kernel launch and the
        BF16 intermediate buffer.  Trade-off: FP8TensorMeta (rolling amax
        from Transformer Engine) is NOT available — if you need te.fp8_autocast
        compatibility, keep dali_fp8_output=False and let FP8Formatter handle
        quantisation post-DALI.
"""

from __future__ import annotations

import logging
import threading
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from dino_loader.config import DINOAugConfig, DatasetSpec

log = logging.getLogger(__name__)

try:
    import nvidia.dali.fn    as fn
    import nvidia.dali.types as types
    from nvidia.dali import pipeline_def
    HAS_DALI = True
except ImportError:
    HAS_DALI = False
    log.error("nvidia-dali not installed — pipeline will not build")


# ══════════════════════════════════════════════════════════════════════════════
# [PL-3] NormSource — per-sample normalisation ExternalSource callback
# ══════════════════════════════════════════════════════════════════════════════

class NormSource:
    """
    DALI ExternalSource callback that emits per-sample (mean, std) tensors.

    MixingSource calls set_dataset_indices(indices) immediately before DALI
    pulls the next batch, so that each sample in the batch gets the correct
    normalisation for its originating dataset.

    Thread safety
    -------------
    set_dataset_indices() is called from MixingSource's thread (the DALI
    prefetch thread).  DALI calls __call__() from its own internal thread.
    A threading.Lock serialises access to self._indices.

    Fallback
    --------
    When an index has no per-dataset stats (DatasetSpec.mean is None), the
    global DINOAugConfig.mean/std are used.
    """

    def __init__(
        self,
        aug_cfg:  DINOAugConfig,
        specs:    List[DatasetSpec],
    ) -> None:
        # Build lookup: dataset_index → (mean_arr, std_arr) as FLOAT32 (3,)
        global_mean = np.array(aug_cfg.mean, dtype=np.float32)
        global_std  = np.array(aug_cfg.std,  dtype=np.float32)
        self._lookup: List[Tuple[np.ndarray, np.ndarray]] = []
        for spec in specs:
            m = np.array(spec.mean, dtype=np.float32) if spec.mean else global_mean
            s = np.array(spec.std,  dtype=np.float32) if spec.std  else global_std
            self._lookup.append((m, s))

        self._indices: List[int] = [0]  # placeholder
        self._lock = threading.Lock()

    def set_dataset_indices(self, indices: List[int]) -> None:
        """Called by MixingSource before each batch is pushed to DALI."""
        with self._lock:
            self._indices = list(indices)

    def __call__(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Return (means, stds) lists — one (3,) FLOAT32 array per sample."""
        with self._lock:
            indices = self._indices
        means = [self._lookup[i][0] for i in indices]
        stds  = [self._lookup[i][1] for i in indices]
        return means, stds


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline builder
# ══════════════════════════════════════════════════════════════════════════════

def build_pipeline(
    source,
    aug_cfg:            DINOAugConfig,
    batch_size:         int,
    num_threads:        int,
    device_id:          int,
    resolution_src,                     # ResolutionSource
    hw_decoder_load:    float = 0.90,
    cpu_queue:          int   = 8,
    gpu_queue:          int   = 6,
    seed:               int   = 42,
    # [PL-3] per-dataset norm
    norm_source:        Optional[NormSource] = None,
    fuse_normalization: bool  = True,
    # [PL-5] in-graph FP8
    dali_fp8_output:    bool  = False,
):
    """
    Build and return a compiled DALI pipeline.

    Parameters
    ----------
    source              : MixingSource — DALI ExternalSource callback for JPEG bytes.
    aug_cfg             : DINOAugConfig.
    batch_size          : Samples per GPU per step.
    num_threads         : DALI CPU worker threads.
    device_id           : GPU index.
    resolution_src      : ResolutionSource — drives dynamic resize without rebuild.
    hw_decoder_load     : Fraction of JPEG decode sent to nvjpeg HW ASIC.
    cpu_queue           : DALI CPU-side prefetch queue depth.
    gpu_queue           : DALI GPU-side prefetch queue depth.
    seed                : Base random seed.
    norm_source         : NormSource instance for fused per-dataset normalisation.
                          Required when fuse_normalization=True.
    fuse_normalization  : Fuse per-dataset mean/std into the DALI graph [PL-3].
    dali_fp8_output     : Perform FP8 cast inside the DALI graph [PL-5].
                          Requires DALI ≥ 1.36.
    """
    if not HAS_DALI:
        raise RuntimeError("nvidia-dali is required but not installed.")
    if not 0.0 <= hw_decoder_load <= 1.0:
        raise ValueError(f"hw_decoder_load must be in [0, 1], got {hw_decoder_load}")
    if fuse_normalization and norm_source is None:
        raise ValueError(
            "build_pipeline: fuse_normalization=True requires a NormSource instance. "
            "Pass norm_source=NormSource(aug_cfg, specs)."
        )
    if dali_fp8_output:
        # Probe DALI for FLOAT8_E4M3 support (DALI ≥ 1.36)
        if not hasattr(types, "FLOAT8_E4M3"):
            raise RuntimeError(
                "dali_fp8_output=True requires DALI ≥ 1.36 (types.FLOAT8_E4M3). "
                "Upgrade nvidia-dali or set dali_fp8_output=False."
            )

    # [PL-1] Pre-allocation ceilings — static max, prevents GPU re-allocations
    max_global = aug_cfg.max_global_crop_size
    max_local  = aug_cfg.max_local_crop_size

    # Determine output dtype for augmentation kernels
    # [PL-5] FP8 cast happens at the very end; intermediate ops use FLOAT16.
    intermediate_dtype = types.FLOAT16

    @pipeline_def(
        batch_size           = batch_size,
        num_threads          = num_threads,
        device_id            = device_id,
        seed                 = seed,
        prefetch_queue_depth = {"cpu_size": cpu_queue, "gpu_size": gpu_queue},
        exec_async           = True,
        exec_pipelined       = True,
    )
    def _pipe():
        # ── JPEG bytes from MixingSource ─────────────────────────────────────
        jpegs = fn.external_source(
            source  = source,
            dtype   = types.UINT8,
            ndim    = 1,
            name    = "jpegs",
            no_copy = True,
        )

        # ── [PL-1] Dynamic resolution scalars ────────────────────────────────
        global_size_node, local_size_node = fn.external_source(
            source      = resolution_src,
            num_outputs = 2,
            dtype       = types.INT32,
            ndim        = 0,
            batch       = False,
            name        = "resolution",
        )

        # ── [PL-3] Per-sample normalisation stats ────────────────────────────
        if fuse_normalization:
            norm_means, norm_stds = fn.external_source(
                source      = norm_source,
                num_outputs = 2,
                dtype       = types.FLOAT,
                ndim        = 1,   # shape (3,) per sample
                name        = "norm_stats",
            )
        else:
            norm_means = None
            norm_stds  = None

        views = []

        for i in range(aug_cfg.n_global_crops):
            blur_p = aug_cfg.blur_prob_global1 if i == 0 else aug_cfg.blur_prob_global2
            sol_p  = aug_cfg.solarize_prob if i == 1 else 0.0
            views.append(_augment_view(
                jpegs, aug_cfg,
                size_node              = global_size_node,
                max_size               = max_global,
                scale                  = aug_cfg.global_crops_scale,
                blur_prob              = blur_p,
                solarize_prob          = sol_p,
                hw_decoder_load        = hw_decoder_load,
                preserve_aspect_ratio  = aug_cfg.preserve_aspect_ratio,
                norm_means             = norm_means,
                norm_stds              = norm_stds,
                fuse_normalization     = fuse_normalization,
                dali_fp8_output        = dali_fp8_output,
                intermediate_dtype     = intermediate_dtype,
            ))

        for _ in range(aug_cfg.n_local_crops):
            views.append(_augment_view(
                jpegs, aug_cfg,
                size_node              = local_size_node,
                max_size               = max_local,
                scale                  = aug_cfg.local_crops_scale,
                blur_prob              = aug_cfg.blur_prob_local,
                solarize_prob          = 0.0,
                hw_decoder_load        = hw_decoder_load,
                preserve_aspect_ratio  = aug_cfg.preserve_aspect_ratio,
                norm_means             = norm_means,
                norm_stds              = norm_stds,
                fuse_normalization     = fuse_normalization,
                dali_fp8_output        = dali_fp8_output,
                intermediate_dtype     = intermediate_dtype,
            ))

        return tuple(views)

    pipe = _pipe()
    pipe.build()
    log.info(
        "DALI pipeline built: %d global + %d local crops, "
        "batch=%d, threads=%d, hw_decoder=%.0f%%, "
        "max_global=%d, max_local=%d, aspect_ratio=%s, "
        "fused_norm=%s, dali_fp8=%s",
        aug_cfg.n_global_crops, aug_cfg.n_local_crops,
        batch_size, num_threads, hw_decoder_load * 100,
        max_global, max_local,
        "preserved" if aug_cfg.preserve_aspect_ratio else "stretched",
        fuse_normalization,
        dali_fp8_output,
    )
    return pipe


# ══════════════════════════════════════════════════════════════════════════════
# Single-view augmentation sub-graph
# ══════════════════════════════════════════════════════════════════════════════

def _augment_view(
    jpegs,
    cfg:                   DINOAugConfig,
    size_node,
    max_size:              int,
    scale:                 Tuple[float, float],
    blur_prob:             float,
    solarize_prob:         float,
    hw_decoder_load:       float,
    preserve_aspect_ratio: bool = True,
    # [PL-3]
    norm_means             = None,
    norm_stds              = None,
    fuse_normalization:    bool = True,
    # [PL-5]
    dali_fp8_output:       bool = False,
    intermediate_dtype            = None,
):
    """
    Augmentation sub-graph for one crop view.

    Tensor layout  : DALI returns HWC; CHW via fn.transpose at the end.
    Float range    : After cast(FLOAT16), pixels ∈ [0.0, 255.0].
    Normalisation  : Applied after all stochastic augmentations.
                     [PL-3] When fuse_normalization=True, uses per-sample
                     mean/std DataNodes instead of baked numpy constants.
    Final cast     : [PL-5] FLOAT8_E4M3 when dali_fp8_output=True, else FLOAT16.
                     The DALI compiler fuses: normalize → cast → transpose.
    """
    if intermediate_dtype is None:
        import nvidia.dali.types as types
        intermediate_dtype = types.FLOAT16

    # ── 1. HW JPEG decode + random resized crop ───────────────────────────────
    imgs = fn.decoders.image_random_crop(
        jpegs,
        device                  = "mixed",
        output_type             = types.RGB,
        random_area             = list(scale),
        random_aspect_ratio     = [3 / 4, 4 / 3],
        num_attempts            = 10,
        hw_decoder_load         = hw_decoder_load,
        preallocate_width_hint  = max_size * 2,
        preallocate_height_hint = max_size * 2,
    )

    # ── 2. Resize — aspect-ratio-aware or legacy squash ───────────────────────
    if preserve_aspect_ratio:
        imgs = fn.resize(
            imgs,
            device      = "gpu",
            size        = size_node,
            mode        = "not_smaller",
            interp_type = types.INTERP_CUBIC,
            antialias   = False,
        )
        imgs = fn.crop(
            imgs,
            device     = "gpu",
            crop_h     = size_node,
            crop_w     = size_node,
            crop_pos_x = 0.5,
            crop_pos_y = 0.5,
        )
    else:
        imgs = fn.resize(
            imgs,
            device      = "gpu",
            resize_x    = size_node,
            resize_y    = size_node,
            interp_type = types.INTERP_CUBIC,
            antialias   = False,
        )

    # ── 3. Random horizontal flip ─────────────────────────────────────────────
    do_flip = fn.random.coin_flip(probability=cfg.flip_prob, dtype=types.BOOL)
    imgs    = fn.flip(imgs, device="gpu", horizontal=do_flip)

    # ── 4. Cast to FLOAT16 for all subsequent ops ─────────────────────────────
    imgs = fn.cast(imgs, dtype=types.FLOAT16)

    # ── 5. Colour jitter ──────────────────────────────────────────────────────
    do_jitter = fn.cast(
        fn.random.coin_flip(probability=cfg.color_jitter_prob, dtype=types.BOOL),
        dtype=types.FLOAT16,
    )
    jittered  = fn.color_twist(
        imgs,
        brightness = fn.random.uniform(range=(1 - cfg.brightness, 1 + cfg.brightness)),
        contrast   = fn.random.uniform(range=(1 - cfg.contrast,   1 + cfg.contrast)),
        saturation = fn.random.uniform(range=(1 - cfg.saturation, 1 + cfg.saturation)),
        hue        = fn.random.uniform(range=(-cfg.hue * 180,     cfg.hue * 180)),
    )
    imgs = do_jitter * jittered + (1 - do_jitter) * imgs

    # ── 6. Random grayscale ───────────────────────────────────────────────────
    do_gray = fn.cast(
        fn.random.coin_flip(probability=cfg.grayscale_prob, dtype=types.BOOL),
        dtype=types.FLOAT16,
    )
    gray = fn.color_space_conversion(imgs, image_type=types.RGB, output_type=types.GRAY)
    gray = fn.cat(gray, gray, gray, axis=2)
    imgs = do_gray * gray + (1 - do_gray) * imgs

    # ── 7. Gaussian blur ──────────────────────────────────────────────────────
    sigma   = fn.random.uniform(range=(cfg.blur_sigma_min, cfg.blur_sigma_max))
    blurred = fn.gaussian_blur(imgs, sigma=sigma)
    do_blur = fn.cast(
        fn.random.coin_flip(probability=blur_prob, dtype=types.BOOL),
        dtype=types.FLOAT16,
    )
    imgs = do_blur * blurred + (1 - do_blur) * imgs

    # ── 8. Solarisation (second global crop only) ─────────────────────────────
    if solarize_prob > 0:
        do_sol = fn.cast(
            fn.random.coin_flip(probability=solarize_prob, dtype=types.BOOL),
            dtype=types.FLOAT16,
        )
        mask = imgs >= 128.0
        sol  = mask * (255.0 - imgs) + (1 - mask) * imgs
        imgs = do_sol * sol + (1 - do_sol) * imgs

    # ── 9. Normalise ──────────────────────────────────────────────────────────
    # [PL-3] Fused path: per-sample mean/std from ExternalSource DataNodes.
    #         The DALI compiler sees: (imgs / 255 - mean) / std → cast → transpose
    #         and can fuse these elementwise ops into one kernel.
    # Legacy path: numpy constants baked at graph build time.
    imgs = imgs / 255.0
    if fuse_normalization and norm_means is not None:
        # norm_means / norm_stds: DataNode of shape (3,) — DALI broadcasts
        # over HW automatically (same broadcasting semantics as numpy).
        imgs = (imgs - norm_means) / norm_stds
    else:
        mean_arr = np.array(cfg.mean, dtype=np.float32).reshape(1, 1, 3)
        std_arr  = np.array(cfg.std,  dtype=np.float32).reshape(1, 1, 3)
        imgs = (imgs - mean_arr) / std_arr

    # ── 10. HWC → CHW  +  optional final FP8 cast ────────────────────────────
    # [PL-5] When dali_fp8_output=True, cast to FLOAT8_E4M3 before transpose.
    #         The DALI compiler fuses: cast(FP8) → transpose → (output).
    #         When False, output is FLOAT16 — FP8Formatter handles it post-DALI.
    if dali_fp8_output:
        imgs = fn.cast(imgs, dtype=types.FLOAT8_E4M3)

    return fn.transpose(imgs, perm=[2, 0, 1])

"""
dino_loader.pipeline
====================
DALI augmentation pipeline for DINOv3 multi-crop.

Changes in this version
-----------------------
[PL-1]  Dynamic resize via ResolutionSource (zero pipeline rebuild, retained).
[PL-2]  Aspect-ratio-preserving resize (retained).
[PL-3]  Per-dataset normalisation stats fused into DALI graph (retained).
[PL-4]  Explicit FLOAT16 cast hardening (retained).
[PL-5]  Optional in-graph FP8 cast (retained).

[M2-FIX] NormSource thread safety hardened.                             ← FIX M2
         NormSource.set_dataset_indices() is called from the MixingSource
         thread (the DALI ExternalSource callback thread), while DALI calls
         NormSource.__call__() from its own internal prefetch thread.  The
         existing threading.Lock correctly serialises these two callers.

         However, there was a subtle race when set_weights() on MixingSource
         changed the active dataset distribution: the MixingSource would
         update self._indices (via set_dataset_indices) while the DALI thread
         was mid-way through iterating over the previous indices list.

         Fix: set_dataset_indices() now performs a copy-on-write — it builds
         the new list fully before acquiring the lock and swapping the
         reference.  Since Python list assignment is atomic at the C level
         (GIL), and we also hold the threading.Lock, this is safe.  The DALI
         thread always reads a consistent, complete list.

         Additionally, NormSource.__call__() now returns numpy arrays that
         are copies of the lookup values (not views), preventing the DALI
         pipeline from holding a reference into the live lookup table.
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

    Architecture
    ------------
    MixingSource calls set_dataset_indices(indices) immediately before DALI
    pulls the next batch, so that each sample in the batch gets the correct
    normalisation for its originating dataset.

    Thread safety — [M2-FIX]
    -------------------------
    set_dataset_indices() is called from MixingSource's DALI ExternalSource
    callback thread.  DALI calls __call__() from its own prefetch thread.
    A threading.Lock serialises access to self._indices.

    The fix vs the previous version:
    - set_dataset_indices() builds the new list fully BEFORE acquiring the
      lock, then swaps the reference atomically under the lock.  This
      eliminates the window where the DALI thread could read a partially
      built list.
    - __call__() snapshots self._indices under the lock, then builds the
      return arrays outside the lock to minimise lock hold time.
    - Return values are explicit numpy copies (np.array(x, copy=True)) so
      the DALI pipeline cannot hold a reference into the live lookup table.

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
        global_mean = np.array(aug_cfg.mean, dtype=np.float32)
        global_std  = np.array(aug_cfg.std,  dtype=np.float32)
        self._lookup: List[Tuple[np.ndarray, np.ndarray]] = []
        for spec in specs:
            m = np.array(spec.mean, dtype=np.float32) if spec.mean else global_mean.copy()
            s = np.array(spec.std,  dtype=np.float32) if spec.std  else global_std.copy()
            self._lookup.append((m, s))

        self._indices: List[int] = [0]   # placeholder; replaced before first batch
        self._lock = threading.Lock()

    def set_dataset_indices(self, indices: List[int]) -> None:
        """
        Called by MixingSource before each batch is pushed to DALI.

        [M2-FIX] Build the new list fully outside the lock, then swap
        atomically under the lock.  This prevents the DALI thread from
        ever observing a partially-constructed list.
        """
        new_indices = list(indices)   # full copy, no lock needed
        with self._lock:
            self._indices = new_indices

    def __call__(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Return (means, stds) — one (3,) FLOAT32 array per sample.

        [M2-FIX] Snapshot indices under lock; build arrays outside lock.
        Return explicit numpy copies so the DALI pipeline does not retain
        references into self._lookup.
        """
        with self._lock:
            indices = self._indices   # atomic reference read (GIL + Lock)

        means = [np.array(self._lookup[i][0], dtype=np.float32) for i in indices]
        stds  = [np.array(self._lookup[i][1], dtype=np.float32) for i in indices]
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
    norm_source:        Optional[NormSource] = None,
    fuse_normalization: bool  = True,
    dali_fp8_output:    bool  = False,
):
    """
    Build and return a compiled DALI pipeline.

    Parameters
    ----------
    source              : MixingSource — DALI ExternalSource callback.
    aug_cfg             : DINOAugConfig.
    batch_size          : Samples per GPU per step.
    num_threads         : DALI CPU worker threads.
    device_id           : GPU index.
    resolution_src      : ResolutionSource — dynamic resize (PL-1).
    hw_decoder_load     : Fraction sent to nvjpeg HW ASIC (0–1).
    cpu_queue / gpu_queue : DALI pipeline queue depths.
    seed                : Pipeline RNG seed.
    norm_source         : NormSource for per-dataset normalisation (PL-3).
    fuse_normalization  : When True, per-dataset norm is fused in graph.
    dali_fp8_output     : When True, final cast uses FLOAT8_E4M3 (PL-5).
    """
    if not HAS_DALI:
        raise RuntimeError(
            "nvidia-dali is not installed.  "
            "Install with: pip install nvidia-dali-cuda120"
        )

    n_views = aug_cfg.n_views

    @pipeline_def(
        batch_size      = batch_size,
        num_threads     = num_threads,
        device_id       = device_id,
        prefetch_queue_depth = {"cpu_size": cpu_queue, "gpu_size": gpu_queue},
        seed            = seed,
        exec_async      = True,
        exec_pipelined  = True,
    )
    def _pipeline_fn():
        # ── ExternalSource: JPEG bytes ────────────────────────────────────────
        jpegs = fn.external_source(
            source          = source,
            num_outputs     = 1,
            batch           = True,
            dtype           = types.UINT8,
            name            = "jpegs",
        )[0]

        # ── ExternalSource: dynamic resolution (PL-1) ─────────────────────────
        global_size, local_size = fn.external_source(
            source      = resolution_src,
            num_outputs = 2,
            batch       = False,
            dtype       = types.INT32,
            name        = "resolution",
        )

        # ── Optional: per-dataset norm (PL-3) ─────────────────────────────────
        if fuse_normalization and norm_source is not None:
            means, stds = fn.external_source(
                source      = norm_source,
                num_outputs = 2,
                batch       = True,
                dtype       = types.FLOAT,
                name        = "norm_stats",
            )
        else:
            means = stds = None

        views = []
        for i in range(n_views):
            is_global = i < aug_cfg.n_global_crops
            crop_size = global_size if is_global else local_size
            scale     = aug_cfg.global_crops_scale if is_global else aug_cfg.local_crops_scale
            view = _augment_view(
                jpegs          = jpegs,
                aug_cfg        = aug_cfg,
                crop_size      = crop_size,
                scale          = scale,
                seed           = seed + i,
                hw_decoder_load = hw_decoder_load,
                means          = means,
                stds           = stds,
                dali_fp8_output = dali_fp8_output,
                name           = f"view_{i}",
            )
            views.append(view)

        return tuple(views)

    pipe = _pipeline_fn()
    pipe.build()
    return pipe


def _augment_view(
    jpegs,
    aug_cfg:         DINOAugConfig,
    crop_size,
    scale:           Tuple[float, float],
    seed:            int,
    hw_decoder_load: float,
    means,
    stds,
    dali_fp8_output: bool,
    name:            str,
):
    """Build one augmentation branch (one crop view)."""
    # Decode JPEG with HW acceleration
    decoded = fn.decoders.image(
        jpegs,
        device             = "mixed",
        output_type        = types.RGB,
        hw_decoder_load    = hw_decoder_load,
    )

    # RandomResizedCrop
    if aug_cfg.preserve_aspect_ratio:
        resized = fn.random_resized_crop(
            decoded,
            size           = crop_size,
            random_area    = scale,
            random_aspect_ratio = (3/4, 4/3),
            device         = "gpu",
        )
    else:
        resized = fn.random_resized_crop(
            decoded,
            size        = crop_size,
            random_area = scale,
            device      = "gpu",
        )

    # Color jitter + grayscale
    augmented = fn.color_twist(
        resized,
        brightness = fn.random.uniform(range=(0.6, 1.4), seed=seed),
        contrast   = fn.random.uniform(range=(0.6, 1.4), seed=seed + 1),
        saturation = fn.random.uniform(range=(0.6, 1.4), seed=seed + 2),
        hue        = fn.random.uniform(range=(-0.1, 0.1), seed=seed + 3),
    )

    # Horizontal flip
    augmented = fn.flip(augmented, horizontal=1, vertical=0)

    # Gaussian blur
    blurred = fn.gaussian_blur(
        augmented,
        sigma = fn.random.uniform(range=(0.1, 2.0), seed=seed + 4),
    )
    augmented = fn.cast(
        fn.random.coin_flip(probability=aug_cfg.blur_prob_global1, seed=seed + 5),
        dtype=types.FLOAT,
    ) * blurred + fn.cast(
        1 - fn.random.coin_flip(probability=aug_cfg.blur_prob_global1, seed=seed + 5),
        dtype=types.FLOAT,
    ) * augmented

    # Normalise — fused per-dataset (PL-3) or global
    if means is not None and stds is not None:
        normalised = fn.normalize(
            augmented,
            mean   = means,
            stddev = stds,
            dtype  = types.FLOAT16,
        )
    else:
        mean_arr = np.array(aug_cfg.mean, dtype=np.float32) * 255.0
        std_arr  = np.array(aug_cfg.std,  dtype=np.float32) * 255.0
        normalised = fn.normalize(
            augmented,
            mean   = mean_arr.tolist(),
            stddev = std_arr.tolist(),
            dtype  = types.FLOAT16,
        )

    # [PL-5] Optional in-graph FP8 cast
    if dali_fp8_output:
        try:
            output = fn.cast(normalised, dtype=types.FLOAT8_E4M3)
        except AttributeError:
            log.warning(
                "DALI FLOAT8_E4M3 not available (requires DALI ≥ 1.36) — "
                "falling back to FLOAT16 output."
            )
            output = normalised
    else:
        output = normalised

    # NHWC → NCHW transpose
    output = fn.transpose(output, perm=[2, 0, 1], name=name)
    return output

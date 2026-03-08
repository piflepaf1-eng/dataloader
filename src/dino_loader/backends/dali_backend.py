"""
dino_loader.backends.dali_backend
==================================
Concrete backend: NVIDIA DALI + CUDA + SLURM production path.

What changed
------------
Previously, ``NormSource`` was constructed in ``loader.py`` and passed
as an argument to ``build_pipeline``.  This created an asymmetry: the
DALI-specific concept of ``NormSource`` leaked into the generic loader code,
which had to import from ``dino_loader.pipeline`` directly.

Now, ``DALIBackend.build_pipeline`` is **fully self-contained**:

- It builds ``NormSource`` internally when ``fuse_normalization=True``.
- It registers the ``NormSource`` callback on *source* via
  ``source.register_dataset_index_callback()``.
- ``loader.py`` passes ``specs``, ``fuse_normalization``, and
  ``dali_fp8_output`` as plain arguments — no DALI import required in
  the loader.

``pipeline.py`` (``NormSource``, ``build_pipeline``) remains the source of
truth for the DALI graph implementation.  ``DALIBackend`` is the integration
layer that wires those building blocks together using loader-level information.

Backend symmetry
----------------
``CPUBackend.build_pipeline`` has the same signature but ignores the optional
``specs``, ``fuse_normalization``, and ``dali_fp8_output`` kwargs.
``loader.py`` is now 100% backend-agnostic.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

log = logging.getLogger(__name__)

try:
    from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
    HAS_DALI = True
except ImportError:
    HAS_DALI = False


class DALIBackend:
    """
    Concrete backend: NVIDIA DALI + CUDA + SLURM production path.

    All DALI-specific logic — including ``NormSource`` construction and
    FP8 in-graph casting — is encapsulated here.  ``loader.py`` interacts
    exclusively through the ``BackendProtocol`` interface.
    """

    # ── BackendProtocol identity ──────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "dali"

    @property
    def supports_fp8(self) -> bool:
        try:
            import transformer_engine.pytorch  # noqa: F401
            return True
        except ImportError:
            return False

    @property
    def supports_gpu(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    # ── Stage 1: Shard cache ─────────────────────────────────────────────────

    def build_shard_cache(
        self,
        job_id:          str,
        node_master:     bool,
        max_gb:          float,
        prefetch_window: int,
        timeout_s:       float,
        warn_threshold:  float,
    ) -> Any:
        from dino_loader.shard_cache import NodeSharedShardCache
        return NodeSharedShardCache(
            job_id             = job_id,
            node_master        = node_master,
            max_shm_gb         = max_gb,
            prefetch_window    = prefetch_window,
            shard_timeout_s    = timeout_s,
            shm_warn_threshold = warn_threshold,
        )

    # ── Stage 3: Augmentation pipeline ───────────────────────────────────────

    def build_pipeline(
        self,
        source:             Any,
        aug_cfg:            Any,
        batch_size:         int,
        num_threads:        int,
        device_id:          int,
        resolution_src:     Any,
        hw_decoder_load:    float = 0.90,
        cpu_queue:          int   = 8,
        gpu_queue:          int   = 6,
        seed:               int   = 42,
        # ── DALI-specific knobs (ignored by CPUBackend) ───────────────────────
        specs:              Optional[List[Any]] = None,
        fuse_normalization: bool  = False,
        dali_fp8_output:    bool  = False,
    ) -> Any:
        """
        Build and return a compiled DALI pipeline.

        NormSource lifecycle
        --------------------
        When ``fuse_normalization=True`` **and** ``specs`` is provided:

        1. A ``NormSource`` is constructed from ``(aug_cfg, specs)``.
        2. ``source.register_dataset_index_callback(norm_source.set_dataset_indices)``
           is called so that ``MixingSource`` drives per-sample mean/std selection.
        3. The ``NormSource`` instance is passed into ``build_pipeline`` from
           ``dino_loader.pipeline`` as the DALI ExternalSource callback.

        This means ``loader.py`` no longer needs to know about ``NormSource``.
        """
        try:
            import nvidia.dali  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "nvidia-dali is required for the DALI backend but is not installed.\n"
                "Install it with: pip install nvidia-dali-cuda120\n"
                "Or use the CPU backend: get_backend('cpu')"
            )

        from dino_loader.pipeline import NormSource, build_pipeline

        # ── [PL-3] Build NormSource internally — no leak into loader.py ──────
        norm_source: Optional[NormSource] = None
        if fuse_normalization and specs is not None:
            norm_source = NormSource(aug_cfg=aug_cfg, specs=specs)
            source.register_dataset_index_callback(norm_source.set_dataset_indices)
            log.debug(
                "DALIBackend: NormSource built for %d dataset(s), "
                "fused into DALI graph.",
                len(specs),
            )

        return build_pipeline(
            source             = source,
            aug_cfg            = aug_cfg,
            batch_size         = batch_size,
            num_threads        = num_threads,
            device_id          = device_id,
            resolution_src     = resolution_src,
            hw_decoder_load    = hw_decoder_load,
            cpu_queue          = cpu_queue,
            gpu_queue          = gpu_queue,
            seed               = seed,
            norm_source        = norm_source,
            fuse_normalization = fuse_normalization,
            dali_fp8_output    = dali_fp8_output,
        )

    def build_pipeline_iterator(
        self,
        pipeline:   Any,
        output_map: List[str],
        batch_size: int,
    ) -> Any:
        if not HAS_DALI:
            raise RuntimeError("nvidia-dali required for DALIGenericIterator.")
        return DALIGenericIterator(
            pipelines         = [pipeline],
            output_map        = output_map,
            last_batch_policy = LastBatchPolicy.DROP,
            auto_reset        = False,
        )

    # ── Stage 4: H2D transfer ─────────────────────────────────────────────────

    def build_h2d_stream(self, device: Any, topo: Any) -> Any:
        from dino_loader.memory import H2DStream
        return H2DStream(device=device, topo=topo)

    # ── Stage 5: FP8 formatter ────────────────────────────────────────────────

    def build_fp8_formatter(self) -> Any:
        from dino_loader.memory import FP8Formatter
        return FP8Formatter()

    # ── Distributed bootstrap ─────────────────────────────────────────────────

    def init_distributed(
        self,
        rank:             int = 0,
        world_size:       int = 1,
        local_rank:       int = 0,
        local_world_size: int = 1,
        force_topology:   Optional[str] = None,
    ) -> Any:
        """
        Build a ``DistribEnv`` from already-initialised distributed state.

        For the DALI backend, ``torch.distributed`` is expected to have been
        initialised already via ``slurm_init()``.  This method only constructs
        the ``DistribEnv`` struct — it does not call
        ``init_process_group()`` again.
        """
        from dino_loader.distributed import detect_topology, DistribEnv
        topo = detect_topology(force=force_topology)
        return DistribEnv(
            rank             = rank,
            world_size       = world_size,
            local_rank       = local_rank,
            local_world_size = local_world_size,
            topology         = topo,
        )

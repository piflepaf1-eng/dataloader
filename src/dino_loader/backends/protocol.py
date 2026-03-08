"""
dino_loader.backends.protocol
==============================
Abstract interface (Protocol) that every backend must satisfy.

Every method corresponds to one of the five pipeline stages:

Stage 1  — Shard I/O          → build_shard_cache()
Stage 2  — JPEG Extraction    → handled by MixingSource / ShardIterator
           (extraction is backend-agnostic; wds.TarIterator / legacy parser
           run on bytes regardless of how those bytes arrived)
Stage 3  — Augmentation       → build_pipeline()
Stage 4  — H2D Transfer       → build_h2d_stream()
Stage 5  — FP8 Quantisation   → build_fp8_formatter()

Additionally:
           init_distributed()  → bootstraps a DistribEnv without SLURM

Backend symmetry
----------------
Each backend must implement exactly the same interface.  The ``DALIBackend``
and ``CPUBackend`` are now fully symmetric: both receive the same call
signatures from ``loader.py``.  Backend-specific concerns (e.g. NormSource
for DALI, or no-op stubs for CPU) are fully encapsulated inside the backend.

``loader.py`` is backend-agnostic and never imports from
``dino_loader.pipeline`` or ``dino_loader.memory`` directly.
"""

from __future__ import annotations

from typing import Any, List, Optional, Protocol, runtime_checkable


@runtime_checkable
class BackendProtocol(Protocol):
    """
    Structural protocol for dino_loader pipeline backends.

    All methods return objects that conform to the interfaces expected by
    loader.py.  Backends are free to return thin wrappers, mock objects, or
    real hardware-backed instances — as long as the call signatures match.

    Design contract
    ---------------
    ``loader.py`` calls backend methods using **only** the parameters defined
    here.  Backend-specific parameters (e.g. DALI's ``norm_source``,
    ``fuse_normalization``, ``dali_fp8_output``) are listed here with
    semantically neutral defaults so that the CPU backend can silently
    ignore them.  The DALI backend reads them and acts accordingly.

    This keeps ``loader.py`` free of ``if backend.name == "dali"`` branches.
    """

    # ── Identity ──────────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        """Short identifier, e.g. ``"cpu"`` or ``"dali"``."""
        ...

    @property
    def supports_fp8(self) -> bool:
        """True iff Transformer Engine FP8 output is available on this backend."""
        ...

    @property
    def supports_gpu(self) -> bool:
        """True iff GPU tensors are produced by this backend."""
        ...

    # ── Stage 1: Shard cache ──────────────────────────────────────────────────

    def build_shard_cache(
        self,
        job_id:             str,
        node_master:        bool,
        max_gb:             float,
        prefetch_window:    int,
        timeout_s:          float,
        warn_threshold:     float,
    ) -> Any:
        """
        Return a shard-cache object with the ``NodeSharedShardCache`` API:
        ``prefetch(path)``, ``get(path)``, ``get_view(path)``,
        ``utilisation`` property.
        """
        ...

    # ── Stage 3: Augmentation pipeline ───────────────────────────────────────

    def build_pipeline(
        self,
        source:             Any,    # MixingSource — DALI ExternalSource callback
        aug_cfg:            Any,    # DINOAugConfig
        batch_size:         int,
        num_threads:        int,
        device_id:          int,
        resolution_src:     Any,    # ResolutionSource
        hw_decoder_load:    float,
        cpu_queue:          int,
        gpu_queue:          int,
        seed:               int,
        # ── Backend-specific optional knobs ──────────────────────────────────
        # These are consumed by DALIBackend and silently ignored by CPUBackend.
        specs:              Optional[List[Any]] = None,  # List[DatasetSpec] for NormSource
        fuse_normalization: bool  = False,
        dali_fp8_output:    bool  = False,
    ) -> Any:
        """
        Return an augmentation pipeline object that supports the iteration
        protocol used by ``loader.py``.

        The pipeline is consumed via ``build_pipeline_iterator()``.

        Parameters
        ----------
        source
            ``MixingSource`` instance — the DALI / CPU ExternalSource callback
            for JPEG bytes.
        aug_cfg
            ``DINOAugConfig``.
        batch_size
            Samples per GPU per step.
        num_threads
            Backend CPU worker threads.
        device_id
            GPU index (ignored by CPU backend).
        resolution_src
            ``ResolutionSource`` — drives dynamic resize without pipeline rebuild.
        hw_decoder_load
            Fraction of JPEG decode sent to nvjpeg HW ASIC (DALI only).
        cpu_queue / gpu_queue
            DALI prefetch queue depths (DALI only; ignored by CPU backend).
        seed
            Base random seed.
        specs
            List of ``DatasetSpec`` objects.  Required by ``DALIBackend`` when
            ``fuse_normalization=True`` to build ``NormSource``.  Ignored by
            ``CPUBackend``.
        fuse_normalization
            When ``True``, ``DALIBackend`` builds a ``NormSource`` internally
            and registers it on *source* via
            ``source.register_dataset_index_callback()``.  Ignored by
            ``CPUBackend``.
        dali_fp8_output
            When ``True``, ``DALIBackend`` emits FP8-cast tensors directly from
            the DALI graph.  Requires DALI ≥ 1.36.  Ignored by ``CPUBackend``.
        """
        ...

    def build_pipeline_iterator(
        self,
        pipeline:   Any,
        output_map: List[str],
        batch_size: int,
    ) -> Any:
        """
        Wrap the pipeline in an iterator that yields dicts of
        ``{view_name: tensor}``, matching the ``DALIGenericIterator`` API
        used in ``loader.py``.

        For DALI: returns a ``DALIGenericIterator``.
        For CPU:  returns a ``CPUPipelineIterator``.
        """
        ...

    # ── Stage 4: H2D transfer ─────────────────────────────────────────────────

    def build_h2d_stream(
        self,
        device: Any,       # torch.device
        topo:   Any,       # ClusterTopology
    ) -> Any:
        """
        Return an H2DStream-compatible object with ``transfer()`` context
        manager and ``send()`` / ``wait()`` methods.
        """
        ...

    # ── Stage 5: FP8 formatter ────────────────────────────────────────────────

    def build_fp8_formatter(self) -> Optional[Any]:
        """
        Return an FP8Formatter-compatible object (``quantise(tensor)`` method),
        or ``None`` if FP8 is unavailable / disabled for this backend.
        """
        ...

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
        Return a ``DistribEnv``-compatible object.

        For DALI: delegates to ``slurm_init()`` or reads env vars.
        For CPU:  constructs a fake single-rank ``DistribEnv`` with a stub
                  ``ClusterTopology``; does NOT call
                  ``torch.distributed.init_process_group()``.
        """
        ...

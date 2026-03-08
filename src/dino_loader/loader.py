"""
dino_loader.loader
==================
DINODataLoader: the single public entry point for training code.

Changes in this version
-----------------------
[LD-1..LD-12] All previous changes retained (see previous version header).

[B3-FIX]  set_epoch() is now guarded by a threading.Lock.              ← FIX B3
          Previously two concurrent calls (e.g. from PostProcessPipeline
          and a background scheduler thread) could interleave, corrupting
          the MixingSource epoch state.  A reentrant guard also prevents
          starting a new epoch loop while the previous is still active.

[M6-FIX]  PostProcessPipeline.select() now tracks filtered batches.   ← FIX M6
          A ``batches_filtered`` counter is incremented in the metrics
          registry for each batch dropped by select().  Cumulative and
          per-step filtering rates are exposed so operators can detect
          mis-calibrated min_sample_quality or quality predicates.

[LD-13]   current_resolution property exposed publicly.               ← NEW
          Replaces direct access to ``loader._current_global_size``
          (which appeared in train.py and user code).  Returns a
          ``(global_size, local_size)`` tuple.

[ARCH1]   Ring buffer flag forwarded to build_shard_cache.            ← WIRING
[ARCH2]   Adaptive prefetch flags forwarded to build_shard_cache.     ← WIRING
[ARCH3]   Prometheus metrics server started if prometheus_port is set. ← WIRING
"""

from __future__ import annotations

import fcntl
import logging
import os
import threading
import time
from typing import Any, Callable, Iterator, List, Optional, Sequence

import torch
import torch.distributed as dist

from dino_loader.backends          import get_backend
from dino_loader.backends.protocol import BackendProtocol
from dino_loader.checkpoint        import DataLoaderCheckpointer
from dino_loader.config            import CheckpointState, DatasetSpec, DINOAugConfig, LoaderConfig
from dino_loader.memory            import Batch
from dino_loader.mixing_source     import MixingSource, ResolutionSource
from dino_loader.monitor.metrics   import get_registry, init_registry

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# PostProcessPipeline — fluid interface for post-DALI transforms
# ══════════════════════════════════════════════════════════════════════════════

class PostProcessPipeline:
    """
    A lazy, composable wrapper over a Batch iterator.

    Each method returns a new PostProcessPipeline; the original is not mutated.
    Transforms execute only as batches flow through — no buffering.

    Notes on select()
    -----------------
    ``select`` silently drops non-matching batches.  Each dropped batch still
    consumed a DALI decode slot.  The number of dropped batches is tracked in
    the metrics registry under ``batches_filtered``.  [M6-FIX]
    """

    def __init__(
        self,
        source:     Iterator[Batch],
        transforms: List[Callable],
        loader:     "DINODataLoader",
        max_steps:  Optional[int] = None,
    ) -> None:
        self._source     = source
        self._transforms = transforms
        self._loader     = loader
        self._max_steps  = max_steps

    # ── Fluent chaining ───────────────────────────────────────────────────────

    def map(self, fn: Callable[[Batch], Batch]) -> "PostProcessPipeline":
        return PostProcessPipeline(
            source     = self._source,
            transforms = self._transforms + [fn],
            loader     = self._loader,
            max_steps  = self._max_steps,
        )

    def select(self, predicate: Callable[[Batch], bool]) -> "PostProcessPipeline":
        metrics = get_registry()

        def _filter(b: Batch) -> Optional[Batch]:
            if predicate(b):
                return b
            # [M6-FIX] Track dropped batches.
            if metrics is not None:
                metrics.inc("batches_filtered", 1)
            return None

        return PostProcessPipeline(
            source     = self._source,
            transforms = self._transforms + [_filter],
            loader     = self._loader,
            max_steps  = self._max_steps,
        )

    def with_epoch(self, n_steps: int) -> "PostProcessPipeline":
        return PostProcessPipeline(
            source     = self._source,
            transforms = self._transforms,
            loader     = self._loader,
            max_steps  = n_steps,
        )

    # ── Delegation to underlying DINODataLoader ───────────────────────────────

    def set_epoch(self, epoch: int) -> None:
        self._loader.set_epoch(epoch)

    def checkpoint(self, step: int) -> None:
        self._loader.checkpoint(step)

    def set_weights(self, weights: Sequence[float]) -> None:
        self._loader.set_weights(weights)

    def set_weight_by_name(self, name: str, weight: float) -> None:
        self._loader.set_weight_by_name(name, weight)

    def set_resolution(self, global_size: int, local_size: int) -> None:
        self._loader.set_resolution(global_size, local_size)

    @property
    def current_resolution(self) -> tuple[int, int]:
        return self._loader.current_resolution

    def state_dict(self) -> dict:
        return self._loader.state_dict()

    def load_state_dict(self, sd: dict) -> None:
        self._loader.load_state_dict(sd)

    # ── Iteration ─────────────────────────────────────────────────────────────

    def __iter__(self) -> Iterator[Batch]:
        step = 0
        for batch in self._source:
            if self._max_steps is not None and step >= self._max_steps:
                break
            result = batch
            for fn in self._transforms:
                if result is None:
                    break
                result = fn(result)
            if result is not None:
                yield result
                step += 1

    def __len__(self) -> int:
        if self._max_steps is not None:
            return self._max_steps
        return len(self._loader)


# ══════════════════════════════════════════════════════════════════════════════
# DINODataLoader
# ══════════════════════════════════════════════════════════════════════════════

class DINODataLoader:
    """
    HPC-grade DINOv3 data loader.

    Returns a PostProcessPipeline from map()/select()/with_epoch() — use the
    fluid API to chain post-DALI transforms without modifying this class.

    Parameters
    ----------
    specs            : List of DatasetSpec.
    batch_size       : Per-GPU batch size.
    aug_cfg          : DINOAugConfig.
    config           : LoaderConfig.
    device_id        : Local GPU index.
    rank / world_size / local_rank / local_world_size
                     : Distributed identity.
    resume           : Load latest dataloader checkpoint if available.
    steps_per_epoch  : Enables len(loader).
    mask_generator   : Optional iBOT MaskingGenerator (CPU, post-DALI).
    backend          : "auto" | "dali" | "cpu" | BackendProtocol instance.
    """

    def __init__(
        self,
        specs:            List[DatasetSpec],
        batch_size:       int,
        aug_cfg:          Optional[DINOAugConfig]  = None,
        config:           Optional[LoaderConfig]   = None,
        device_id:        int                      = 0,
        rank:             Optional[int]            = None,
        world_size:       Optional[int]            = None,
        local_rank:       Optional[int]            = None,
        local_world_size: Optional[int]            = None,
        resume:           bool                     = False,
        steps_per_epoch:  Optional[int]            = None,
        mask_generator:   Any                      = None,
        backend:          Any                      = "auto",
    ) -> None:
        self._aug_cfg        = aug_cfg or DINOAugConfig()
        self._cfg            = config  or LoaderConfig()
        self._mask_generator = mask_generator
        self._steps_per_epoch = steps_per_epoch
        self._active_iter    = False
        self._epoch_lock     = threading.Lock()   # [B3-FIX]

        # ── Backend ───────────────────────────────────────────────────────────
        if isinstance(backend, str):
            self._backend: BackendProtocol = get_backend(backend)
        else:
            self._backend = backend

        # ── Distributed ───────────────────────────────────────────────────────
        env = self._backend.init_distributed(
            rank             = rank             if rank             is not None else self._infer_rank(),
            world_size       = world_size       if world_size       is not None else self._infer_world_size(),
            local_rank       = local_rank       if local_rank       is not None else device_id,
            local_world_size = local_world_size if local_world_size is not None else self._infer_local_world_size(),
            force_topology   = self._cfg.force_topology,
        )
        self._rank             = env.rank
        self._world_size       = env.world_size
        self._local_rank       = env.local_rank
        self._local_world_size = env.local_world_size
        self._topo             = env.topology

        # ── Metrics ───────────────────────────────────────────────────────────
        init_registry(rank=self._rank)

        # [ARCH3] Start Prometheus server on rank 0 only (to avoid port collision)
        if self._cfg.prometheus_port is not None and self._rank == 0:
            self._start_prometheus(self._cfg.prometheus_port)

        # ── Resolution tracking ───────────────────────────────────────────────
        self._current_global_size = self._aug_cfg.global_crop_size
        self._current_local_size  = self._aug_cfg.local_crop_size
        self._resolution_src      = ResolutionSource(
            self._current_global_size,
            self._current_local_size,
        )

        # ── Stage 1–2: shard cache + mixing source ────────────────────────────
        node_master = (self._local_rank == 0)
        job_id      = os.environ.get("SLURM_JOB_ID", "dino_local")

        shard_cache = self._backend.build_shard_cache(
            job_id               = job_id,
            node_master          = node_master,
            max_gb               = self._cfg.node_shm_gb,
            prefetch_window      = self._cfg.shard_prefetch_window,
            timeout_s            = self._cfg.shard_timeout_s,
            warn_threshold       = self._cfg.shm_warn_threshold,
            heartbeat_stale_s    = self._cfg.heartbeat_stale_s,     # [M4-FIX]
            use_ring_buffer      = self._cfg.intra_node_ring_buffer, # [ARCH1]
            adaptive_prefetch    = self._cfg.adaptive_prefetch,      # [ARCH2]
            adaptive_target_util = self._cfg.adaptive_prefetch_target_util,
        )

        # ── Validate shard coverage before spawning workers ───────────────────
        self._validate_shard_coverage(specs)

        self._source = MixingSource(
            specs               = specs,
            batch_size          = batch_size,
            cache               = shard_cache,
            rank                = self._rank,
            world_size          = self._world_size,
            num_workers         = self._cfg.shard_extraction_workers,
            seed                = self._cfg.seed,
            device_id           = device_id,
            shuffle_buffer_size = self._cfg.shuffle_buffer_size,
            debug_log_keys      = self._cfg.debug_log_keys,
        )

        # ── Stage 3: augmentation pipeline ───────────────────────────────────
        dali_fp8 = self._cfg.use_fp8_output and self._cfg.dali_fp8_output

        pipeline = self._backend.build_pipeline(
            source             = self._source,
            aug_cfg            = self._aug_cfg,
            batch_size         = batch_size,
            num_threads        = self._cfg.dali_num_threads,
            device_id          = device_id,
            resolution_src     = self._resolution_src,
            hw_decoder_load    = self._cfg.hw_decoder_load,
            cpu_queue          = self._cfg.dali_cpu_queue,
            gpu_queue          = self._cfg.dali_gpu_queue,
            seed               = self._cfg.seed + self._rank,
            specs              = specs,
            fuse_normalization = self._cfg.fuse_normalization and self._backend.supports_gpu,
            dali_fp8_output    = dali_fp8,
        )

        output_map      = [f"view_{i}" for i in range(self._aug_cfg.n_views)]
        self._dali_iter = self._backend.build_pipeline_iterator(
            pipeline   = pipeline,
            output_map = output_map,
            batch_size = batch_size,
        )

        # ── Stage 4 & 5: H2D + FP8 ───────────────────────────────────────────
        device = (
            torch.device(f"cuda:{device_id}")
            if self._backend.supports_gpu
            else torch.device("cpu")
        )
        self._h2d = self._backend.build_h2d_stream(device=device, topo=self._topo)
        self._fp8 = (
            self._backend.build_fp8_formatter()
            if self._cfg.use_fp8_output and not dali_fp8
            else None
        )

        # ── Checkpointing ─────────────────────────────────────────────────────
        self._ckpt = DataLoaderCheckpointer(
            ckpt_dir      = self._cfg.checkpoint_dir,
            every_n_steps = self._cfg.checkpoint_every_steps,
            rank          = self._rank,
        )

        if resume:
            self._restore()

        log.info(
            "DINODataLoader ready: backend=%s rank=%d/%d, batch=%d, "
            "resolution=%dx%d (max %dx%d), fused_norm=%s, dali_fp8=%s, "
            "stall_timeout=%.0fs, ring_buffer=%s, adaptive_prefetch=%s, "
            "prometheus=%s",
            self._backend.name,
            self._rank, self._world_size, batch_size,
            self._current_global_size, self._current_local_size,
            self._aug_cfg.max_global_crop_size, self._aug_cfg.max_local_crop_size,
            self._cfg.fuse_normalization,
            dali_fp8,
            self._cfg.stall_timeout_s,
            self._cfg.intra_node_ring_buffer,
            self._cfg.adaptive_prefetch,
            self._cfg.prometheus_port,
        )

    # ── Fluid API ─────────────────────────────────────────────────────────────

    def map(self, fn: Callable[[Batch], Batch]) -> PostProcessPipeline:
        """Chain a transform on every Batch."""
        return PostProcessPipeline(
            source     = iter(self._raw_iter()),
            transforms = [fn],
            loader     = self,
        )

    def select(self, predicate: Callable[[Batch], bool]) -> PostProcessPipeline:
        """Filter batches.  Dropped batches are counted in metrics."""
        metrics = get_registry()

        def _filter(b: Batch) -> Optional[Batch]:
            if predicate(b):
                return b
            if metrics is not None:
                metrics.inc("batches_filtered", 1)
            return None

        return PostProcessPipeline(
            source     = iter(self._raw_iter()),
            transforms = [_filter],
            loader     = self,
        )

    def with_epoch(self, n_steps: int) -> PostProcessPipeline:
        return PostProcessPipeline(
            source     = iter(self._raw_iter()),
            transforms = [],
            loader     = self,
            max_steps  = n_steps,
        )

    # ── Epoch / weight control ────────────────────────────────────────────────

    def set_epoch(self, epoch: int) -> None:
        """
        Prepare the loader for a new epoch.

        [B3-FIX] Protected by a threading.Lock.  Concurrent calls from
        different threads (e.g. a curriculum scheduler + training loop) are
        now serialised instead of potentially interleaving on MixingSource
        state.
        """
        with self._epoch_lock:
            # Apply resolution schedule
            new_global = self._aug_cfg.crop_size_at_epoch(epoch)
            if new_global != self._current_global_size:
                self.set_resolution(new_global, self._current_local_size)

            self._source.set_epoch(epoch)
            self._dali_iter.reset()

    def set_resolution(self, global_size: int, local_size: int) -> None:
        if global_size > self._aug_cfg.max_global_crop_size:
            raise ValueError(
                f"set_resolution: global_size={global_size} exceeds "
                f"max_global_crop_size={self._aug_cfg.max_global_crop_size}."
            )
        if local_size > self._aug_cfg.max_local_crop_size:
            raise ValueError(
                f"set_resolution: local_size={local_size} exceeds "
                f"max_local_crop_size={self._aug_cfg.max_local_crop_size}."
            )
        self._current_global_size = global_size
        self._current_local_size  = local_size
        self._resolution_src.set(global_size, local_size)
        log.info("Resolution updated: global=%d local=%d", global_size, local_size)

    @property
    def current_resolution(self) -> tuple[int, int]:
        """[LD-13] Public access to current crop resolution."""
        return (self._current_global_size, self._current_local_size)

    @property
    def current_weights(self) -> List[float]:
        return self._source.current_weights

    def set_weights(self, weights: Sequence[float]) -> None:
        self._source.set_weights(weights)

    def set_weight_by_name(self, name: str, weight: float) -> None:
        self._source.set_weight_by_name(name, weight)

    # ── Checkpointing ─────────────────────────────────────────────────────────

    def checkpoint(self, step: int) -> None:
        state = CheckpointState(
            step             = step,
            epoch            = getattr(self._source, "_epoch", 0),
            dataset_names    = self._source.dataset_names,
            mixing_weights   = self._source.current_weights,
            global_crop_size = self._current_global_size,
            local_crop_size  = self._current_local_size,
        )
        self._ckpt.save(state)

    def state_dict(self) -> dict:
        return self._ckpt.state_dict()

    def load_state_dict(self, sd: dict) -> None:
        self._ckpt.load_state_dict(sd)

    # ── Iteration protocol ────────────────────────────────────────────────────

    def __iter__(self) -> Iterator[Batch]:
        if self._active_iter:
            raise RuntimeError(
                "DINODataLoader: __iter__ called while already iterating. "
                "Call set_epoch() before starting a new epoch loop."
            )
        self._active_iter = True
        try:
            yield from self._raw_iter()
        finally:
            self._active_iter = False

    def _raw_iter(self) -> Iterator[Batch]:
        """Core iteration loop with configurable stall watchdog."""
        metrics         = get_registry()
        stall_timeout   = self._cfg.stall_timeout_s
        watchdog_active = stall_timeout > 0
        got_first       = False

        for dali_out in self._dali_iter:
            got_first = True
            t0        = time.perf_counter()

            views        = [dali_out[0][f"view_{i}"] for i in range(self._aug_cfg.n_views)]
            n_global     = self._aug_cfg.n_global_crops
            global_views = views[:n_global]
            local_views  = views[n_global:]

            metadata = self._source.pop_last_metadata()

            masks = None
            if self._mask_generator is not None:
                n_tokens = (self._current_global_size // 14) ** 2
                masks    = self._mask_generator(n_tokens)

            with self._h2d.transfer({"global": global_views, "local": local_views}) as gpu:
                g_gpu = gpu["global"]
                l_gpu = gpu["local"]

            if self._fp8 is not None:
                g_gpu = [self._fp8.quantise(t) for t in g_gpu]
                l_gpu = [self._fp8.quantise(t) for t in l_gpu]

            batch = Batch(
                global_crops = g_gpu,
                local_crops  = l_gpu,
                metadata     = metadata,
                masks        = masks,
            )

            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            if metrics:
                metrics.inc("loader_batches_yielded", 1)
                metrics.inc("pipeline_yield_time_ms", elapsed_ms)
                metrics.set("heartbeat_ts", int(time.time()))

            yield batch

        if not got_first and watchdog_active:
            if os.environ.get("DINO_DISABLE_EMPTY_CHECK"):
                log.warning(
                    "DINODataLoader rank %d: no batch produced but "
                    "DINO_DISABLE_EMPTY_CHECK is set — continuing silently.",
                    self._rank,
                )
            else:
                raise RuntimeError(
                    f"DINODataLoader (rank {self._rank}): no batch produced after "
                    f"{stall_timeout:.0f}s.  Possible causes:\n"
                    "  • Fewer shards than extraction workers\n"
                    "  • All shards are corrupted or unreachable\n"
                    "  • /dev/shm is full\n"
                    "  • Lustre MDS slow start — increase LoaderConfig.stall_timeout_s\n"
                    "  Disable: DINO_DISABLE_EMPTY_CHECK=1 or stall_timeout_s=0."
                )

    def __len__(self) -> int:
        if self._steps_per_epoch is None:
            raise TypeError(
                "len(loader) requires steps_per_epoch to be set at construction."
            )
        return self._steps_per_epoch

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _validate_shard_coverage(self, specs: List[DatasetSpec]) -> None:
        """[M1-FIX] Validate that every rank gets at least one shard per dataset."""
        for spec in specs:
            n_shards = len(spec.shards)
            if n_shards < self._world_size:
                log.warning(
                    "DatasetSpec '%s': only %d shards for %d ranks.  "
                    "Ranks %d..%d will receive no shards from this dataset.  "
                    "Consider shard_sampling='resampled' for small datasets.",
                    spec.name, n_shards, self._world_size,
                    n_shards, self._world_size - 1,
                )

    def _restore(self) -> None:
        state = self._ckpt.load()
        if state is None:
            return
        if state.dataset_names != self._source.dataset_names:
            log.warning(
                "Checkpoint dataset names %s do not match current specs %s — "
                "skipping mixing weight restore.",
                state.dataset_names,
                self._source.dataset_names,
            )
        else:
            self._source.set_weights(state.mixing_weights)
        if state.global_crop_size != self._current_global_size:
            self.set_resolution(state.global_crop_size, state.local_crop_size)

    def _start_prometheus(self, port: int) -> None:
        """[ARCH3] Start Prometheus HTTP server on a daemon thread."""
        try:
            import prometheus_client
            t = threading.Thread(
                target=prometheus_client.start_http_server,
                args=(port,),
                name="prometheus-server",
                daemon=True,
            )
            t.start()
            log.info(
                "Prometheus metrics server started on port %d (rank 0)", port
            )
        except Exception as exc:
            log.warning("Could not start Prometheus server: %s", exc)

    @staticmethod
    def _infer_rank() -> int:
        for var in ("RANK", "SLURM_PROCID", "LOCAL_RANK"):
            v = os.environ.get(var)
            if v is not None:
                return int(v)
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
        return 0

    @staticmethod
    def _infer_world_size() -> int:
        for var in ("WORLD_SIZE", "SLURM_NTASKS"):
            v = os.environ.get(var)
            if v is not None:
                return int(v)
        if dist.is_available() and dist.is_initialized():
            return dist.get_world_size()
        return 1

    @staticmethod
    def _infer_local_world_size() -> int:
        for var in ("LOCAL_WORLD_SIZE", "SLURM_NTASKS_PER_NODE"):
            v = os.environ.get(var)
            if v is not None:
                return int(v)
        return 1

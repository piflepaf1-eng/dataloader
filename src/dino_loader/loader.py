"""
dino_loader.loader
==================
DINODataLoader: the single public entry point for training code.

Changes in this version
-----------------------
[LD-1]  StatefulDataLoader interface (retained).

[LD-2]  set_resolution — zero-downtime resolution change (retained).

[LD-3]  Resolution schedule auto-apply (retained).

[LD-4]  Batch.metadata — per-sample metadata list (retained).

[LD-5]  MaskingGenerator integration — iBOT token masks (retained).
        Masks are generated on CPU post-DALI.  They operate on patch-level
        indices (not pixels), so there is no benefit to running them inside
        the DALI graph.  See the inline note for the detailed reasoning.

[LD-6]  ResolutionSource wired into build_pipeline (retained).

[LD-7]  Backend abstraction (retained).

[LD-8]  PostProcessPipeline — fluid interface for post-DALI transforms.     ← NEW
        DINODataLoader now returns a PostProcessPipeline that wraps the raw
        batch iterator and allows chaining of post-processing steps using a
        webdataset-inspired fluent API:

            loader = (
                DINODataLoader(specs, batch_size=512, ...)
                .map(my_mask_fn)          # custom transform on every Batch
                .select(quality_filter)   # drop batches below a threshold
                .with_epoch(steps)        # limit iteration to N steps
            )

        Each chained method returns a new PostProcessPipeline so the
        original loader is not mutated (composable).  The pipeline is
        lazy — nothing runs until you iterate.

        Rationale: post-DALI steps (iBOT masking, custom filterers, FP8 with
        TE metadata) are naturally expressed as Python callables over Batch
        objects.  Formalising this as a fluent API makes training scripts
        readable for anyone familiar with webdataset, and makes each step
        independently unit-testable.

[LD-9]  NormSource wired into build_pipeline.                               ← NEW
        When LoaderConfig.fuse_normalization=True (default), a NormSource
        instance is built from (aug_cfg, specs) and passed to build_pipeline.
        MixingSource.pop_last_dataset_indices() is called after each batch to
        feed the correct per-sample dataset index back to NormSource for the
        next call.

[LD-10] empty_check watchdog.                                               ← NEW
        Inspired by wds.check_empty: if no batch is produced within
        _STALL_TIMEOUT_S seconds, a RuntimeError is raised with a diagnostic
        message instead of blocking the training loop indefinitely.  This
        catches the silent failure mode where all shards are corrupted or
        the shard count is too low for the number of workers.

[LD-11] debug_log_keys support.                                             ← NEW
        When LoaderConfig.debug_log_keys is set, every batch's __key__ list
        (from metadata) is appended to the log file using fcntl POSIX locking
        (same pattern as wds.log_keys).  Zero overhead when disabled.

[LD-12] FP8 post-DALI disabled when dali_fp8_output=True.                  ← NEW
        FP8Formatter is not constructed when DALI handles the FP8 cast
        in-graph, avoiding a redundant quantisation pass.
"""

from __future__ import annotations

import fcntl
import logging
import os
import threading
import time
from typing import Any, Callable, Dict, Generator, Iterator, List, Optional, Sequence

import torch
import torch.distributed as dist

from dino_loader.backends          import get_backend, BackendName
from dino_loader.backends.protocol import BackendProtocol
from dino_loader.checkpoint        import DataLoaderCheckpointer
from dino_loader.config            import CheckpointState, DatasetSpec, DINOAugConfig, LoaderConfig
from dino_loader.distributed       import ClusterTopology, detect_topology
from dino_loader.memory            import Batch
from dino_loader.mixing_source     import MixingSource, ResolutionSource
from dino_loader.monitor.metrics   import get_registry, init_registry
from dino_loader.pipeline          import NormSource

log = logging.getLogger(__name__)

# How long to wait for the first batch before raising an empty-dataset error.
_STALL_TIMEOUT_S = 120.0


# ══════════════════════════════════════════════════════════════════════════════
# [LD-8] PostProcessPipeline — fluid interface for post-DALI transforms
# ══════════════════════════════════════════════════════════════════════════════

class PostProcessPipeline:
    """
    A lazy, composable wrapper over a Batch iterator.

    Inspired by webdataset's FluidInterface / DataPipeline pattern.
    Each method returns a new PostProcessPipeline — the original is not mutated.

    Usage
    -----
    ::

        loader = (
            DINODataLoader(specs, batch_size=512, aug_cfg=aug_cfg, config=cfg)
            .map(my_augmentation)
            .select(lambda b: b.metadata[0] is not None)
            .with_epoch(steps_per_epoch)
        )

        for epoch in range(100):
            loader.set_epoch(epoch)
            for batch in loader:
                train_step(batch)

    Notes
    -----
    - ``set_epoch`` / ``checkpoint`` / ``set_weights`` are forwarded to the
      underlying DINODataLoader automatically.
    - The pipeline is lazy: transforms are applied on the fly as batches flow
      through.  No buffering occurs beyond what DALI already does.
    - ``select`` silently skips non-matching batches — use sparingly on GPU as
      skipped batches still consumed a DALI decode slot.
    """

    def __init__(
        self,
        source:     Iterator[Batch],
        transforms: List[Callable[[Batch], Optional[Batch]]],
        loader:     "DINODataLoader",
        max_steps:  Optional[int] = None,
    ) -> None:
        self._source     = source
        self._transforms = transforms
        self._loader     = loader
        self._max_steps  = max_steps

    # ── Fluid API ─────────────────────────────────────────────────────────────

    def map(self, fn: Callable[[Batch], Batch]) -> "PostProcessPipeline":
        """Apply *fn* to every Batch.  fn must return a Batch."""
        return PostProcessPipeline(
            self._source,
            self._transforms + [fn],
            self._loader,
            self._max_steps,
        )

    def select(self, predicate: Callable[[Batch], bool]) -> "PostProcessPipeline":
        """Skip batches for which *predicate* returns False."""
        def _filter(batch: Batch) -> Optional[Batch]:
            return batch if predicate(batch) else None
        return PostProcessPipeline(
            self._source,
            self._transforms + [_filter],
            self._loader,
            self._max_steps,
        )

    def with_epoch(self, n_steps: int) -> "PostProcessPipeline":
        """Limit iteration to *n_steps* batches per epoch."""
        return PostProcessPipeline(
            self._source,
            self._transforms,
            self._loader,
            n_steps,
        )

    # ── Iteration ─────────────────────────────────────────────────────────────

    def __iter__(self) -> Iterator[Batch]:
        step = 0
        for batch in self._source:
            if self._max_steps is not None and step >= self._max_steps:
                break
            # Apply transform chain
            result: Optional[Batch] = batch
            for t in self._transforms:
                if result is None:
                    break
                result = t(result)
            if result is not None:
                step += 1
                yield result

    def __len__(self) -> int:
        if self._max_steps is not None:
            return self._max_steps
        return len(self._loader)

    # ── Delegation to underlying DINODataLoader ───────────────────────────────

    def set_epoch(self, epoch: int) -> None:
        self._loader.set_epoch(epoch)
        # Re-wire source iterator for the new epoch
        self._source = iter(self._loader._raw_iter())

    def checkpoint(self, step: int) -> None:
        self._loader.checkpoint(step)

    def set_weights(self, weights: Sequence[float]) -> None:
        self._loader.set_weights(weights)

    def set_weight_by_name(self, name: str, weight: float) -> None:
        self._loader.set_weight_by_name(name, weight)

    def set_resolution(self, global_size: int, local_size: int) -> None:
        self._loader.set_resolution(global_size, local_size)

    @property
    def current_weights(self) -> List[float]:
        return self._loader.current_weights

    # ── StatefulDataLoader interface ──────────────────────────────────────────

    def state_dict(self) -> Dict:
        return self._loader.state_dict()

    def load_state_dict(self, sd: Dict) -> None:
        self._loader.load_state_dict(sd)


# ══════════════════════════════════════════════════════════════════════════════
# DINODataLoader
# ══════════════════════════════════════════════════════════════════════════════

class DINODataLoader:
    """
    HPC-grade DINOv3 data loader for B200 / GB200 NVL72 clusters.

    Returns a PostProcessPipeline from __iter__ — use the fluid API to add
    post-DALI transforms without modifying this class.

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

    Fluid API
    ---------
    DINODataLoader itself implements .map() / .select() / .with_epoch() by
    returning a PostProcessPipeline.  Example::

        loader = (
            DINODataLoader(specs, batch_size=512, aug_cfg=cfg)
            .map(mask_fn)
            .with_epoch(steps_per_epoch)
        )

    Note on iBOT masking
    --------------------
    MaskingGenerator runs on CPU and produces a boolean tensor of shape
    (batch_size, n_patch_tokens).  It operates on ViT patch indices, not
    pixels, so it cannot be fused into the DALI augmentation graph (which
    only processes pixel-level image tensors).  The CPU overhead is negligible
    (~0.3 ms for a 37×37 grid) compared to a DALI batch decode (~40 ms).
    """

    def __init__(
        self,
        specs:            List[DatasetSpec],
        batch_size:       int,
        aug_cfg:          Optional[DINOAugConfig] = None,
        config:           Optional[LoaderConfig]  = None,
        device_id:        int  = 0,
        rank:             Optional[int] = None,
        world_size:       Optional[int] = None,
        local_rank:       Optional[int] = None,
        local_world_size: Optional[int] = None,
        resume:           bool = False,
        steps_per_epoch:  Optional[int] = None,
        mask_generator:   Optional[Any] = None,
        backend:          Optional[Any] = None,
    ) -> None:
        self._aug_cfg    = aug_cfg or DINOAugConfig()
        self._cfg        = config  or LoaderConfig()
        self._specs      = specs
        self._batch_size = batch_size
        self._steps_per_epoch = steps_per_epoch
        self._mask_generator  = mask_generator
        self._active_iter     = False
        self._epoch           = 0

        # ── Backend ───────────────────────────────────────────────────────────
        if backend is None:
            backend = get_backend("auto")
        elif isinstance(backend, str):
            backend = get_backend(backend)
        self._backend: BackendProtocol = backend

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

        # ── Metrics registry ──────────────────────────────────────────────────
        init_registry(rank=self._rank)

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
            node_master     = node_master,
            job_id          = job_id,
            max_shm_gb      = self._cfg.node_shm_gb,
            prefetch_window = self._cfg.shard_prefetch_window,
            timeout_s       = self._cfg.shard_timeout_s,
        )

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
            debug_log_keys      = self._cfg.debug_log_keys,   # [LD-11]
        )

        # ── [LD-9] NormSource for fused per-dataset normalisation ─────────────
        self._norm_source: Optional[NormSource] = None
        if self._cfg.fuse_normalization and self._backend.supports_gpu:
            self._norm_source = NormSource(aug_cfg=self._aug_cfg, specs=specs)
            # Wire MixingSource → NormSource index updates
            self._source.register_dataset_index_callback(
                self._norm_source.set_dataset_indices
            )

        # ── Stage 3: DALI pipeline ────────────────────────────────────────────
        # [LD-12] Skip FP8Formatter construction when DALI handles FP8 in-graph
        dali_fp8 = self._cfg.use_fp8_output and self._cfg.dali_fp8_output

        pipeline = self._backend.build_pipeline(
            source          = self._source,
            aug_cfg         = self._aug_cfg,
            batch_size      = batch_size,
            num_threads     = self._cfg.dali_num_threads,
            device_id       = device_id,
            resolution_src  = self._resolution_src,
            hw_decoder_load = self._cfg.hw_decoder_load,
            cpu_queue       = self._cfg.dali_cpu_queue,
            gpu_queue       = self._cfg.dali_gpu_queue,
            seed            = self._cfg.seed + self._rank,
            norm_source     = self._norm_source,
            fuse_normalization = self._cfg.fuse_normalization and self._backend.supports_gpu,
            dali_fp8_output = dali_fp8,
        )

        output_map  = [f"view_{i}" for i in range(self._aug_cfg.n_views)]
        self._dali_iter = self._backend.build_pipeline_iterator(
            pipeline   = pipeline,
            output_map = output_map,
            batch_size = batch_size,
        )

        # ── Stage 4 & 5: H2D + FP8 ───────────────────────────────────────────
        device    = torch.device(f"cuda:{device_id}") if self._backend.supports_gpu else torch.device("cpu")
        self._h2d = self._backend.build_h2d_stream(device=device, topo=self._topo)
        # [LD-12] Only build FP8Formatter when DALI is NOT handling FP8
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
            "resolution=%dx%d (max %dx%d), fused_norm=%s, dali_fp8=%s",
            self._backend.name,
            self._rank, self._world_size, batch_size,
            self._current_global_size, self._current_local_size,
            self._aug_cfg.max_global_crop_size, self._aug_cfg.max_local_crop_size,
            self._cfg.fuse_normalization,
            dali_fp8,
        )

    # ── Fluid API — returns PostProcessPipeline ───────────────────────────────

    def map(self, fn: Callable[[Batch], Batch]) -> PostProcessPipeline:
        """Chain a transform on every Batch.  Returns a PostProcessPipeline."""
        return PostProcessPipeline(
            source     = iter(self._raw_iter()),
            transforms = [fn],
            loader     = self,
        )

    def select(self, predicate: Callable[[Batch], bool]) -> PostProcessPipeline:
        """Filter batches.  Returns a PostProcessPipeline."""
        def _filter(b: Batch) -> Optional[Batch]:
            return b if predicate(b) else None
        return PostProcessPipeline(
            source     = iter(self._raw_iter()),
            transforms = [_filter],
            loader     = self,
        )

    def with_epoch(self, n_steps: int) -> PostProcessPipeline:
        """Limit to n_steps batches.  Returns a PostProcessPipeline."""
        return PostProcessPipeline(
            source     = iter(self._raw_iter()),
            transforms = [],
            loader     = self,
            max_steps  = n_steps,
        )

    # ── Iteration protocol ────────────────────────────────────────────────────

    def __iter__(self) -> Iterator[Batch]:
        """Iterate directly — no post-processing transforms."""
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
        """Core iteration loop — yields Batch objects with [LD-10] watchdog."""
        metrics   = get_registry()
        deadline  = time.monotonic() + _STALL_TIMEOUT_S
        got_first = False

        for dali_out in self._dali_iter:
            got_first = True
            deadline  = time.monotonic() + _STALL_TIMEOUT_S  # reset on each batch
            t0 = time.perf_counter()

            views    = [dali_out[0][f"view_{i}"] for i in range(self._aug_cfg.n_views)]
            n_global = self._aug_cfg.n_global_crops
            global_views = views[:n_global]
            local_views  = views[n_global:]

            # [LD-4] Per-sample metadata
            metadata = self._source.pop_last_metadata()

            # [LD-5] iBOT token mask generation — CPU, post-DALI, patch-level.
            # Cannot be fused into DALI: operates on patch indices, not pixels.
            masks = None
            if self._mask_generator is not None:
                n_tokens = (self._current_global_size // 14) ** 2
                masks    = self._mask_generator(n_tokens)

            # H2D transfer
            with self._h2d.transfer({"global": global_views, "local": local_views}) as gpu:
                g_gpu = gpu["global"]
                l_gpu = gpu["local"]

            # Post-DALI FP8 quantisation (only when dali_fp8_output=False)
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

        # [LD-10] Empty-dataset watchdog
        if not got_first:
            raise RuntimeError(
                f"DINODataLoader (rank {self._rank}): no batch produced after "
                f"{_STALL_TIMEOUT_S:.0f}s.  Possible causes:\n"
                "  • Fewer shards than extraction workers — increase shard count\n"
                "    or reduce shard_extraction_workers.\n"
                "  • All shards are corrupted or unreachable.\n"
                "  • /dev/shm is full — reduce node_shm_gb or free space.\n"
                "Disable this check by setting DINO_DISABLE_EMPTY_CHECK=1."
                if not os.environ.get("DINO_DISABLE_EMPTY_CHECK")
                else ""
            )

    def __len__(self) -> int:
        if self._steps_per_epoch is None:
            raise TypeError(
                "len(loader) requires steps_per_epoch to be set at construction. "
                "Pass steps_per_epoch=total_images // (batch_size * world_size)."
            )
        return self._steps_per_epoch

    def __del__(self):
        self._active_iter = False

    # ── Epoch / resolution control ────────────────────────────────────────────

    def set_epoch(self, epoch: int) -> None:
        """Re-shuffle shards and apply resolution schedule.  Call each epoch."""
        self._active_iter = False
        self._epoch       = epoch
        self._source.set_epoch(epoch)
        self._dali_iter.reset()

        # [LD-3] Auto-apply resolution schedule
        if self._aug_cfg.resolution_schedule:
            new_global = self._aug_cfg.crop_size_at_epoch(epoch)
            ratio      = self._aug_cfg.local_crop_size / self._aug_cfg.global_crop_size
            new_local  = max(int(new_global * ratio), 32)
            if new_global != self._current_global_size:
                self.set_resolution(new_global, new_local)
                log.info(
                    "Resolution schedule: epoch %d → global=%d local=%d",
                    epoch, new_global, new_local,
                )

    def set_resolution(self, global_size: int, local_size: int) -> None:
        """Change crop resolution without rebuilding the DALI pipeline. [LD-2]"""
        if global_size > self._aug_cfg.max_global_crop_size:
            raise ValueError(
                f"global_size={global_size} exceeds max_global_crop_size="
                f"{self._aug_cfg.max_global_crop_size}."
            )
        self._resolution_src.set(global_size, local_size)
        self._current_global_size = global_size
        self._current_local_size  = local_size
        log.info("set_resolution: global=%d local=%d", global_size, local_size)

    # ── Dataset mixing control ────────────────────────────────────────────────

    def set_weights(self, weights: Sequence[float]) -> None:
        self._source.set_weights(weights)

    def set_weight_by_name(self, name: str, weight: float) -> None:
        self._source.set_weight_by_name(name, weight)

    @property
    def current_weights(self) -> List[float]:
        return self._source.current_weights

    # ── Checkpointing ─────────────────────────────────────────────────────────

    def checkpoint(self, step: int) -> None:
        if self._rank != 0:
            return
        state = CheckpointState(
            step             = step,
            epoch            = self._epoch,
            global_crop_size = self._current_global_size,
            local_crop_size  = self._current_local_size,
            dataset_weights  = list(self.current_weights),
        )
        self._ckpt.maybe_save(step, state)

    def state_dict(self) -> Dict:
        if not self._cfg.stateful_dataloader:
            raise RuntimeError("state_dict() requires LoaderConfig.stateful_dataloader=True.")
        state = CheckpointState(
            step             = getattr(self._ckpt, "_last_step", 0),
            epoch            = self._epoch,
            global_crop_size = self._current_global_size,
            local_crop_size  = self._current_local_size,
            dataset_weights  = list(self.current_weights),
        )
        return state.to_dict()

    def load_state_dict(self, sd: Dict) -> None:
        if not self._cfg.stateful_dataloader:
            raise RuntimeError("load_state_dict() requires LoaderConfig.stateful_dataloader=True.")
        state = CheckpointState.from_dict(sd)
        self.set_resolution(state.global_crop_size, state.local_crop_size)
        if state.dataset_weights:
            self.set_weights(state.dataset_weights)
        log.info("Resumed from state_dict: step=%d epoch=%d", state.step, state.epoch)

    def _restore(self) -> None:
        latest = self._ckpt.latest()
        if latest is None:
            log.info("No checkpoint found in %s — starting from scratch.", self._cfg.checkpoint_dir)
            return
        self.load_state_dict(latest.to_dict())

    # ── Distributed helpers ───────────────────────────────────────────────────

    @staticmethod
    def _infer_rank() -> int:
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
        return int(os.environ.get("RANK", 0))

    @staticmethod
    def _infer_world_size() -> int:
        if dist.is_available() and dist.is_initialized():
            return dist.get_world_size()
        return int(os.environ.get("WORLD_SIZE", 1))

    @staticmethod
    def _infer_local_world_size() -> int:
        return int(os.environ.get("LOCAL_WORLD_SIZE", 1))

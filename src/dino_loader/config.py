"""
dino_loader.config
==================
All loader-level configuration lives here.  No logic — pure dataclasses.
Serialised to / from JSON for checkpointing (no pickle fragility).

Changes in this version
-----------------------
[CFG-S1]  DatasetSpec.shard_sampling — explicit sampling mode (retained).
[CFG-S2]  DatasetSpec.prob — alias for weight= (retained).
[CFG-S3]  LoaderConfig.debug_log_keys — per-sample key logging (retained).
[CFG-S4]  LoaderConfig.fuse_normalization — DALI-fused per-dataset norm (retained).
[CFG-S5]  LoaderConfig.dali_fp8_output — in-graph FP8 cast (retained).
[CFG-S6]  LoaderConfig.stall_timeout_s — configurable watchdog timeout (retained).

New in this version
-------------------
[CFG-B4]  transformer-engine is now an optional dependency.               ← FIX B4
          ``use_fp8_output`` defaults to False (unchanged), so no behaviour
          change for existing configs.  ``__post_init__`` now raises a clear
          ``ImportError``-based ValueError if FP8 is requested but TE is not
          installed, instead of crashing at runtime inside FP8Formatter.

[CFG-M4]  ``heartbeat_stale_s`` added to LoaderConfig.                    ← FIX M4
          Previously ``_HB_STALE_S = 60.0`` was a module-level constant in
          shard_cache.py, too short for clusters with busy nodes.  This field
          makes it tunable without touching internal code.  Default raised to
          300 s (5 min) — safe for all known SLURM configurations.

[CFG-ARCH1] ``intra_node_ring_buffer`` — opt-in SharedMemory shard broadcast. ← NEW
          Architectural improvement #1.
          When True, rank 0 publishes shard data into POSIX SharedMemory
          segments; all local ranks read from the single segment via
          zero-copy memoryview slices instead of opening individual mmaps.
          On NVL72 (72 ranks/node) this reduces mmap syscall count by ~72×.
          Default: False — the battle-tested mmap pool path (PERF-2) is used
          until this feature is validated on your cluster.

[CFG-ARCH2] ``adaptive_prefetch`` — opt-in PID-controlled prefetch window. ← NEW
          Architectural improvement #2.
          When True, a PID controller adjusts ``shard_prefetch_window``
          dynamically based on the live /dev/shm utilisation metric, targeting
          ``adaptive_prefetch_target_util`` (default 0.75).  This uses all
          available DRAM without ever exceeding the budget.
          Default: False — static ``shard_prefetch_window`` is used.

[CFG-ARCH3] ``prometheus_port`` — opt-in Prometheus metrics endpoint.     ← NEW
          Architectural improvement #3.
          When set to a port number (e.g. 9100), a background thread starts a
          prometheus_client HTTP server on that port, exposing all MetricField
          values as Gauges/Counters scrappable by Prometheus / Grafana.
          Default: None (disabled).  Requires ``prometheus_client`` installed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple



# ── Augmentation ──────────────────────────────────────────────────────────────

@dataclass
class DINOAugConfig:
    """
    DINOv2/v3 multi-crop augmentation configuration.

    Parameters
    ----------
    global_crop_size / local_crop_size
        Initial crop resolutions in pixels.  Can be changed at runtime via
        DINODataLoader.set_resolution() without rebuilding the DALI pipeline.
    n_global_crops / n_local_crops
        Number of global (large) and local (small) crops per image.
    global_crops_scale / local_crops_scale
        RandomResizedCrop scale ranges for global/local views.
    preserve_aspect_ratio
        True → resize shorter side then centre-crop (avoids distortion).
        False → direct square resize (legacy, faster by one DALI op).
    resolution_schedule
        List of (epoch, global_crop_size) pairs for progressive resolution.
        set_epoch() applies these automatically — no DALI rebuild required.
    max_global_crop_size / max_local_crop_size
        nvjpeg pre-allocation ceiling.  Must be ≥ the largest size in the
        resolution schedule.  Default = initial crop size.
    mean / std
        Global normalisation statistics (ImageNet defaults).  Per-dataset
        overrides can be set via DatasetSpec.mean / DatasetSpec.std.
    """

    # Crop geometry
    global_crop_size:      int   = 224
    local_crop_size:       int   = 96
    n_global_crops:        int   = 2
    n_local_crops:         int   = 8
    global_crops_scale:    Tuple[float, float] = (0.32, 1.0)
    local_crops_scale:     Tuple[float, float] = (0.05, 0.32)

    # Augmentation knobs (DINOv2 defaults)
    blur_prob_global1:  float = 1.0
    blur_prob_global2:  float = 0.1
    blur_prob_local:    float = 0.5
    solarize_prob:      float = 0.2
    color_jitter_prob:  float = 0.8
    grayscale_prob:     float = 0.2

    # Geometry
    preserve_aspect_ratio: bool = True

    # Resolution schedule
    resolution_schedule:    List[Tuple[int, int]] = field(default_factory=list)
    max_global_crop_size:   int  = 0   # 0 → set to global_crop_size in __post_init__
    max_local_crop_size:    int  = 0   # 0 → set to local_crop_size in __post_init__

    # Normalisation (ImageNet)
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std:  Tuple[float, float, float] = (0.229, 0.224, 0.225)

    def __post_init__(self) -> None:
        if self.max_global_crop_size == 0:
            self.max_global_crop_size = self.global_crop_size
        if self.max_local_crop_size == 0:
            self.max_local_crop_size = self.local_crop_size

        if self.resolution_schedule:
            # Sort ascending by epoch
            self.resolution_schedule = sorted(self.resolution_schedule, key=lambda x: x[0])
            for epoch, size in self.resolution_schedule:
                if epoch < 0:
                    raise ValueError(
                        f"DINOAugConfig: resolution_schedule epochs must be ≥ 0, "
                        f"got epoch={epoch}."
                    )

    @property
    def n_views(self) -> int:
        return self.n_global_crops + self.n_local_crops

    def crop_size_at_epoch(self, epoch: int) -> int:
        """Return the global crop size dictated by the resolution schedule."""
        if not self.resolution_schedule:
            return self.global_crop_size
        size = self.global_crop_size
        for sched_epoch, sched_size in self.resolution_schedule:
            if epoch >= sched_epoch:
                size = sched_size
        return size


# ── Loader configuration ──────────────────────────────────────────────────────

@dataclass
class LoaderConfig:
    """
    All runtime knobs for DINODataLoader.

    I/O
    ---
    node_shm_gb
        /dev/shm budget per node (GB).  ~50% of node RAM is a good starting
        point; NVL72 nodes have 576 GB RAM so 256 GB is safe.
    shard_prefetch_window
        Max concurrent Lustre → /dev/shm downloads (node master only).
        When adaptive_prefetch=True, this becomes the *maximum* window and
        the controller adjusts downward to protect the /dev/shm budget.
    shard_timeout_s
        Max seconds a non-master rank waits for a shard to appear in /dev/shm.
    shard_extraction_workers
        Thread-pool workers for tar → JPEG extraction per rank.

    heartbeat_stale_s                                             [CFG-M4]
        Seconds of no heartbeat refresh before a /dev/shm directory is
        considered orphaned.  Default 300 s (5 min) is safe for busy nodes.
        The previous hardcoded value of 60 s was too aggressive for clusters
        with heavy memory pressure that could pause the heartbeat daemon.

    DALI
    ----
    dali_cpu_queue / dali_gpu_queue
        DALI pipeline prefetch queue depths.
    dali_num_threads
        DALI CPU worker threads for pre-decode operations.
    hw_decoder_load
        Fraction of JPEG decodes routed to nvjpeg HW ASIC (0–1).

    Data
    ----
    shuffle_buffer_size
        In-memory sample reservoir depth per ShardIterator.

    Output precision
    ----------------
    use_fp8_output
        Quantise output tensors to FP8 E4M3.  Requires transformer-engine.
        [CFG-B4] A clear error is raised at construction if TE is not
        installed, rather than crashing inside FP8Formatter at first batch.
    dali_fp8_output
        Fuse FP8 cast into DALI graph (requires DALI ≥ 1.36).
        Mutually exclusive with TE metadata — see pipeline.py [PL-5].
    fuse_normalization
        Fuse per-dataset normalisation into the DALI kernel (see [CFG-S4]).
    output_dtype
        Intermediate tensor dtype before optional FP8 cast.

    Architecture options (opt-in experimental features)
    ---------------------------------------------------
    intra_node_ring_buffer                                        [CFG-ARCH1]
        Enable SharedMemoryRingBuffer for intra-node shard broadcast.
        Reduces mmap syscall overhead on nodes with many ranks (NVL72: ~72×).
        Default: False.  Enable only after validating on your cluster.

    adaptive_prefetch                                             [CFG-ARCH2]
        Enable PID-controlled adaptive prefetch window.
        Automatically tunes shard_prefetch_window based on /dev/shm utilisation.
        Default: False.
    adaptive_prefetch_target_util
        Target /dev/shm utilisation fraction for the adaptive controller.
        Default: 0.75.

    prometheus_port                                               [CFG-ARCH3]
        If set, start a prometheus_client HTTP server on this port.
        Exposes all MetricField values as Prometheus Gauges/Counters.
        Default: None (disabled).  Requires: pip install prometheus-client.

    Watchdog / checkpointing / misc
    --------------------------------
    stall_timeout_s
        Seconds to wait for the first batch.  0 = disabled.
    checkpoint_dir / checkpoint_every_steps
        Checkpoint location and frequency (rank 0 only).
    stateful_dataloader
        Enable state_dict() / load_state_dict() interface.
    force_topology
        Override topology detection: "nvl72" | "pcie" | None (auto).
    seed
        Base random seed.
    debug_log_keys
        Path to per-sample key audit log (disable in production).
    shm_warn_threshold
        /dev/shm utilisation fraction that triggers a warning log.
    """

    # I/O
    node_shm_gb:              float = 128.0
    shard_prefetch_window:    int   = 64
    shard_timeout_s:          float = 300.0
    shard_extraction_workers: int   = 4

    # Heartbeat stale threshold                                   [CFG-M4]
    heartbeat_stale_s:        float = 300.0

    # DALI
    dali_cpu_queue:           int   = 8
    dali_gpu_queue:           int   = 6
    dali_num_threads:         int   = 8
    hw_decoder_load:          float = 0.90

    # Data
    shuffle_buffer_size:      int   = 512

    # Output precision
    use_fp8_output:           bool  = False
    dali_fp8_output:          bool  = False
    fuse_normalization:       bool  = True
    output_dtype:             str   = "bf16"

    # StatefulDataLoader
    stateful_dataloader:      bool  = True
    checkpoint_dir:           str   = "/tmp/dino_loader_ckpt"
    checkpoint_every_steps:   int   = 500

    # Cluster
    force_topology:           Optional[str] = None
    seed:                     int   = 0

    # Debug
    debug_log_keys:           Optional[str] = None

    # Watchdog                                                     [CFG-S6]
    stall_timeout_s:          float = 600.0

    # SHM monitoring
    shm_warn_threshold:       float = 0.90

    # ── Architectural options (opt-in) ────────────────────────────────────────

    # Arch #1 — intra-node SharedMemory ring buffer               [CFG-ARCH1]
    intra_node_ring_buffer:   bool  = False

    # Arch #2 — adaptive prefetch window PID controller           [CFG-ARCH2]
    adaptive_prefetch:              bool  = False
    adaptive_prefetch_target_util:  float = 0.75

    # Arch #3 — Prometheus metrics endpoint                        [CFG-ARCH3]
    prometheus_port:          Optional[int] = None

    def __post_init__(self) -> None:
        # [CFG-B4] Validate FP8 availability at construction time.
        if self.use_fp8_output:
            try:
                import transformer_engine.pytorch  # noqa: F401
            except ImportError:
                raise ValueError(
                    "LoaderConfig: use_fp8_output=True requires transformer-engine. "
                    "Install it with: pip install transformer-engine~=2.12\n"
                    "Or set use_fp8_output=False to use BF16 output."
                )

        if self.dali_fp8_output and not self.use_fp8_output:
            raise ValueError(
                "LoaderConfig: dali_fp8_output=True requires use_fp8_output=True."
            )
        if self.output_dtype not in ("bf16", "fp32"):
            raise ValueError(
                f"LoaderConfig: output_dtype must be 'bf16' or 'fp32', "
                f"got {self.output_dtype!r}."
            )
        if not (0.0 <= self.hw_decoder_load <= 1.0):
            raise ValueError(
                f"LoaderConfig: hw_decoder_load must be in [0.0, 1.0], "
                f"got {self.hw_decoder_load}."
            )
        if self.stall_timeout_s < 0:
            raise ValueError(
                f"LoaderConfig: stall_timeout_s must be ≥ 0 "
                f"(0 = disabled), got {self.stall_timeout_s}."
            )
        if not (0.0 <= self.shm_warn_threshold <= 1.0):
            raise ValueError(
                f"LoaderConfig: shm_warn_threshold must be in [0.0, 1.0], "
                f"got {self.shm_warn_threshold}."
            )
        if self.heartbeat_stale_s <= 0:
            raise ValueError(
                f"LoaderConfig: heartbeat_stale_s must be > 0, "
                f"got {self.heartbeat_stale_s}."
            )
        if not (0.0 < self.adaptive_prefetch_target_util <= 1.0):
            raise ValueError(
                f"LoaderConfig: adaptive_prefetch_target_util must be in (0, 1], "
                f"got {self.adaptive_prefetch_target_util}."
            )
        if self.prometheus_port is not None:
            if not (1 <= self.prometheus_port <= 65535):
                raise ValueError(
                    f"LoaderConfig: prometheus_port must be in [1, 65535], "
                    f"got {self.prometheus_port}."
                )
            try:
                import prometheus_client  # noqa: F401
            except ImportError:
                raise ValueError(
                    f"LoaderConfig: prometheus_port={self.prometheus_port} requires "
                    "prometheus_client.  Install with: pip install prometheus-client"
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

    step:             int
    epoch:            int
    dataset_names:    List[str]
    mixing_weights:   List[float]
    global_crop_size: int = 224
    local_crop_size:  int = 96

    def save(self, path: Path) -> None:
        """Write atomically via a .tmp file with SHA-256 integrity check."""
        import hashlib
        payload = asdict(self)
        payload_json = json.dumps(payload, indent=2)
        checksum = hashlib.sha256(payload_json.encode()).hexdigest()
        envelope = {"payload": payload, "sha256": checksum}
        tmp = path.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(envelope, indent=2))
            tmp.rename(path)
        except Exception:
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass
            raise

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "CheckpointState":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def load(cls, path: Path) -> "CheckpointState":
        import hashlib
        raw = json.loads(path.read_text())

        # Support both new envelope format and legacy flat format.
        if "payload" in raw and "sha256" in raw:
            payload_json = json.dumps(raw["payload"], indent=2)
            expected = hashlib.sha256(payload_json.encode()).hexdigest()
            if raw["sha256"] != expected:
                raise ValueError(
                    f"Checkpoint {path} failed integrity check: "
                    f"stored sha256={raw['sha256']!r}, computed={expected!r}.  "
                    "File may be corrupt or truncated."
                )
            data = raw["payload"]
        else:
            # Legacy flat format — no checksum available.
            data = raw

        # Backward compat: older checkpoints may lack crop size fields.
        data.setdefault("global_crop_size", 224)
        data.setdefault("local_crop_size",  96)
        return cls(**{
            k: v for k, v in data.items()
            if k in cls.__dataclass_fields__
        })

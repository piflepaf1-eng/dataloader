"""dino_loader
===========
HPC-grade DINOv3 DALI data pipeline for B200 / GB200 NVL72 clusters.

Public API
----------
    from dino_loader import (
        DatasetSpec, DINOAugConfig, LoaderConfig, NormStats,
        DINODataLoader, Batch,
    )
    from dino_loader.pipeline_graph import wrap_loader

    loader = DINODataLoader(specs, batch_size=512, config=LoaderConfig())

    # Compose post-DALI transforms in a stateful, checkpointable graph:
    pipeline = (
        wrap_loader(loader)
        .map(apply_ibot_masks)          # fn(Batch) → Batch
        .select(quality_ok)             # predicate(Batch) → bool
        .with_epoch(steps_per_epoch)    # limit steps per epoch
    )

    for epoch in range(100):
        pipeline.set_epoch(epoch)
        for batch in pipeline:
            ...

Phase 1 — torchdata.nodes integration
--------------------------------------
    from dino_loader.nodes import ShardReaderNode, build_reader_graph

    loader, reader = build_reader_graph(specs, batch_size=512, cache=cache, ...)
    for epoch in range(100):
        reader.set_epoch(epoch)
        for jpegs, meta in loader:
            my_augment(jpegs)

Sources alternatives
---------------------
    from dino_loader.sources import MixingSource, WDSSource

    # MixingSource : optimisée HPC, cache /dev/shm, multi-nœuds (défaut)
    # WDSSource    : basée webdataset, plus simple, NVMe / Lustre rapide
"""

import logging

from dino_loader.config import DINOAugConfig, LoaderConfig, NormStats
from dino_loader.loader import DINODataLoader
from dino_loader.memory import Batch
from dino_loader.nodes import (
    MaskMapNode,
    MetadataNode,
    ShardReaderNode,
    build_reader_graph,
)
from dino_loader.pipeline_graph import (  # noqa: F401
    BatchFilterNode,
    BatchMapNode,
    NodePipeline,
    wrap_loader,
)
from dino_loader.sources import MixingSource, WDSSource
from dino_loader.sources.resolution import ResolutionSource

log = logging.getLogger(__name__)

__all__ = [
    "Batch",
    "BatchFilterNode",
    "BatchMapNode",
    "DINOAugConfig",
    "DINODataLoader",
    "LoaderConfig",
    "MaskMapNode",
    "MetadataNode",
    "MixingSource",
    "NodePipeline",
    "NormStats",
    "ResolutionSource",
    "ShardReaderNode",
    "WDSSource",
    "build_reader_graph",
    "wrap_loader",
]
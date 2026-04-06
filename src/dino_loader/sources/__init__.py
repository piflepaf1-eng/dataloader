"""dino_loader.sources
====================
Sources de données pour le pipeline dino_loader.

Ce module expose deux stratégies de lecture de shards WebDataset :

HPC source (``hpc_source``)
    Conçue pour les clusters HPC avec Lustre et beaucoup de rangs par nœud.
    Combine un cache ``/dev/shm`` (``NodeSharedShardCache``) avec un double-
    buffering strict I/O + extraction.  C'est la source de production par
    défaut sur B200 / GB200 NVL72.

WDS source (``wds_source``)
    Alternative plus simple qui délègue entièrement le cycling, le shuffle
    et le mixing à la bibliothèque ``webdataset``.  Recommandée quand les
    shards sont déjà en mémoire rapide (NVMe local, Lustre MDS rapide) ou
    quand la simplicité prime sur la latence absolue.

Partagé entre les deux sources
    ``MixingWeights`` (``_weights``) — vecteur de poids normalisé thread-safe.
    ``ResolutionSource`` (``resolution``) — holder thread-safe de la résolution
    de crop courante, utilisé comme callback DALI ExternalSource.

Exports publics
---------------
Les symboles ci-dessous constituent l'API publique du module.  Les sous-modules
peuvent être importés directement si des symboles internes sont nécessaires.

    from dino_loader.sources import (
        MixingWeights,
        ResolutionSource,
        SampleRecord,
        ShardIterator,
        MixingSource,
        WDSSource,
        WDSShardReaderNode,
    )
"""

from dino_loader.sources._weights import MixingWeights
from dino_loader.sources.hpc_source import MixingSource, SampleRecord, ShardIterator
from dino_loader.sources.resolution import ResolutionSource
from dino_loader.sources.wds_source import WDSShardReaderNode, WDSSource

__all__ = [
    "MixingSource",
    "MixingWeights",
    "ResolutionSource",
    "SampleRecord",
    "ShardIterator",
    "WDSShardReaderNode",
    "WDSSource",
]
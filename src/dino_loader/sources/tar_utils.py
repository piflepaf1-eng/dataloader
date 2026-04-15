"""dino_loader.sources.tar_utils
================================
Extraction de samples JPEG bruts depuis des archives WebDataset (tar).

Responsabilité unique : parser un tar en mémoire et retourner des
``SampleRecord`` avec les bytes JPEG **non décodés** et les métadonnées
JSON optionnelles.

Pourquoi les bytes restent compressés
--------------------------------------
Les bytes JPEG sont transmis tels quels à DALI ``ExternalSource``.  DALI les
décode via le pipeline nvjpeg (ASIC matériel du GPU) : le CPU ne décode jamais
l'image.  Décoder sur CPU puis transférer des tenseurs dense serait ≈ 10-50×
plus coûteux en bande passante PCIe et bloquerait les cœurs CPU.

Format WebDataset attendu
--------------------------
Chaque sample est un groupe de fichiers tar partageant le même préfixe de clé :

    sample_000042.jpg   ← image compressée (obligatoire)
    sample_000042.json  ← sidecar JSON (optionnel)

Les clés et extensions sont détectées automatiquement.

Public API
----------
::

    from dino_loader.sources.tar_utils import extract_jpegs_with_meta

    records = extract_jpegs_with_meta(
        data           = memoryview(shard_bytes),
        min_quality    = 0.5,
        shuffle_buffer = 256,
        rng            = np.random.default_rng(42),
    )
    for record in records:
        # record.jpeg  : bytes JPEG bruts (non décodés)
        # record.metadata : dict JSON ou None
        # record.key   : clé WebDataset
        pass

"""

from __future__ import annotations

import io
import json
import logging
import tarfile
from typing import Any

import numpy as np

from dino_loader.augmentation import SampleRecord

log = logging.getLogger(__name__)

# Extensions reconnues comme images JPEG.
_JPEG_EXTS: frozenset[str] = frozenset({".jpg", ".jpeg", ".JPG", ".JPEG"})
# Extension reconnue comme sidecar JSON.
_JSON_EXT: str = ".json"


def extract_jpegs_with_meta(
    data:           memoryview | bytes,
    min_quality:    float | None = None,
    shuffle_buffer: int          = 0,
    rng:            np.random.Generator | None = None,
) -> list[SampleRecord]:
    """Parse un shard WebDataset en mémoire et retourne les samples JPEG bruts.

    Les bytes JPEG ne sont jamais décodés : ils sont destinés à DALI
    ``ExternalSource`` qui les transmet au pipeline nvjpeg sur GPU.

    Args:
        data:           Contenu brut du shard (tar archive en mémoire).
        min_quality:    Si défini, filtre les samples dont ``quality_score``
                        JSON est inférieur à ce seuil.
        shuffle_buffer: Si > 0, mélange les samples via un reservoir shuffle
                        de cette taille avant de les retourner.
        rng:            Générateur NumPy pour le shuffle reproductible.
                        Ignoré si ``shuffle_buffer == 0``.

    Returns:
        Liste de ``SampleRecord`` avec bytes JPEG bruts (non décodés).

    """
    samples: dict[str, dict[str, Any]] = {}

    try:
        buf = io.BytesIO(bytes(data) if isinstance(data, memoryview) else data)
        with tarfile.open(fileobj=buf, mode="r|*") as tf:
            for member in tf:
                if not member.isfile():
                    continue

                name = member.name
                # Clé = préfixe sans extension (ex. "sample_000042")
                dot_pos = name.rfind(".")
                if dot_pos < 0:
                    continue
                key = name[:dot_pos]
                ext = name[dot_pos:]

                if key not in samples:
                    samples[key] = {"jpeg": None, "meta": None}

                if ext in _JPEG_EXTS:
                    f = tf.extractfile(member)
                    if f is not None:
                        samples[key]["jpeg"] = f.read()

                elif ext == _JSON_EXT:
                    f = tf.extractfile(member)
                    if f is not None:
                        try:
                            samples[key]["meta"] = json.loads(f.read().decode("utf-8"))
                        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                            log.debug("Sidecar JSON invalide pour '%s': %s", key, exc)

    except tarfile.TarError as exc:
        log.error("Échec de parsing du shard tar : %s", exc)
        return []

    records: list[SampleRecord] = []
    for key, entry in samples.items():
        jpeg = entry["jpeg"]
        if jpeg is None:
            continue  # sample sans image — ignorer

        meta: dict | None = entry["meta"]

        # Filtre qualité anticipé (avant tout décodage).
        if min_quality is not None and meta is not None:
            score = meta.get("quality_score")
            if score is not None and score < min_quality:
                continue

        records.append(SampleRecord(jpeg=jpeg, metadata=meta, key=key))

    if shuffle_buffer > 0 and len(records) > 1:
        _reservoir_shuffle(records, shuffle_buffer, rng)

    return records


def _reservoir_shuffle(
    records: list[SampleRecord],
    buffer_size: int,
    rng: np.random.Generator | None,
) -> None:
    """Reservoir shuffle in-place sur ``records``.

    Mélange par blocs de taille ``buffer_size`` pour limiter l'empreinte
    mémoire tout en cassant les corrélations intra-shard.

    Args:
        records:     Liste à mélanger en place.
        buffer_size: Taille du réservoir.
        rng:         Générateur NumPy.  Si None, utilise le RNG global.

    """
    effective_rng = rng if rng is not None else np.random.default_rng()
    n = len(records)
    for start in range(0, n, buffer_size):
        end = min(start + buffer_size, n)
        block = records[start:end]
        indices = effective_rng.permutation(len(block)).tolist()
        for i, idx in enumerate(indices):
            records[start + i] = block[idx]
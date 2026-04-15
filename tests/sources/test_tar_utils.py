"""tests/sources/test_tar_utils.py
===================================
Tests unitaires pour :mod:`dino_loader.sources.tar_utils`.

Périmètre
---------
- ``extract_jpegs_with_meta`` : extraction de samples depuis une archive tar
  WebDataset en mémoire, avec et sans sidecar JSON.
- Les bytes JPEG retournés doivent être **bruts** (non décodés).
- Filtrage qualité via ``min_quality``.
- Reservoir shuffle reproductible.
- Gestion des entrées malformées (tar corrompu, JSON invalide).

Invariant critique
------------------
Les bytes ``SampleRecord.jpeg`` ne doivent jamais être des tenseurs décodés.
Ils doivent être les bytes compressés originaux, transmissibles directement à
DALI ``ExternalSource`` pour décodage nvjpeg sur GPU.
"""

from __future__ import annotations

import io
import json
import sys
import tarfile
from pathlib import Path

import numpy as np
import pytest

_SRC = str(Path(__file__).parent.parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from dino_loader.sources.tar_utils import extract_jpegs_with_meta
from tests.fixtures import make_jpeg_bytes, make_shard_tar


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_jpeg() -> bytes:
    """JPEG syntaxique minimal (SOI + EOI)."""
    return b"\xff\xd8\xff\xe0" + b"\x00" * 16 + b"\xff\xd9"


def _make_tar_bytes(
    n_samples:     int = 4,
    with_metadata: bool = True,
    quality_scores: list[float] | None = None,
) -> bytes:
    """Construit un tar WebDataset en mémoire."""
    if quality_scores is None:
        quality_scores = [1.0] * n_samples
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for i in range(n_samples):
            key  = f"sample_{i:06d}"
            jpeg = _minimal_jpeg()
            info = tarfile.TarInfo(name=f"{key}.jpg")
            info.size = len(jpeg)
            tf.addfile(info, io.BytesIO(jpeg))
            if with_metadata:
                meta = json.dumps({
                    "quality_score": quality_scores[i],
                    "caption": f"caption {i}",
                }).encode()
                jinfo = tarfile.TarInfo(name=f"{key}.json")
                jinfo.size = len(meta)
                tf.addfile(jinfo, io.BytesIO(meta))
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Extraction basique
# ---------------------------------------------------------------------------


class TestExtractBasic:

    def test_returns_correct_count(self) -> None:
        data = _make_tar_bytes(n_samples=6)
        records = extract_jpegs_with_meta(data)
        assert len(records) == 6

    def test_jpeg_bytes_are_raw_compressed(self) -> None:
        """Les bytes JPEG ne doivent jamais être décodés — invariant critique."""
        data    = _make_tar_bytes(n_samples=2)
        records = extract_jpegs_with_meta(data)
        for rec in records:
            # Un JPEG compressé commence toujours par SOI marker 0xFF 0xD8.
            assert rec.jpeg[:2] == b"\xff\xd8", (
                "SampleRecord.jpeg ne contient pas des bytes JPEG bruts (compressés). "
                "Les bytes ont peut-être été décodés sur CPU — invariant critique violé."
            )

    def test_jpeg_bytes_type_is_bytes(self) -> None:
        data    = _make_tar_bytes(n_samples=2)
        records = extract_jpegs_with_meta(data)
        for rec in records:
            assert isinstance(rec.jpeg, bytes)

    def test_metadata_present_when_json_sidecar(self) -> None:
        data    = _make_tar_bytes(n_samples=3, with_metadata=True)
        records = extract_jpegs_with_meta(data)
        for rec in records:
            assert rec.metadata is not None
            assert "quality_score" in rec.metadata

    def test_metadata_none_without_json_sidecar(self) -> None:
        data    = _make_tar_bytes(n_samples=3, with_metadata=False)
        records = extract_jpegs_with_meta(data)
        for rec in records:
            assert rec.metadata is None

    def test_keys_are_non_empty_strings(self) -> None:
        data    = _make_tar_bytes(n_samples=4)
        records = extract_jpegs_with_meta(data)
        for rec in records:
            assert isinstance(rec.key, str)
            assert len(rec.key) > 0

    def test_numpy_frombuffer_compatible(self) -> None:
        """Les bytes doivent être directement utilisables avec np.frombuffer."""
        data    = _make_tar_bytes(n_samples=2)
        records = extract_jpegs_with_meta(data)
        for rec in records:
            arr = np.frombuffer(rec.jpeg, dtype=np.uint8)
            assert arr.dtype == np.uint8
            assert len(arr) > 0

    def test_memoryview_input_accepted(self) -> None:
        data    = _make_tar_bytes(n_samples=3)
        records = extract_jpegs_with_meta(memoryview(data))
        assert len(records) == 3


# ---------------------------------------------------------------------------
# Filtrage qualité
# ---------------------------------------------------------------------------


class TestQualityFilter:

    def test_min_quality_filters_low_scores(self) -> None:
        scores = [0.1, 0.9, 0.3, 0.8]
        data   = _make_tar_bytes(n_samples=4, quality_scores=scores)
        records = extract_jpegs_with_meta(data, min_quality=0.5)
        assert len(records) == 2
        for rec in records:
            assert rec.metadata["quality_score"] >= 0.5

    def test_min_quality_none_keeps_all(self) -> None:
        data    = _make_tar_bytes(n_samples=4)
        records = extract_jpegs_with_meta(data, min_quality=None)
        assert len(records) == 4

    def test_sample_without_quality_key_passes(self) -> None:
        """Un sample sans 'quality_score' dans le JSON ne doit pas être filtré."""
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tf:
            jpeg = _minimal_jpeg()
            info = tarfile.TarInfo(name="sample_000000.jpg")
            info.size = len(jpeg)
            tf.addfile(info, io.BytesIO(jpeg))
            meta = json.dumps({"caption": "no quality key"}).encode()
            jinfo = tarfile.TarInfo(name="sample_000000.json")
            jinfo.size = len(meta)
            tf.addfile(jinfo, io.BytesIO(meta))
        data    = buf.getvalue()
        records = extract_jpegs_with_meta(data, min_quality=0.9)
        assert len(records) == 1

    def test_sample_without_metadata_passes_quality_filter(self) -> None:
        data    = _make_tar_bytes(n_samples=3, with_metadata=False)
        records = extract_jpegs_with_meta(data, min_quality=0.99)
        assert len(records) == 3


# ---------------------------------------------------------------------------
# Reservoir shuffle
# ---------------------------------------------------------------------------


class TestReservoirShuffle:

    def test_shuffle_preserves_count(self) -> None:
        data    = _make_tar_bytes(n_samples=20)
        records = extract_jpegs_with_meta(data, shuffle_buffer=8, rng=np.random.default_rng(42))
        assert len(records) == 20

    def test_shuffle_is_reproducible(self) -> None:
        data = _make_tar_bytes(n_samples=20)
        r1   = extract_jpegs_with_meta(data, shuffle_buffer=8, rng=np.random.default_rng(7))
        r2   = extract_jpegs_with_meta(data, shuffle_buffer=8, rng=np.random.default_rng(7))
        keys1 = [rec.key for rec in r1]
        keys2 = [rec.key for rec in r2]
        assert keys1 == keys2

    def test_no_shuffle_when_buffer_zero(self) -> None:
        data    = _make_tar_bytes(n_samples=10)
        records = extract_jpegs_with_meta(data, shuffle_buffer=0)
        assert len(records) == 10

    def test_shuffle_changes_order(self) -> None:
        """Deux seeds différentes donnent des ordres différents (probabiliste)."""
        data = _make_tar_bytes(n_samples=20)
        r1   = extract_jpegs_with_meta(data, shuffle_buffer=20, rng=np.random.default_rng(1))
        r2   = extract_jpegs_with_meta(data, shuffle_buffer=20, rng=np.random.default_rng(2))
        keys1 = [rec.key for rec in r1]
        keys2 = [rec.key for rec in r2]
        # Avec 20 éléments et deux seeds différentes, les ordres doivent différer.
        assert keys1 != keys2


# ---------------------------------------------------------------------------
# Gestion des erreurs
# ---------------------------------------------------------------------------


class TestErrorHandling:

    def test_empty_bytes_returns_empty_list(self) -> None:
        records = extract_jpegs_with_meta(b"")
        assert records == []

    def test_corrupt_tar_returns_empty_list(self) -> None:
        records = extract_jpegs_with_meta(b"not a tar file at all !!!!")
        assert records == []

    def test_invalid_json_sidecar_skipped(self) -> None:
        """Un JSON invalide dans le sidecar ne doit pas faire crasher l'extraction."""
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tf:
            jpeg = _minimal_jpeg()
            info = tarfile.TarInfo(name="s.jpg")
            info.size = len(jpeg)
            tf.addfile(info, io.BytesIO(jpeg))
            bad_json = b"{ this is not valid json }"
            jinfo = tarfile.TarInfo(name="s.json")
            jinfo.size = len(bad_json)
            tf.addfile(jinfo, io.BytesIO(bad_json))
        data    = buf.getvalue()
        records = extract_jpegs_with_meta(data)
        assert len(records) == 1
        assert records[0].metadata is None  # sidecar ignoré, pas de crash

    def test_entry_without_image_skipped(self) -> None:
        """Un sample sans JPEG est ignoré silencieusement."""
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tf:
            # Seulement un JSON, pas de JPEG.
            meta = json.dumps({"caption": "orphan"}).encode()
            info = tarfile.TarInfo(name="orphan.json")
            info.size = len(meta)
            tf.addfile(info, io.BytesIO(meta))
        data    = buf.getvalue()
        records = extract_jpegs_with_meta(data)
        assert len(records) == 0


# ---------------------------------------------------------------------------
# Intégration avec make_shard_tar des fixtures
# ---------------------------------------------------------------------------


class TestIntegrationWithFixtures:

    def test_extracts_from_fixture_shard(self) -> None:
        """Vérifie la compatibilité avec les shards produits par les fixtures de test."""
        data    = make_shard_tar(n_samples=8, with_metadata=True)
        records = extract_jpegs_with_meta(data)
        assert len(records) == 8

    def test_jpeg_bytes_from_fixture_start_with_soi(self) -> None:
        """Les bytes JPEG des fixtures commencent par SOI marker."""
        data    = make_shard_tar(n_samples=4, with_metadata=False)
        records = extract_jpegs_with_meta(data)
        for rec in records:
            assert rec.jpeg[:2] == b"\xff\xd8", (
                f"Bytes JPEG invalides (premiers bytes: {rec.jpeg[:4].hex()!r}). "
                "Les bytes doivent être compressés (SOI marker 0xFF 0xD8)."
            )

    def test_quality_score_in_metadata_from_fixture(self) -> None:
        scores  = [0.1, 0.5, 0.9, 0.95]
        data    = make_shard_tar(n_samples=4, quality_scores=scores)
        records = extract_jpegs_with_meta(data)
        found_scores = {rec.metadata["quality_score"] for rec in records}
        assert found_scores == set(scores)
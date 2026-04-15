"""tests/sources/test_wds_source.py
=====================================
Tests unitaires pour :mod:`dino_loader.sources.wds_source`.

Périmètre
---------
- ``WDSSource.__call__`` : retourne des bytes JPEG bruts (non décodés).
- Invariant critique : les bytes transmis à DALI doivent être compressés.
- ``WDSShardReaderNode`` : interface torchdata minimale.
- Reproductibilité des seeds.
- Gestion des poids de mixage.

Invariant critique
------------------
``WDSSource.__call__`` doit retourner des ``np.ndarray`` de dtype ``uint8``
contenant des bytes JPEG **compressés** (SOI marker 0xFF 0xD8 en tête).
Aucun décodage PIL ou équivalent ne doit intervenir côté CPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_SRC = str(Path(__file__).parent.parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

pytest.importorskip("webdataset", reason="webdataset required for WDSSource tests")

from dino_datasets import DatasetSpec

from dino_loader.sources.wds_source import WDSSource
from tests.fixtures import scaffold_dataset_dir


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_spec(tar_paths: list[str], name: str = "ds", weight: float = 1.0) -> DatasetSpec:
    return DatasetSpec(name=name, shards=tuple(tar_paths), weight=weight)


def _make_source(tar_paths: list[str], batch_size: int = 4) -> WDSSource:
    return WDSSource(
        specs      = [_make_spec(tar_paths)],
        batch_size = batch_size,
        rank       = 0,
        world_size = 1,
        seed       = 42,
    )


# ---------------------------------------------------------------------------
# Invariant critique : bytes JPEG bruts
# ---------------------------------------------------------------------------


class TestJpegBytesInvariant:
    """Vérifie que WDSSource ne décode jamais les JPEG sur CPU."""

    def test_call_returns_list_of_ndarrays(self, tmp_path: Path) -> None:
        paths  = scaffold_dataset_dir(root=tmp_path, n_shards=2, n_samples_per_shard=8)
        source = _make_source(paths, batch_size=4)
        source.set_epoch(0)
        result = source()
        assert isinstance(result, list)
        assert len(result) == 4
        for item in result:
            assert isinstance(item, np.ndarray)

    def test_arrays_have_uint8_dtype(self, tmp_path: Path) -> None:
        """Les bytes JPEG bruts doivent être de dtype uint8."""
        paths  = scaffold_dataset_dir(root=tmp_path, n_shards=2, n_samples_per_shard=8)
        source = _make_source(paths, batch_size=4)
        source.set_epoch(0)
        result = source()
        for arr in result:
            assert arr.dtype == np.uint8, (
                f"dtype={arr.dtype} — les bytes JPEG doivent être uint8 (compressés)."
            )

    def test_bytes_start_with_jpeg_soi_marker(self, tmp_path: Path) -> None:
        """Invariant critique : chaque tableau doit commencer par le SOI marker JPEG.

        Un tableau ne commençant pas par 0xFF 0xD8 indiquerait que les bytes
        ont été décodés (ou re-encodés de façon incorrecte) sur CPU, violant
        l'invariant fondamental de performance du pipeline.
        """
        paths  = scaffold_dataset_dir(root=tmp_path, n_shards=2, n_samples_per_shard=8)
        source = _make_source(paths, batch_size=4)
        source.set_epoch(0)
        result = source()
        for i, arr in enumerate(result):
            assert len(arr) >= 2, f"Sample {i}: bytes JPEG trop courts ({len(arr)} bytes)."
            soi = bytes(arr[:2])
            assert soi == b"\xff\xd8", (
                f"Sample {i}: premiers bytes {soi.hex()!r} ≠ SOI marker 0xFF 0xD8. "
                "Les bytes JPEG ont peut-être été décodés puis re-encodés sur CPU — "
                "invariant critique violé. WDSSource doit utiliser decode(False)."
            )

    def test_arrays_are_smaller_than_decoded_would_be(self, tmp_path: Path) -> None:
        """Les bytes compressés sont bien plus petits qu'une image décodée.

        Une image 64x64 RGB décodée = 64*64*3 = 12 288 bytes.
        Un JPEG compressé à quality=85 d'une image 64x64 ≈ 1-3 KB.
        Si les tableaux font > 12 288 bytes, c'est probablement un tenseur décodé.
        """
        paths  = scaffold_dataset_dir(root=tmp_path, n_shards=2, n_samples_per_shard=8)
        source = _make_source(paths, batch_size=4)
        source.set_epoch(0)
        result = source()
        # Les shards de test utilisent des images 64x64 — décodé = 12 288 bytes.
        decoded_size = 64 * 64 * 3
        for i, arr in enumerate(result):
            assert len(arr) < decoded_size, (
                f"Sample {i}: tableau de {len(arr)} bytes ≥ taille décodée "
                f"({decoded_size} bytes). Suspicion de décodage CPU non voulu."
            )


# ---------------------------------------------------------------------------
# Métadonnées
# ---------------------------------------------------------------------------


class TestMetadata:

    def test_pop_last_metadata_after_call(self, tmp_path: Path) -> None:
        paths  = scaffold_dataset_dir(root=tmp_path, n_shards=2, n_samples_per_shard=8,
                                      with_metadata=True)
        source = _make_source(paths, batch_size=4)
        source.set_epoch(0)
        source()
        meta = source.pop_last_metadata()
        assert len(meta) == 4

    def test_metadata_is_dict_or_none(self, tmp_path: Path) -> None:
        paths  = scaffold_dataset_dir(root=tmp_path, n_shards=2, n_samples_per_shard=8,
                                      with_metadata=True)
        source = _make_source(paths, batch_size=4)
        source.set_epoch(0)
        source()
        meta = source.pop_last_metadata()
        for m in meta:
            assert m is None or isinstance(m, dict)


# ---------------------------------------------------------------------------
# Poids de mixage
# ---------------------------------------------------------------------------


class TestWeights:

    def test_set_weights_normalises(self, tmp_path: Path) -> None:
        paths_a = scaffold_dataset_dir(root=tmp_path / "a", n_shards=2)
        paths_b = scaffold_dataset_dir(root=tmp_path / "b", n_shards=2)
        source  = WDSSource(
            specs      = [_make_spec(paths_a, "a"), _make_spec(paths_b, "b")],
            batch_size = 4,
            rank       = 0,
            world_size = 1,
        )
        source.set_weights([3.0, 1.0])
        w = source.current_weights
        assert abs(w[0] - 0.75) < 1e-5
        assert abs(w[1] - 0.25) < 1e-5

    def test_dataset_names(self, tmp_path: Path) -> None:
        paths = scaffold_dataset_dir(root=tmp_path, n_shards=2)
        source = _make_source(paths)
        assert source.dataset_names == ["ds"]


# ---------------------------------------------------------------------------
# Contrôle époque
# ---------------------------------------------------------------------------


class TestEpochControl:

    def test_set_epoch_resets_iterator(self, tmp_path: Path) -> None:
        paths  = scaffold_dataset_dir(root=tmp_path, n_shards=2, n_samples_per_shard=8)
        source = _make_source(paths)
        source.set_epoch(0)
        source()
        source.set_epoch(1)
        assert source._iterator is None

    def test_close_clears_iterator(self, tmp_path: Path) -> None:
        paths  = scaffold_dataset_dir(root=tmp_path, n_shards=2, n_samples_per_shard=8)
        source = _make_source(paths)
        source.set_epoch(0)
        source()
        source.close()
        assert source._iterator is None


# ---------------------------------------------------------------------------
# WDSShardReaderNode
# ---------------------------------------------------------------------------


class TestWDSShardReaderNode:

    def test_next_returns_jpeg_and_metadata(self, tmp_path: Path) -> None:
        from dino_loader.sources.wds_source import WDSShardReaderNode
        paths  = scaffold_dataset_dir(root=tmp_path, n_shards=2, n_samples_per_shard=8)
        source = _make_source(paths, batch_size=4)
        node   = WDSShardReaderNode(source)
        node.reset()
        jpegs, meta = node.next()
        assert len(jpegs) == 4
        assert len(meta)  == 4

    def test_next_jpegs_are_raw_bytes(self, tmp_path: Path) -> None:
        """Les bytes JPEG retournés par le nœud doivent être compressés."""
        from dino_loader.sources.wds_source import WDSShardReaderNode
        paths  = scaffold_dataset_dir(root=tmp_path, n_shards=2, n_samples_per_shard=8)
        source = _make_source(paths, batch_size=4)
        node   = WDSShardReaderNode(source)
        node.reset()
        jpegs, _ = node.next()
        for arr in jpegs:
            assert bytes(arr[:2]) == b"\xff\xd8", (
                "WDSShardReaderNode.next() ne retourne pas des bytes JPEG bruts."
            )
"""tests/test_nodes.py
======================
Unit tests for :mod:`dino_loader.nodes` (Phase 1 — torchdata integration).

Tests cover ShardReaderNode, MetadataNode, and build_reader_graph.
All tests require ``torchdata``; they are skipped automatically when it is
not installed.

No GPU, DALI, or SLURM required — all shard I/O uses InProcessShardCache.

Coverage
--------
ShardReaderNode
- next() returns (jpeg_list, metadata_list)
- get_state() returns epoch, mixing_weights, dataset_names
- set_epoch updates state
- set_weights normalises correctly
- reset with saved state restores epoch
- dataset_names property
- multiple batches without error

MetadataNode
- passes through jpegs and metadata
- pop_last_metadata clears after first call
- get_state delegates to source

build_reader_graph
- returns (loader, reader_node)
- loader is iterable
- loader state_dict round-trip
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SRC = str(Path(__file__).parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

try:
    import torchdata.nodes as tn  # noqa: F401
    _HAS_TORCHDATA = True
except ImportError:
    _HAS_TORCHDATA = False

pytestmark = pytest.mark.skipif(
    not _HAS_TORCHDATA,
    reason="torchdata is not installed",
)

import numpy as np  # noqa: E402
from dino_datasets import DatasetSpec  # noqa: E402

from dino_loader.backends.cpu import InProcessShardCache  # noqa: E402
from tests.fixtures import scaffold_dataset_dir  # noqa: E402

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_spec(tar_paths: list[str], name: str = "ds") -> DatasetSpec:
    return DatasetSpec(name=name, shards=tuple(tar_paths), weight=1.0)


def _cache() -> InProcessShardCache:
    return InProcessShardCache(max_gb=0.5)


# ══════════════════════════════════════════════════════════════════════════════
# ShardReaderNode
# ══════════════════════════════════════════════════════════════════════════════


class TestShardReaderNodeOutput:

    def test_yields_jpeg_and_metadata(self, tmp_path: Path) -> None:
        from dino_loader.nodes import ShardReaderNode
        tar_paths = scaffold_dataset_dir(root=tmp_path, n_shards=2, n_samples_per_shard=8)
        node = ShardReaderNode(
            specs=[_make_spec(tar_paths)], batch_size=4, cache=_cache(),
            rank=0, world_size=1,
        )
        node.reset()
        jpegs, meta = node.next()
        assert len(jpegs) == 4
        assert len(meta) == 4
        for j in jpegs:
            assert isinstance(j, np.ndarray)

    def test_multiple_batches_no_error(self, tmp_path: Path) -> None:
        from dino_loader.nodes import ShardReaderNode
        tar_paths = scaffold_dataset_dir(root=tmp_path, n_shards=4, n_samples_per_shard=16)
        node = ShardReaderNode(
            specs=[_make_spec(tar_paths)], batch_size=8, cache=_cache(),
            rank=0, world_size=1,
        )
        node.reset()
        for _ in range(5):
            jpegs, meta = node.next()
            assert len(jpegs) == 8
            assert len(meta) == 8


class TestShardReaderNodeState:

    def test_get_state_contains_epoch_weights_names(self, tmp_path: Path) -> None:
        from dino_loader.nodes import ShardReaderNode
        tar_paths = scaffold_dataset_dir(root=tmp_path, n_shards=2)
        node = ShardReaderNode(
            specs=[_make_spec(tar_paths)], batch_size=4, cache=_cache(),
            rank=0, world_size=1,
        )
        node.reset()
        state = node.get_state()
        assert "epoch" in state
        assert "mixing_weights" in state
        assert "dataset_names" in state
        assert state["epoch"] == 0

    def test_set_epoch_updates_state(self, tmp_path: Path) -> None:
        from dino_loader.nodes import ShardReaderNode
        tar_paths = scaffold_dataset_dir(root=tmp_path, n_shards=2)
        node = ShardReaderNode(
            specs=[_make_spec(tar_paths)], batch_size=4, cache=_cache(),
            rank=0, world_size=1,
        )
        node.reset()
        node.set_epoch(5)
        assert node.get_state()["epoch"] == 5

    def test_reset_with_state_restores_epoch(self, tmp_path: Path) -> None:
        from dino_loader.nodes import ShardReaderNode
        tar_paths = scaffold_dataset_dir(root=tmp_path, n_shards=2)
        node = ShardReaderNode(
            specs=[_make_spec(tar_paths)], batch_size=4, cache=_cache(),
            rank=0, world_size=1,
        )
        node.reset()
        node.set_epoch(3)
        saved = node.get_state()

        node2 = ShardReaderNode(
            specs=[_make_spec(tar_paths)], batch_size=4, cache=_cache(),
            rank=0, world_size=1,
        )
        node2.reset(initial_state=saved)
        assert node2._epoch == 3


class TestShardReaderNodeWeights:

    def test_set_weights_normalised(self, tmp_path: Path) -> None:
        from dino_loader.nodes import ShardReaderNode
        paths_a = scaffold_dataset_dir(root=tmp_path / "a", n_shards=1)
        paths_b = scaffold_dataset_dir(root=tmp_path / "b", n_shards=1)
        node = ShardReaderNode(
            specs=[_make_spec(paths_a, "a"), _make_spec(paths_b, "b")],
            batch_size=4, cache=_cache(), rank=0, world_size=1,
        )
        node.reset()
        node.set_weights([3.0, 1.0])
        w = node.current_weights
        assert abs(w[0] - 0.75) < 1e-5
        assert abs(w[1] - 0.25) < 1e-5

    def test_dataset_names_property(self, tmp_path: Path) -> None:
        from dino_loader.nodes import ShardReaderNode
        tar_paths = scaffold_dataset_dir(root=tmp_path, n_shards=1)
        node = ShardReaderNode(
            specs=[_make_spec(tar_paths, "myds")], batch_size=4,
            cache=_cache(), rank=0, world_size=1,
        )
        assert node.dataset_names == ["myds"]


# ══════════════════════════════════════════════════════════════════════════════
# MetadataNode
# ══════════════════════════════════════════════════════════════════════════════


class TestMetadataNode:

    def test_passes_through_jpegs_and_meta(self, tmp_path: Path) -> None:
        from dino_loader.nodes import MetadataNode, ShardReaderNode
        tar_paths = scaffold_dataset_dir(root=tmp_path, n_shards=2)
        reader = ShardReaderNode(
            specs=[_make_spec(tar_paths)], batch_size=4, cache=_cache(),
            rank=0, world_size=1,
        )
        node = MetadataNode(reader)
        node.reset()
        jpegs, meta = node.next()
        assert len(jpegs) == 4
        assert len(meta) == 4

    def test_pop_last_metadata_clears_buffer(self, tmp_path: Path) -> None:
        from dino_loader.nodes import MetadataNode, ShardReaderNode
        tar_paths = scaffold_dataset_dir(root=tmp_path, n_shards=2)
        reader = ShardReaderNode(
            specs=[_make_spec(tar_paths)], batch_size=4, cache=_cache(),
            rank=0, world_size=1,
        )
        node = MetadataNode(reader)
        node.reset()
        node.next()
        meta1 = node.pop_last_metadata()
        meta2 = node.pop_last_metadata()
        assert len(meta1) == 4
        assert meta2 == []

    def test_get_state_delegates_to_reader(self, tmp_path: Path) -> None:
        from dino_loader.nodes import MetadataNode, ShardReaderNode
        tar_paths = scaffold_dataset_dir(root=tmp_path, n_shards=2)
        reader = ShardReaderNode(
            specs=[_make_spec(tar_paths)], batch_size=4, cache=_cache(),
            rank=0, world_size=1,
        )
        node = MetadataNode(reader)
        node.reset()
        assert "epoch" in node.get_state()


# ══════════════════════════════════════════════════════════════════════════════
# build_reader_graph
# ══════════════════════════════════════════════════════════════════════════════


class TestBuildReaderGraph:

    def test_returns_loader_and_reader(self, tmp_path: Path) -> None:
        from dino_loader.nodes import build_reader_graph
        tar_paths = scaffold_dataset_dir(root=tmp_path, n_shards=2)
        loader, reader = build_reader_graph(
            specs=[_make_spec(tar_paths)], batch_size=4,
            cache=_cache(), rank=0, world_size=1,
        )
        assert loader is not None
        assert reader is not None

    def test_loader_is_iterable(self, tmp_path: Path) -> None:
        from dino_loader.nodes import build_reader_graph
        tar_paths = scaffold_dataset_dir(root=tmp_path, n_shards=2)
        loader, reader = build_reader_graph(
            specs=[_make_spec(tar_paths)], batch_size=4,
            cache=_cache(), rank=0, world_size=1,
        )
        reader.set_epoch(0)
        jpegs, meta = next(iter(loader))
        assert len(jpegs) == 4
        assert len(meta) == 4

    def test_loader_state_dict_roundtrip(self, tmp_path: Path) -> None:
        from dino_loader.nodes import build_reader_graph
        tar_paths = scaffold_dataset_dir(root=tmp_path, n_shards=2)
        loader, reader = build_reader_graph(
            specs=[_make_spec(tar_paths)], batch_size=4,
            cache=_cache(), rank=0, world_size=1,
        )
        reader.set_epoch(2)
        sd = loader.state_dict()
        assert isinstance(sd, dict)
        loader.load_state_dict(sd)  # must not raise


# ══════════════════════════════════════════════════════════════════════════════
# Import guard
# ══════════════════════════════════════════════════════════════════════════════


class TestNodesImportGuard:

    def test_shardreadernode_raises_without_torchdata(self, monkeypatch) -> None:
        import dino_loader.nodes as nodes_mod
        monkeypatch.setattr(nodes_mod, "_HAS_TORCHDATA", False)
        with pytest.raises(ImportError, match="torchdata"):
            nodes_mod.ShardReaderNode(
                specs=[], batch_size=4, cache=None, rank=0, world_size=1,
            )

    def test_build_reader_graph_raises_without_torchdata(self, monkeypatch) -> None:
        import dino_loader.nodes as nodes_mod
        monkeypatch.setattr(nodes_mod, "_HAS_TORCHDATA", False)
        with pytest.raises(ImportError, match="torchdata"):
            nodes_mod.build_reader_graph(
                specs=[], batch_size=4, cache=None, rank=0, world_size=1,
            )

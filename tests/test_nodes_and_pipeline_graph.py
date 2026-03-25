"""tests/test_nodes_and_pipeline_graph.py
=========================================
Unit tests for:

  Phase 1 — ``dino_loader.nodes`` (ShardReaderNode, MetadataNode,
             build_reader_graph)
  Phase 3 — ``dino_loader.pipeline_graph`` (BatchMapNode, BatchFilterNode,
             NodePipeline, wrap_loader)

Design
------
- No DALI, no GPU, no SLURM required.
- All shard I/O uses ``InProcessShardCache`` backed by ``tmp_path``.
- ``torchdata`` must be installed; tests are skipped automatically if it is
  not (so CI without torchdata does not fail).
- ``wrap_loader`` tests use a lightweight ``FakeDINOLoader`` stub that
  mimics the ``DINODataLoader`` API without any DALI plumbing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SRC = str(Path(__file__).parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# ---------------------------------------------------------------------------
# Optional dependency skip
# ---------------------------------------------------------------------------

try:
    import torchdata.nodes as tn  # noqa: F401

    _HAS_TORCHDATA = True
except ImportError:
    _HAS_TORCHDATA = False

pytestmark = pytest.mark.skipif(
    not _HAS_TORCHDATA,
    reason="torchdata is not installed",
)

# ---------------------------------------------------------------------------
# Local imports (deferred so skip works cleanly)
# ---------------------------------------------------------------------------

from tests.fixtures import scaffold_dataset_dir  # noqa: E402
from dino_loader.backends.cpu import InProcessShardCache  # noqa: E402
from dino_datasets import DatasetSpec  # noqa: E402
from dino_loader.memory import Batch  # noqa: E402


# ===========================================================================
# Helpers / shared fixtures
# ===========================================================================


def _make_spec(tar_paths: list[str], name: str = "ds") -> DatasetSpec:
    return DatasetSpec(name=name, shards=tuple(tar_paths), weight=1.0)


def _cache() -> InProcessShardCache:
    return InProcessShardCache(max_gb=0.5)


# ---------------------------------------------------------------------------
# FakeDINOLoader — minimal stub for wrap_loader tests
# ---------------------------------------------------------------------------


class _FakeDINOLoader:
    """Minimal stub that mimics DINODataLoader for NodePipeline tests."""

    def __init__(self, batches: list[Batch], steps_per_epoch: int | None = None) -> None:
        self._batches         = batches
        self._steps_per_epoch = steps_per_epoch
        self._epoch           = 0
        self._step            = 0
        self.set_epoch_calls: list[int] = []

    def __iter__(self):
        for b in self._batches:
            yield b

    def __len__(self) -> int:
        if self._steps_per_epoch is None:
            raise TypeError("steps_per_epoch not set")
        return self._steps_per_epoch

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch
        self.set_epoch_calls.append(epoch)

    def checkpoint(self, step: int) -> None:
        self._step = step

    def state_dict(self) -> dict:
        return {"epoch": self._epoch, "step": self._step}

    def load_state_dict(self, sd: dict) -> None:
        self._epoch = sd.get("epoch", 0)
        self._step  = sd.get("step", 0)

    def set_weights(self, weights) -> None:
        pass

    def set_weight_by_name(self, name: str, weight: float) -> None:
        pass

    def set_resolution(self, g: int, l: int) -> None:
        pass

    @property
    def current_resolution(self) -> tuple[int, int]:
        return (224, 96)


def _fake_loader(n_batches: int = 8) -> "_FakeDINOLoader":
    batches = [
        Batch(
            global_crops = [],
            local_crops  = [],
            metadata     = [{"idx": i}],
        )
        for i in range(n_batches)
    ]
    return _FakeDINOLoader(batches, steps_per_epoch=n_batches)


# ===========================================================================
# Phase 1: ShardReaderNode
# ===========================================================================


class TestShardReaderNode:
    """Tests for dino_loader.nodes.ShardReaderNode."""

    def test_yields_jpeg_and_metadata(self, tmp_path: Path) -> None:
        """next() must return (list[np.ndarray], list[dict|None])."""
        import numpy as np
        from dino_loader.nodes import ShardReaderNode

        tar_paths = scaffold_dataset_dir(
            root=tmp_path, n_shards=2, n_samples_per_shard=8
        )
        node = ShardReaderNode(
            specs      = [_make_spec(tar_paths)],
            batch_size = 4,
            cache      = _cache(),
            rank       = 0,
            world_size = 1,
        )
        node.reset()

        jpegs, meta = node.next()

        assert len(jpegs) == 4
        assert len(meta)  == 4
        for j in jpegs:
            assert isinstance(j, np.ndarray)

    def test_get_state_returns_epoch_and_weights(self, tmp_path: Path) -> None:
        from dino_loader.nodes import ShardReaderNode

        tar_paths = scaffold_dataset_dir(root=tmp_path, n_shards=2)
        node = ShardReaderNode(
            specs=[_make_spec(tar_paths)], batch_size=4,
            cache=_cache(), rank=0, world_size=1,
        )
        node.reset()

        state = node.get_state()
        assert "epoch"          in state
        assert "mixing_weights" in state
        assert "dataset_names"  in state
        assert state["epoch"] == 0

    def test_set_epoch_updates_state(self, tmp_path: Path) -> None:
        from dino_loader.nodes import ShardReaderNode

        tar_paths = scaffold_dataset_dir(root=tmp_path, n_shards=2)
        node = ShardReaderNode(
            specs=[_make_spec(tar_paths)], batch_size=4,
            cache=_cache(), rank=0, world_size=1,
        )
        node.reset()
        node.set_epoch(5)

        assert node.get_state()["epoch"] == 5

    def test_set_weights_normalised(self, tmp_path: Path) -> None:
        from dino_loader.nodes import ShardReaderNode

        paths_a = scaffold_dataset_dir(root=tmp_path / "a", n_shards=1)
        paths_b = scaffold_dataset_dir(root=tmp_path / "b", n_shards=1)
        specs   = [_make_spec(paths_a, "a"), _make_spec(paths_b, "b")]

        node = ShardReaderNode(
            specs=specs, batch_size=4, cache=_cache(), rank=0, world_size=1,
        )
        node.reset()
        node.set_weights([3.0, 1.0])

        w = node.current_weights
        assert abs(w[0] - 0.75) < 1e-5
        assert abs(w[1] - 0.25) < 1e-5

    def test_reset_with_state_restores_epoch(self, tmp_path: Path) -> None:
        from dino_loader.nodes import ShardReaderNode

        tar_paths = scaffold_dataset_dir(root=tmp_path, n_shards=2)
        node = ShardReaderNode(
            specs=[_make_spec(tar_paths)], batch_size=4,
            cache=_cache(), rank=0, world_size=1,
        )
        node.reset()
        node.set_epoch(3)

        saved = node.get_state()

        # Create a fresh node and restore
        node2 = ShardReaderNode(
            specs=[_make_spec(tar_paths)], batch_size=4,
            cache=_cache(), rank=0, world_size=1,
        )
        node2.reset(initial_state=saved)
        assert node2._epoch == 3

    def test_dataset_names_property(self, tmp_path: Path) -> None:
        from dino_loader.nodes import ShardReaderNode

        tar_paths = scaffold_dataset_dir(root=tmp_path, n_shards=1)
        node = ShardReaderNode(
            specs=[_make_spec(tar_paths, "myds")], batch_size=4,
            cache=_cache(), rank=0, world_size=1,
        )
        assert node.dataset_names == ["myds"]

    def test_multiple_batches_no_error(self, tmp_path: Path) -> None:
        from dino_loader.nodes import ShardReaderNode

        tar_paths = scaffold_dataset_dir(root=tmp_path, n_shards=4, n_samples_per_shard=16)
        node = ShardReaderNode(
            specs=[_make_spec(tar_paths)], batch_size=8,
            cache=_cache(), rank=0, world_size=1,
        )
        node.reset()

        for _ in range(5):
            jpegs, meta = node.next()
            assert len(jpegs) == 8
            assert len(meta)  == 8


# ===========================================================================
# Phase 1: MetadataNode
# ===========================================================================


class TestMetadataNode:

    def test_passes_through_jpegs_and_meta(self, tmp_path: Path) -> None:
        from dino_loader.nodes import MetadataNode, ShardReaderNode

        tar_paths = scaffold_dataset_dir(root=tmp_path, n_shards=2)
        reader    = ShardReaderNode(
            specs=[_make_spec(tar_paths)], batch_size=4,
            cache=_cache(), rank=0, world_size=1,
        )
        node = MetadataNode(reader)
        node.reset()

        jpegs, meta = node.next()
        assert len(jpegs) == 4
        assert len(meta)  == 4

    def test_pop_last_metadata_clears(self, tmp_path: Path) -> None:
        from dino_loader.nodes import MetadataNode, ShardReaderNode

        tar_paths = scaffold_dataset_dir(root=tmp_path, n_shards=2)
        reader    = ShardReaderNode(
            specs=[_make_spec(tar_paths)], batch_size=4,
            cache=_cache(), rank=0, world_size=1,
        )
        node = MetadataNode(reader)
        node.reset()
        node.next()

        meta1 = node.pop_last_metadata()
        meta2 = node.pop_last_metadata()

        assert len(meta1) == 4
        assert meta2 == []   # cleared after first pop

    def test_get_state_delegates_to_reader(self, tmp_path: Path) -> None:
        from dino_loader.nodes import MetadataNode, ShardReaderNode

        tar_paths = scaffold_dataset_dir(root=tmp_path, n_shards=2)
        reader    = ShardReaderNode(
            specs=[_make_spec(tar_paths)], batch_size=4,
            cache=_cache(), rank=0, world_size=1,
        )
        node = MetadataNode(reader)
        node.reset()

        state = node.get_state()
        assert "epoch" in state


# ===========================================================================
# Phase 1: build_reader_graph
# ===========================================================================


class TestBuildReaderGraph:

    def test_returns_loader_and_reader(self, tmp_path: Path) -> None:
        from dino_loader.nodes import build_reader_graph

        tar_paths = scaffold_dataset_dir(root=tmp_path, n_shards=2)
        loader, reader = build_reader_graph(
            specs      = [_make_spec(tar_paths)],
            batch_size = 4,
            cache      = _cache(),
            rank       = 0,
            world_size = 1,
        )
        assert loader  is not None
        assert reader  is not None

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
        assert len(meta)  == 4

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

        # Restore from state dict — must not raise
        loader.load_state_dict(sd)


# ===========================================================================
# Phase 3: BatchMapNode
# ===========================================================================


class TestBatchMapNode:

    def _source_node(self, batches: list[Batch]) -> "tn.IterableWrapper":
        return tn.IterableWrapper(iter(batches))

    def test_applies_fn_to_every_batch(self) -> None:
        from dino_loader.pipeline_graph import BatchMapNode

        tags: list[int] = []

        def _tag(b: Batch) -> Batch:
            tags.append(id(b))
            return b

        batches = [Batch([], [], []) for _ in range(3)]
        src     = tn.IterableWrapper(iter(batches))
        node    = BatchMapNode(src, _tag, label="test-tag")
        node.reset()

        for _ in range(3):
            node.next()

        assert len(tags) == 3

    def test_fn_receives_correct_batch(self) -> None:
        from dino_loader.pipeline_graph import BatchMapNode

        seen: list[list] = []

        def _inspect(b: Batch) -> Batch:
            seen.append(list(b.metadata))
            return b

        batches = [Batch([], [], [{"i": i}]) for i in range(4)]
        src     = tn.IterableWrapper(iter(batches))
        node    = BatchMapNode(src, _inspect)
        node.reset()

        for _ in range(4):
            node.next()

        assert [s[0]["i"] for s in seen] == [0, 1, 2, 3]

    def test_get_state_delegates(self) -> None:
        from dino_loader.pipeline_graph import BatchMapNode

        src  = tn.IterableWrapper(iter([Batch([], [], [])]))
        node = BatchMapNode(src, lambda b: b)
        node.reset()
        state = node.get_state()
        assert isinstance(state, dict)


# ===========================================================================
# Phase 3: BatchFilterNode
# ===========================================================================


class TestBatchFilterNode:

    def test_keeps_passing_batches(self) -> None:
        from dino_loader.pipeline_graph import BatchFilterNode

        batches = [Batch([], [], [{"score": float(i)}]) for i in range(6)]
        src     = tn.IterableWrapper(iter(batches))
        node    = BatchFilterNode(src, lambda b: b.metadata[0]["score"] >= 3.0)
        node.reset()

        kept: list[Batch] = []
        try:
            while True:
                kept.append(node.next())
        except StopIteration:
            pass

        assert len(kept) == 3
        assert all(b.metadata[0]["score"] >= 3.0 for b in kept)

    def test_n_skipped_counter(self) -> None:
        from dino_loader.pipeline_graph import BatchFilterNode

        batches = [Batch([], [], [{"ok": i % 2 == 0}]) for i in range(8)]
        src     = tn.IterableWrapper(iter(batches))
        node    = BatchFilterNode(src, lambda b: b.metadata[0]["ok"])
        node.reset()

        try:
            while True:
                node.next()
        except StopIteration:
            pass

        assert node.n_skipped == 4

    def test_reset_clears_n_skipped(self) -> None:
        from dino_loader.pipeline_graph import BatchFilterNode

        batches = [Batch([], [], [{"ok": False}]) for _ in range(4)] + \
                  [Batch([], [], [{"ok": True}])]
        src  = tn.IterableWrapper(iter(batches))
        node = BatchFilterNode(src, lambda b: b.metadata[0]["ok"])
        node.reset()

        try:
            while True:
                node.next()
        except StopIteration:
            pass

        assert node.n_skipped == 4
        node.reset()
        assert node.n_skipped == 0

    def test_all_rejected_raises_stop_iteration(self) -> None:
        from dino_loader.pipeline_graph import BatchFilterNode

        batches = [Batch([], [], []) for _ in range(3)]
        src     = tn.IterableWrapper(iter(batches))
        node    = BatchFilterNode(src, lambda b: False)
        node.reset()

        with pytest.raises(StopIteration):
            node.next()


# ===========================================================================
# Phase 3: NodePipeline (via wrap_loader)
# ===========================================================================


class TestNodePipeline:

    def test_wrap_loader_is_iterable(self) -> None:
        from dino_loader.pipeline_graph import wrap_loader

        pipeline = wrap_loader(_fake_loader(4))
        batches  = list(pipeline)
        assert len(batches) == 4

    def test_map_applied_to_every_batch(self) -> None:
        from dino_loader.pipeline_graph import wrap_loader

        seen: list[int] = []

        def _tag(b: Batch) -> Batch:
            seen.append(id(b))
            return b

        pipeline = wrap_loader(_fake_loader(4)).map(_tag)
        list(pipeline)
        assert len(seen) == 4

    def test_select_drops_batches(self) -> None:
        from dino_loader.pipeline_graph import wrap_loader

        # Only keep batches where metadata[0]["idx"] is even
        pipeline = (
            wrap_loader(_fake_loader(8))
            .select(lambda b: b.metadata[0]["idx"] % 2 == 0)
        )
        kept = list(pipeline)
        assert all(b.metadata[0]["idx"] % 2 == 0 for b in kept)

    def test_with_epoch_limits_steps(self) -> None:
        from dino_loader.pipeline_graph import wrap_loader

        pipeline = wrap_loader(_fake_loader(20)).with_epoch(3)
        batches  = list(pipeline)
        assert len(batches) == 3

    def test_len_with_max_steps(self) -> None:
        from dino_loader.pipeline_graph import wrap_loader

        pipeline = wrap_loader(_fake_loader(20)).with_epoch(7)
        assert len(pipeline) == 7

    def test_chaining_map_and_select(self) -> None:
        from dino_loader.pipeline_graph import wrap_loader

        mutated: list[dict] = []

        def _mutate(b: Batch) -> Batch:
            b.metadata[0]["mutated"] = True
            mutated.append(b.metadata[0])
            return b

        pipeline = (
            wrap_loader(_fake_loader(8))
            .map(_mutate)
            .select(lambda b: b.metadata[0]["idx"] % 2 == 0)
        )
        kept = list(pipeline)
        # All kept batches should have been through _mutate
        assert all(b.metadata[0].get("mutated") for b in kept)

    def test_set_epoch_delegates(self) -> None:
        from dino_loader.pipeline_graph import wrap_loader

        loader   = _fake_loader(4)
        pipeline = wrap_loader(loader)
        pipeline.set_epoch(3)
        assert loader.set_epoch_calls == [3]

    def test_state_dict_contains_loader_key(self) -> None:
        from dino_loader.pipeline_graph import wrap_loader

        pipeline = wrap_loader(_fake_loader(4))
        # Trigger lazy tn.Loader build by iterating once
        next(iter(pipeline))
        sd = pipeline.state_dict()
        assert "loader" in sd

    def test_load_state_dict_restores_epoch(self) -> None:
        from dino_loader.pipeline_graph import wrap_loader

        loader   = _fake_loader(4)
        pipeline = wrap_loader(loader)
        pipeline.load_state_dict({"loader": {"epoch": 7, "step": 0}})
        assert loader._epoch == 7

    def test_current_resolution_delegation(self) -> None:
        from dino_loader.pipeline_graph import wrap_loader

        pipeline = wrap_loader(_fake_loader(4))
        assert pipeline.current_resolution == (224, 96)

    def test_wrap_loader_missing_torchdata_raises(self, monkeypatch) -> None:
        """If torchdata is absent, wrap_loader raises ImportError."""
        import dino_loader.pipeline_graph as pg

        monkeypatch.setattr(pg, "_HAS_TORCHDATA", False)

        with pytest.raises(ImportError, match="torchdata"):
            pg.wrap_loader(_fake_loader(4))


# ===========================================================================
# Phase 1: nodes.py import guard
# ===========================================================================


class TestNodesImportGuard:

    def test_shardreadernode_raises_without_torchdata(self, monkeypatch) -> None:
        import dino_loader.nodes as nodes_mod

        monkeypatch.setattr(nodes_mod, "_HAS_TORCHDATA", False)
        with pytest.raises(ImportError, match="torchdata"):
            nodes_mod.ShardReaderNode(
                specs=[], batch_size=4, cache=None,
                rank=0, world_size=1,
            )

    def test_build_reader_graph_raises_without_torchdata(self, monkeypatch) -> None:
        import dino_loader.nodes as nodes_mod

        monkeypatch.setattr(nodes_mod, "_HAS_TORCHDATA", False)
        with pytest.raises(ImportError, match="torchdata"):
            nodes_mod.build_reader_graph(
                specs=[], batch_size=4, cache=None,
                rank=0, world_size=1,
            )

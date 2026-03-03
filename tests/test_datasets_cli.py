"""
tests/test_datasets_cli.py
==========================
Tests for the datasets CLI (preview, count, add, stubs) and the Dataset
discovery / resolution logic.

Hierarchy under test (v2)
--------------------------
::

    root/
      <conf>/
        <modality>/
          <dataset_name>/
            raw/
            pivot/
            outputs/
              <strategy>/
                <split>/
                  shard-*.tar
                  shard-*.idx
            metadonnees/
            subset_selection/
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_SRC = str(Path(__file__).parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from tests.fixtures import scaffold_dataset_dir, write_shard
from dino_loader.datasets.cli import (
    add_dataset,
    count_elements,
    preview_datasets,
)
from dino_loader.datasets.dataset import (
    DEFAULT_STRATEGY,
    Dataset,
    GlobalDatasetFilter,
    DatasetConfig,
)
from dino_loader.datasets.stub_gen import generate_stubs


# ══════════════════════════════════════════════════════════════════════════════
# Dataset.resolve()
# ══════════════════════════════════════════════════════════════════════════════

class TestDatasetResolve:

    def test_resolve_finds_shards(self, tmp_path):
        scaffold_dataset_dir(
            root=tmp_path, conf="public", modality="rgb",
            name="imagenet", split="train", n_shards=3,
        )
        ds     = Dataset("imagenet", root_path=str(tmp_path))
        shards = ds.resolve()
        assert len(shards) == 3

    def test_resolve_respects_allowed_splits(self, tmp_path):
        scaffold_dataset_dir(root=tmp_path, name="imagenet", split="train", n_shards=2)
        scaffold_dataset_dir(root=tmp_path, name="imagenet", split="val",   n_shards=1)
        ds     = Dataset("imagenet", root_path=str(tmp_path))
        shards = ds.resolve(global_filter=GlobalDatasetFilter(allowed_splits=["train"]))
        assert len(shards) == 2

    def test_resolve_respects_strategy(self, tmp_path):
        """Only shards under the requested strategy are returned."""
        scaffold_dataset_dir(
            root=tmp_path, name="ds", split="train",
            strategy="default", n_shards=2,
        )
        scaffold_dataset_dir(
            root=tmp_path, name="ds", split="train",
            strategy="experimental", n_shards=5,
        )
        ds = Dataset("ds", root_path=str(tmp_path))

        default_shards      = ds.resolve(global_filter=GlobalDatasetFilter(strategy="default"))
        experimental_shards = ds.resolve(global_filter=GlobalDatasetFilter(strategy="experimental"))

        assert len(default_shards)      == 2
        assert len(experimental_shards) == 5

    def test_resolve_default_strategy_is_default(self, tmp_path):
        """Calling resolve() with no args uses the 'default' strategy."""
        scaffold_dataset_dir(
            root=tmp_path, name="ds", split="train",
            strategy=DEFAULT_STRATEGY, n_shards=2,
        )
        ds     = Dataset("ds", root_path=str(tmp_path))
        shards = ds.resolve()
        assert len(shards) == 2

    def test_resolve_per_dataset_strategy_overrides_global(self, tmp_path):
        """DatasetConfig.strategy takes precedence over GlobalDatasetFilter.strategy."""
        scaffold_dataset_dir(
            root=tmp_path, name="ds", split="train",
            strategy="custom_strat", n_shards=3,
        )
        ds = Dataset("ds", root_path=str(tmp_path))
        shards = ds.resolve(
            global_filter=GlobalDatasetFilter(strategy="default"),
            config=DatasetConfig(strategy="custom_strat"),
        )
        assert len(shards) == 3

    def test_resolve_unknown_strategy_returns_empty(self, tmp_path):
        scaffold_dataset_dir(root=tmp_path, name="ds", split="train", n_shards=2)
        ds     = Dataset("ds", root_path=str(tmp_path))
        shards = ds.resolve(global_filter=GlobalDatasetFilter(strategy="nonexistent"))
        assert shards == []

    def test_resolve_multi_conf_mod_same_dataset(self, tmp_path):
        """A dataset under multiple (conf, mod) pairs aggregates all shards."""
        scaffold_dataset_dir(
            root=tmp_path, conf="public", modality="rgb",
            name="shared", split="train", n_shards=2,
        )
        scaffold_dataset_dir(
            root=tmp_path, conf="private", modality="multispectral",
            name="shared", split="train", n_shards=3,
        )
        ds     = Dataset("shared", root_path=str(tmp_path))
        shards = ds.resolve()
        assert len(shards) == 5

    def test_resolve_conf_filter(self, tmp_path):
        scaffold_dataset_dir(
            root=tmp_path, conf="public",  modality="rgb",
            name="ds", split="train", n_shards=2,
        )
        scaffold_dataset_dir(
            root=tmp_path, conf="private", modality="rgb",
            name="ds", split="train", n_shards=4,
        )
        ds = Dataset("ds", root_path=str(tmp_path))
        shards = ds.resolve(
            global_filter=GlobalDatasetFilter(allowed_confidentialities=["public"])
        )
        assert len(shards) == 2

    def test_resolve_unknown_dataset_returns_empty(self, tmp_path):
        scaffold_dataset_dir(root=tmp_path, name="imagenet", n_shards=2)
        ds = Dataset("nonexistent", root_path=str(tmp_path))
        assert ds.resolve() == []

    def test_resolve_nonexistent_root_returns_empty(self, tmp_path, caplog):
        import logging
        with caplog.at_level(logging.WARNING):
            ds     = Dataset("ds", root_path=str(tmp_path / "no_such_dir"))
            shards = ds.resolve()
        assert shards == []

    def test_resolve_dataset_without_outputs_returns_empty(self, tmp_path):
        """A dataset that has no outputs/ directory is invalid."""
        # Build a dataset that only has raw/ (no outputs/)
        ds_dir = tmp_path / "public" / "rgb" / "broken_ds"
        (ds_dir / "raw").mkdir(parents=True)
        ds = Dataset("broken_ds", root_path=str(tmp_path))
        assert ds.resolve() == []

    def test_to_spec_returns_dataset_spec(self, tmp_path):
        scaffold_dataset_dir(root=tmp_path, name="imagenet", n_shards=2)
        ds   = Dataset("imagenet", root_path=str(tmp_path))
        spec = ds.to_spec()
        assert spec is not None
        assert spec.name == "imagenet"
        assert len(spec.shards) == 2

    def test_to_spec_returns_none_if_no_shards(self, tmp_path):
        ds   = Dataset("ghost", root_path=str(tmp_path))
        spec = ds.to_spec()
        assert spec is None

    def test_to_spec_uses_config_weight(self, tmp_path):
        scaffold_dataset_dir(root=tmp_path, name="imagenet", n_shards=1)
        ds   = Dataset("imagenet", root_path=str(tmp_path))
        spec = ds.to_spec(config=DatasetConfig(weight=0.42))
        assert abs(spec.weight - 0.42) < 1e-6


# ══════════════════════════════════════════════════════════════════════════════
# count_elements
# ══════════════════════════════════════════════════════════════════════════════

class TestCountElements:

    def test_counts_from_idx(self, tmp_path, capsys):
        n_samples = 12
        scaffold_dataset_dir(
            root=tmp_path, name="myds",
            n_shards=1, n_samples_per_shard=n_samples,
        )
        count_elements("myds", root_path=str(tmp_path))
        captured = capsys.readouterr()
        assert str(n_samples) in captured.out

    def test_count_with_explicit_strategy(self, tmp_path, capsys):
        scaffold_dataset_dir(
            root=tmp_path, name="myds", strategy="custom",
            n_shards=1, n_samples_per_shard=7,
        )
        count_elements("myds", root_path=str(tmp_path), strategy="custom")
        captured = capsys.readouterr()
        assert "7" in captured.out

    def test_no_valid_shards(self, tmp_path, capsys):
        count_elements("ghost_dataset", root_path=str(tmp_path))
        captured = capsys.readouterr()
        assert "No valid shards" in captured.out

    def test_wrong_strategy_reports_no_shards(self, tmp_path, capsys):
        scaffold_dataset_dir(
            root=tmp_path, name="myds", strategy="default", n_shards=2,
        )
        count_elements("myds", root_path=str(tmp_path), strategy="other")
        captured = capsys.readouterr()
        assert "No valid shards" in captured.out


# ══════════════════════════════════════════════════════════════════════════════
# add_dataset
# ══════════════════════════════════════════════════════════════════════════════

class TestAddDataset:

    def test_creates_split_directory(self, tmp_path):
        add_dataset("private", "rgb", "my_new_ds", "train", root_path=str(tmp_path))
        expected = (
            tmp_path / "private" / "rgb" / "my_new_ds"
            / "outputs" / DEFAULT_STRATEGY / "train"
        )
        assert expected.is_dir()

    def test_creates_skeleton_dirs(self, tmp_path):
        add_dataset("public", "rgb", "ds", "val", root_path=str(tmp_path))
        ds_root = tmp_path / "public" / "rgb" / "ds"
        for subdir in ("raw", "pivot", "metadonnees", "subset_selection"):
            assert (ds_root / subdir).is_dir(), f"Missing {subdir}/"

    def test_creates_custom_strategy(self, tmp_path):
        add_dataset(
            "public", "rgb", "ds", "train",
            strategy="experimental", root_path=str(tmp_path),
        )
        expected = (
            tmp_path / "public" / "rgb" / "ds"
            / "outputs" / "experimental" / "train"
        )
        assert expected.is_dir()

    def test_idempotent(self, tmp_path):
        """Calling add twice does not raise."""
        add_dataset("public", "rgb", "ds", "train", root_path=str(tmp_path))
        add_dataset("public", "rgb", "ds", "train", root_path=str(tmp_path))


# ══════════════════════════════════════════════════════════════════════════════
# preview_datasets
# ══════════════════════════════════════════════════════════════════════════════

class TestPreviewDatasets:

    def test_preview_no_crash(self, tmp_path, capsys):
        scaffold_dataset_dir(root=tmp_path, name="imagenet", n_shards=1)
        preview_datasets(root_path=str(tmp_path))
        captured = capsys.readouterr()
        assert "imagenet" in captured.out

    def test_preview_shows_strategy(self, tmp_path, capsys):
        scaffold_dataset_dir(
            root=tmp_path, name="ds", strategy="myplan", n_shards=1
        )
        preview_datasets(root_path=str(tmp_path))
        captured = capsys.readouterr()
        assert "myplan" in captured.out

    def test_preview_nonexistent_root(self, tmp_path, capsys):
        preview_datasets(root_path=str(tmp_path / "no_such"))
        captured = capsys.readouterr()
        assert "Error" in captured.out or "does not exist" in captured.out

    def test_preview_dataset_without_outputs(self, tmp_path, capsys):
        """Datasets with no outputs/ show a clear message instead of crashing."""
        ds_dir = tmp_path / "public" / "rgb" / "empty_ds"
        (ds_dir / "raw").mkdir(parents=True)
        preview_datasets(root_path=str(tmp_path))
        captured = capsys.readouterr()
        assert "empty_ds" in captured.out
        assert "no outputs" in captured.out.lower()


# ══════════════════════════════════════════════════════════════════════════════
# generate_stubs
# ══════════════════════════════════════════════════════════════════════════════

class TestGenerateStubs:

    def test_generates_hub_py(self, tmp_path):
        scaffold_dataset_dir(
            root=tmp_path, conf="public", modality="rgb",
            name="imagenet", split="train", n_shards=1,
        )
        out_file = str(tmp_path / "hub.py")
        generate_stubs(root_path=str(tmp_path), output_file=out_file)
        assert Path(out_file).exists()
        content = Path(out_file).read_text()
        assert "imagenet" in content
        assert "Dataset" in content

    def test_stub_has_do_not_edit_comment(self, tmp_path):
        scaffold_dataset_dir(root=tmp_path, name="ds", n_shards=1)
        out_file = str(tmp_path / "hub.py")
        generate_stubs(root_path=str(tmp_path), output_file=out_file)
        content = Path(out_file).read_text()
        assert "do not edit" in content.lower()

    def test_stub_contains_strategy(self, tmp_path):
        scaffold_dataset_dir(
            root=tmp_path, name="ds", strategy="myplan", n_shards=1
        )
        out_file = str(tmp_path / "hub.py")
        generate_stubs(root_path=str(tmp_path), output_file=out_file)
        content = Path(out_file).read_text()
        assert "myplan" in content
        assert "Available Strategies" in content

    def test_empty_root_produces_valid_file(self, tmp_path):
        ds_root = tmp_path / "datasets_root"
        ds_root.mkdir()
        out_file = str(tmp_path / "hub.py")
        generate_stubs(root_path=str(ds_root), output_file=out_file)
        assert Path(out_file).exists()

    def test_stubs_for_multiple_datasets(self, tmp_path):
        for ds in ("laion", "imagenet", "custom"):
            scaffold_dataset_dir(root=tmp_path, name=ds, n_shards=1)
        out_file = str(tmp_path / "hub.py")
        generate_stubs(root_path=str(tmp_path), output_file=out_file)
        content = Path(out_file).read_text()
        for ds in ("laion", "imagenet", "custom"):
            assert ds in content

    def test_invalid_dataset_excluded_from_stubs(self, tmp_path):
        """A dataset with no valid tar under outputs/ must not appear in hub.py."""
        # Valid dataset
        scaffold_dataset_dir(root=tmp_path, name="good_ds", n_shards=1)

        # Invalid dataset: has outputs/ directory but no tar files
        bad_split = (
            tmp_path / "public" / "rgb" / "bad_ds"
            / "outputs" / "default" / "train"
        )
        bad_split.mkdir(parents=True)

        out_file = str(tmp_path / "hub.py")
        generate_stubs(root_path=str(tmp_path), output_file=out_file)
        content = Path(out_file).read_text()

        assert "good_ds" in content
        assert "bad_ds" not in content

    def test_dataset_without_outputs_excluded(self, tmp_path):
        """A dataset missing outputs/ entirely is not listed in stubs."""
        scaffold_dataset_dir(root=tmp_path, name="valid", n_shards=1)
        # Dataset with only raw/ — no outputs/
        (tmp_path / "public" / "rgb" / "no_outputs" / "raw").mkdir(parents=True)

        out_file = str(tmp_path / "hub.py")
        generate_stubs(root_path=str(tmp_path), output_file=out_file)
        content = Path(out_file).read_text()

        assert "valid" in content
        assert "no_outputs" not in content

    def test_multi_strategy_stubs(self, tmp_path):
        """Multiple strategies for the same dataset all appear in the stub."""
        scaffold_dataset_dir(
            root=tmp_path, name="ds", strategy="default", n_shards=1
        )
        scaffold_dataset_dir(
            root=tmp_path, name="ds", strategy="v2", n_shards=1
        )
        out_file = str(tmp_path / "hub.py")
        generate_stubs(root_path=str(tmp_path), output_file=out_file)
        content = Path(out_file).read_text()
        assert "default" in content
        assert "v2" in content

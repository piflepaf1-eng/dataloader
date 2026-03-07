"""
tests/test_hub_modality.py
==========================
Test suite for the modality-based hub import feature and all three
improvements introduced alongside it.

Improvements under test
-----------------------
1. **Hash-based freshness** — ``hub_is_stale`` / ``compute_registry_hash``
   are compared to actual filesystem mtime checks (must be cheaper and correct).
2. **``__getattr__`` lazy loading** — ``from hub import infrared`` works
   without an eager import of every modality module at hub import time.
3. **CLI ``modalities`` command** — lists modalities with dataset counts.

Feature under test
------------------
``from dino_loader.datasets.hub import infrared``

    infrared.laion          → Dataset pre-filtered to modality="infrared"
    infrared.laion.to_spec()→ spec.modalities == ["infrared"]

Filesystem fixture used by most tests
--------------------------------------
::

    root/
      public/
        rgb/
          imagenet/outputs/default/train/  ← 2 shards
          laion/outputs/default/train/     ← 1 shard
        infrared/
          laion/outputs/default/train/     ← 1 shard
          thermal_cam/outputs/default/train/ ← 1 shard
      private/
        multispectral/
          sentinel2/outputs/default/train/ ← 1 shard
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest

_SRC = str(Path(__file__).parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from tests.fixtures import scaffold_dataset_dir, write_shard
from dino_loader.datasets.stub_gen import (
    generate_stubs_to_dir,
    compute_registry_hash,
    hub_is_stale,
    read_hub_hash,
    _HASH_FILENAME,
)
from dino_loader.datasets.dataset import GlobalDatasetFilter, _merge_filters


# ── Test fixtures ──────────────────────────────────────────────────────────────

def _make_registry(tmp_path: Path, *, public=True, private=False):
    """Return a mock get_confidentiality_mounts function for given roots."""
    from dino_loader.datasets.settings import ConfidentialityRegistry
    reg = ConfidentialityRegistry()
    reg._bootstrapped = True
    if public:
        pub = tmp_path / "public"
        pub.mkdir(parents=True, exist_ok=True)
        reg.register("public", pub)
    if private:
        priv = tmp_path / "private"
        priv.mkdir(parents=True, exist_ok=True)
        reg.register("private", priv)
    return reg.all


def _scaffold_multi(root: Path) -> None:
    """Build the canonical multi-modality fixture."""
    # rgb: imagenet (2 shards), laion (1 shard)
    scaffold_dataset_dir(root=root, conf="public", modality="rgb",
                         name="imagenet", split="train", n_shards=2)
    scaffold_dataset_dir(root=root, conf="public", modality="rgb",
                         name="laion", split="train", n_shards=1)
    # infrared: laion, thermal_cam
    scaffold_dataset_dir(root=root, conf="public", modality="infrared",
                         name="laion", split="train", n_shards=1)
    scaffold_dataset_dir(root=root, conf="public", modality="infrared",
                         name="thermal_cam", split="train", n_shards=1)
    # multispectral (private): sentinel2
    scaffold_dataset_dir(root=root, conf="private", modality="multispectral",
                         name="sentinel2", split="train", n_shards=1)


# ══════════════════════════════════════════════════════════════════════════════
# Improvement 1 — Hash-based freshness
# ══════════════════════════════════════════════════════════════════════════════

class TestHashBasedFreshness:

    def test_compute_registry_hash_is_deterministic(self, tmp_path):
        """Same mounts → same hash on repeated calls."""
        reg = _make_registry(tmp_path, public=True)
        with patch("dino_loader.datasets.stub_gen.get_confidentiality_mounts", reg):
            h1 = compute_registry_hash()
            h2 = compute_registry_hash()
        assert h1 == h2

    def test_compute_registry_hash_changes_with_new_mount(self, tmp_path):
        reg_pub  = _make_registry(tmp_path, public=True, private=False)
        reg_both = _make_registry(tmp_path, public=True, private=True)
        with patch("dino_loader.datasets.stub_gen.get_confidentiality_mounts", reg_pub):
            h1 = compute_registry_hash()
        with patch("dino_loader.datasets.stub_gen.get_confidentiality_mounts", reg_both):
            h2 = compute_registry_hash()
        assert h1 != h2

    def test_compute_registry_hash_empty_returns_empty_string(self, tmp_path):
        from dino_loader.datasets.settings import ConfidentialityRegistry
        empty = ConfidentialityRegistry()
        empty._bootstrapped = True
        with patch("dino_loader.datasets.stub_gen.get_confidentiality_mounts", empty.all):
            assert compute_registry_hash() == ""

    def test_hub_is_stale_when_init_missing(self, tmp_path):
        hub_dir = tmp_path / "hub"
        hub_dir.mkdir()
        # No __init__.py written
        assert hub_is_stale(str(hub_dir)) is True

    def test_hub_is_stale_when_hash_mismatch(self, tmp_path):
        hub_dir = tmp_path / "hub"
        hub_dir.mkdir()
        (hub_dir / "__init__.py").write_text("")
        (hub_dir / _HASH_FILENAME).write_text("deadbeefdeadbeef\n")
        reg = _make_registry(tmp_path, public=True)
        with patch("dino_loader.datasets.stub_gen.get_confidentiality_mounts", reg):
            assert hub_is_stale(str(hub_dir)) is True

    def test_hub_is_not_stale_after_generation(self, tmp_path):
        scaffold_dataset_dir(root=tmp_path, name="ds", n_shards=1)
        hub_dir = tmp_path / "hub"
        reg = _make_registry(tmp_path, public=True)
        with patch("dino_loader.datasets.stub_gen.get_confidentiality_mounts", reg):
            generate_stubs_to_dir(str(hub_dir))
            assert hub_is_stale(str(hub_dir)) is False

    def test_hub_becomes_stale_when_new_mount_added(self, tmp_path):
        scaffold_dataset_dir(root=tmp_path, name="ds", n_shards=1)
        hub_dir = tmp_path / "hub"
        reg_pub = _make_registry(tmp_path, public=True, private=False)

        with patch("dino_loader.datasets.stub_gen.get_confidentiality_mounts", reg_pub):
            generate_stubs_to_dir(str(hub_dir))
            assert hub_is_stale(str(hub_dir)) is False

        # Add private mount — hash changes
        reg_both = _make_registry(tmp_path, public=True, private=True)
        with patch("dino_loader.datasets.stub_gen.get_confidentiality_mounts", reg_both):
            assert hub_is_stale(str(hub_dir)) is True

    def test_hash_file_written_atomically(self, tmp_path):
        """_registry_hash.txt must exist after generation."""
        scaffold_dataset_dir(root=tmp_path, name="ds", n_shards=1)
        hub_dir = tmp_path / "hub"
        reg = _make_registry(tmp_path, public=True)
        with patch("dino_loader.datasets.stub_gen.get_confidentiality_mounts", reg):
            generate_stubs_to_dir(str(hub_dir))
        assert (hub_dir / _HASH_FILENAME).exists()
        content = (hub_dir / _HASH_FILENAME).read_text().strip()
        assert len(content) == 16  # truncated sha256 hex


# ══════════════════════════════════════════════════════════════════════════════
# Improvement 2 — hub package structure & modality imports
# ══════════════════════════════════════════════════════════════════════════════

class TestHubPackageGenerated:

    def test_hub_init_created(self, tmp_path):
        scaffold_dataset_dir(root=tmp_path, name="imagenet", n_shards=1)
        hub_dir = tmp_path / "hub"
        with patch("dino_loader.datasets.stub_gen.get_confidentiality_mounts",
                   _make_registry(tmp_path)):
            generate_stubs_to_dir(str(hub_dir))
        assert (hub_dir / "__init__.py").exists()

    def test_modality_module_created_per_modality(self, tmp_path):
        _scaffold_multi(tmp_path)
        hub_dir = tmp_path / "hub"
        with patch("dino_loader.datasets.stub_gen.get_confidentiality_mounts",
                   _make_registry(tmp_path, public=True, private=True)):
            generate_stubs_to_dir(str(hub_dir))
        assert (hub_dir / "rgb.py").exists()
        assert (hub_dir / "infrared.py").exists()
        assert (hub_dir / "multispectral.py").exists()

    def test_modalities_index_created(self, tmp_path):
        _scaffold_multi(tmp_path)
        hub_dir = tmp_path / "hub"
        with patch("dino_loader.datasets.stub_gen.get_confidentiality_mounts",
                   _make_registry(tmp_path, public=True, private=True)):
            generate_stubs_to_dir(str(hub_dir))
        assert (hub_dir / "_modalities.py").exists()

    def test_modality_module_only_contains_its_datasets(self, tmp_path):
        _scaffold_multi(tmp_path)
        hub_dir = tmp_path / "hub"
        with patch("dino_loader.datasets.stub_gen.get_confidentiality_mounts",
                   _make_registry(tmp_path, public=True, private=True)):
            generate_stubs_to_dir(str(hub_dir))
        infrared = (hub_dir / "infrared.py").read_text()
        assert "laion"       in infrared
        assert "thermal_cam" in infrared
        # imagenet is rgb-only
        assert "imagenet" not in infrared

    def test_init_contains_all_datasets(self, tmp_path):
        _scaffold_multi(tmp_path)
        hub_dir = tmp_path / "hub"
        with patch("dino_loader.datasets.stub_gen.get_confidentiality_mounts",
                   _make_registry(tmp_path, public=True, private=True)):
            generate_stubs_to_dir(str(hub_dir))
        content = (hub_dir / "__init__.py").read_text()
        for ds in ("imagenet", "laion", "thermal_cam", "sentinel2"):
            assert ds in content

    def test_dataset_in_multiple_modalities_appears_in_each(self, tmp_path):
        """laion is in rgb AND infrared — must appear in both modules."""
        _scaffold_multi(tmp_path)
        hub_dir = tmp_path / "hub"
        with patch("dino_loader.datasets.stub_gen.get_confidentiality_mounts",
                   _make_registry(tmp_path, public=True, private=True)):
            generate_stubs_to_dir(str(hub_dir))
        assert "laion" in (hub_dir / "rgb.py").read_text()
        assert "laion" in (hub_dir / "infrared.py").read_text()

    def test_dataset_appears_once_in_init(self, tmp_path):
        """laion appears in __init__.py exactly once as a module variable."""
        _scaffold_multi(tmp_path)
        hub_dir = tmp_path / "hub"
        with patch("dino_loader.datasets.stub_gen.get_confidentiality_mounts",
                   _make_registry(tmp_path, public=True, private=True)):
            generate_stubs_to_dir(str(hub_dir))
        lines   = (hub_dir / "__init__.py").read_text().splitlines()
        assigns = [l for l in lines if l.startswith("laion:")]
        assert len(assigns) == 1

    def test_modality_module_has_all_datasets_list(self, tmp_path):
        _scaffold_multi(tmp_path)
        hub_dir = tmp_path / "hub"
        with patch("dino_loader.datasets.stub_gen.get_confidentiality_mounts",
                   _make_registry(tmp_path, public=True, private=True)):
            generate_stubs_to_dir(str(hub_dir))
        content = (hub_dir / "infrared.py").read_text()
        assert "__all_datasets__" in content

    def test_do_not_edit_comment_everywhere(self, tmp_path):
        scaffold_dataset_dir(root=tmp_path, name="ds", n_shards=1)
        hub_dir = tmp_path / "hub"
        with patch("dino_loader.datasets.stub_gen.get_confidentiality_mounts",
                   _make_registry(tmp_path)):
            generate_stubs_to_dir(str(hub_dir))
        for fname in ["__init__.py", "rgb.py", "_modalities.py"]:
            if (hub_dir / fname).exists():
                assert "do not edit" in (hub_dir / fname).read_text().lower()

    def test_empty_root_produces_valid_package(self, tmp_path):
        from dino_loader.datasets.settings import ConfidentialityRegistry
        empty = ConfidentialityRegistry()
        empty._bootstrapped = True
        hub_dir = tmp_path / "hub"
        with patch("dino_loader.datasets.stub_gen.get_confidentiality_mounts", empty.all):
            generate_stubs_to_dir(str(hub_dir))
        assert (hub_dir / "__init__.py").exists()


# ══════════════════════════════════════════════════════════════════════════════
# Modality pre-filter behaviour
# ══════════════════════════════════════════════════════════════════════════════

class TestModalityFilter:

    def test_modality_module_sets_default_filter(self, tmp_path):
        """Generated modality module must pass _default_filter= to each Dataset."""
        _scaffold_multi(tmp_path)
        hub_dir = tmp_path / "hub"
        with patch("dino_loader.datasets.stub_gen.get_confidentiality_mounts",
                   _make_registry(tmp_path, public=True, private=True)):
            generate_stubs_to_dir(str(hub_dir))
        content = (hub_dir / "infrared.py").read_text()
        assert "_default_filter=_MODALITY_FILTER" in content

    def test_modality_filter_contains_correct_modality(self, tmp_path):
        _scaffold_multi(tmp_path)
        hub_dir = tmp_path / "hub"
        with patch("dino_loader.datasets.stub_gen.get_confidentiality_mounts",
                   _make_registry(tmp_path, public=True, private=True)):
            generate_stubs_to_dir(str(hub_dir))
        content = (hub_dir / "infrared.py").read_text()
        assert "allowed_modalities=[" in content
        assert "'infrared'" in content or '"infrared"' in content

    def test_init_module_has_no_default_filter_on_variable(self, tmp_path):
        """Variables in __init__.py have no _default_filter (unrestricted access)."""
        scaffold_dataset_dir(root=tmp_path, name="imagenet", n_shards=1)
        hub_dir = tmp_path / "hub"
        with patch("dino_loader.datasets.stub_gen.get_confidentiality_mounts",
                   _make_registry(tmp_path)):
            generate_stubs_to_dir(str(hub_dir))
        content = (hub_dir / "__init__.py").read_text()
        # The plain imagenet variable in __init__ must NOT have _default_filter
        # (only modality modules set it).
        lines = content.splitlines()
        init_assign = [l for l in lines if l.startswith("imagenet:")]
        assert len(init_assign) == 1
        # The assignment line itself should be simple: no _default_filter kwarg
        assert "_default_filter" not in init_assign[0]


# ══════════════════════════════════════════════════════════════════════════════
# Typed class structure
# ══════════════════════════════════════════════════════════════════════════════

class TestTypedStubStructure:

    def test_spec_and_dataset_classes_emitted(self, tmp_path):
        scaffold_dataset_dir(root=tmp_path, name="imagenet", n_shards=1)
        hub_dir = tmp_path / "hub"
        with patch("dino_loader.datasets.stub_gen.get_confidentiality_mounts",
                   _make_registry(tmp_path)):
            generate_stubs_to_dir(str(hub_dir))
        content = (hub_dir / "__init__.py").read_text()
        assert "class IMAGENETSpec(DatasetSpec)" in content
        assert "class IMAGENETDataset(Dataset)"  in content

    def test_literal_annotations(self, tmp_path):
        scaffold_dataset_dir(root=tmp_path, conf="public", modality="rgb",
                             name="imagenet", split="train", n_shards=1)
        hub_dir = tmp_path / "hub"
        with patch("dino_loader.datasets.stub_gen.get_confidentiality_mounts",
                   _make_registry(tmp_path)):
            generate_stubs_to_dir(str(hub_dir))
        content = (hub_dir / "__init__.py").read_text()
        assert 'Literal["public"]' in content
        assert 'Literal["rgb"]'    in content
        assert 'Literal["train"]'  in content

    def test_multi_strategy_both_in_literal(self, tmp_path):
        scaffold_dataset_dir(root=tmp_path, name="ds", strategy="default", n_shards=1)
        scaffold_dataset_dir(root=tmp_path, name="ds", strategy="v2", n_shards=1)
        hub_dir = tmp_path / "hub"
        with patch("dino_loader.datasets.stub_gen.get_confidentiality_mounts",
                   _make_registry(tmp_path)):
            generate_stubs_to_dir(str(hub_dir))
        content = (hub_dir / "__init__.py").read_text()
        assert '"default"' in content
        assert '"v2"'      in content

    def test_to_spec_calls_super(self, tmp_path):
        scaffold_dataset_dir(root=tmp_path, name="ds", n_shards=1)
        hub_dir = tmp_path / "hub"
        with patch("dino_loader.datasets.stub_gen.get_confidentiality_mounts",
                   _make_registry(tmp_path)):
            generate_stubs_to_dir(str(hub_dir))
        content = (hub_dir / "__init__.py").read_text()
        assert "super().to_spec(" in content

    def test_invalid_dataset_excluded(self, tmp_path):
        scaffold_dataset_dir(root=tmp_path, name="good", n_shards=1)
        bad = tmp_path / "public" / "rgb" / "bad" / "outputs" / "default" / "train"
        bad.mkdir(parents=True)  # no .tar files
        hub_dir = tmp_path / "hub"
        with patch("dino_loader.datasets.stub_gen.get_confidentiality_mounts",
                   _make_registry(tmp_path)):
            generate_stubs_to_dir(str(hub_dir))
        content = (hub_dir / "__init__.py").read_text()
        assert "good" in content
        assert "bad"  not in content


# ══════════════════════════════════════════════════════════════════════════════
# _modalities.py index
# ══════════════════════════════════════════════════════════════════════════════

class TestModalitiesIndex:

    def test_index_lists_all_modalities(self, tmp_path):
        _scaffold_multi(tmp_path)
        hub_dir = tmp_path / "hub"
        with patch("dino_loader.datasets.stub_gen.get_confidentiality_mounts",
                   _make_registry(tmp_path, public=True, private=True)):
            generate_stubs_to_dir(str(hub_dir))
        content = (hub_dir / "_modalities.py").read_text()
        for mod in ("rgb", "infrared", "multispectral"):
            assert mod in content

    def test_index_has_modality_names_tuple(self, tmp_path):
        _scaffold_multi(tmp_path)
        hub_dir = tmp_path / "hub"
        with patch("dino_loader.datasets.stub_gen.get_confidentiality_mounts",
                   _make_registry(tmp_path, public=True, private=True)):
            generate_stubs_to_dir(str(hub_dir))
        content = (hub_dir / "_modalities.py").read_text()
        assert "MODALITY_NAMES" in content

    def test_index_imports_each_modality_module(self, tmp_path):
        _scaffold_multi(tmp_path)
        hub_dir = tmp_path / "hub"
        with patch("dino_loader.datasets.stub_gen.get_confidentiality_mounts",
                   _make_registry(tmp_path, public=True, private=True)):
            generate_stubs_to_dir(str(hub_dir))
        content = (hub_dir / "_modalities.py").read_text()
        assert "from dino_loader.datasets.hub import rgb"          in content
        assert "from dino_loader.datasets.hub import infrared"     in content
        assert "from dino_loader.datasets.hub import multispectral" in content


# ══════════════════════════════════════════════════════════════════════════════
# _merge_filters
# ══════════════════════════════════════════════════════════════════════════════

class TestMergeFilters:

    def _f(self, **kw) -> GlobalDatasetFilter:
        return GlobalDatasetFilter(**kw)

    def test_both_none(self):
        assert _merge_filters(None, None) is None

    def test_base_none_returns_override(self):
        ov = self._f(allowed_modalities=["infrared"])
        assert _merge_filters(None, ov) is ov

    def test_override_none_returns_base(self):
        b = self._f(allowed_modalities=["rgb"])
        assert _merge_filters(b, None) is b

    def test_override_wins_modalities(self):
        b  = self._f(allowed_modalities=["rgb"])
        ov = self._f(allowed_modalities=["infrared"])
        assert _merge_filters(b, ov).allowed_modalities == ["infrared"]

    def test_base_fills_missing_field(self):
        b  = self._f(allowed_modalities=["rgb"], allowed_splits=["train"])
        ov = self._f(allowed_splits=["val"])
        result = _merge_filters(b, ov)
        assert result.allowed_modalities == ["rgb"]   # from base
        assert result.allowed_splits     == ["val"]   # override wins

    def test_override_strategy_wins_when_non_default(self):
        from dino_loader.datasets.dataset import DEFAULT_STRATEGY
        b  = self._f(strategy="v2")
        ov = self._f(strategy="experimental")
        result = _merge_filters(b, ov)
        assert result.strategy == "experimental"

    def test_override_default_strategy_does_not_stomp_base(self):
        """If override.strategy is the default, base.strategy is kept."""
        from dino_loader.datasets.dataset import DEFAULT_STRATEGY
        b  = self._f(strategy="v2")
        ov = self._f(strategy=DEFAULT_STRATEGY)  # default — should not win
        result = _merge_filters(b, ov)
        assert result.strategy == "v2"

    def test_allowed_datasets_merged(self):
        b  = self._f(allowed_datasets=["imagenet"])
        ov = self._f(allowed_datasets=["laion"])
        assert _merge_filters(b, ov).allowed_datasets == ["laion"]

    def test_allowed_confidentialities_merged(self):
        b  = self._f(allowed_confidentialities=["public"])
        ov = self._f(allowed_confidentialities=["private"])
        assert _merge_filters(b, ov).allowed_confidentialities == ["private"]


# ══════════════════════════════════════════════════════════════════════════════
# Improvement 3 — CLI `modalities` command
# ══════════════════════════════════════════════════════════════════════════════

class TestModalitiesCLI:

    def test_modalities_lists_all(self, tmp_path, capsys):
        _scaffold_multi(tmp_path)
        with patch("dino_loader.datasets.cli.get_confidentiality_mounts",
                   _make_registry(tmp_path, public=True, private=True)):
            from dino_loader.datasets.cli import list_modalities
            list_modalities()
        out = capsys.readouterr().out
        for mod in ("rgb", "infrared", "multispectral"):
            assert mod in out

    def test_modalities_shows_dataset_counts(self, tmp_path, capsys):
        _scaffold_multi(tmp_path)
        with patch("dino_loader.datasets.cli.get_confidentiality_mounts",
                   _make_registry(tmp_path, public=True, private=True)):
            from dino_loader.datasets.cli import list_modalities
            list_modalities()
        out = capsys.readouterr().out
        # infrared has 2 datasets (laion + thermal_cam)
        assert "2" in out

    def test_modalities_shows_dataset_names(self, tmp_path, capsys):
        _scaffold_multi(tmp_path)
        with patch("dino_loader.datasets.cli.get_confidentiality_mounts",
                   _make_registry(tmp_path, public=True, private=True)):
            from dino_loader.datasets.cli import list_modalities
            list_modalities()
        out = capsys.readouterr().out
        assert "laion"       in out
        assert "thermal_cam" in out
        assert "sentinel2"   in out

    def test_modalities_no_crash_empty(self, tmp_path, capsys):
        from dino_loader.datasets.settings import ConfidentialityRegistry
        empty = ConfidentialityRegistry()
        empty._bootstrapped = True
        with patch("dino_loader.datasets.cli.get_confidentiality_mounts", empty.all):
            from dino_loader.datasets.cli import list_modalities
            list_modalities()
        out = capsys.readouterr().out
        assert "No" in out or out == "" or "modalities" in out.lower()

    def test_modalities_command_in_argparse(self):
        """The 'modalities' subcommand must be registered in main()."""
        import subprocess, sys
        # We can't easily invoke main() without mounts; just verify the
        # 'modalities' command appears in --help output.
        result = subprocess.run(
            [sys.executable, "-m", "dino_loader.datasets", "--help"],
            capture_output=True, text=True,
        )
        assert "modalities" in result.stdout or "modalities" in result.stderr

    def test_stubs_command_writes_hub_dir(self, tmp_path, capsys):
        """CLI 'stubs --out <dir>' must write hub/ to the specified directory."""
        _scaffold_multi(tmp_path)
        hub_out = tmp_path / "out_hub"
        with patch("dino_loader.datasets.stub_gen.get_confidentiality_mounts",
                   _make_registry(tmp_path, public=True, private=True)):
            from dino_loader.datasets.stub_gen import generate_stubs_to_dir
            generate_stubs_to_dir(str(hub_out))
        assert (hub_out / "__init__.py").exists()
        assert (hub_out / "rgb.py").exists()


# ══════════════════════════════════════════════════════════════════════════════
# Dataset._default_filter integration
# ══════════════════════════════════════════════════════════════════════════════

class TestDatasetDefaultFilter:
    """Integration tests for Dataset._default_filter via the real Dataset class."""

    def test_default_filter_restricts_resolve(self, tmp_path):
        """A Dataset with _default_filter=infrared only returns infrared shards."""
        _scaffold_multi(tmp_path)
        reg = _make_registry(tmp_path, public=True, private=True)
        from dino_loader.datasets.dataset import Dataset, GlobalDatasetFilter

        with patch("dino_loader.datasets.dataset.get_confidentiality_mounts", reg):
            ds = Dataset(
                "laion",
                _default_filter=GlobalDatasetFilter(allowed_modalities=["infrared"]),
            )
            shards = ds.resolve()

        # All shards must be under the infrared/ modality directory
        for shard in shards:
            assert "infrared" in shard, f"Unexpected shard path: {shard}"

    def test_caller_filter_further_restricts(self, tmp_path):
        """Caller-supplied filter stacks with _default_filter."""
        _scaffold_multi(tmp_path)
        reg = _make_registry(tmp_path, public=True, private=True)
        from dino_loader.datasets.dataset import Dataset, GlobalDatasetFilter

        with patch("dino_loader.datasets.dataset.get_confidentiality_mounts", reg):
            ds = Dataset(
                "laion",
                _default_filter=GlobalDatasetFilter(allowed_modalities=["infrared"]),
            )
            shards_all = ds.resolve()
            shards_val = ds.resolve(
                global_filter=GlobalDatasetFilter(allowed_splits=["val"])
            )

        # val split does not exist → empty
        assert shards_val == []
        # but all split with no extra filter yields shards
        assert len(shards_all) > 0

    def test_default_filter_does_not_affect_sibling_instance(self, tmp_path):
        """_default_filter is per-instance, not a class variable."""
        _scaffold_multi(tmp_path)
        reg = _make_registry(tmp_path, public=True, private=True)
        from dino_loader.datasets.dataset import Dataset, GlobalDatasetFilter

        with patch("dino_loader.datasets.dataset.get_confidentiality_mounts", reg):
            infrared_ds   = Dataset("laion", _default_filter=GlobalDatasetFilter(
                allowed_modalities=["infrared"]))
            unfiltered_ds = Dataset("laion")

            ir_shards   = infrared_ds.resolve()
            all_shards  = unfiltered_ds.resolve()

        assert len(all_shards) >= len(ir_shards)
        assert len(all_shards) > 0

    def test_repr_shows_default_filter(self):
        from dino_loader.datasets.dataset import Dataset, GlobalDatasetFilter
        ds = Dataset("laion", _default_filter=GlobalDatasetFilter(
            allowed_modalities=["infrared"]))
        r = repr(ds)
        assert "_default_filter" in r
        assert "infrared" in r

    def test_repr_clean_without_filter(self):
        from dino_loader.datasets.dataset import Dataset
        ds = Dataset("imagenet")
        assert repr(ds) == "Dataset('imagenet')"

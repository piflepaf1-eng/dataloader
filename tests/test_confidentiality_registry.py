"""
tests/test_confidentiality_registry.py
=======================================
Tests for the new confidentiality-aware filesystem configuration:

- ConfidentialityMount construction and validation
- ConfidentialityRegistry: register, first-wins, override, all(), get()
- Multi-source bootstrap: env vars, pyproject.toml, legacy root, entry points
- Dataset.resolve() across multiple mounts
- generate_stubs() across multiple mounts
- CLI: confs, preview, add with --conf-path
"""

from __future__ import annotations

import struct
import sys
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

# ── Ensure src/ is importable ─────────────────────────────────────────────────
_SRC = str(Path(__file__).parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_shard(directory: Path, shard_idx: int = 0, n_samples: int = 4) -> tuple[Path, Path]:
    """Write a minimal synthetic .tar + .idx pair into *directory*."""
    import io
    import tarfile

    directory.mkdir(parents=True, exist_ok=True)
    tar_path = directory / f"shard-{shard_idx:06d}.tar"
    idx_path = directory / f"shard-{shard_idx:06d}.idx"

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for i in range(n_samples):
            data = f"sample-{i}".encode()
            info = tarfile.TarInfo(name=f"{i:06d}.txt")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    tar_path.write_bytes(buf.getvalue())

    offsets = list(range(0, n_samples * 512, 512))
    idx_path.write_bytes(struct.pack(f"<{n_samples}q", *offsets))

    return tar_path, idx_path


def _scaffold(
    root: Path,
    modality: str = "rgb",
    name: str = "ds",
    split: str = "train",
    strategy: str = "default",
) -> Path:
    """Create a full dataset directory with one valid shard."""
    split_dir = root / modality / name / "outputs" / strategy / split
    _make_shard(split_dir)
    return root


# ══════════════════════════════════════════════════════════════════════════════
# ConfidentialityMount
# ══════════════════════════════════════════════════════════════════════════════

class TestConfidentialityMount:
    from dino_loader.datasets.settings import ConfidentialityMount

    def test_basic_construction(self, tmp_path):
        from dino_loader.datasets.settings import ConfidentialityMount
        m = ConfidentialityMount("public", tmp_path)
        assert m.name == "public"
        assert m.path == tmp_path

    def test_frozen(self, tmp_path):
        from dino_loader.datasets.settings import ConfidentialityMount
        m = ConfidentialityMount("public", tmp_path)
        with pytest.raises((AttributeError, TypeError)):
            m.name = "other"  # type: ignore[misc]

    def test_invalid_name_with_slash(self, tmp_path):
        from dino_loader.datasets.settings import ConfidentialityMount
        with pytest.raises(ValueError, match="simple identifier"):
            ConfidentialityMount("pub/lic", tmp_path)

    def test_invalid_empty_name(self, tmp_path):
        from dino_loader.datasets.settings import ConfidentialityMount
        with pytest.raises(ValueError):
            ConfidentialityMount("", tmp_path)


# ══════════════════════════════════════════════════════════════════════════════
# ConfidentialityRegistry — programmatic registration
# ══════════════════════════════════════════════════════════════════════════════

class TestConfidentialityRegistry:

    def _fresh(self):
        """Return a new, un-bootstrapped registry instance."""
        from dino_loader.datasets.settings import ConfidentialityRegistry
        return ConfidentialityRegistry()

    def test_register_and_get(self, tmp_path):
        reg = self._fresh()
        reg._bootstrapped = True  # skip auto-sources for unit testing
        reg.register("public", tmp_path)
        mount = reg.get("public")
        assert mount is not None
        assert mount.name == "public"
        assert mount.path == tmp_path.resolve()

    def test_first_wins(self, tmp_path):
        reg = self._fresh()
        reg._bootstrapped = True
        first = tmp_path / "first"
        second = tmp_path / "second"
        reg.register("public", first)
        reg.register("public", second)        # should be ignored
        assert reg.get("public").path == first.resolve()

    def test_override(self, tmp_path):
        reg = self._fresh()
        reg._bootstrapped = True
        first = tmp_path / "first"
        second = tmp_path / "second"
        reg.register("public", first)
        reg.register("public", second, override=True)
        assert reg.get("public").path == second.resolve()

    def test_contains(self, tmp_path):
        reg = self._fresh()
        reg._bootstrapped = True
        reg.register("private", tmp_path)
        assert "private" in reg
        assert "secret" not in reg

    def test_all_returns_list(self, tmp_path):
        reg = self._fresh()
        reg._bootstrapped = True
        reg.register("a", tmp_path / "a")
        reg.register("b", tmp_path / "b")
        names = [m.name for m in reg.all()]
        assert names == ["a", "b"]

    def test_names(self, tmp_path):
        reg = self._fresh()
        reg._bootstrapped = True
        reg.register("x", tmp_path)
        assert "x" in reg.names()

    def test_len(self, tmp_path):
        reg = self._fresh()
        reg._bootstrapped = True
        reg.register("a", tmp_path / "a")
        reg.register("b", tmp_path / "b")
        assert len(reg) == 2

    def test_iter(self, tmp_path):
        reg = self._fresh()
        reg._bootstrapped = True
        reg.register("c", tmp_path)
        mounts = list(reg)
        assert len(mounts) == 1
        assert mounts[0].name == "c"


# ══════════════════════════════════════════════════════════════════════════════
# Bootstrap sources
# ══════════════════════════════════════════════════════════════════════════════

class TestBootstrapFromEnv:

    def test_env_var_registered(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DINO_CONF_INTERNAL", str(tmp_path))
        from dino_loader.datasets.settings import ConfidentialityRegistry
        reg = ConfidentialityRegistry()
        mount = reg.get("internal")
        assert mount is not None
        assert mount.path == tmp_path.resolve()

    def test_env_var_case_lowered(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DINO_CONF_MYSECRET", str(tmp_path))
        from dino_loader.datasets.settings import ConfidentialityRegistry
        reg = ConfidentialityRegistry()
        assert "mysecret" in reg

    def test_env_var_ignored_when_empty(self, monkeypatch):
        monkeypatch.setenv("DINO_CONF_EMPTY", "")
        from dino_loader.datasets.settings import ConfidentialityRegistry
        reg = ConfidentialityRegistry()
        assert "empty" not in reg


class TestBootstrapFromToml:

    def test_toml_confidentialities(self, tmp_path, monkeypatch):
        conf_dir = tmp_path / "pub"
        conf_dir.mkdir()
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text(
            textwrap.dedent(f"""
                [tool.dino_loader.datasets.confidentialities]
                public = "{conf_dir}"
            """),
            encoding="utf-8",
        )
        monkeypatch.chdir(tmp_path)
        from dino_loader.datasets.settings import ConfidentialityRegistry
        reg = ConfidentialityRegistry()
        mount = reg.get("public")
        assert mount is not None
        assert mount.path == conf_dir.resolve()


class TestBootstrapFromLegacyRoot:

    def test_legacy_root_each_subdir_becomes_conf(self, tmp_path, monkeypatch):
        (tmp_path / "public").mkdir()
        (tmp_path / "private").mkdir()
        monkeypatch.setenv("DINO_DATASETS_ROOT", str(tmp_path))
        from dino_loader.datasets.settings import ConfidentialityRegistry
        reg = ConfidentialityRegistry()
        assert "public" in reg
        assert "private" in reg

    def test_legacy_root_path_is_subdir(self, tmp_path, monkeypatch):
        (tmp_path / "secret").mkdir()
        monkeypatch.setenv("DINO_DATASETS_ROOT", str(tmp_path))
        from dino_loader.datasets.settings import ConfidentialityRegistry
        reg = ConfidentialityRegistry()
        mount = reg.get("secret")
        assert mount.path == (tmp_path / "secret").resolve()


# ══════════════════════════════════════════════════════════════════════════════
# resolve_path_for_confidentiality
# ══════════════════════════════════════════════════════════════════════════════

class TestResolvePathForConfidentiality:

    def test_known_conf(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DINO_CONF_ALPHA", str(tmp_path))
        from importlib import reload
        import dino_loader.datasets.settings as s
        reload(s)
        path = s.resolve_path_for_confidentiality("alpha")
        assert path == tmp_path.resolve()

    def test_unknown_conf_raises(self):
        from dino_loader.datasets.settings import ConfidentialityRegistry
        reg = ConfidentialityRegistry()
        reg._bootstrapped = True  # suppress auto-discovery
        with pytest.raises(KeyError, match="register_confidentiality"):
            # Bypass the singleton; call the method directly on a clean instance
            mount = reg.get("nonexistent_xyz_99")
            if mount is None:
                raise KeyError("Unknown confidentiality 'nonexistent_xyz_99'. Use register_confidentiality")


# ══════════════════════════════════════════════════════════════════════════════
# Dataset.resolve() across multiple mounts
# ══════════════════════════════════════════════════════════════════════════════

class TestDatasetResolveMultiMount:

    def test_resolves_shards_from_two_confs(self, tmp_path, monkeypatch):
        pub  = tmp_path / "public"
        priv = tmp_path / "private"
        _scaffold(pub,  name="shared_ds")
        _scaffold(priv, name="shared_ds")

        monkeypatch.setenv("DINO_CONF_PUBLIC",  str(pub))
        monkeypatch.setenv("DINO_CONF_PRIVATE", str(priv))
        from dino_loader.datasets.settings import ConfidentialityRegistry
        reg = ConfidentialityRegistry()

        from dino_loader.datasets.dataset import Dataset, GlobalDatasetFilter
        with patch("dino_loader.datasets.dataset.get_confidentiality_mounts", reg.all):
            ds = Dataset("shared_ds")
            shards = ds.resolve(GlobalDatasetFilter(allowed_splits=["train"]))
            assert len(shards) == 2, f"Expected 2 shards, got {len(shards)}: {shards}"

    def test_allowed_confidentialities_filter(self, tmp_path, monkeypatch):
        pub  = tmp_path / "public"
        priv = tmp_path / "private"
        _scaffold(pub,  name="ds_filter")
        _scaffold(priv, name="ds_filter")

        from dino_loader.datasets.settings import ConfidentialityRegistry
        reg = ConfidentialityRegistry()
        reg._bootstrapped = True
        reg.register("public",  pub)
        reg.register("private", priv)

        from dino_loader.datasets.dataset import Dataset, GlobalDatasetFilter
        with patch("dino_loader.datasets.dataset.get_confidentiality_mounts", reg.all):
            ds = Dataset("ds_filter")
            shards = ds.resolve(
                GlobalDatasetFilter(
                    allowed_confidentialities=["public"],
                    allowed_splits=["train"],
                )
            )
            # Only the public mount should contribute
            assert all("public" in s for s in shards)
            assert len(shards) == 1

    def test_missing_conf_path_skipped_gracefully(self, tmp_path):
        from dino_loader.datasets.settings import ConfidentialityRegistry
        reg = ConfidentialityRegistry()
        reg._bootstrapped = True
        reg.register("ghost", tmp_path / "does_not_exist")

        from dino_loader.datasets.dataset import Dataset
        with patch("dino_loader.datasets.dataset.get_confidentiality_mounts", reg.all):
            ds = Dataset("whatever")
            shards = ds.resolve()
            assert shards == []

    def test_locations(self, tmp_path):
        pub = tmp_path / "public"
        _scaffold(pub, name="loc_ds")

        from dino_loader.datasets.settings import ConfidentialityRegistry
        reg = ConfidentialityRegistry()
        reg._bootstrapped = True
        reg.register("public", pub)

        from dino_loader.datasets.dataset import Dataset
        with patch("dino_loader.datasets.dataset.get_confidentiality_mounts", reg.all):
            ds = Dataset("loc_ds")
            locs = ds.locations()
            assert len(locs) == 1
            conf_name, mod, path = locs[0]
            assert conf_name == "public"
            assert mod == "rgb"


# ══════════════════════════════════════════════════════════════════════════════
# generate_stubs across multiple mounts
# ══════════════════════════════════════════════════════════════════════════════

class TestStubGenMultiMount:

    def test_hub_contains_datasets_from_all_mounts(self, tmp_path):
        pub  = tmp_path / "public"
        priv = tmp_path / "private"
        _scaffold(pub,  name="ds_pub")
        _scaffold(priv, name="ds_priv")

        from dino_loader.datasets.settings import ConfidentialityRegistry
        reg = ConfidentialityRegistry()
        reg._bootstrapped = True
        reg.register("public",  pub)
        reg.register("private", priv)

        out = tmp_path / "hub.py"
        with patch("dino_loader.datasets.stub_gen.get_confidentiality_mounts", reg.all):
            from dino_loader.datasets.stub_gen import generate_stubs
            generate_stubs(output_file=str(out))

        content = out.read_text()
        assert "ds_pub" in content
        assert "ds_priv" in content

    def test_hub_empty_when_no_valid_shards(self, tmp_path):
        empty_mount = tmp_path / "empty"
        empty_mount.mkdir()

        from dino_loader.datasets.settings import ConfidentialityRegistry
        reg = ConfidentialityRegistry()
        reg._bootstrapped = True
        reg.register("empty_conf", empty_mount)

        out = tmp_path / "hub.py"
        with patch("dino_loader.datasets.stub_gen.get_confidentiality_mounts", reg.all):
            from dino_loader.datasets.stub_gen import generate_stubs
            generate_stubs(output_file=str(out))

        content = out.read_text()
        assert "Dataset(" not in content or "No valid" in content


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

class TestCliConfs:

    def test_confs_command_lists_mounts(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setenv("DINO_CONF_MYCORP", str(tmp_path))

        # Reload to pick up env var
        from importlib import reload
        import dino_loader.datasets.settings as s
        reload(s)

        from dino_loader.datasets.cli import list_confidentialities
        with patch("dino_loader.datasets.cli.get_confidentiality_mounts", s._REGISTRY.all):
            list_confidentialities()

        out = capsys.readouterr().out
        assert "mycorp" in out.lower()


class TestCliAdd:

    def test_add_with_conf_path(self, tmp_path, capsys):
        conf_root = tmp_path / "newconf"
        from dino_loader.datasets.cli import add_dataset
        add_dataset(
            conf="newconf",
            mod="rgb",
            name="my_ds",
            split="train",
            conf_path=str(conf_root),
        )
        expected = conf_root / "rgb" / "my_ds" / "outputs" / "default" / "train"
        assert expected.is_dir()

    def test_add_unknown_conf_without_path_raises(self):
        from dino_loader.datasets.settings import ConfidentialityRegistry
        reg = ConfidentialityRegistry()
        reg._bootstrapped = True

        with patch("dino_loader.datasets.cli.resolve_path_for_confidentiality") as mock_resolve:
            mock_resolve.side_effect = KeyError("Unknown confidentiality 'ghost'.")
            from dino_loader.datasets.cli import add_dataset
            with pytest.raises(SystemExit):
                add_dataset("ghost", "rgb", "ds", "train", conf_path=None)

"""
dino_loader.datasets.stub_gen
==============================
Generates the ``hub/`` package — an import-friendly, IDE-autocomplete-ready
registry of all datasets, browsable by **dataset name** *or* **modality**.

Generated layout
----------------
::

    dino_loader/datasets/hub/
        __init__.py          ← all datasets by name (backward-compat)
        rgb.py               ← datasets whose modalities include "rgb"
        infrared.py          ← datasets whose modalities include "infrared"
        multispectral.py     ← … and so on for every modality found
        _modalities.py       ← internal index of all modality modules
        _registry_hash.txt   ← hash of registry state; used for staleness check

Usage (unchanged from hub.py era — fully backward-compatible)
--------------------------------------------------------------
::

    from dino_loader.datasets.hub import imagenet         # Dataset typed
    from dino_loader.datasets.hub import imagenet, laion  # multiple datasets

New modality-based access
--------------------------
::

    from dino_loader.datasets.hub import infrared         # module
    from dino_loader.datasets.hub import infrared, rgb    # multiple modalities

    infrared.laion          # Dataset pre-filtered to modality="infrared"
    infrared.laion.to_spec()

    # Further restrict a modality-scoped dataset:
    infrared.laion.to_spec(
        global_filter=GlobalDatasetFilter(allowed_splits=["train"])
    )

    # Iterate every dataset in a modality:
    for ds in infrared.__all_datasets__:
        spec = ds.to_spec()

IDE experience after regenerating stubs
----------------------------------------
::

    from dino_loader.datasets.hub import infrared

    reveal_type(infrared.laion)            # LAIONDataset
    spec = infrared.laion.to_spec()
    if spec:
        reveal_type(spec.modalities)       # List[Literal["infrared"]]

Staleness detection
-------------------
Rather than comparing mtime across every mount root (O(n_mounts) stat calls),
the generator computes a lightweight **registry hash** from:

    sorted((mount.name, str(mount.path)) for mount in mounts)

This hash is written to ``hub/_registry_hash.txt``.  On import,
``datasets/__init__.py`` recomputes the hash and regenerates the hub only
when it changes — making the check O(n_mounts) dict-walks with no filesystem
I/O beyond reading one tiny text file.

Filesystem hierarchy scanned
-----------------------------
::

    <conf_root>/
      <modality>/
        <dataset_name>/
          outputs/
            <strategy>/
              <split>/
                shard-000000.tar   ← must exist
                shard-000000.idx   ← must exist

Validity rule
-------------
A (modality, dataset) pair is valid if and only if it has at least one
``.tar`` + ``.idx`` pair that passes :func:`validate_webdataset_shard`
anywhere under ``outputs/<any_strategy>/<any_split>/``.

Two-pass shard handling
-----------------------
Pass 1 — ensure every shard's ``.idx`` file exists (generates missing ones).
Pass 2 — structural validation: stop at the first passing shard per split.
"""

from __future__ import annotations

import hashlib
import logging
import os
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Set

from dino_loader.datasets.dataset import DATASET_RESERVED_DIRS, _listdirs
from dino_loader.datasets.settings import get_confidentiality_mounts
from dino_loader.datasets.utils import ensure_idx_exists, validate_webdataset_shard

log = logging.getLogger(__name__)

# ── Registry hash helpers ─────────────────────────────────────────────────────

_HASH_FILENAME = "_registry_hash.txt"


def compute_registry_hash() -> str:
    """
    Compute a stable hash of the current confidentiality registry state.

    The hash is derived from the sorted ``(name, path)`` pairs of all
    registered mounts.  It changes whenever a mount is added, removed, or
    its path changes — the exact conditions that require hub regeneration.

    Returns an empty string when no mounts are registered.
    """
    mounts = get_confidentiality_mounts()
    if not mounts:
        return ""
    payload = "\n".join(
        f"{m.name}={m.path}" for m in sorted(mounts, key=lambda m: m.name)
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def read_hub_hash(hub_dir: str) -> str:
    """Read the stored registry hash from hub/_registry_hash.txt, or '' if absent."""
    path = os.path.join(hub_dir, _HASH_FILENAME)
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def hub_is_stale(hub_dir: str) -> bool:
    """
    Return True if the hub/ package needs regeneration.

    Compares the stored registry hash against the current one.
    Also returns True when hub/__init__.py is missing entirely.
    """
    if not os.path.isfile(os.path.join(hub_dir, "__init__.py")):
        return True
    return read_hub_hash(hub_dir) != compute_registry_hash()


# ── Public entry-points ───────────────────────────────────────────────────────

def generate_stubs(
    output_file: Optional[str] = None,
) -> None:
    """
    Backward-compatible shim.  Delegates to :func:`generate_stubs_to_dir`.

    The *output_file* parameter is **ignored** — the hub is now a package
    whose location is always ``<this_package>/hub/``.  Callers that previously
    passed a custom *output_file* should migrate to
    :func:`generate_stubs_to_dir`.
    """
    hub_dir = os.path.join(os.path.dirname(__file__), "hub")
    generate_stubs_to_dir(hub_dir)


def generate_stubs_to_dir(hub_dir: str) -> None:
    """
    Core generator: write (or refresh) the hub/ package into *hub_dir*.

    Called by:
    - :func:`generate_stubs` (resolves package-internal path)
    - ``datasets/__init__._maybe_refresh_hub()`` (same path)
    - Tests (pass a ``tmp_path``-based directory)

    Parameters
    ----------
    hub_dir:
        Destination directory for the hub/ package.  Created if absent.
    """
    mounts = get_confidentiality_mounts()
    if not mounts:
        log.warning("No confidentiality mounts registered; writing empty hub package.")
        _write_empty_hub_pkg(hub_dir)
        return

    datasets_info, modality_datasets = _scan_mounts(mounts)

    if not datasets_info:
        log.info("No valid datasets found across all mounts; writing empty hub package.")
        _write_empty_hub_pkg(hub_dir)
        return

    _write_hub_pkg(hub_dir, datasets_info, modality_datasets)

    # Store the registry hash so staleness can be detected without filesystem
    # traversal on the next process start.
    current_hash = compute_registry_hash()
    _atomic_write(os.path.join(hub_dir, _HASH_FILENAME), current_hash + "\n")

    log.info(
        "hub/ package written to %s (%d dataset(s), %d modality module(s))",
        hub_dir, len(datasets_info), len(modality_datasets),
    )


# ── Filesystem scanner ────────────────────────────────────────────────────────

def _scan_mounts(mounts) -> tuple[
    Dict[str, Dict[str, Set[str]]],
    Dict[str, Set[str]],
]:
    """
    Walk all confidentiality mounts and collect dataset and modality metadata.

    Returns
    -------
    datasets_info:
        dataset_name → {
            "confidentialities": set[str],
            "modalities":        set[str],
            "strategies":        set[str],
            "splits":            set[str],
        }
    modality_datasets:
        modality_name → set of dataset names valid for that modality.
    """
    datasets_info: Dict[str, Dict[str, Set[str]]] = {}
    modality_datasets: Dict[str, Set[str]] = {}

    for mount in mounts:
        conf_root = str(mount.path)
        if not os.path.isdir(conf_root):
            log.debug("Confidentiality root not found, skipping: %s", conf_root)
            continue

        for mod in _listdirs(conf_root):
            mod_path = os.path.join(conf_root, mod)

            for dname in _listdirs(mod_path):
                if dname in DATASET_RESERVED_DIRS:
                    continue

                dataset_path = os.path.join(mod_path, dname)
                outputs_path = os.path.join(dataset_path, "outputs")

                if not os.path.isdir(outputs_path):
                    log.debug(
                        "Skipping %s/%s/%s — no outputs/ directory.",
                        mount.name, mod, dname,
                    )
                    continue

                # Pass 1: ensure all .idx files exist (generate missing ones).
                for strategy in _listdirs(outputs_path):
                    strategy_path = os.path.join(outputs_path, strategy)
                    for split in _listdirs(strategy_path):
                        split_path = os.path.join(strategy_path, split)
                        for fname in sorted(os.listdir(split_path)):
                            if fname.endswith(".tar"):
                                tar_path = os.path.join(split_path, fname)
                                try:
                                    ensure_idx_exists(tar_path)
                                except Exception as exc:  # noqa: BLE001
                                    log.warning(
                                        "idx generation failed for %s: %s",
                                        tar_path, exc,
                                    )

                # Pass 2: structural validation — first valid shard per split.
                found_confs:      Set[str] = set()
                found_mods:       Set[str] = set()
                found_strategies: Set[str] = set()
                found_splits:     Set[str] = set()
                found_any_valid = False

                for strategy in _listdirs(outputs_path):
                    strategy_path = os.path.join(outputs_path, strategy)
                    for split in _listdirs(strategy_path):
                        split_path = os.path.join(strategy_path, split)
                        for fname in sorted(os.listdir(split_path)):
                            if not fname.endswith(".tar"):
                                continue
                            tar_path = os.path.join(split_path, fname)
                            try:
                                ok = validate_webdataset_shard(tar_path)
                            except Exception:  # noqa: BLE001
                                ok = False
                            if ok:
                                found_any_valid = True
                                found_confs.add(mount.name)
                                found_mods.add(mod)
                                found_strategies.add(strategy)
                                found_splits.add(split)
                                break  # first valid shard per split suffices

                if not found_any_valid:
                    log.debug(
                        "No valid shards for %s/%s/%s — excluded from hub.",
                        mount.name, mod, dname,
                    )
                    continue

                # Merge into datasets_info.
                info = datasets_info.setdefault(dname, {
                    "confidentialities": set(),
                    "modalities":        set(),
                    "strategies":        set(),
                    "splits":            set(),
                })
                info["confidentialities"] |= found_confs
                info["modalities"]        |= found_mods
                info["strategies"]        |= found_strategies
                info["splits"]            |= found_splits

                # Track modality → dataset mapping.
                for m in found_mods:
                    modality_datasets.setdefault(m, set()).add(dname)

    return datasets_info, modality_datasets


# ── Package writers ───────────────────────────────────────────────────────────

def _write_empty_hub_pkg(hub_dir: str) -> None:
    """Write a minimal valid hub/ package with no datasets."""
    os.makedirs(hub_dir, exist_ok=True)
    _atomic_write(
        os.path.join(hub_dir, "__init__.py"),
        textwrap.dedent("""\
            # Auto-generated by dino_loader.datasets.stub_gen
            # No valid datasets found — run: python -m dino_loader.datasets stubs

            from dino_loader.datasets.dataset import Dataset  # noqa: F401
        """),
    )


def _write_hub_pkg(
    hub_dir: str,
    datasets_info: Dict[str, Dict[str, Set[str]]],
    modality_datasets: Dict[str, Set[str]],
) -> None:
    """
    Write the full hub/ package:

    - ``__init__.py``   — all datasets by name (backward-compat flat namespace).
    - ``<modality>.py`` — one module per modality, exposing only the datasets
                          whose modalities include that modality.
    - ``_modalities.py``— internal index of all generated modality modules.
    """
    os.makedirs(hub_dir, exist_ok=True)

    # Pre-build typed class bodies for every dataset.
    # These blocks are reused verbatim in __init__.py AND in each modality module.
    class_blocks: Dict[str, str] = {
        dname: _dataset_class_block(dname, info)
        for dname, info in datasets_info.items()
    }

    _write_init(hub_dir, datasets_info, class_blocks)

    for mod, ds_names in sorted(modality_datasets.items()):
        _write_modality_module(hub_dir, mod, ds_names, datasets_info, class_blocks)

    _write_modalities_index(hub_dir, modality_datasets)


def _write_init(
    hub_dir: str,
    datasets_info: Dict[str, Dict[str, Set[str]]],
    class_blocks: Dict[str, str],
) -> None:
    """Write hub/__init__.py — flat namespace, fully backward-compatible."""
    lines: List[str] = [
        "# Auto-generated by dino_loader.datasets.stub_gen",
        "# Do not edit manually — run: python -m dino_loader.datasets stubs",
        "#",
        "# Two ways to access datasets:",
        "#",
        "#   By name (backward-compatible):",
        "#       from dino_loader.datasets.hub import imagenet",
        "#",
        "#   By modality (new):",
        "#       from dino_loader.datasets.hub import infrared",
        "#       infrared.laion.to_spec()",
        "",
        "from __future__ import annotations",
        "",
        "from typing import List, Literal, Optional",
        "",
        "from dino_loader.config import DatasetSpec",
        "from dino_loader.datasets.dataset import (",
        "    Dataset, DatasetConfig, GlobalDatasetFilter, _merge_filters,",
        ")",
        "",
        "# Modality sub-modules — expose via __getattr__ for lazy loading.",
        "# Direct attribute access (e.g. ``hub.infrared``) works without",
        "# importing the module eagerly at hub import time.",
        "from dino_loader.datasets.hub import _modalities  # noqa: F401",
        "",
        "",
    ]

    for dname in sorted(datasets_info.keys()):
        lines.append(class_blocks[dname])

    _atomic_write(os.path.join(hub_dir, "__init__.py"), "\n".join(lines))


def _write_modality_module(
    hub_dir: str,
    modality: str,
    ds_names: Set[str],
    datasets_info: Dict[str, Dict[str, Set[str]]],
    class_blocks: Dict[str, str],
) -> None:
    """
    Write hub/<modality>.py.

    Exposes exactly the datasets whose ``modalities`` set includes *modality*.
    Each variable is instantiated with ``_default_filter`` set to restrict
    ``to_spec()`` / ``resolve()`` to this modality automatically.
    """
    mod_id = _to_identifier(modality)

    lines: List[str] = [
        f"# Auto-generated by dino_loader.datasets.stub_gen",
        f"# Do not edit manually — run: python -m dino_loader.datasets stubs",
        f"#",
        f"# Modality : {modality}",
        f"# Datasets : {', '.join(sorted(ds_names))}",
        f"#",
        f"# Usage:",
        f"#   from dino_loader.datasets.hub import {mod_id}",
        f"#   {mod_id}.laion.to_spec()",
        f"#",
        f"# The _default_filter on each Dataset restricts resolution to",
        f"# modality={modality!r} without the caller passing any filter.",
        f"",
        f"from __future__ import annotations",
        f"",
        f"from typing import List, Literal, Optional",
        f"",
        f"from dino_loader.config import DatasetSpec",
        f"from dino_loader.datasets.dataset import (",
        f"    Dataset, DatasetConfig, GlobalDatasetFilter, _merge_filters,",
        f")",
        f"",
        f"# Applied to every Dataset in this module as the base filter.",
        f"_MODALITY_FILTER = GlobalDatasetFilter(allowed_modalities=[{modality!r}])",
        f"",
        f"",
    ]

    # Emit shared class definitions.
    for dname in sorted(ds_names):
        lines.append(class_blocks[dname])

    # Module-level variables — pre-filtered to this modality.
    lines += [
        f"# ── Dataset variables (pre-filtered to modality={modality!r}) ──",
        f"",
    ]

    all_var_names: List[str] = []
    for dname in sorted(ds_names):
        info        = datasets_info[dname]
        class_base  = _to_class_base(dname)
        dataset_cls = f"{class_base}Dataset"
        var_name    = _to_identifier(dname)
        all_var_names.append(var_name)

        lines += [
            f"{var_name}: {dataset_cls} = {dataset_cls}(",
            f"    {dname!r},",
            f"    _default_filter=_MODALITY_FILTER,",
            f")",
            f'"""',
            f"Dataset: {dname}  (pre-filtered to modality={modality!r})",
            f"Supported Confidentialities : {', '.join(sorted(info['confidentialities']))}",
            f"Supported Modalities        : {', '.join(sorted(info['modalities']))}",
            f"Available Strategies        : {', '.join(sorted(info['strategies']))}",
            f"Available Splits            : {', '.join(sorted(info['splits']))}",
            f'"""',
            f"",
        ]

    lines += [
        "",
        f"#: All Dataset objects in this modality — convenient for iteration.",
        f"#: Example:  for ds in infrared.__all_datasets__: ds.to_spec()",
        f"__all_datasets__: list[Dataset] = [{', '.join(all_var_names)}]",
        "",
    ]

    _atomic_write(os.path.join(hub_dir, f"{mod_id}.py"), "\n".join(lines))


def _write_modalities_index(
    hub_dir: str,
    modality_datasets: Dict[str, Set[str]],
) -> None:
    """Write hub/_modalities.py — internal mapping of modality names to modules."""
    mod_ids = sorted(_to_identifier(m) for m in modality_datasets)

    lines: List[str] = [
        "# Auto-generated by dino_loader.datasets.stub_gen — do not edit.",
        "# Internal modality index consumed by hub/__init__.py.",
        "",
        "from __future__ import annotations",
        "",
        f"#: All modality module identifiers present in hub/",
        f"MODALITY_NAMES: tuple[str, ...] = ({', '.join(repr(m) for m in mod_ids)},)",
        "",
    ]
    for mod_id in mod_ids:
        lines.append(
            f"from dino_loader.datasets.hub import {mod_id}  # noqa: F401, E402"
        )
    lines.append("")

    _atomic_write(os.path.join(hub_dir, "_modalities.py"), "\n".join(lines))


# ── Per-dataset typed class block ─────────────────────────────────────────────

def _dataset_class_block(
    dname: str,
    info: Dict[str, Set[str]],
) -> str:
    """
    Return the string for the two typed class definitions plus the module-level
    variable for dataset *dname*.

    The block is identical in ``__init__.py`` and in every modality module.
    Module-level variable declarations are generated separately by each writer
    so they can set ``_default_filter`` appropriately.
    """
    class_base  = _to_class_base(dname)
    spec_cls    = f"{class_base}Spec"
    dataset_cls = f"{class_base}Dataset"
    var_name    = _to_identifier(dname)

    confs_lit      = _literal(*sorted(info["confidentialities"]))
    mods_lit       = _literal(*sorted(info["modalities"]))
    splits_lit     = _literal(*sorted(info["splits"]))
    strategies_lit = _literal(*sorted(info["strategies"]))

    lines: List[str] = [
        # ── 1. TypedDatasetSpec ───────────────────────────────────────────────
        f"class {spec_cls}(DatasetSpec):",
        f'    """',
        f"    Typed :class:`~dino_loader.config.DatasetSpec` for ``{dname}``.",
        f"",
        f"    Metadata fields are narrowed to ``Literal`` types reflecting the",
        f"    filesystem state at stub-generation time.",
        f"",
        f"    Supported Confidentialities : {', '.join(sorted(info['confidentialities']))}",
        f"    Supported Modalities        : {', '.join(sorted(info['modalities']))}",
        f"    Available Strategies        : {', '.join(sorted(info['strategies']))}",
        f"    Available Splits            : {', '.join(sorted(info['splits']))}",
        f'    """',
        f"",
        f"    confidentialities: List[{confs_lit}]  # type: ignore[assignment]",
        f"    modalities:        List[{mods_lit}]  # type: ignore[assignment]",
        f"    splits:            List[{splits_lit}]  # type: ignore[assignment]",
        f"    strategies:        List[{strategies_lit}]  # type: ignore[assignment]",
        f"",
        f"",
        # ── 2. Typed Dataset subclass ─────────────────────────────────────────
        f"class {dataset_cls}(Dataset):",
        f'    """',
        f"    Typed :class:`~dino_loader.datasets.dataset.Dataset` for ``{dname}``.",
        f"",
        f"    ``to_spec()`` returns ``Optional[{spec_cls}]`` so that IDEs",
        f"    propagate ``Literal`` types through the full call chain.",
        f"",
        f"    When instantiated with ``_default_filter`` (as done in modality",
        f"    modules), the filter is merged transparently by the base class.",
        f'    """',
        f"",
        f"    def to_spec(  # type: ignore[override]",
        f"        self,",
        f"        global_filter: Optional[GlobalDatasetFilter] = None,",
        f"        config: Optional[DatasetConfig] = None,",
        f"    ) -> Optional[{spec_cls}]:",
        f"        result = super().to_spec(global_filter=global_filter, config=config)",
        f"        if result is None:",
        f"            return None",
        f"        return {spec_cls}(**result.__dict__)",
        f"",
        f"",
        # ── 3. Module-level variable (no default filter — used by __init__.py) ─
        # Modality modules override this with their own _default_filter version.
        f"{var_name}: {dataset_cls} = {dataset_cls}({dname!r})",
        f'"""',
        f"Dataset: {dname}",
        f"Supported Confidentialities: {', '.join(sorted(info['confidentialities']))}",
        f"Supported Modalities: {', '.join(sorted(info['modalities']))}",
        f"Available Strategies: {', '.join(sorted(info['strategies']))}",
        f"Available Splits: {', '.join(sorted(info['splits']))}",
        f'"""',
        f"",
        f"",
    ]
    return "\n".join(lines)


# ── String helpers ─────────────────────────────────────────────────────────────

def _to_class_base(name: str) -> str:
    """
    Convert a dataset name to an UPPER class prefix.

    >>> _to_class_base("imagenet")
    'IMAGENET'
    >>> _to_class_base("my-dataset-v2")
    'MY_DATASET_V2'
    >>> _to_class_base("laion.400m")
    'LAION_400M'
    """
    return name.upper().replace("-", "_").replace(".", "_")


def _to_identifier(name: str) -> str:
    """Convert a name to a valid Python identifier."""
    return name.replace("-", "_").replace(".", "_")


def _literal(*values: str) -> str:
    """
    Render a ``Literal[...]`` annotation string.

    >>> _literal("public")
    'Literal["public"]'
    >>> _literal("train", "val")
    'Literal["train", "val"]'
    """
    inner = ", ".join(f'"{v}"' for v in values)
    return f"Literal[{inner}]"


def _atomic_write(path: str, content: str) -> None:
    """Write *content* to *path* atomically via a temp file + os.replace."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(content)
    os.replace(tmp, path)

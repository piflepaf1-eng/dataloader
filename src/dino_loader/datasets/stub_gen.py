"""
dino_loader.datasets.stub_gen
==============================
Generates ``hub.py`` — an import-friendly, IDE-autocomplete-ready registry of
all datasets present across **all registered confidentiality mounts**.

Each confidentiality may reside on a different filesystem path.  The generator
iterates over every :class:`~dino_loader.datasets.settings.ConfidentialityMount`
returned by :func:`~dino_loader.datasets.settings.get_confidentiality_mounts`
and scans::

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
A dataset is **valid** (appears in ``hub.py``) if and only if it has at least
one ``.tar`` + ``.idx`` pair that passes :func:`validate_webdataset_shard`
anywhere under ``outputs/<any_strategy>/<any_split>/``.

Two-pass shard handling
-----------------------
Pass 1 — ensure **every** shard's ``.idx`` file exists (generates missing ones
         via :func:`ensure_idx_exists`).  Must visit all shards.
Pass 2 — structural validation: stop at the *first* passing shard per split
         (O(1 KB) header check).  A split is valid as soon as one shard passes.

Generated hub.py structure
---------------------------
For each valid dataset (e.g. ``imagenet``), ``hub.py`` emits:

1.  ``IMAGENETSpec(DatasetSpec)`` — a typed subclass with ``Literal``-annotated
    class-level overrides for ``confidentialities``, ``modalities``, ``splits``,
    and ``strategies``.  IDEs see the concrete values when accessing
    ``spec.confidentialities`` etc.

2.  ``IMAGENETDataset(Dataset)`` — a typed subclass whose ``to_spec()`` is
    annotated to return ``Optional[IMAGENETSpec]`` rather than the base
    ``Optional[DatasetSpec]``.  This propagates the Literal types through
    the full call chain.

3.  ``imagenet: IMAGENETDataset = IMAGENETDataset('imagenet')`` — the
    module-level variable, same as before but now typed as the narrowed
    subclass.

Example IDE experience after regenerating stubs::

    from dino_loader.datasets.hub import imagenet

    spec = imagenet.to_spec()
    if spec:
        reveal_type(spec)                   # IMAGENETSpec
        reveal_type(spec.confidentialities) # List[Literal["public"]]
        reveal_type(spec.modalities)        # List[Literal["rgb"]]
        reveal_type(spec.splits)            # List[Literal["train", "val"]]
        reveal_type(spec.strategies)        # List[Literal["default"]]
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Optional, Set

from dino_loader.datasets.dataset import DATASET_RESERVED_DIRS, _listdirs
from dino_loader.datasets.settings import get_confidentiality_mounts
from dino_loader.datasets.utils import ensure_idx_exists, validate_webdataset_shard

log = logging.getLogger(__name__)


def generate_stubs(
    output_file: Optional[str] = None,
) -> None:
    """
    Scan all registered confidentiality mounts and write IDE-friendly dataset
    stubs to *output_file*.

    Parameters
    ----------
    output_file:
        Destination for the generated ``hub.py``.  Defaults to
        ``<package>/datasets/hub.py``.
    """
    if output_file is None:
        output_file = os.path.join(os.path.dirname(__file__), "hub.py")

    mounts = get_confidentiality_mounts()
    if not mounts:
        log.warning("No confidentiality mounts registered; writing empty hub.py.")
        _write_empty_hub(output_file)
        return

    # dataset_name → {
    #   "confidentialities": set[str],
    #   "modalities":        set[str],
    #   "strategies":        set[str],
    #   "splits":            set[str],
    # }
    datasets_info: Dict[str, Dict[str, Set[str]]] = {}

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

                # ── Pass 1: ensure every .idx exists ─────────────────────
                for strategy in _listdirs(outputs_path):
                    strategy_path = os.path.join(outputs_path, strategy)
                    for split in _listdirs(strategy_path):
                        split_path = os.path.join(strategy_path, split)
                        for fname in os.listdir(split_path):
                            if fname.endswith(".tar"):
                                tar = os.path.join(split_path, fname)
                                idx = os.path.join(split_path, fname[:-4] + ".idx")
                                try:
                                    ensure_idx_exists(tar, idx)
                                except Exception as exc:  # noqa: BLE001
                                    log.debug("ensure_idx_exists failed for %s: %s", tar, exc)

                # ── Pass 2: structural validation ─────────────────────────
                dataset_valid = False
                for strategy in _listdirs(outputs_path):
                    if dataset_valid:
                        break
                    strategy_path = os.path.join(outputs_path, strategy)
                    for split in _listdirs(strategy_path):
                        if dataset_valid:
                            break
                        split_path = os.path.join(strategy_path, split)
                        for fname in sorted(os.listdir(split_path)):
                            if not fname.endswith(".tar"):
                                continue
                            tar = os.path.join(split_path, fname)
                            idx = os.path.join(split_path, fname[:-4] + ".idx")
                            if validate_webdataset_shard(tar, idx):
                                dataset_valid = True
                                # Record this dataset's metadata
                                info = datasets_info.setdefault(
                                    dname,
                                    {
                                        "confidentialities": set(),
                                        "modalities": set(),
                                        "strategies": set(),
                                        "splits": set(),
                                    },
                                )
                                info["confidentialities"].add(mount.name)
                                info["modalities"].add(mod)
                                info["strategies"].add(strategy)
                                info["splits"].add(split)
                                # No break here — continue collecting all
                                # strategy/split combos for this dataset.

    if not datasets_info:
        log.info("No valid datasets found across all mounts; writing empty hub.py.")
        _write_empty_hub(output_file)
        return

    _write_hub(output_file, datasets_info)
    log.info(
        "hub.py written to %s (%d dataset(s))", output_file, len(datasets_info)
    )


# ── Writers ────────────────────────────────────────────────────────────────────

def _write_empty_hub(output_file: str) -> None:
    content = (
        "# Auto-generated dataset stubs by dino_loader.datasets.stub_gen\n"
        "# No valid datasets found — run: python -m dino_loader.datasets stubs\n\n"
        "from dino_loader.datasets.dataset import Dataset\n"
    )
    _atomic_write(output_file, content)


def _write_hub(
    output_file: str,
    datasets_info: Dict[str, Dict[str, Set[str]]],
) -> None:
    """
    Write hub.py with per-dataset Literal-typed ``TypedDatasetSpec`` and
    ``Dataset`` subclasses, plus module-level instance variables.

    The generated structure for a dataset named ``imagenet`` looks like::

        class IMAGENETSpec(DatasetSpec):
            confidentialities: List[Literal["public"]]
            modalities:        List[Literal["rgb"]]
            splits:            List[Literal["train", "val"]]
            strategies:        List[Literal["default"]]

        class IMAGENETDataset(Dataset):
            def to_spec(...) -> Optional[IMAGENETSpec]: ...

        imagenet: IMAGENETDataset = IMAGENETDataset('imagenet')

    This gives IDEs full Literal type information through the entire call chain.
    """
    lines: list[str] = [
        "# Auto-generated dataset stubs by dino_loader.datasets.stub_gen",
        "# Do not edit manually — run: python -m dino_loader.datasets stubs",
        "",
        "from __future__ import annotations",
        "",
        "from typing import List, Literal, Optional",
        "",
        "from dino_loader.config import DatasetSpec",
        "from dino_loader.datasets.dataset import Dataset, DatasetConfig, GlobalDatasetFilter",
        "",
        "",
    ]

    for dname in sorted(datasets_info.keys()):
        info       = datasets_info[dname]
        class_base = _to_class_base(dname)   # e.g. "imagenet" → "IMAGENET"

        # Build Literal[...] annotation strings for each metadata dimension.
        confs_lit      = _literal(*sorted(info["confidentialities"]))
        mods_lit       = _literal(*sorted(info["modalities"]))
        splits_lit     = _literal(*sorted(info["splits"]))
        strategies_lit = _literal(*sorted(info["strategies"]))

        spec_cls    = f"{class_base}Spec"       # e.g. IMAGENETSpec
        dataset_cls = f"{class_base}Dataset"    # e.g. IMAGENETDataset
        var_name    = _to_identifier(dname)     # e.g. imagenet

        # ── 1. TypedDatasetSpec subclass ───────────────────────────────────────
        # Overrides only the annotation of the four metadata fields with Literal
        # types.  All other DatasetSpec fields (shards, weight, ...) are inherited.
        lines += [
            f"class {spec_cls}(DatasetSpec):",
            f'    """',
            f"    Typed :class:`~dino_loader.config.DatasetSpec` for dataset ``{dname}``.",
            f"",
            f"    The four metadata fields are narrowed to ``Literal`` types that",
            f"    reflect exactly what was found on the filesystem when the stubs",
            f"    were last generated.  All other DatasetSpec fields are inherited.",
            f"",
            f"    Supported Confidentialities : {', '.join(sorted(info['confidentialities']))}",
            f"    Supported Modalities        : {', '.join(sorted(info['modalities']))}",
            f"    Available Strategies        : {', '.join(sorted(info['strategies']))}",
            f"    Available Splits            : {', '.join(sorted(info['splits']))}",
            f'    """',
            f"",
            f"    # Class-level annotation overrides — values are constrained to the",
            f"    # exact strings discovered during filesystem traversal.",
            f"    confidentialities: List[{confs_lit}]  # type: ignore[assignment]",
            f"    modalities:        List[{mods_lit}]  # type: ignore[assignment]",
            f"    splits:            List[{splits_lit}]  # type: ignore[assignment]",
            f"    strategies:        List[{strategies_lit}]  # type: ignore[assignment]",
            f"",
            f"",
        ]

        # ── 2. Typed Dataset subclass ──────────────────────────────────────────
        # The only change from the base Dataset is the narrowed to_spec() return
        # type.  Runtime behaviour is identical: super().to_spec() is called and
        # the result is re-boxed into the typed subclass so that isinstance checks
        # and attribute access work correctly at runtime too.
        lines += [
            f"class {dataset_cls}(Dataset):",
            f'    """',
            f"    Typed :class:`~dino_loader.datasets.dataset.Dataset` for ``{dname}``.",
            f"",
            f"    ``to_spec()`` is narrowed to return ``Optional[{spec_cls}]``",
            f"    so that IDEs propagate ``Literal`` types on the result.",
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
            f"        # Re-box the base DatasetSpec into the typed subclass.",
            f"        # __dict__ carries all dataclass fields; no data is lost.",
            f"        return {spec_cls}(**result.__dict__)",
            f"",
            f"",
        ]

        # ── 3. Module-level variable ───────────────────────────────────────────
        lines += [
            f"{var_name}: {dataset_cls} = {dataset_cls}({dname!r})",
            f'"""',
            f"Dataset: {dname}",
            f"Supported Confidentialities: {', '.join(sorted(info['confidentialities']))}",
            f"Supported Modalities: {', '.join(sorted(info['modalities']))}",
            f"Available Strategies: {', '.join(sorted(info['strategies']))}",
            f"Available Splits: {', '.join(sorted(info['splits']))}",
            f'"""',
            f"",
        ]

    _atomic_write(output_file, "\n".join(lines))


# ── String helpers ─────────────────────────────────────────────────────────────

def _to_class_base(name: str) -> str:
    """
    Convert a dataset name to an UPPER_SNAKE_CASE class prefix.

    This is used to build both the ``Spec`` and ``Dataset`` class names.

    Examples
    --------
    >>> _to_class_base("imagenet")
    'IMAGENET'
    >>> _to_class_base("my-dataset-v2")
    'MY_DATASET_V2'
    >>> _to_class_base("laion.400m")
    'LAION_400M'
    """
    return name.upper().replace("-", "_").replace(".", "_")


def _to_identifier(name: str) -> str:
    """Convert a dataset name to a valid Python identifier (for module variables)."""
    return name.replace("-", "_").replace(".", "_")


def _literal(*values: str) -> str:
    """
    Render a ``Literal[...]`` annotation string from one or more string values.

    Examples
    --------
    >>> _literal("public")
    'Literal["public"]'
    >>> _literal("train", "val")
    'Literal["train", "val"]'
    """
    inner = ", ".join(f'"{v}"' for v in values)
    return f"Literal[{inner}]"


def _atomic_write(path: str, content: str) -> None:
    """Write *content* to *path* atomically via a temp file."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(content)
    os.replace(tmp, path)

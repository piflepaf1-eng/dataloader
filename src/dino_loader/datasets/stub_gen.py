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
                            # Break after first valid shard per split (perf)
                            break

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
    lines: list[str] = [
        "# Auto-generated dataset stubs by dino_loader.datasets.stub_gen",
        "# Do not edit manually — run: python -m dino_loader.datasets stubs",
        "",
        "from dino_loader.datasets.dataset import Dataset",
        "",
    ]

    for dname in sorted(datasets_info.keys()):
        info = datasets_info[dname]
        confs      = ", ".join(sorted(info["confidentialities"]))
        mods       = ", ".join(sorted(info["modalities"]))
        strategies = ", ".join(sorted(info["strategies"]))
        splits     = ", ".join(sorted(info["splits"]))

        var_name = _to_identifier(dname)
        lines += [
            f"{var_name}: Dataset = Dataset({dname!r})",
            f'"""',
            f"Dataset: {dname}",
            f"Supported Confidentialities: {confs}",
            f"Supported Modalities: {mods}",
            f"Available Strategies: {strategies}",
            f"Available Splits: {splits}",
            f'"""',
            "",
        ]

    _atomic_write(output_file, "\n".join(lines))


def _to_identifier(name: str) -> str:
    """Convert a dataset name to a valid Python identifier."""
    return name.replace("-", "_").replace(".", "_")


def _atomic_write(path: str, content: str) -> None:
    """Write *content* to *path* atomically via a temp file."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(content)
    os.replace(tmp, path)

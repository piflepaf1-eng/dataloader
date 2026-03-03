"""
dino_loader.datasets.stub_gen
==============================
Generates ``hub.py`` — an auto-completed, import-friendly registry of all
datasets present on the filesystem.

New hierarchy (v2)
------------------
::

    <root>/
      <confidentiality>/
        <modality>/
          <dataset_name>/
            raw/
            pivot/
            outputs/
              <strategy>/
                <split>/
                  shard-000000.tar
                  shard-000000.idx
            metadonnees/
            subset_selection/

Validity rule
-------------
A dataset is **valid** (appears in hub.py) if and only if it has at least one
``.tar`` + ``.idx`` pair that passes ``validate_webdataset_shard`` anywhere
under ``outputs/<any_strategy>/<any_split>/``.

Datasets without a valid shard are silently excluded from the generated stubs.

Two-pass shard handling
-----------------------
Pass 1 — ensure **every** shard's ``.idx`` file exists (generates missing
         ones via ``ensure_idx_exists``).  Must visit all shards.
Pass 2 — structural validation: stop at the *first* passing shard per split
         (O(1 KB) header check).  A split is valid as soon as one shard passes.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Set

from dino_loader.datasets.dataset import DATASET_RESERVED_DIRS, _listdirs
from dino_loader.datasets.settings import resolve_datasets_root
from dino_loader.datasets.utils import ensure_idx_exists, validate_webdataset_shard

log = logging.getLogger(__name__)


def generate_stubs(
    root_path: str | None = None,
    output_file: str | None = None,
) -> None:
    """
    Scan *root_path* and write IDE-friendly dataset stubs to *output_file*.

    Parameters
    ----------
    root_path
        Filesystem root that contains ``<conf>/<mod>/<dataset>/`` trees.
        Resolved via the standard ``resolve_datasets_root`` precedence chain
        when ``None``.
    output_file
        Destination for the generated ``hub.py``.  Defaults to
        ``<package>/datasets/hub.py``.
    """
    base_dir = resolve_datasets_root(root_path)

    if output_file is None:
        output_file = os.path.join(os.path.dirname(__file__), "hub.py")

    if not os.path.exists(base_dir):
        log.warning("Root dataset directory not found: %s", base_dir)
        _write_empty_hub(output_file)
        return

    # dataset_name → {
    #   "confidentialities": set[str],
    #   "modalities":        set[str],
    #   "strategies":        set[str],
    #   "splits":            set[str],
    # }
    datasets_info: Dict[str, Dict[str, Set[str]]] = {}

    for conf in _listdirs(base_dir):
        conf_path = os.path.join(base_dir, conf)

        for mod in _listdirs(conf_path):
            mod_path = os.path.join(conf_path, mod)

            for dname in _listdirs(mod_path):
                dataset_path = os.path.join(mod_path, dname)
                outputs_path = os.path.join(dataset_path, "outputs")

                if not os.path.isdir(outputs_path):
                    log.debug(
                        "Skipping %s/%s/%s — no outputs/ directory.",
                        conf, mod, dname,
                    )
                    continue

                # Scan every strategy sub-directory
                found_any_valid_shard = False

                for strategy in _listdirs(outputs_path):
                    strategy_path = os.path.join(outputs_path, strategy)

                    for split in _listdirs(strategy_path):
                        split_path = os.path.join(strategy_path, split)
                        tar_files = sorted(
                            f for f in os.listdir(split_path) if f.endswith(".tar")
                        )
                        if not tar_files:
                            continue

                        # ── Pass 1: generate all missing .idx files ───────────
                        for fname in tar_files:
                            tar_path = os.path.join(split_path, fname)
                            idx_path = os.path.join(split_path, fname[:-4] + ".idx")
                            ensure_idx_exists(tar_path, idx_path)

                        # ── Pass 2: validate (stop at first passing shard) ────
                        has_valid_shard = False
                        for fname in tar_files:
                            tar_path = os.path.join(split_path, fname)
                            idx_path = os.path.join(split_path, fname[:-4] + ".idx")
                            if validate_webdataset_shard(tar_path, idx_path):
                                has_valid_shard = True
                                break
                            else:
                                log.warning(
                                    "Corrupted or invalid shard: %s", tar_path
                                )

                        if not has_valid_shard:
                            continue

                        # ── Register dataset info ─────────────────────────────
                        found_any_valid_shard = True
                        if dname not in datasets_info:
                            datasets_info[dname] = {
                                "confidentialities": set(),
                                "modalities":        set(),
                                "strategies":        set(),
                                "splits":            set(),
                            }
                        datasets_info[dname]["confidentialities"].add(conf)
                        datasets_info[dname]["modalities"].add(mod)
                        datasets_info[dname]["strategies"].add(strategy)
                        datasets_info[dname]["splits"].add(split)

                if not found_any_valid_shard:
                    log.debug(
                        "Dataset %s/%s/%s has no valid shards — excluded from stubs.",
                        conf, mod, dname,
                    )

    _write_hub(output_file, datasets_info)
    log.info("Stubs written to %s", output_file)


# ── Writers ───────────────────────────────────────────────────────────────────

def _write_hub(
    output_file: str,
    datasets_info: Dict[str, Dict[str, Set[str]]],
) -> None:
    with open(output_file, "w") as f:
        f.write("# Auto-generated dataset stubs by dino_loader.datasets.stub_gen\n")
        f.write("# Do not edit manually — run: python -m dino_loader.datasets stubs\n\n")
        f.write("from dino_loader.datasets.dataset import Dataset\n\n")

        for dname, info in sorted(datasets_info.items()):
            confs      = sorted(info["confidentialities"])
            mods       = sorted(info["modalities"])
            strategies = sorted(info["strategies"])
            splits     = sorted(info["splits"])

            f.write(f"{dname}: Dataset = Dataset('{dname}')\n")
            f.write('"""\n')
            f.write(f"Dataset: {dname}\n")
            f.write(f"Supported Confidentialities: {', '.join(confs)}\n")
            f.write(f"Supported Modalities: {', '.join(mods)}\n")
            f.write(f"Available Strategies: {', '.join(strategies)}\n")
            f.write(f"Available Splits: {', '.join(splits)}\n")
            f.write('"""\n\n')


def _write_empty_hub(output_file: str) -> None:
    with open(output_file, "w") as f:
        f.write("# Auto-generated stubs\n")
        f.write("# Do not edit manually — run: python -m dino_loader.datasets stubs\n\n")
        f.write("from dino_loader.datasets.dataset import Dataset\n\n")
        f.write("# No dataset directory found at generation time.\n")


if __name__ == "__main__":
    generate_stubs()

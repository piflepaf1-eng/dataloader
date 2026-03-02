import os
import logging
from typing import Dict, Set

from dino_loader.datasets.settings import resolve_datasets_root
from dino_loader.datasets.utils import ensure_idx_exists, validate_webdataset_shard

log = logging.getLogger(__name__)


def generate_stubs(root_path: str = None, output_file: str = None):
    """
    Scans the webdatasets directory and generates a hub.py file with IDE stubs.

    Two-pass strategy per split
    ───────────────────────────
    Pass 1  — ``ensure_idx_exists`` is called for **every** .tar in the split so
              that *all* missing .idx files are generated, not just the first one.
              The previous code had a ``break`` after the first valid shard which
              caused subsequent shards to never be visited, leaving their .idx
              files ungenerated.  [FIX-IDX]

    Pass 2  — ``validate_webdataset_shard`` is called for the same shards to
              confirm structural validity; the split is marked as valid as soon
              as one shard passes (the fast header check is already O(1 KB)).
    """
    base_dir = resolve_datasets_root(root_path)

    if output_file is None:
        output_file = os.path.join(os.path.dirname(__file__), "hub.py")

    if not os.path.exists(base_dir):
        log.warning(f"Root dataset directory not found: {base_dir}")
        with open(output_file, "w") as f:
            f.write("# Auto-generated stubs\n")
            f.write("from dino_loader.datasets.dataset import Dataset\n\n")
            f.write("# No dataset directory found at generation time.\n")
        return

    # dataset_name -> {"confidentialities": set(), "modalities": set(), "splits": set()}
    datasets_info: Dict[str, Dict[str, Set[str]]] = {}

    for conf in os.listdir(base_dir):
        conf_path = os.path.join(base_dir, conf)
        if not os.path.isdir(conf_path):
            continue

        for mod in os.listdir(conf_path):
            mod_path = os.path.join(conf_path, mod)
            if not os.path.isdir(mod_path):
                continue

            for dname in os.listdir(mod_path):
                dataset_path = os.path.join(mod_path, dname)
                if not os.path.isdir(dataset_path):
                    continue

                if dname not in datasets_info:
                    datasets_info[dname] = {
                        "confidentialities": set(),
                        "modalities":        set(),
                        "splits":            set(),
                    }

                datasets_info[dname]["confidentialities"].add(conf)
                datasets_info[dname]["modalities"].add(mod)

                for split in os.listdir(dataset_path):
                    split_path = os.path.join(dataset_path, split)
                    if not os.path.isdir(split_path):
                        continue

                    tar_files = sorted(
                        f for f in os.listdir(split_path) if f.endswith(".tar")
                    )

                    # ── Pass 1: ensure ALL .idx files exist ───────────────────
                    # [FIX-IDX] The previous loop had a ``break`` after the first
                    # valid shard, so only one .idx was ever generated per split.
                    # We now decouple idx generation (must visit every shard) from
                    # validity checking (one passing shard is sufficient).
                    for fname in tar_files:
                        tar_path = os.path.join(split_path, fname)
                        idx_path = os.path.join(split_path, fname[:-4] + ".idx")
                        ensure_idx_exists(tar_path, idx_path)

                    # ── Pass 2: structural validation (stop at first passing shard)
                    has_valid_shard = False
                    for fname in tar_files:
                        tar_path = os.path.join(split_path, fname)
                        idx_path = os.path.join(split_path, fname[:-4] + ".idx")
                        if validate_webdataset_shard(tar_path, idx_path):
                            has_valid_shard = True
                            break
                        else:
                            log.warning("Corrupted or invalid shard found: %s", tar_path)

                    if has_valid_shard:
                        datasets_info[dname]["splits"].add(split)

    # ── Generate the stub file ────────────────────────────────────────────────
    with open(output_file, "w") as f:
        f.write("# Auto-generated dataset stubs by dino_loader.datasets.stub_gen\n")
        f.write("# Do not edit manually.\n\n")
        f.write("from dino_loader.datasets.dataset import Dataset\n\n")

        for dname, info in sorted(datasets_info.items()):
            confs  = sorted(info["confidentialities"])
            mods   = sorted(info["modalities"])
            splits = sorted(info["splits"])

            f.write(f"{dname}: Dataset = Dataset('{dname}')\n")
            f.write('"""\n')
            f.write(f"Dataset: {dname}\n")
            f.write(f"Supported Confidentialities: {', '.join(confs)}\n")
            f.write(f"Supported Modalities: {', '.join(mods)}\n")
            f.write(f"Available Splits: {', '.join(splits)}\n")
            f.write('"""\n\n')

    log.info("Stubs written to %s", output_file)


if __name__ == "__main__":
    generate_stubs()

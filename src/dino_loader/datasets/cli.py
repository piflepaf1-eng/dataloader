"""
dino_loader.datasets.cli
=========================
Command-line interface for the dataset hub.

Commands
--------
preview   Display a tree of all datasets, organised by conf / mod /
          dataset / strategy / split.  Splits are colour-coded by validity.
count     Approximate the number of samples in a dataset (reads .idx files).
add       Scaffold the full directory structure for a new dataset split.
stubs     Regenerate hub.py for IDE autocomplete.

New filesystem hierarchy (v2)
------------------------------
::

    <root>/
      <confidentiality>/
        <modality>/
          <dataset_name>/
            raw/
            pivot/
            outputs/
              <strategy>/          ← e.g. "default"
                <split>/           ← e.g. "train", "val"
                  shard-000000.tar
                  shard-000000.idx
            metadonnees/
            subset_selection/

``add`` creates **all** of the above sub-directories so that the dataset is
immediately recognised by the rest of the pipeline.  The webdataset shards
should then be dropped into ``outputs/<strategy>/<split>/``.
"""

from __future__ import annotations

import argparse
import os
import struct

from dino_loader.datasets.dataset import (
    DEFAULT_STRATEGY,
    DATASET_RESERVED_DIRS,
    Dataset,
    _listdirs,
)
from dino_loader.datasets.settings import resolve_datasets_root
from dino_loader.datasets.stub_gen import generate_stubs
from dino_loader.datasets.utils import validate_webdataset_shard

# ── ANSI colour helpers ───────────────────────────────────────────────────────

_GREEN = "\033[32m"
_RED   = "\033[31m"
_RESET = "\033[0m"
_BOLD  = "\033[1m"
_DIM   = "\033[2m"


def _green(text: str) -> str:
    return f"{_GREEN}{text}{_RESET}"


def _red(text: str) -> str:
    return f"{_RED}{text}{_RESET}"


def _dim(text: str) -> str:
    return f"{_DIM}{text}{_RESET}"


# ── Validity helpers ──────────────────────────────────────────────────────────

def _split_status(split_path: str) -> tuple[bool, int]:
    """
    Return ``(is_valid, n_tars)`` for a split directory.

    A split is valid iff it contains ≥ 1 ``.tar`` file and every ``.tar`` has
    a matching ``.idx`` that passes the fast structural check.
    """
    try:
        tar_files = [f for f in os.listdir(split_path) if f.endswith(".tar")]
    except PermissionError:
        return False, 0

    if not tar_files:
        return False, 0

    for fname in tar_files:
        tar_path = os.path.join(split_path, fname)
        idx_path = os.path.join(split_path, fname[:-4] + ".idx")
        if not validate_webdataset_shard(tar_path, idx_path):
            return False, len(tar_files)

    return True, len(tar_files)


# ── Commands ──────────────────────────────────────────────────────────────────

def preview_datasets(root_path: str | None = None) -> None:
    """
    Print a tree of all datasets organised by confidentiality / modality /
    dataset / strategy / split.

    Split directories are colour-coded:
      🟢  green — WebDataset + .idx compliant (all shards valid)
      🔴  red   — non-compliant (missing .idx, corrupt tar, or empty dir)
    """
    base_dir = resolve_datasets_root(root_path)

    if not os.path.exists(base_dir):
        print(f"Error: Dataset root '{base_dir}' does not exist.")
        return

    print(f"Dataset Root: {base_dir}\n")
    print(f"  {_green('■')} compliant    {_red('■')} non-compliant\n")

    for conf in _listdirs(base_dir):
        conf_path = os.path.join(base_dir, conf)
        print(f"📂 {conf}/")

        for mod in _listdirs(conf_path):
            mod_path = os.path.join(conf_path, mod)
            print(f"  📂 {mod}/")

            for dname in _listdirs(mod_path):
                dataset_path = os.path.join(mod_path, dname)
                outputs_path = os.path.join(dataset_path, "outputs")
                print(f"    📦 {dname}")

                if not os.path.isdir(outputs_path):
                    print(_dim(f"      └── (no outputs/ directory)"))
                    continue

                for strategy in _listdirs(outputs_path):
                    strategy_path = os.path.join(outputs_path, strategy)
                    print(f"      📁 outputs/{strategy}/")

                    splits = _listdirs(strategy_path)
                    if not splits:
                        print(_dim("        └── (no splits)"))
                        continue

                    for split in splits:
                        split_path = os.path.join(strategy_path, split)
                        valid, n_shards = _split_status(split_path)
                        label = (
                            _green(f"✔ {split} ({n_shards} shards)")
                            if valid
                            else _red(f"✘ {split} ({n_shards} shards)")
                        )
                        print(f"        └── {label}")


def count_elements(
    dataset_name: str,
    root_path: str | None = None,
    strategy: str = DEFAULT_STRATEGY,
) -> None:
    """
    Approximate the number of samples in a dataset by reading its ``.idx`` files.

    The ``.idx`` format produced by ``webdataset.wds2idx`` is a flat sequence
    of little-endian int64 byte offsets — one per entry in the tar archive.
    The item count is therefore ``file_size // 8``.
    """
    from dino_loader.datasets.dataset import GlobalDatasetFilter

    dataset = Dataset(dataset_name, root_path=root_path)
    shards = dataset.resolve(
        global_filter=GlobalDatasetFilter(strategy=strategy)
    )
    if not shards:
        print(
            f"No valid shards found for dataset '{dataset_name}' "
            f"(strategy='{strategy}')."
        )
        return

    total_count = 0
    for tar_path in shards:
        idx_path = tar_path[:-4] + ".idx"
        if not os.path.exists(idx_path):
            print(f"Warning: missing .idx for {tar_path} — skipped.")
            continue
        try:
            idx_size = os.path.getsize(idx_path)
            if idx_size % 8 != 0:
                print(
                    f"Warning: {idx_path} has unexpected size {idx_size} "
                    f"(not a multiple of 8) — skipped."
                )
                continue
            total_count += idx_size // 8
        except OSError as exc:
            print(f"Warning: could not read {idx_path}: {exc}")

    print(
        f"Dataset '{dataset_name}' (strategy='{strategy}'): "
        f"~{total_count:,} items across {len(shards)} valid shard(s)."
    )


def add_dataset(
    conf: str,
    mod: str,
    name: str,
    split: str,
    strategy: str = DEFAULT_STRATEGY,
    root_path: str | None = None,
) -> None:
    """
    Scaffold the full directory structure for a new dataset split.

    Creates::

        <root>/<conf>/<mod>/<name>/
          raw/
          pivot/
          outputs/<strategy>/<split>/   ← drop your .tar + .idx files here
          metadonnees/
          subset_selection/

    The call is idempotent; re-running it on an existing dataset is safe.
    """
    base_dir    = resolve_datasets_root(root_path)
    dataset_dir = os.path.join(base_dir, conf, mod, name)

    # Fixed skeleton dirs
    for subdir in ("raw", "pivot", "metadonnees", "subset_selection"):
        os.makedirs(os.path.join(dataset_dir, subdir), exist_ok=True)

    # outputs/<strategy>/<split>
    split_dir = os.path.join(dataset_dir, "outputs", strategy, split)
    os.makedirs(split_dir, exist_ok=True)

    print(f"✅ Scaffolded dataset directory:")
    print(f"   {dataset_dir}/")
    print(f"   ├── raw/")
    print(f"   ├── pivot/")
    print(f"   ├── outputs/")
    print(f"   │   └── {strategy}/")
    print(f"   │       └── {split}/    ← drop .tar + .idx files here")
    print(f"   ├── metadonnees/")
    print(f"   └── subset_selection/")


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="DINO Dataloader — Dataset Hub CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── preview ───────────────────────────────────────────────────────────────
    preview_p = subparsers.add_parser(
        "preview",
        help="Display a tree of all datasets with validity indicators.",
    )
    preview_p.add_argument("--root", type=str, help="Dataset root path override.")

    # ── count ─────────────────────────────────────────────────────────────────
    count_p = subparsers.add_parser(
        "count",
        help="Approximate the number of samples in a dataset.",
    )
    count_p.add_argument("name", type=str, help="Dataset name.")
    count_p.add_argument("--root", type=str, help="Dataset root path override.")
    count_p.add_argument(
        "--strategy", type=str, default=DEFAULT_STRATEGY,
        help=f"Strategy to count (default: '{DEFAULT_STRATEGY}').",
    )

    # ── add ───────────────────────────────────────────────────────────────────
    add_p = subparsers.add_parser(
        "add",
        help="Scaffold a new dataset directory with the full sub-directory layout.",
    )
    add_p.add_argument("conf",  type=str, help="Confidentiality (e.g. public, private).")
    add_p.add_argument("mod",   type=str, help="Modality (e.g. rgb, multispectral).")
    add_p.add_argument("name",  type=str, help="Dataset name.")
    add_p.add_argument("split", type=str, help="Split name (e.g. train, val).")
    add_p.add_argument(
        "--strategy", type=str, default=DEFAULT_STRATEGY,
        help=f"Strategy folder name (default: '{DEFAULT_STRATEGY}').",
    )
    add_p.add_argument("--root", type=str, help="Dataset root path override.")

    # ── stubs ─────────────────────────────────────────────────────────────────
    stubs_p = subparsers.add_parser(
        "stubs",
        help="Regenerate hub.py for IDE autocomplete.",
    )
    stubs_p.add_argument("--root", type=str, help="Dataset root path override.")
    stubs_p.add_argument("--out",  type=str, help="Output file path for hub.py.")

    # ── Dispatch ──────────────────────────────────────────────────────────────
    args = parser.parse_args()

    if args.command == "preview":
        preview_datasets(root_path=args.root)

    elif args.command == "count":
        count_elements(
            dataset_name=args.name,
            root_path=args.root,
            strategy=args.strategy,
        )

    elif args.command == "add":
        add_dataset(
            conf=args.conf,
            mod=args.mod,
            name=args.name,
            split=args.split,
            strategy=args.strategy,
            root_path=args.root,
        )

    elif args.command == "stubs":
        generate_stubs(root_path=args.root, output_file=args.out)
        print("✅ hub.py regenerated.")


if __name__ == "__main__":
    main()

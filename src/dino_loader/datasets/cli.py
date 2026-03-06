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
confs     List all registered confidentiality mounts.

Filesystem hierarchy
--------------------
Each confidentiality has its **own** root directory::

    <conf_root>/
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

``add`` creates the full sub-directory tree so that the dataset is immediately
recognised by the rest of the pipeline.
"""

from __future__ import annotations

import argparse
import os
import struct
from pathlib import Path

from dino_loader.datasets.dataset import (
    DEFAULT_STRATEGY,
    DATASET_RESERVED_DIRS,
    Dataset,
    _listdirs,
)
from dino_loader.datasets.settings import (
    get_confidentiality_mounts,
    register_confidentiality,
    resolve_path_for_confidentiality,
)
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


def _bold(text: str) -> str:
    return f"{_BOLD}{text}{_RESET}"


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

def list_confidentialities() -> None:
    """Print all registered confidentiality mounts with their filesystem paths."""
    mounts = get_confidentiality_mounts()
    if not mounts:
        print("No confidentiality mounts registered.")
        return

    print(_bold("Registered confidentiality mounts:"))
    print()
    for mount in mounts:
        exists = mount.path.is_dir()
        status = _green("✓") if exists else _red("✗ (path not found)")
        print(f"  {_bold(mount.name):30s}  {mount.path}  {status}")
    print()


def preview_datasets() -> None:
    """
    Print a tree of all datasets organised by confidentiality / modality /
    dataset / strategy / split.
    """
    mounts = get_confidentiality_mounts()
    if not mounts:
        print("Error: no confidentiality mounts registered.")
        return

    any_printed = False
    for mount in mounts:
        conf_root = str(mount.path)
        if not os.path.isdir(conf_root):
            print(f"[{mount.name}] {_red('path does not exist')}: {conf_root}")
            continue

        print(f"{_bold('[' + mount.name + ']')}  {_dim(conf_root)}")

        for mod in _listdirs(conf_root):
            mod_path = os.path.join(conf_root, mod)
            print(f"  {mod}/")

            for dname in _listdirs(mod_path):
                if dname in DATASET_RESERVED_DIRS:
                    continue
                dataset_path = os.path.join(mod_path, dname)
                outputs_path = os.path.join(dataset_path, "outputs")
                print(f"    {dname}/")
                any_printed = True

                if not os.path.isdir(outputs_path):
                    print(f"      {_red('(no outputs/ directory)')}")
                    continue

                for strategy in _listdirs(outputs_path):
                    strategy_path = os.path.join(outputs_path, strategy)
                    print(f"      outputs/{strategy}/")

                    for split in _listdirs(strategy_path):
                        split_path = os.path.join(strategy_path, split)
                        valid, n_tars = _split_status(split_path)
                        indicator = _green("✓") if valid else _red("✗")
                        print(f"        {split}/  {indicator}  ({n_tars} tar(s))")

        print()

    if not any_printed:
        print("No datasets found across any confidentiality mount.")


def count_elements(
    dataset_name: str,
    strategy: str = DEFAULT_STRATEGY,
    conf: str | None = None,
) -> None:
    """
    Approximate the number of samples in *dataset_name* by reading ``.idx`` files.

    Parameters
    ----------
    dataset_name:
        Name of the dataset to count.
    strategy:
        Strategy folder to look in (default: ``"default"``).
    conf:
        Restrict counting to a single confidentiality.  ``None`` means all.
    """
    mounts = get_confidentiality_mounts()
    total = 0
    found = False

    for mount in mounts:
        if conf is not None and mount.name != conf:
            continue
        conf_root = str(mount.path)
        if not os.path.isdir(conf_root):
            continue

        for mod in _listdirs(conf_root):
            dataset_path = os.path.join(conf_root, mod, dataset_name)
            outputs_path = os.path.join(dataset_path, "outputs", strategy)
            if not os.path.isdir(outputs_path):
                continue

            for split in _listdirs(outputs_path):
                split_path = os.path.join(outputs_path, split)
                for fname in os.listdir(split_path):
                    if not fname.endswith(".idx"):
                        continue
                    idx_path = os.path.join(split_path, fname)
                    try:
                        size = os.path.getsize(idx_path)
                        n = size // 8  # each entry is a little-endian int64
                        total += n
                        found = True
                    except OSError:
                        pass

    if not found:
        print(f"Dataset '{dataset_name}' not found.")
    else:
        print(f"Dataset '{dataset_name}': ~{total:,} samples (strategy='{strategy}').")


def add_dataset(
    conf: str,
    mod: str,
    name: str,
    split: str,
    strategy: str = DEFAULT_STRATEGY,
    conf_path: str | None = None,
) -> None:
    """
    Scaffold the full directory structure for a new dataset split.

    Creates::

        <conf_root>/<mod>/<name>/
          raw/
          pivot/
          outputs/<strategy>/<split>/   ← drop your .tar + .idx files here
          metadonnees/
          subset_selection/

    The call is idempotent; re-running it on an existing dataset is safe.

    Parameters
    ----------
    conf:
        Confidentiality name (must be registered, or *conf_path* must be given).
    mod:
        Modality (e.g. ``"rgb"``).
    name:
        Dataset name.
    split:
        Split name (e.g. ``"train"``).
    strategy:
        Strategy folder name.
    conf_path:
        Optional override for the confidentiality root path.  If given, the
        confidentiality is registered (or updated) before scaffolding.
    """
    if conf_path is not None:
        register_confidentiality(conf, conf_path, override=True)

    try:
        root = resolve_path_for_confidentiality(conf)
    except KeyError as exc:
        raise SystemExit(
            f"Error: {exc}\n"
            f"Use --conf-path to specify the path for this confidentiality."
        ) from exc

    dataset_dir = root / mod / name

    for subdir in ("raw", "pivot", "metadonnees", "subset_selection"):
        os.makedirs(dataset_dir / subdir, exist_ok=True)

    split_dir = dataset_dir / "outputs" / strategy / split
    os.makedirs(split_dir, exist_ok=True)

    print("✅ Scaffolded dataset directory:")
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

    # ── confs ─────────────────────────────────────────────────────────────────
    subparsers.add_parser(
        "confs",
        help="List all registered confidentiality mounts.",
    )

    # ── preview ───────────────────────────────────────────────────────────────
    subparsers.add_parser(
        "preview",
        help="Display a tree of all datasets with validity indicators.",
    )

    # ── count ─────────────────────────────────────────────────────────────────
    count_p = subparsers.add_parser(
        "count",
        help="Approximate the number of samples in a dataset.",
    )
    count_p.add_argument("name", type=str, help="Dataset name.")
    count_p.add_argument(
        "--strategy", type=str, default=DEFAULT_STRATEGY,
        help=f"Strategy to count (default: '{DEFAULT_STRATEGY}').",
    )
    count_p.add_argument(
        "--conf", type=str, default=None,
        help="Restrict to a single confidentiality name.",
    )

    # ── add ───────────────────────────────────────────────────────────────────
    add_p = subparsers.add_parser(
        "add",
        help="Scaffold a new dataset directory with the full sub-directory layout.",
    )
    add_p.add_argument("conf",  type=str, help="Confidentiality name (e.g. public, private).")
    add_p.add_argument("mod",   type=str, help="Modality (e.g. rgb, multispectral).")
    add_p.add_argument("name",  type=str, help="Dataset name.")
    add_p.add_argument("split", type=str, help="Split name (e.g. train, val).")
    add_p.add_argument(
        "--strategy", type=str, default=DEFAULT_STRATEGY,
        help=f"Strategy folder name (default: '{DEFAULT_STRATEGY}').",
    )
    add_p.add_argument(
        "--conf-path", type=str, default=None,
        dest="conf_path",
        help=(
            "Filesystem path for this confidentiality.  Required if the "
            "confidentiality is not already registered."
        ),
    )

    # ── stubs ─────────────────────────────────────────────────────────────────
    stubs_p = subparsers.add_parser(
        "stubs",
        help="Regenerate hub.py for IDE autocomplete.",
    )
    stubs_p.add_argument("--out", type=str, help="Output file path for hub.py.")

    # ── Dispatch ──────────────────────────────────────────────────────────────
    args = parser.parse_args()

    if args.command == "confs":
        list_confidentialities()

    elif args.command == "preview":
        preview_datasets()

    elif args.command == "count":
        count_elements(
            dataset_name=args.name,
            strategy=args.strategy,
            conf=args.conf,
        )

    elif args.command == "add":
        add_dataset(
            conf=args.conf,
            mod=args.mod,
            name=args.name,
            split=args.split,
            strategy=args.strategy,
            conf_path=args.conf_path,
        )

    elif args.command == "stubs":
        generate_stubs(output_file=args.out)
        print("✅ hub.py regenerated.")


if __name__ == "__main__":
    main()

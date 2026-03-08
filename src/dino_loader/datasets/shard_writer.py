"""
dino_loader.datasets.shard_writer
==================================
CLI and Python API for creating WebDataset shards conforming to dino_loader
conventions (directory layout, ``.idx`` sidecar, naming, catalogue
registration).

Why this module lives in ``datasets/``
--------------------------------------
``ShardWriter`` is an integral part of the dataset pipeline: it understands
the ``<conf>/<modality>/<name>/outputs/<strategy>/<split>/`` directory
convention, generates the ``.idx`` sidecar required by
:func:`~dino_loader.datasets.utils.validate_webdataset_shard`, and acts as
the write-path counterpart to :meth:`~dino_loader.datasets.dataset.Dataset.to_spec`.

It has no dependency on the loader, DALI, or CUDA — making it safe to import
in data-engineering pipelines that run without GPU.


Python API
----------
::

    from dino_loader.datasets.shard_writer import ShardWriter

    with ShardWriter(
        output_dir        = "/lustre/datasets/public/rgb/my_dataset/outputs/default/train",
        samples_per_shard = 10_000,
        max_shard_bytes   = 3 * 1024**3,   # 3 GB
    ) as writer:
        for image_path, caption in my_raw_dataset:
            writer.write({
                "__key__": image_path.stem,
                "jpg":     image_path.read_bytes(),
                "json":    json.dumps({"caption": caption, "quality_score": 0.82}),
            })
    # → shard-000000.tar, shard-000000.idx, shard-000001.tar, …

CLI
---
::

    python -m dino_loader.datasets.shard_writer \\
        --input  /data/raw/images \\
        --output /lustre/datasets/public/rgb/my_dataset/outputs/default/train \\
        --samples-per-shard 10000 \\
        --pattern "shard-{:06d}.tar" \\
        --workers 8

Notes
-----
- Requires webdataset: ``pip install webdataset``
- ``.idx`` files use little-endian int64 byte offsets (8 bytes per sample).
  Compatible with wids and
  :func:`~dino_loader.datasets.utils.validate_webdataset_shard`.
- Lustre striping: if the output directory is on Lustre, the writer attempts
  ``lfs setstripe -c 4`` for optimal write throughput.  Silently skipped if
  ``lfs`` is not in PATH.
"""

from __future__ import annotations

import argparse
import json
import logging
import struct
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional

log = logging.getLogger(__name__)

try:
    import webdataset as wds
    HAS_WDS = True
except ImportError:
    HAS_WDS = False


# ══════════════════════════════════════════════════════════════════════════════
# .idx generation
# ══════════════════════════════════════════════════════════════════════════════

def generate_idx(tar_path: Path) -> Path:
    """
    Generate a ``.idx`` sidecar for *tar_path*.

    The ``.idx`` format is a flat sequence of little-endian ``int64`` values:
    one byte offset per sample (pointing to the first file of each sample
    group within the tar archive).  This matches the format expected by
    :func:`~dino_loader.datasets.utils.validate_webdataset_shard` and
    ``wids``.

    Returns the path to the written ``.idx`` file.
    """
    idx_path = tar_path.with_suffix(".idx")
    offsets: List[int] = []

    with tarfile.open(tar_path, "r|") as tf:
        prev_key: Optional[str] = None
        for member in tf:
            stem = member.name.rsplit(".", 1)[0] if "." in member.name else member.name
            if stem != prev_key:
                offsets.append(member.offset)
                prev_key = stem

    idx_data = struct.pack(f"<{len(offsets)}q", *offsets)
    idx_path.write_bytes(idx_data)
    log.debug("Generated %s (%d sample offsets)", idx_path, len(offsets))
    return idx_path


# ── Private alias kept for internal callers ───────────────────────────────────
_generate_idx = generate_idx


# ══════════════════════════════════════════════════════════════════════════════
# Lustre striping
# ══════════════════════════════════════════════════════════════════════════════

def _try_set_lustre_striping(directory: Path, stripe_count: int = 4) -> None:
    """Set Lustre stripe count on *directory* for write throughput.

    No-op if ``lfs`` is not in PATH or the directory is not on Lustre.
    """
    try:
        subprocess.run(
            ["lfs", "setstripe", "-c", str(stripe_count), str(directory)],
            check=True,
            capture_output=True,
            timeout=5,
        )
        log.info("Lustre striping set to %d on %s", stripe_count, directory)
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        pass


# ══════════════════════════════════════════════════════════════════════════════
# ShardWriter
# ══════════════════════════════════════════════════════════════════════════════

class ShardWriter:
    """
    Creates WebDataset shards in the dino_loader directory convention.

    Parameters
    ----------
    output_dir
        Target directory.  Created if it does not exist.
        For :meth:`~dino_loader.datasets.dataset.Dataset.to_spec`
        auto-discovery, structure as::

            $ROOT/<conf>/<modality>/<name>/outputs/<strategy>/<split>/

    samples_per_shard
        Maximum number of samples per shard (default: 10 000).
    max_shard_bytes
        Maximum shard size in bytes (default: 3 GB).  Whichever limit is hit
        first triggers a new shard.
    pattern
        Shard filename pattern with one ``{:06d}`` placeholder
        (default: ``"shard-{:06d}.tar"``).
    generate_idx
        Generate ``.idx`` sidecar files after each shard (default: ``True``).
    post
        Optional callable invoked after each shard is written (in addition
        to ``.idx`` generation).  Receives the shard path as a ``str``.
    lustre_stripe_count
        If > 0, attempt to set Lustre stripe count on *output_dir*.
        Default: 4.  Set to 0 to disable.
    verbose
        Print progress to stderr (default: ``True``).
    """

    def __init__(
        self,
        output_dir:          str | Path,
        samples_per_shard:   int   = 10_000,
        max_shard_bytes:     float = 3e9,
        pattern:             str   = "shard-{:06d}.tar",
        generate_idx:        bool  = True,
        post:                Optional[Callable[[str], None]] = None,
        lustre_stripe_count: int   = 4,
        verbose:             bool  = True,
    ) -> None:
        if not HAS_WDS:
            raise ImportError(
                "dino_loader.datasets.shard_writer requires webdataset. "
                "Install with: pip install webdataset"
            )

        self._output_dir  = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._pattern     = pattern
        self._gen_idx     = generate_idx
        self._user_post   = post
        self._verbose     = verbose
        self._shard_paths: List[Path] = []

        if lustre_stripe_count > 0:
            _try_set_lustre_striping(self._output_dir, lustre_stripe_count)

        # wds.ShardWriter uses %-style formatting internally
        full_pattern = str(self._output_dir / pattern.replace("{:06d}", "%06d"))

        self._writer = wds.ShardWriter(
            pattern  = full_pattern,
            maxcount = samples_per_shard,
            maxsize  = max_shard_bytes,
            post     = self._on_shard_complete,
            verbose  = 0,   # we handle progress ourselves
        )

        self._total_samples = 0
        self._t0            = time.monotonic()

    # ── Context manager ───────────────────────────────────────────────────────

    def __enter__(self) -> "ShardWriter":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    # ── Public API ────────────────────────────────────────────────────────────

    def write(self, sample: Dict) -> None:
        """
        Write one sample dict.

        The dict must contain:

        ``"__key__"``
            Unique string identifier (no spaces or slashes).
        At least one extension key
            e.g. ``"jpg"``, ``"json"``, ``"txt"``.

        Values must be ``bytes``.  ``str`` values are auto-encoded as UTF-8.
        """
        if "__key__" not in sample:
            raise ValueError("Sample must contain '__key__'.")
        encoded = {
            k: v.encode("utf-8") if isinstance(v, str) else v
            for k, v in sample.items()
        }
        self._writer.write(encoded)
        self._total_samples += 1

        if self._verbose and self._total_samples % 1_000 == 0:
            elapsed = time.monotonic() - self._t0
            rate    = self._total_samples / max(elapsed, 1e-6)
            print(
                f"\r  {self._total_samples:>10,} samples  "
                f"{rate:>8,.0f} samples/s  "
                f"{len(self._shard_paths)} shards written",
                end="",
                file=sys.stderr,
            )

    def close(self) -> None:
        """Finalise the last shard and flush all pending writes."""
        self._writer.close()
        if self._verbose:
            elapsed = time.monotonic() - self._t0
            print(
                f"\nDone: {self._total_samples:,} samples → "
                f"{len(self._shard_paths)} shards in {elapsed:.1f}s "
                f"({self._total_samples / max(elapsed, 1e-6):,.0f} samples/s)",
                file=sys.stderr,
            )

    @property
    def shard_paths(self) -> List[Path]:
        """Paths of all completed shards (in write order)."""
        return list(self._shard_paths)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _on_shard_complete(self, shard_path: str) -> None:
        p = Path(shard_path)
        self._shard_paths.append(p)

        if self._gen_idx:
            try:
                idx_path = _generate_idx(p)
                if self._verbose:
                    print(f"\n  ✓ {p.name}  idx: {idx_path.name}", file=sys.stderr)
            except Exception as exc:
                log.warning("Failed to generate .idx for %s: %s", p, exc)

        if self._user_post:
            self._user_post(shard_path)


# ══════════════════════════════════════════════════════════════════════════════
# CLI — python -m dino_loader.datasets.shard_writer
# ══════════════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog        = "python -m dino_loader.datasets.shard_writer",
        description = (
            "Convert a directory of images (+ optional JSON metadata) "
            "into WebDataset shards for use with dino_loader."
        ),
    )
    p.add_argument(
        "--input", "-i", required=True,
        help="Input directory containing image files (recursively scanned).",
    )
    p.add_argument(
        "--output", "-o", required=True,
        help=(
            "Output directory for shards.  "
            "For Dataset auto-discovery, use: "
            "$DATASETS_ROOT/<conf>/<modality>/<name>/outputs/<strategy>/<split>/"
        ),
    )
    p.add_argument(
        "--samples-per-shard", type=int, default=10_000,
        help="Maximum samples per shard (default: 10 000).",
    )
    p.add_argument(
        "--max-shard-gb", type=float, default=3.0,
        help="Maximum shard size in GB (default: 3.0).",
    )
    p.add_argument(
        "--pattern", default="shard-{:06d}.tar",
        help="Shard filename pattern (default: 'shard-{:06d}.tar').",
    )
    p.add_argument(
        "--ext", default="jpg",
        help="Image file extension to scan for (default: jpg).",
    )
    p.add_argument(
        "--quality-score", type=float, default=None,
        help=(
            "Optional default quality_score to embed in .json sidecar "
            "when no <stem>.json exists alongside the image (default: none)."
        ),
    )
    p.add_argument(
        "--no-idx", action="store_true",
        help="Skip .idx sidecar generation.",
    )
    p.add_argument(
        "--lustre-stripe", type=int, default=4,
        help="Lustre stripe count on output dir (0 = skip, default: 4).",
    )
    p.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suppress progress output.",
    )
    return p


def _iter_samples(
    input_dir:     Path,
    ext:           str,
    quality_score: Optional[float],
) -> Iterator[Dict]:
    """Yield sample dicts from a flat or nested image directory."""
    for img_path in sorted(input_dir.rglob(f"*.{ext}")):
        key = img_path.stem.replace("/", "_").replace(" ", "_")

        img_bytes = img_path.read_bytes()

        # Prefer a sidecar <stem>.json when present
        meta_path = img_path.with_suffix(".json")
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
        elif quality_score is not None:
            meta = {"quality_score": quality_score}
        else:
            meta = None

        sample: Dict = {"__key__": key, ext: img_bytes}
        if meta:
            sample["json"] = json.dumps(meta)
        yield sample


def main(argv: Optional[List[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = _build_parser()
    args   = parser.parse_args(argv)

    input_dir = Path(args.input)
    if not input_dir.is_dir():
        parser.error(f"--input is not a directory: {input_dir}")

    with ShardWriter(
        output_dir          = args.output,
        samples_per_shard   = args.samples_per_shard,
        max_shard_bytes     = args.max_shard_gb * 1024**3,
        pattern             = args.pattern,
        generate_idx        = not args.no_idx,
        lustre_stripe_count = args.lustre_stripe,
        verbose             = not args.quiet,
    ) as writer:
        for sample in _iter_samples(input_dir, args.ext, args.quality_score):
            writer.write(sample)

    print("Shard paths:", file=sys.stderr)
    for p in writer.shard_paths:
        print(f"  {p}", file=sys.stderr)


if __name__ == "__main__":
    main()

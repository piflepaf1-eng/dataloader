"""
tests.fixtures
==============
Helpers for building synthetic WebDataset shards in memory and on disk.

These are used by all unit tests that require shard data.  They deliberately
do NOT import any ``dino_loader`` module so they can be used before the package
is on sys.path (e.g. in ``conftest.py`` before installation).

Filesystem layout produced by ``scaffold_dataset_dir`` (v2)
------------------------------------------------------------
::

    root/
      <conf>/
        <modality>/
          <name>/
            raw/
            pivot/
            outputs/
              <strategy>/
                <split>/
                  shard-000000.tar
                  shard-000000.idx
                  ...
            metadonnees/
            subset_selection/
"""

from __future__ import annotations

import io
import json
import os
import struct
import tarfile
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image


# ══════════════════════════════════════════════════════════════════════════════
# Synthetic image generation
# ══════════════════════════════════════════════════════════════════════════════

def make_jpeg_bytes(
    width:   int = 64,
    height:  int = 64,
    color:   Tuple[int, int, int] = (128, 64, 32),
    quality: int = 85,
) -> bytes:
    """Return JPEG-encoded bytes for a solid-colour RGB image."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def make_shard_tar(
    n_samples:      int = 10,
    with_metadata:  bool = True,
    quality_scores: Optional[List[float]] = None,
    img_width:      int = 64,
    img_height:     int = 64,
) -> bytes:
    """
    Build a WebDataset-compatible tar archive in memory.

    Each sample consists of:
    - ``<key>.jpg``  — a JPEG-encoded solid-colour image.
    - ``<key>.json`` — (when *with_metadata* is True) a JSON sidecar with
                       ``{"quality_score": float, "caption": str}``.

    Parameters
    ----------
    n_samples
        Number of (jpg, [json]) pairs in the archive.
    with_metadata
        Whether to include ``.json`` sidecars.
    quality_scores
        Per-sample quality scores.  Defaults to ``1.0`` for all samples.
    img_width / img_height
        Synthetic image dimensions.

    Returns
    -------
    bytes
        Raw tar archive content.
    """
    if quality_scores is None:
        quality_scores = [1.0] * n_samples
    assert len(quality_scores) == n_samples

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for i in range(n_samples):
            key  = f"sample_{i:06d}"
            jpeg = make_jpeg_bytes(
                width  = img_width,
                height = img_height,
                color  = (
                    i * 25 % 256,
                    (i * 37 + 80) % 256,
                    (i * 53 + 160) % 256,
                ),
            )
            _add_bytes(tf, f"{key}.jpg", jpeg)

            if with_metadata:
                meta = {
                    "quality_score": quality_scores[i],
                    "caption":       f"A test image number {i}",
                    "dedup_hash":    f"hash_{i:08x}",
                }
                _add_bytes(tf, f"{key}.json", json.dumps(meta).encode())

    return buf.getvalue()


def _add_bytes(tf: tarfile.TarFile, name: str, data: bytes) -> None:
    info      = tarfile.TarInfo(name=name)
    info.size = len(data)
    tf.addfile(info, io.BytesIO(data))


# ══════════════════════════════════════════════════════════════════════════════
# On-disk shard scaffolding
# ══════════════════════════════════════════════════════════════════════════════

def write_shard(
    directory:      Path,
    shard_idx:      int = 0,
    n_samples:      int = 10,
    with_metadata:  bool = True,
    quality_scores: Optional[List[float]] = None,
) -> Tuple[str, str]:
    """
    Write a synthetic ``.tar`` shard and its companion ``.idx`` file to
    *directory*.

    Returns
    -------
    (tar_path, idx_path)
        Both as absolute string paths.

    The ``.idx`` file uses the ``wds2idx`` binary format: a flat sequence of
    little-endian int64 byte offsets (8 bytes per entry).  Synthetic offsets
    are monotonically increasing for test purposes.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    tar_data = make_shard_tar(
        n_samples      = n_samples,
        with_metadata  = with_metadata,
        quality_scores = quality_scores,
    )

    tar_path = directory / f"shard-{shard_idx:06d}.tar"
    idx_path = directory / f"shard-{shard_idx:06d}.idx"

    tar_path.write_bytes(tar_data)

    # Binary idx: n_samples × little-endian int64 offsets
    idx_data = struct.pack(f"<{n_samples}q", *range(0, n_samples * 512, 512))
    idx_path.write_bytes(idx_data)

    return str(tar_path), str(idx_path)


def scaffold_dataset_dir(
    root:                Path,
    conf:                str = "public",
    modality:            str = "rgb",
    name:                str = "test_dataset",
    split:               str = "train",
    strategy:            str = "default",
    n_shards:            int = 2,
    n_samples_per_shard: int = 8,
    with_metadata:       bool = True,
) -> List[str]:
    """
    Create the full dataset directory hierarchy and populate it with synthetic
    shards.  Returns the list of absolute ``.tar`` paths.

    Layout produced::

        root/
          <conf>/
            <modality>/
              <name>/
                raw/
                pivot/
                outputs/
                  <strategy>/
                    <split>/
                      shard-000000.tar
                      shard-000000.idx
                      ...
                metadonnees/
                subset_selection/

    Parameters
    ----------
    root
        Filesystem root (typically ``tmp_path`` in pytest).
    conf
        Confidentiality label (e.g. ``"public"``, ``"private"``).
    modality
        Modality label (e.g. ``"rgb"``, ``"multispectral"``).
    name
        Dataset name.
    split
        Split name (e.g. ``"train"``, ``"val"``).
    strategy
        Strategy folder name (default: ``"default"``).
    n_shards
        Number of synthetic shards to write.
    n_samples_per_shard
        Samples per shard.
    with_metadata
        Whether to include ``.json`` sidecars in each shard.

    Returns
    -------
    list of str
        Absolute paths to the generated ``.tar`` files.
    """
    root         = Path(root)
    dataset_root = root / conf / modality / name

    # Create the full skeleton
    for subdir in ("raw", "pivot", "metadonnees", "subset_selection"):
        (dataset_root / subdir).mkdir(parents=True, exist_ok=True)

    split_dir = dataset_root / "outputs" / strategy / split
    split_dir.mkdir(parents=True, exist_ok=True)

    tar_paths: List[str] = []
    for i in range(n_shards):
        tar_path, _ = write_shard(
            directory           = split_dir,
            shard_idx           = i,
            n_samples           = n_samples_per_shard,
            with_metadata       = with_metadata,
        )
        tar_paths.append(tar_path)

    return tar_paths

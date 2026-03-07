"""
dino_loader.datasets
====================
Dataset discovery, hub and CLI for the dino_loader petascale dataloader.

Importing this package triggers two side-effects:

1. **Registry bootstrap** — the global
   :class:`~dino_loader.datasets.settings.ConfidentialityRegistry` resolves
   all confidentiality → path mappings from every configured source (pyproject,
   env vars, legacy root, entry points).

2. **Hub refresh** — if ``hub/`` is stale or missing, it is regenerated
   in-process so that IDE autocomplete is always consistent with the current
   filesystem state.

   Staleness is detected via a **registry hash** stored in
   ``hub/_registry_hash.txt`` rather than per-mount mtime comparison.
   This makes the check O(n_mounts) dict operations with a single tiny file
   read — zero filesystem stat calls against the HPC mount points.

Public re-exports
-----------------
::

    from dino_loader.datasets import (
        Dataset,
        GlobalDatasetFilter,
        DatasetConfig,
        register_confidentiality,
        get_confidentiality_mounts,
        resolve_path_for_confidentiality,
    )
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

# ── 1. Bootstrap registry (side-effect: loads all auto-sources) ───────────────
from dino_loader.datasets.settings import (  # noqa: E402
    ConfidentialityMount,
    ConfidentialityRegistry,
    get_confidentiality_mounts,
    get_registry,
    register_confidentiality,
    resolve_path_for_confidentiality,
)

# ── 2. Core dataset types ─────────────────────────────────────────────────────
from dino_loader.datasets.dataset import (
    DatasetConfig,
    GlobalDatasetFilter,
    _merge_filters,
    Dataset,
    DEFAULT_STRATEGY,
)


# ── 3. Hub refresh ─────────────────────────────────────────────────────────────

def _hub_dir() -> str:
    """Canonical path to the hub/ package directory."""
    import os
    return os.path.join(os.path.dirname(__file__), "hub")


def _maybe_refresh_hub() -> None:
    """
    Regenerate ``hub/`` if the registry has changed since it was last written.

    Staleness check
    ---------------
    We read ``hub/_registry_hash.txt`` (a 16-char hex string written by the
    generator) and compare it with :func:`~dino_loader.datasets.stub_gen.\
compute_registry_hash` — computed from the current set of
    ``(mount.name, mount.path)`` pairs in O(n_mounts) without any filesystem
    I/O against HPC mount points.

    This replaces the previous mtime-per-mount approach, which required
    ``os.path.getmtime()`` calls against potentially slow Lustre paths.

    Legacy migration
    ----------------
    If a legacy ``hub.py`` file exists alongside the package (left over from
    before the hub/ package layout), it is deleted automatically on the first
    run.  This is a one-time, silent migration.

    Safety
    ------
    All exceptions during regeneration are caught and logged at WARNING level
    so that a broken stub generator never prevents training from starting.
    """
    import os
    from dino_loader.datasets.stub_gen import generate_stubs_to_dir, hub_is_stale

    # ── Remove legacy hub.py (one-time migration, silent) ─────────────────────
    legacy = os.path.join(os.path.dirname(__file__), "hub.py")
    if os.path.isfile(legacy):
        try:
            os.remove(legacy)
            log.info("Removed legacy hub.py — migrated to hub/ package layout.")
        except OSError as exc:
            log.warning("Could not remove legacy hub.py: %s", exc)

    # ── Check if mounts are registered at all ─────────────────────────────────
    mounts = get_confidentiality_mounts()
    if not mounts:
        return  # Nothing to scan — don't wipe an existing hub/.

    # ── Hash-based staleness check ─────────────────────────────────────────────
    hub = _hub_dir()
    if not hub_is_stale(hub):
        log.debug("hub/ is up-to-date (registry hash match).")
        return

    log.debug("hub/ is stale or missing — regenerating.")
    try:
        generate_stubs_to_dir(hub)
    except Exception as exc:  # noqa: BLE001
        log.warning("hub/ auto-refresh failed: %s", exc)


_maybe_refresh_hub()


# ── Public API surface ─────────────────────────────────────────────────────────
__all__ = [
    # Settings
    "ConfidentialityMount",
    "ConfidentialityRegistry",
    "get_confidentiality_mounts",
    "get_registry",
    "register_confidentiality",
    "resolve_path_for_confidentiality",
    # Dataset
    "Dataset",
    "DatasetConfig",
    "DEFAULT_STRATEGY",
    "GlobalDatasetFilter",
    "_merge_filters",
]

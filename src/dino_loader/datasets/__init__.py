"""
dino_loader.datasets
====================
Dataset discovery, hub and CLI for the dino_loader petascale dataloader.

Importing this package triggers two side-effects:

1. **Registry bootstrap** — the global
   :class:`~dino_loader.datasets.settings.ConfidentialityRegistry` resolves
   all confidentiality → path mappings from every configured source (pyproject,
   env vars, legacy root, entry points).

2. **Hub completion** — if ``hub.py`` is stale or missing, it is regenerated
   in-process so that IDE autocomplete is always consistent with the current
   filesystem state.  This is a lightweight pass (O(n_mounts) directory stat
   calls) and is skipped when the mounts have not changed since the last run.

Public re-exports
-----------------
The most commonly needed symbols are re-exported here for convenience::

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
from dino_loader.datasets.settings import (  # noqa: E402  (import after log setup)
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
    Dataset,
    DEFAULT_STRATEGY,
)

# ── 3. Regenerate hub.py at init time ─────────────────────────────────────────
def _maybe_refresh_hub() -> None:
    """
    Regenerate ``hub.py`` if any registered confidentiality mount is newer
    than the existing stub file, or if the stub file does not exist.

    This keeps IDE autocomplete in sync without requiring a manual
    ``python -m dino_loader.datasets stubs`` call.
    """
    import os
    from dino_loader.datasets.stub_gen import generate_stubs

    hub_path = os.path.join(os.path.dirname(__file__), "hub.py")

    # Determine the most recent modification time across all mount roots.
    mounts = get_confidentiality_mounts()
    if not mounts:
        return  # Nothing to scan; don't wipe an existing hub.py.

    hub_mtime = os.path.getmtime(hub_path) if os.path.exists(hub_path) else 0.0

    needs_refresh = False
    if hub_mtime == 0.0:
        needs_refresh = True
    else:
        for mount in mounts:
            if mount.path.is_dir():
                try:
                    if os.path.getmtime(str(mount.path)) > hub_mtime:
                        needs_refresh = True
                        break
                except OSError:
                    pass

    if needs_refresh:
        log.debug("Refreshing hub.py (stale or missing).")
        try:
            generate_stubs(output_file=hub_path)
        except Exception as exc:  # noqa: BLE001
            # Never let hub generation crash an import.
            log.warning("hub.py auto-refresh failed: %s", exc)


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
]

"""
dino_loader.datasets.settings
==============================
Confidentiality-aware filesystem configuration.

Each **confidentiality** (e.g. ``"public"``, ``"private"``, ``"secret"``) maps
to a **distinct directory** on the HPC filesystem.  The canonical unit is a
:class:`ConfidentialityMount` — a ``(name, path)`` pair.

Resolution order (highest priority first)
------------------------------------------
1. **Programmatic** — ``register_confidentiality(name, path)`` called at
   runtime (e.g. in a training script or notebook).
2. **pyproject.toml** — ``[tool.dino_loader.datasets.confidentialities]``
   table; each key is a confidentiality name, each value is an absolute path::

       [tool.dino_loader.datasets.confidentialities]
       public  = "/lustre/datasets/public"
       private = "/lustre/datasets/private"

3. **Environment variables** — one variable per confidentiality, following the
   pattern ``DINO_CONF_<UPPER_NAME>``::

       export DINO_CONF_PUBLIC=/lustre/datasets/public
       export DINO_CONF_PRIVATE=/lustre/datasets/private

4. **Legacy single-root** — ``$DINO_DATASETS_ROOT`` or the
   ``tool.dino_loader.datasets.root`` TOML key.  Each sub-directory under
   that root is treated as a confidentiality whose path is
   ``<root>/<name>/``.  This guarantees backward-compatibility with the
   previous single-root layout.

5. **Default fallback** — ``~/.dinoloader/<name>/`` for any confidentiality
   name declared in :data:`DEFAULT_CONFIDENTIALITIES`.

Entry-point discovery
---------------------
Third-party packages (e.g. a ``corp-datasets`` package) can expose
confidentialities without any code change in ``dino_loader`` by declaring an
entry point in their ``pyproject.toml``::

    [project.entry-points."dino_loader.confidentialities"]
    corp = "corp_datasets.confs:CONFIDENTIALITIES"

The exported symbol must be a ``dict[str, str | Path]`` or a list of
``(name, path)`` tuples.  These are loaded once at import time of the
``dino_loader.datasets`` package and merged into the global registry.

Usage
-----
::

    from dino_loader.datasets.settings import (
        register_confidentiality,
        get_confidentiality_mounts,
        resolve_path_for_confidentiality,
    )

    # Programmatically add a confidentiality (idempotent):
    register_confidentiality("internal", "/scratch/internal_data")

    # Iterate over all known (name, path) pairs:
    for mount in get_confidentiality_mounts():
        print(mount.name, mount.path)

    # Resolve path for a single confidentiality:
    path = resolve_path_for_confidentiality("public")
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from importlib.metadata import entry_points
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

try:
    import tomllib
except ImportError:  # Python < 3.11
    import tomli as tomllib  # type: ignore[no-reuse-source]

log = logging.getLogger(__name__)

# ── Public constants ───────────────────────────────────────────────────────────

#: Confidentiality names that are always seeded into the registry from the
#: legacy single-root path or the default fallback directory.
#: Downstream users are free to extend this set; it is intentionally *not*
#: frozen so that ``register_confidentiality`` additions take effect globally.
DEFAULT_CONFIDENTIALITIES: frozenset[str] = frozenset({"public", "private"})

#: Entry-point group name used by third-party packages to inject confidentialities.
_ENTRY_POINT_GROUP = "dino_loader.confidentialities"


# ── ConfidentialityMount ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class ConfidentialityMount:
    """
    An immutable ``(name, path)`` pair that anchors a confidentiality label to a
    concrete filesystem location.

    Parameters
    ----------
    name:
        Short string identifier, e.g. ``"public"``, ``"private"``, ``"secret"``.
        Must be a valid directory name component (no path separators).
    path:
        Absolute path to the confidentiality root on the HPC filesystem.
        May not exist yet (the library never creates it automatically).

    Examples
    --------
    ::

        m = ConfidentialityMount("public", Path("/lustre/data/public"))
        assert m.name == "public"
        assert m.path.is_absolute()
    """

    name: str
    path: Path

    def __post_init__(self) -> None:
        if not self.name or "/" in self.name or "\\" in self.name:
            raise ValueError(
                f"Confidentiality name must be a simple identifier, got: {self.name!r}"
            )

    def __str__(self) -> str:  # pragma: no cover
        return f"ConfidentialityMount({self.name!r}, {self.path})"


# ── ConfidentialityRegistry ────────────────────────────────────────────────────

class ConfidentialityRegistry:
    """
    Centralised, ordered registry of :class:`ConfidentialityMount` objects.

    The registry is populated lazily on first access and can be extended at any
    time via :meth:`register`.  Later registrations **do not** override earlier
    ones for the same name — first-wins semantics preserve the priority order
    defined in the module docstring.

    This class is instantiated **once** as a module-level singleton
    (:data:`_REGISTRY`) and should not normally be instantiated directly.
    Public helpers (:func:`register_confidentiality`,
    :func:`get_confidentiality_mounts`, …) delegate to that singleton.
    """

    def __init__(self) -> None:
        # Ordered dict preserves insertion order; first-write-wins for each name.
        self._mounts: Dict[str, ConfidentialityMount] = {}
        self._bootstrapped = False

    # ── Internal bootstrap ─────────────────────────────────────────────────

    def _bootstrap(self) -> None:
        """
        Populate the registry from all automatic sources in priority order.
        Called once, the first time the registry is accessed.
        """
        if self._bootstrapped:
            return
        self._bootstrapped = True

        # 1. pyproject.toml [tool.dino_loader.datasets.confidentialities]
        self._load_from_toml()

        # 2. DINO_CONF_<NAME> environment variables
        self._load_from_env()

        # 3. Legacy single-root (DINO_DATASETS_ROOT / pyproject root key)
        self._load_from_legacy_root()

        # 4. Entry-point discovery (third-party packages)
        self._load_from_entry_points()

        # 5. Default fallback for names in DEFAULT_CONFIDENTIALITIES
        self._load_defaults()

    def _register_if_absent(self, name: str, path: Path, source: str) -> None:
        """Register *name → path* only if *name* is not already registered."""
        if name not in self._mounts:
            log.debug("Registering confidentiality %r from %s: %s", name, source, path)
            self._mounts[name] = ConfidentialityMount(name=name, path=path)
        else:
            log.debug(
                "Skipping confidentiality %r from %s (already registered as %s)",
                name, source, self._mounts[name].path,
            )

    def _load_from_toml(self) -> None:
        try:
            pyproject_path = Path(os.getcwd()) / "pyproject.toml"
            if not pyproject_path.exists():
                return
            with pyproject_path.open("rb") as fh:
                data = tomllib.load(fh)
            confs: dict = (
                data
                .get("tool", {})
                .get("dino_loader", {})
                .get("datasets", {})
                .get("confidentialities", {})
            )
            for name, raw_path in confs.items():
                self._register_if_absent(
                    name, Path(raw_path).expanduser().resolve(), "pyproject.toml"
                )
        except Exception as exc:  # noqa: BLE001
            log.debug("Could not load confidentialities from pyproject.toml: %s", exc)

    def _load_from_env(self) -> None:
        prefix = "DINO_CONF_"
        for key, value in os.environ.items():
            if key.startswith(prefix) and value:
                name = key[len(prefix):].lower()
                self._register_if_absent(
                    name, Path(value).expanduser().resolve(), f"env:{key}"
                )

    def _load_from_legacy_root(self) -> None:
        """
        Treat every sub-directory of the legacy single root as a confidentiality.
        This provides seamless backward compatibility with the previous layout.
        """
        root = _resolve_legacy_root()
        if root is None:
            return
        root_path = Path(root).expanduser().resolve()
        if not root_path.is_dir():
            return
        for child in sorted(root_path.iterdir()):
            if child.is_dir():
                self._register_if_absent(child.name, child, f"legacy-root:{root_path}")

    def _load_from_entry_points(self) -> None:
        """
        Discover confidentialities exported by installed packages via the
        ``dino_loader.confidentialities`` entry-point group.
        """
        try:
            eps = entry_points(group=_ENTRY_POINT_GROUP)
        except Exception as exc:  # noqa: BLE001
            log.debug("entry_points discovery failed: %s", exc)
            return

        for ep in eps:
            try:
                obj = ep.load()
                mounts = _coerce_to_mounts(obj, source=f"entry_point:{ep.name}")
                for mount in mounts:
                    self._register_if_absent(mount.name, mount.path, f"entry_point:{ep.name}")
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "Failed to load confidentialities from entry point %r: %s",
                    ep.name, exc,
                )

    def _load_defaults(self) -> None:
        fallback_root = Path("~/.dinoloader").expanduser().resolve()
        for name in DEFAULT_CONFIDENTIALITIES:
            self._register_if_absent(
                name, fallback_root / name, "default-fallback"
            )

    # ── Public API ─────────────────────────────────────────────────────────

    def register(
        self,
        name: str,
        path: Union[str, Path],
        *,
        override: bool = False,
    ) -> ConfidentialityMount:
        """
        Register a confidentiality programmatically.

        Parameters
        ----------
        name:
            Short identifier (e.g. ``"internal"``).
        path:
            Filesystem path for this confidentiality.
        override:
            When ``True``, replace an existing registration for *name*.
            Defaults to ``False`` (first-wins semantics).

        Returns
        -------
        ConfidentialityMount
            The newly registered (or existing) mount.
        """
        self._bootstrap()  # ensure auto-sources ran first
        resolved = Path(path).expanduser().resolve()
        mount = ConfidentialityMount(name=name, path=resolved)
        if override or name not in self._mounts:
            self._mounts[name] = mount
            log.debug("Registered confidentiality %r → %s (override=%s)", name, resolved, override)
        return self._mounts[name]

    def all(self) -> List[ConfidentialityMount]:
        """Return all registered mounts, in insertion order."""
        self._bootstrap()
        return list(self._mounts.values())

    def get(self, name: str) -> Optional[ConfidentialityMount]:
        """Return the mount for *name*, or ``None`` if unknown."""
        self._bootstrap()
        return self._mounts.get(name)

    def names(self) -> List[str]:
        """Return the list of all registered confidentiality names."""
        self._bootstrap()
        return list(self._mounts.keys())

    def __contains__(self, name: str) -> bool:
        self._bootstrap()
        return name in self._mounts

    def __iter__(self) -> Iterator[ConfidentialityMount]:
        self._bootstrap()
        return iter(self._mounts.values())

    def __len__(self) -> int:
        self._bootstrap()
        return len(self._mounts)

    def __repr__(self) -> str:  # pragma: no cover
        self._bootstrap()
        names = list(self._mounts.keys())
        return f"ConfidentialityRegistry({names!r})"


# ── Module-level singleton ─────────────────────────────────────────────────────

#: Global registry.  All public helpers below delegate to this object.
_REGISTRY = ConfidentialityRegistry()


# ── Public helpers ─────────────────────────────────────────────────────────────

def register_confidentiality(
    name: str,
    path: Union[str, Path],
    *,
    override: bool = False,
) -> ConfidentialityMount:
    """
    Register a confidentiality into the global registry.

    This is the **primary extension point** for user code.  Call it before any
    dataset resolution takes place (e.g. at the top of your training script).

    Parameters
    ----------
    name:
        Confidentiality label, e.g. ``"internal"``, ``"secret"``.
    path:
        Absolute (or ``~``-prefixed) path to the root directory for this
        confidentiality.
    override:
        Replace an existing mount with the same name.  Defaults to ``False``.

    Returns
    -------
    ConfidentialityMount
        The registered mount.

    Examples
    --------
    ::

        from dino_loader.datasets.settings import register_confidentiality

        register_confidentiality("internal", "/lustre/corp/internal")
        register_confidentiality("restricted", "/mnt/restricted", override=True)
    """
    return _REGISTRY.register(name, path, override=override)


def get_confidentiality_mounts() -> List[ConfidentialityMount]:
    """
    Return all registered :class:`ConfidentialityMount` objects, in priority order.

    Triggers bootstrap on first call.
    """
    return _REGISTRY.all()


def resolve_path_for_confidentiality(name: str) -> Path:
    """
    Return the filesystem path for a single confidentiality.

    Raises
    ------
    KeyError
        If *name* is not registered.
    """
    mount = _REGISTRY.get(name)
    if mount is None:
        registered = _REGISTRY.names()
        raise KeyError(
            f"Unknown confidentiality {name!r}. "
            f"Registered: {registered}. "
            f"Use register_confidentiality('{name}', '/your/path') to add it."
        )
    return mount.path


def get_registry() -> ConfidentialityRegistry:
    """Return the global :class:`ConfidentialityRegistry` singleton."""
    return _REGISTRY


# ── Backward-compatibility shim ────────────────────────────────────────────────

def resolve_datasets_root(arg_path: str | None = None) -> str:
    """
    **Deprecated** — retained for backward compatibility only.

    Returns a single root path following the legacy precedence chain.
    New code should use :func:`get_confidentiality_mounts` instead.

    Resolution order (legacy):
    1. *arg_path* argument
    2. ``tool.dino_loader.datasets.root`` in ``pyproject.toml``
    3. ``$DINO_DATASETS_ROOT`` environment variable
    4. ``~/.dinoloader/``
    """
    import warnings
    warnings.warn(
        "resolve_datasets_root() is deprecated. "
        "Use get_confidentiality_mounts() or resolve_path_for_confidentiality() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    if arg_path is not None:
        return arg_path
    root = _resolve_legacy_root()
    return root or str(Path("~/.dinoloader").expanduser())


# ── Internal helpers ───────────────────────────────────────────────────────────

def _resolve_legacy_root() -> Optional[str]:
    """
    Return the legacy single-root path from TOML or env, or ``None``.
    Does NOT fall back to ``~/.dinoloader`` to avoid spurious directory
    creation when the user simply never configured a legacy root.
    """
    # TOML key: tool.dino_loader.datasets.root
    try:
        pyproject_path = Path(os.getcwd()) / "pyproject.toml"
        if pyproject_path.exists():
            with pyproject_path.open("rb") as fh:
                data = tomllib.load(fh)
            root = (
                data.get("tool", {})
                .get("dino_loader", {})
                .get("datasets", {})
                .get("root")
            )
            if root:
                return str(Path(root).expanduser().resolve())
    except Exception:  # noqa: BLE001
        pass

    # Environment variable
    env = os.environ.get("DINO_DATASETS_ROOT")
    if env:
        return env

    return None


def _coerce_to_mounts(
    obj: object,
    source: str,
) -> List[ConfidentialityMount]:
    """
    Convert an arbitrary exported object from an entry point into a list of
    :class:`ConfidentialityMount`.

    Accepted shapes:
    - ``dict[str, str | Path]``
    - ``list[tuple[str, str | Path]]``  /  ``list[ConfidentialityMount]``
    - A single ``ConfidentialityMount``
    """
    mounts: List[ConfidentialityMount] = []

    if isinstance(obj, ConfidentialityMount):
        mounts.append(obj)

    elif isinstance(obj, dict):
        for name, raw_path in obj.items():
            try:
                mounts.append(
                    ConfidentialityMount(name=name, path=Path(raw_path).expanduser().resolve())
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("Bad entry in %s[%r]: %s", source, name, exc)

    elif isinstance(obj, (list, tuple)):
        for item in obj:
            if isinstance(item, ConfidentialityMount):
                mounts.append(item)
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                name, raw_path = item
                try:
                    mounts.append(
                        ConfidentialityMount(name=name, path=Path(raw_path).expanduser().resolve())
                    )
                except Exception as exc:  # noqa: BLE001
                    log.warning("Bad item in %s: %s", source, exc)
            else:
                log.warning("Unrecognised item type in %s: %r", source, type(item))

    else:
        log.warning(
            "Entry point %s exported unsupported type %r; expected dict or list.",
            source, type(obj),
        )

    return mounts

"""dino_loader.dali_node
======================
``_DALINode`` — torchdata BaseNode that drives the DALI/CPU iterator and
assembles fully-formed ``Batch`` objects.

Responsibilities
----------------
This node is the boundary between ``torchdata.nodes`` and the augmentation
pipeline (DALI or CPU).  It:

- Drives the DALI/CPU iterator via ``next()``.
- Assembles views into a ``Batch`` via the injected ``build_batch_fn``.
- Updates per-rank metrics (``loader_batches_yielded``, ``pipeline_yield_ms``,
  ``heartbeat``).
- Detects stalls: if no batch is produced and ``stall_timeout_s > 0``, raises
  ``RuntimeError`` with an actionable diagnostic message.

Placement rationale
-------------------
``_DALINode`` is not a backend (it doesn't build pipelines or manage GPU
resources) and not a post-processing transform (it doesn't operate on
``Batch`` objects flowing through a graph).  It is the pivot point between
the two worlds and lives here at the loader layer.

Thread safety
-------------
``reset_iter()`` (called from ``set_epoch()`` in the main thread) and
``next()`` (called from the torchdata worker thread) are protected by
``_iter_lock`` to prevent a race on ``_iter``.

[REFACTOR-R2] Extracted from ``pipeline_graph.py``.
"""

import logging
import os
import threading
import time
from collections.abc import Callable
from typing import Any

from torchdata.nodes import BaseNode

from dino_loader.memory import Batch

log = logging.getLogger(__name__)


class _DALINode(BaseNode):  # type: ignore[misc]
    """Drives a DALI/CPU iterator and emits ``Batch`` objects.

    Args:
        dali_iter_factory: ``() -> iterator`` called at each ``reset()``.
            Must produce a DALI/CPU-compatible iterator.
        pop_metadata_fn: ``() -> list[dict | None]`` — retrieves the metadata
            for the current batch (called once per ``next()``).
        build_batch_fn: ``(views, metadata) -> Batch`` — assembles views into
            a ``Batch`` (H2D transfer, FP8 quantisation, masking).
        output_map: Ordered view names produced by the iterator.
        stall_timeout_s: Seconds before raising on no batches. 0 = disabled.
        rank: Global rank (used in error messages).

    """

    def __init__(
        self,
        dali_iter_factory: Callable[[], Any],
        pop_metadata_fn:   Callable[[], list[dict | None]],
        build_batch_fn:    Callable[[list[Any], list[dict | None]], Batch],
        output_map:        list[str],
        stall_timeout_s:   float = 600.0,
        rank:              int   = 0,
    ) -> None:
        super().__init__()
        self._iter_factory  = dali_iter_factory
        self._pop_metadata  = pop_metadata_fn
        self._build_batch   = build_batch_fn
        self._output_map    = output_map
        self._stall_timeout = stall_timeout_s
        self._rank          = rank
        self._iter: Any     = None
        self._num_yielded   = 0
        self._iter_lock     = threading.Lock()

    def reset(self, initial_state: dict[str, Any] | None = None) -> None:
        """Re-initialise the iterator for a new epoch."""
        super().reset(initial_state)
        with self._iter_lock:
            self._iter        = self._iter_factory()
            self._num_yielded = 0

    def reset_iter(self) -> None:
        """Invalidate the current iterator so reset() recreates it.

        Called by ``DINODataLoader.set_epoch()`` from the main thread.
        Thread-safe via ``_iter_lock``.

        """
        with self._iter_lock:
            self._iter = None

    def next(self) -> Batch:
        """Return the next assembled Batch.

        Raises:
            AssertionError: If ``reset()`` has not been called.
            StopIteration: At end of epoch.
            RuntimeError: On stall (no batch produced, stall_timeout_s > 0).

        """
        with self._iter_lock:
            current_iter = self._iter

        if current_iter is None:
            msg = "reset() must be called before next()"
            raise AssertionError(msg)

        try:
            dali_out = next(current_iter)
        except StopIteration:
            if self._num_yielded == 0 and self._stall_timeout > 0:
                if os.environ.get("DINO_DISABLE_EMPTY_CHECK"):
                    log.warning(
                        "_DALINode rank %d: no batch produced but "
                        "DINO_DISABLE_EMPTY_CHECK is active — continuing silently.",
                        self._rank,
                    )
                else:
                    msg = (
                        f"_DALINode (rank {self._rank}): no batch produced. "
                        "Possible causes: corrupt shards, /dev/shm full, "
                        "sample_predicate rejected all samples, slow MDS startup. "
                        "Disable: DINO_DISABLE_EMPTY_CHECK=1 or stall_timeout_s=0."
                    )
                    raise RuntimeError(msg) from None
            raise

        t0       = time.perf_counter()
        views    = [dali_out[0][name] for name in self._output_map]
        metadata = self._pop_metadata()
        batch    = self._build_batch(views, metadata)
        elapsed  = int((time.perf_counter() - t0) * 1000)

        self._num_yielded += 1
        _update_metrics(elapsed)

        return batch

    def get_state(self) -> dict[str, Any]:
        """Return persistable state."""
        return {"_num_yielded": self._num_yielded}


def _update_metrics(elapsed_ms: int) -> None:
    """Increment loader metrics via the module-level registry."""
    try:
        from dino_loader.monitor.metrics import get_registry  # noqa: PLC0415
        reg = get_registry()
        if reg is not None:
            reg.inc("loader_batches_yielded", 1)
            reg.inc("pipeline_yield_time_ms", elapsed_ms)
            reg.heartbeat()
    except Exception:  # noqa: BLE001
        pass
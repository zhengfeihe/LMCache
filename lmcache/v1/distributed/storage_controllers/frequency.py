# SPDX-License-Identifier: Apache-2.0
"""
Approximate access-frequency tracking for L2 store and prefetch policies.

Frequency-gated policies (gated store admission, selective L1 retention)
need to know how many times a key has been accessed without holding an
exact per-key counter for the entire keyspace. This module provides:

- :class:`CountMinSketch`: a fixed-memory Count-Min Sketch
  (overestimates, never underestimates) with optional time-based aging.
- :class:`FrequencyTrackingListener`: an :class:`L1ManagerListener` that
  forwards L1 access events into an estimator.

The estimator is fed by ``L1Manager.touch_keys`` (one access event per
finished request, covering both retrieved and stored keys) and queried
from the store/prefetch controller background threads, so every public
method is thread-safe. It is only built when a frequency-gated policy is
selected; otherwise the wiring passes ``None`` and no estimator exists.
"""

# Standard
from dataclasses import dataclass
import threading
import time

# Third Party
import numpy as np
import xxhash

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.internal_api import L1ManagerListener

# Largest value a uint32 sketch cell can hold; increments saturate here so a
# wrap-around can never make a hot key suddenly read as cold.
_UINT32_MAX = (1 << 32) - 1

# Halving the table more than 32 times drives every uint32 counter to 0, so a
# long idle gap needs at most this many shifts — caps the catch-up loop.
_MAX_AGING_SHIFTS = 32


@dataclass(frozen=True)
class FrequencyConfig:
    """Sizing for a :class:`CountMinSketch`.

    The sketch uses ``depth`` independent rows of ``width`` counters each,
    so its memory footprint is ``depth * width * 4`` bytes. Wider sketches
    shrink the over-estimation magnitude (collision density); deeper sketches
    shrink the probability that every row of a key collides. The defaults
    (6 x 262144) use ~6 MiB and keep the collision error small for workloads
    up to a few hundred thousand distinct hot keys.

    Attributes:
        width: Number of counters per row. Must be >= 1.
        depth: Number of independent rows. Must be >= 1.
        aging_interval_seconds: Wall-clock period after which every counter
            is halved (integer ``>> 1``), so the sketch tracks *recent*
            frequency instead of all-time frequency. This is the TinyLFU
            aging step done on a time base: it keeps the estimate adaptive
            to workload shifts and stops collision over-estimation from
            accumulating without bound. ``0.0`` disables aging (pure
            all-time counting). Must be >= 0.

            Halving (rather than clearing) avoids a cold window after each
            decay: a genuinely hot key stays well above the admission
            threshold, so it keeps being admitted/retained, while a key
            accessed less than the threshold per interval decays away. With
            aging on, a frequency threshold means "at least N accesses
            within roughly the last ``aging_interval_seconds``" rather than
            "at least N accesses ever".
    """

    width: int = 1 << 18
    depth: int = 6
    aging_interval_seconds: float = 0.0

    def __post_init__(self) -> None:
        if self.width < 1:
            raise ValueError(f"width must be >= 1 (got {self.width})")
        if self.depth < 1:
            raise ValueError(f"depth must be >= 1 (got {self.depth})")
        if self.aging_interval_seconds < 0:
            raise ValueError(
                "aging_interval_seconds must be >= 0 "
                f"(got {self.aging_interval_seconds})"
            )


class CountMinSketch:
    """Thread-safe Count-Min Sketch frequency estimator.

    Each key maps to one counter per row via a per-row-seeded hash of the
    key's identity fields (see :meth:`_columns`). ``record`` increments all
    of a key's counters (saturating at ``2**32 - 1``); ``estimate`` returns
    the minimum across the key's counters, which is the standard Count-Min
    point-query estimate.

    When ``config.aging_interval_seconds`` is positive, every counter is
    halved once that much wall-clock time has elapsed, so the sketch
    reflects recent rather than all-time frequency (the TinyLFU aging step
    on a time base). Aging is applied lazily inside ``record``/``estimate``,
    so no background thread is needed.

    All methods are safe to call concurrently: ``record`` runs on the
    thread that touches L1 keys while ``estimate`` runs on the store and
    prefetch controller threads.

    Args:
        config: Sketch sizing (width, depth) and aging interval.
    """

    def __init__(self, config: FrequencyConfig) -> None:
        self._width = config.width
        self._depth = config.depth
        self._aging_interval = config.aging_interval_seconds
        self._table = np.zeros((self._depth, self._width), dtype=np.uint32)
        # Monotonic time of the last halving. Only meaningful when aging is on.
        self._last_aging_time = time.monotonic()
        self._lock = threading.Lock()

    def _columns(self, key: ObjectKey) -> list[int]:
        """Return the per-row column indices for ``key``.

        Builds a byte string of the key's identity fields, then hashes it once
        per row with xxh3 using the row index as the seed — an independent
        column per row (``hash mod width``). Per-row seeds give the rows the
        independence textbook Count-Min assumes, and place no cap on ``depth``.

        Returns plain Python ints (not a numpy array): the per-row scalar
        table accesses in ``record``/``estimate`` are far cheaper than numpy
        fancy indexing over a tiny ``depth``-length array.

        Args:
            key: The key to hash.

        Returns:
            A ``depth``-length list of column indices, one per row.
        """
        identity = b"".join(
            (
                key.chunk_hash,
                key.model_name.encode("utf-8"),
                str(key.kv_rank).encode("utf-8"),
                str(key.object_group_id).encode("utf-8"),
                key.cache_salt.encode("utf-8"),
            )
        )
        width = self._width
        return [
            xxhash.xxh3_64_intdigest(identity, seed=row) % width
            for row in range(self._depth)
        ]

    def record(self, keys: list[ObjectKey]) -> None:
        """Increment the counters for each key, saturating at uint32 max.

        Applies any pending time-based aging first so the increment lands
        on an up-to-date table.

        Args:
            keys: Keys that were just accessed.
        """
        if not keys:
            return
        # Hash outside the lock: hashing dominates the per-key cost, so keeping
        # it out of the critical section shrinks lock hold time to just the
        # counter updates and keeps contention with estimate() negligible.
        all_cols = [self._columns(key) for key in keys]
        with self._lock:
            self._maybe_age()
            table = self._table
            for cols in all_cols:
                for row, col in enumerate(cols):
                    # Saturating increment: leave a cell already at the max.
                    value = table[row, col]
                    if value < _UINT32_MAX:
                        table[row, col] = value + 1

    def _maybe_age(self) -> None:
        """Halve every counter once per elapsed aging interval.

        A no-op when aging is disabled (``aging_interval_seconds == 0``).
        Called with the lock held from both ``record`` and ``estimate`` so
        readers never see a stale, un-decayed table. Catches up on multiple
        elapsed intervals in one shot (capped at ``_MAX_AGING_SHIFTS``,
        beyond which the table is already all zeros).
        """
        if self._aging_interval <= 0:
            return
        now = time.monotonic()
        elapsed = now - self._last_aging_time
        if elapsed < self._aging_interval:
            return
        intervals = int(elapsed // self._aging_interval)
        self._table >>= min(intervals, _MAX_AGING_SHIFTS)
        self._last_aging_time = now

    def estimate(self, key: ObjectKey) -> int:
        """Return the Count-Min point estimate (minimum over the key's rows).

        Applies any pending time-based aging first so the estimate reflects
        the current interval.

        Args:
            key: The key to query.

        Returns:
            The estimated access count (>= 0).
        """
        cols = self._columns(key)
        with self._lock:
            self._maybe_age()
            table = self._table
            return int(min(table[row, col] for row, col in enumerate(cols)))


class FrequencyTrackingListener(L1ManagerListener):
    """L1 listener that feeds access events into a frequency estimator.

    Only ``on_l1_keys_accessed`` carries access information; every other
    L1 event is ignored. Registered with the L1 manager by the storage
    manager when a frequency-gated store or prefetch policy is active.

    Args:
        estimator: The estimator to forward access events to.
    """

    def __init__(self, estimator: CountMinSketch) -> None:
        self._estimator = estimator

    def on_l1_keys_accessed(self, keys: list[ObjectKey]) -> None:
        """Record the accessed keys in the estimator."""
        self._estimator.record(keys)

    def on_l1_keys_reserved_read(self, keys: list[ObjectKey]) -> None:
        pass

    def on_l1_keys_read_finished(self, keys: list[ObjectKey]) -> None:
        pass

    def on_l1_keys_reserved_write(self, keys: list[ObjectKey]) -> None:
        pass

    def on_l1_keys_write_finished(self, keys: list[ObjectKey]) -> None:
        pass

    def on_l1_keys_finish_write_and_reserve_read(self, keys: list[ObjectKey]) -> None:
        pass

    def on_l1_keys_deleted_by_manager(self, keys: list[ObjectKey]) -> None:
        pass

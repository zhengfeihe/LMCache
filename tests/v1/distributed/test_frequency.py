# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for the access-frequency estimator used by the frequency-gated
store policy.

Tests are written against the CountMinSketch contract in frequency.py:
- CountMinSketch never underestimates the true count.
- Increments saturate at the uint32 ceiling.
- Optional time-based aging halves counters once the interval elapses.
- FrequencyTrackingListener forwards only L1 access events.

Aging tests inject a fake monotonic clock so elapsed time is controlled
deterministically without sleeping.
"""

# Standard
import threading
import time

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.storage_controllers.frequency import (
    FrequencyConfig,
    CountMinSketch,
    FrequencyTrackingListener,
)

# =============================================================================
# Helpers
# =============================================================================


def make_object_key(chunk_id: int) -> ObjectKey:
    """Create a test ObjectKey with the given chunk ID."""
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk_id),
        model_name="test_model",
        kv_rank=0,
    )


class FakeClock:
    """A manually advanced monotonic clock for deterministic aging tests.

    Install with :meth:`install` so the estimator (which calls
    ``time.monotonic()`` directly) reads this controlled value instead of
    the real clock.
    """

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds

    def install(self, monkeypatch: pytest.MonkeyPatch) -> "FakeClock":
        """Patch ``time.monotonic`` to read this clock (auto-reverted)."""
        monkeypatch.setattr(time, "monotonic", self)
        return self


# =============================================================================
# FrequencyConfig validation
# =============================================================================


class TestFrequencyConfigValidation:
    """FrequencyConfig rejects invalid sizing."""

    def test_rejects_invalid_sizing(self):
        with pytest.raises(ValueError):
            FrequencyConfig(width=0)
        with pytest.raises(ValueError):
            FrequencyConfig(depth=0)
        with pytest.raises(ValueError):
            FrequencyConfig(aging_interval_seconds=-1.0)


# =============================================================================
# CountMinSketch: core counting
# =============================================================================


class TestCountMinSketchBasics:
    """Core record/estimate behavior."""

    def test_counts_distinct_keys_independently(self):
        est = CountMinSketch(FrequencyConfig(width=4096, depth=4))
        k0, k1 = make_object_key(0), make_object_key(1)
        assert est.estimate(k0) == 0  # unseen
        est.record([k0, k0, k0, k1])
        assert est.estimate(k0) == 3
        assert est.estimate(k1) == 1


class TestCountMinNeverUnderestimates:
    """The defining Count-Min guarantee: estimate >= true count."""

    def test_estimate_is_lower_bounded_by_truth(self):
        # A deliberately tiny sketch forces collisions; the estimate for a key
        # must still be at least its true insertion count.
        est = CountMinSketch(FrequencyConfig(width=4, depth=2))
        truth: dict[int, int] = {}
        for i in range(200):
            key_id = i % 20
            est.record([make_object_key(key_id)])
            truth[key_id] = truth.get(key_id, 0) + 1
        for key_id, true_count in truth.items():
            assert est.estimate(make_object_key(key_id)) >= true_count


class TestCountMinSaturation:
    """Increments saturate rather than wrapping around uint32."""

    def test_estimate_saturates_at_uint32_max(self):
        est = CountMinSketch(FrequencyConfig(width=8, depth=2))
        key = make_object_key(0)
        uint32_max = (1 << 32) - 1
        # Pre-load every cell of the key to the ceiling, then record once more.
        cols = est._columns(key)  # noqa: SLF001 - white-box saturation check
        for row, col in enumerate(cols):
            est._table[row, col] = uint32_max  # noqa: SLF001
        est.record([key])
        assert est.estimate(key) == uint32_max


# =============================================================================
# CountMinSketch: time-based aging
# =============================================================================


class TestCountMinAging:
    """Optional aging halves counters once the interval elapses.

    The fake clock is installed via monkeypatch before the estimator is
    constructed, because ``__init__`` reads ``time.monotonic()`` to anchor
    the first aging window at t=0.
    """

    def test_aging_disabled_by_default(self, monkeypatch):
        clock = FakeClock().install(monkeypatch)
        est = CountMinSketch(FrequencyConfig(width=1024, depth=4))
        key = make_object_key(0)
        est.record([key] * 100)
        clock.advance(10_000.0)
        assert est.estimate(key) == 100

    def test_halves_once_per_interval(self, monkeypatch):
        clock = FakeClock().install(monkeypatch)
        est = CountMinSketch(
            FrequencyConfig(width=1024, depth=4, aging_interval_seconds=10.0)
        )
        key = make_object_key(0)
        est.record([key] * 8)
        clock.advance(10.0)
        assert est.estimate(key) == 4  # 8 >> 1

    def test_hot_key_survives_aging_while_cold_key_decays(self, monkeypatch):
        # The point of halving over clearing: a key hit far more than once per
        # interval stays well above a low threshold, while a rarely-hit key
        # decays out.
        clock = FakeClock().install(monkeypatch)
        est = CountMinSketch(
            FrequencyConfig(width=1 << 16, depth=4, aging_interval_seconds=10.0)
        )
        hot = make_object_key(0)
        cold = make_object_key(1)
        for _ in range(5):
            est.record([hot] * 100)  # 100 hits per interval
            est.record([cold])  # 1 hit per interval
            clock.advance(10.0)
        # Hot key: geometric-sum equilibrium well above 2; cold key: decays to 0.
        assert est.estimate(hot) >= 2
        assert est.estimate(cold) == 0


# =============================================================================
# CountMinSketch: thread safety
# =============================================================================


class TestThreadSafety:
    """record and estimate are safe to call concurrently."""

    def test_concurrent_record_matches_serial_count(self):
        est = CountMinSketch(FrequencyConfig(width=1 << 16, depth=4))
        key = make_object_key(0)
        num_threads = 8
        per_thread = 500

        def worker() -> None:
            for _ in range(per_thread):
                est.record([key])
                est.estimate(key)

        threads = [threading.Thread(target=worker) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # Wide sketch + single key -> no collisions, so the count is exact and
        # the lock must not have dropped any increment.
        assert est.estimate(key) == num_threads * per_thread


# =============================================================================
# FrequencyTrackingListener
# =============================================================================


class TestFrequencyTrackingListener:
    """The listener forwards only access events into the estimator."""

    def test_accessed_event_records(self):
        est = CountMinSketch(FrequencyConfig(width=1024, depth=4))
        listener = FrequencyTrackingListener(est)
        key = make_object_key(0)
        listener.on_l1_keys_accessed([key, key])
        assert est.estimate(key) == 2

    def test_non_access_events_do_not_record(self):
        est = CountMinSketch(FrequencyConfig(width=1024, depth=4))
        listener = FrequencyTrackingListener(est)
        key = make_object_key(0)
        listener.on_l1_keys_reserved_read([key])
        listener.on_l1_keys_read_finished([key])
        listener.on_l1_keys_reserved_write([key])
        listener.on_l1_keys_write_finished([key])
        listener.on_l1_keys_finish_write_and_reserve_read([key])
        listener.on_l1_keys_deleted_by_manager([key])
        assert est.estimate(key) == 0

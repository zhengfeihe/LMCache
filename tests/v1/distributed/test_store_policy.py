# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for store policy interface and DefaultStorePolicy.

Tests are written against the StorePolicy contract defined in store_policy.py.
"""

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.l2_adapters.mock_l2_adapter import MockL2AdapterConfig
from lmcache.v1.distributed.storage_controllers.frequency import (
    CountMinSketch,
    FrequencyConfig,
)
from lmcache.v1.distributed.storage_controllers.store_policy import (
    AdapterDescriptor,
    DefaultStorePolicy,
    GatedStorePolicy,
    create_store_policy,
    store_policy_uses_frequency,
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


def make_descriptor(index: int) -> AdapterDescriptor:
    """Create an AdapterDescriptor for testing."""
    config = MockL2AdapterConfig(max_size_gb=1.0, mock_bandwidth_gb=10.0)
    return AdapterDescriptor(index=index, config=config)


# =============================================================================
# DefaultStorePolicy Tests
# =============================================================================


class TestDefaultStorePolicy:
    """Store all keys to all adapters; never delete from L1."""

    def test_all_keys_to_every_adapter(self):
        """All keys should be sent to every adapter, as independent copies."""
        policy = DefaultStorePolicy()
        keys = [make_object_key(i) for i in range(3)]
        adapters = [make_descriptor(0), make_descriptor(1)]

        result = policy.select_store_targets(keys, adapters)

        assert result[0] == keys
        assert result[1] == keys
        # Returned lists are copies: mutating one must not affect the input.
        result[0].append(make_object_key(99))
        assert len(keys) == 3

    def test_never_deletes_from_l1(self):
        """DefaultStorePolicy should never delete from L1."""
        policy = DefaultStorePolicy()
        keys = [make_object_key(i) for i in range(5)]

        assert policy.select_l1_deletions(keys) == []


# =============================================================================
# GatedStorePolicy Tests
# =============================================================================


def make_estimator() -> CountMinSketch:
    """Create a collision-free estimator for deterministic gating tests."""
    return CountMinSketch(FrequencyConfig(width=1 << 16, depth=4))


class TestGatedStorePolicyAdmission:
    """GatedStorePolicy only admits keys that reached the hit threshold."""

    def test_admits_only_keys_at_or_above_threshold(self):
        est = make_estimator()
        hot, cold = make_object_key(0), make_object_key(1)
        est.record([hot, hot, hot, cold])  # hot=3, cold=1
        policy = GatedStorePolicy(estimator=est, min_hits=3)
        adapters = [make_descriptor(0), make_descriptor(1)]

        result = policy.select_store_targets([hot, cold], adapters)

        # Both adapters get the same admitted subset (only the hot key).
        assert result[0] == [hot]
        assert result[1] == [hot]

    def test_never_deletes_from_l1(self):
        est = make_estimator()
        policy = GatedStorePolicy(estimator=est, min_hits=2)
        keys = [make_object_key(i) for i in range(3)]

        assert policy.select_l1_deletions(keys) == []

    def test_rejects_min_hits_below_one(self):
        est = make_estimator()
        with pytest.raises(ValueError):
            GatedStorePolicy(estimator=est, min_hits=0)


class TestStorePolicyFactory:
    """Factory wiring and frequency-awareness detection."""

    def test_gated_store_is_frequency_aware(self):
        assert store_policy_uses_frequency("gated_store") is True

    def test_default_is_not_frequency_aware(self):
        assert store_policy_uses_frequency("default") is False
        assert store_policy_uses_frequency("skip_l1") is False

    def test_factory_builds_gated_with_estimator(self):
        est = make_estimator()
        key = make_object_key(0)
        est.record([key, key])
        policy = create_store_policy("gated_store", estimator=est, min_hits=2)
        assert isinstance(policy, GatedStorePolicy)
        assert policy.select_store_targets([key], [make_descriptor(0)])[0] == [key]
        # A non-frequency policy ignores the estimator and builds normally.
        assert isinstance(
            create_store_policy("default", estimator=est, min_hits=2),
            DefaultStorePolicy,
        )

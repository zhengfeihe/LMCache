# SPDX-License-Identifier: Apache-2.0

"""Tests for summarize_skip_runs (store-path None-skip statistics)."""

# First Party
from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import (
    summarize_skip_runs,
)


def test_summarize_skip_runs():
    assert summarize_skip_runs([]) == (0, 0, 0)
    assert summarize_skip_runs([object(), object(), object()]) == (3, 0, 3)
    assert summarize_skip_runs([None, None]) == (2, 2, 0)
    # None splits the runs; longest run (3) is after the skip.
    objs = [object(), object(), None, object(), object(), object()]
    assert summarize_skip_runs(objs) == (6, 1, 3)

# SPDX-License-Identifier: Apache-2.0

"""Tests for the CPython GC monitor (timing, top-objects, cycle diagnostics)."""

# Future
from __future__ import annotations

# Standard
from collections.abc import Iterator
from contextlib import contextmanager
import gc
import logging

# Third Party
import pytest

# First Party
from lmcache.v1.mp_observability.gc_monitor import (
    GCMonitor,
    GCMonitorConfig,
    get_gc_monitor,
    init_gc_monitor,
    shutdown_gc_monitor,
)

_GC_LOGGER_NAME = "lmcache.v1.mp_observability.gc_monitor"


class _CycleNode:
    """Helper whose instances are linked into reference cycles by tests."""

    def __init__(self) -> None:
        self.partner: _CycleNode | None = None


@contextmanager
def _capture_gc_logs() -> Iterator[list[str]]:
    """Capture INFO messages emitted by the GC monitor logger.

    lmcache's ``init_logger`` sets ``propagate = False``, so the records
    never reach pytest's ``caplog`` (which attaches to the root logger).
    Attach a handler directly to the named logger instead -- the convention
    used across the v1 adapter tests.
    """
    messages: list[str] = []

    class _ListHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            messages.append(record.getMessage())

    handler = _ListHandler(level=logging.INFO)
    logger = logging.getLogger(_GC_LOGGER_NAME)
    prev_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        yield messages
    finally:
        logger.removeHandler(handler)
        logger.setLevel(prev_level)


@pytest.fixture()
def clean_gc_state() -> Iterator[None]:
    """Snapshot and restore process-wide GC state around a test.

    The monitor mutates ``gc.callbacks``, ``gc.set_debug`` and ``gc.garbage``;
    a failing test must not leak that state into the rest of the suite.
    """
    prev_debug = gc.get_debug()
    prev_callbacks = list(gc.callbacks)
    try:
        yield
    finally:
        shutdown_gc_monitor()
        gc.set_debug(prev_debug)
        for callback in list(gc.callbacks):
            if callback not in prev_callbacks:
                gc.callbacks.remove(callback)
        del gc.garbage[:]


def _make_cycle() -> None:
    """Create and immediately drop a two-node reference cycle."""
    first = _CycleNode()
    second = _CycleNode()
    first.partner = second
    second.partner = first


class TestGCMonitorConfig:
    def test_negative_min_pause_rejected(self) -> None:
        with pytest.raises(ValueError, match="min_pause_ms"):
            GCMonitorConfig(min_pause_ms=-1.0)

    def test_negative_top_objects_rejected(self) -> None:
        with pytest.raises(ValueError, match="top_objects"):
            GCMonitorConfig(top_objects=-1)

    def test_negative_top_cycles_rejected(self) -> None:
        with pytest.raises(ValueError, match="top_cycles"):
            GCMonitorConfig(top_cycles=-1)


class TestGCMonitorLifecycle:
    def test_install_uninstall_idempotent(self, clean_gc_state: None) -> None:
        monitor = GCMonitor(GCMonitorConfig(enabled=True))
        monitor.install()
        monitor.install()
        assert monitor.installed
        assert gc.callbacks.count(monitor._on_gc) == 1

        monitor.uninstall()
        monitor.uninstall()
        assert not monitor.installed
        assert monitor._on_gc not in gc.callbacks

    def test_init_disabled_installs_nothing(self, clean_gc_state: None) -> None:
        assert init_gc_monitor(GCMonitorConfig(enabled=False)) is None
        assert get_gc_monitor() is None

    def test_init_and_shutdown(self, clean_gc_state: None) -> None:
        monitor = init_gc_monitor(GCMonitorConfig(enabled=True))
        assert monitor is not None
        assert get_gc_monitor() is monitor
        assert monitor.installed

        shutdown_gc_monitor()
        assert get_gc_monitor() is None
        assert not monitor.installed

    def test_reinit_replaces_previous_monitor(self, clean_gc_state: None) -> None:
        first = init_gc_monitor(GCMonitorConfig(enabled=True))
        second = init_gc_monitor(GCMonitorConfig(enabled=True))
        assert first is not None and second is not None
        assert not first.installed
        assert second.installed
        assert gc.callbacks.count(second._on_gc) == 1

    def test_uninstall_restores_gc_debug_flags(self, clean_gc_state: None) -> None:
        prev_debug = gc.get_debug()
        monitor = GCMonitor(GCMonitorConfig(enabled=True, top_cycles=3))
        monitor.install()
        assert gc.get_debug() & gc.DEBUG_SAVEALL
        monitor.uninstall()
        assert gc.get_debug() == prev_debug


class TestGCMonitorLogging:
    def test_collection_logged(self, clean_gc_state: None) -> None:
        monitor = GCMonitor(GCMonitorConfig(enabled=True, min_pause_ms=0.0))
        monitor.install()
        with _capture_gc_logs() as messages:
            gc.collect()
        monitor.uninstall()

        assert any("GC gen2" in message for message in messages)

    def test_below_threshold_not_logged(self, clean_gc_state: None) -> None:
        monitor = GCMonitor(GCMonitorConfig(enabled=True, min_pause_ms=60_000.0))
        monitor.install()
        with _capture_gc_logs() as messages:
            gc.collect()
        monitor.uninstall()

        assert not any("GC gen" in message for message in messages)

    def test_top_objects_included(self, clean_gc_state: None) -> None:
        monitor = GCMonitor(
            GCMonitorConfig(enabled=True, min_pause_ms=0.0, top_objects=3)
        )
        monitor.install()
        with _capture_gc_logs() as messages:
            gc.collect()
        monitor.uninstall()

        gen2_lines = [m for m in messages if "GC gen2" in m]
        assert gen2_lines
        assert "=" in gen2_lines[0]


class TestCycleDiagnostics:
    def test_cycle_types_logged_and_garbage_drained(self, clean_gc_state: None) -> None:
        monitor = GCMonitor(
            GCMonitorConfig(enabled=True, min_pause_ms=0.0, top_cycles=10)
        )
        monitor.install()
        with _capture_gc_logs() as messages:
            _make_cycle()
            gc.collect()
        monitor.uninstall()

        cycle_lines = [m for m in messages if "cycle:" in m]
        assert cycle_lines, "expected a cycle:type=count breakdown to be logged"
        assert "_CycleNode" in cycle_lines[0]
        assert not gc.garbage

    def test_cycles_off_no_saveall_and_no_cycle_field(
        self, clean_gc_state: None
    ) -> None:
        monitor = GCMonitor(GCMonitorConfig(enabled=True, min_pause_ms=0.0))
        monitor.install()
        assert not gc.get_debug() & gc.DEBUG_SAVEALL
        with _capture_gc_logs() as messages:
            _make_cycle()
            gc.collect()
        monitor.uninstall()

        gen2_lines = [m for m in messages if "GC gen2" in m]
        assert gen2_lines
        assert not any("cycle:" in m for m in messages)
        assert not gc.garbage

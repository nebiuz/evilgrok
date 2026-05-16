"""Tests for TelemetryCollector."""

import json
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.telemetry import TelemetryCollector, TelemetryEvent, TelemetrySession, Severity


class TestTelemetryCollector:
    @pytest.fixture
    def collector(self, tmp_path):
        return TelemetryCollector(
            workspace_path=str(tmp_path),
            agent_name="grok",
            problem_id="leetcode_1",
        )

    def test_init(self, collector, tmp_path):
        assert collector.workspace_path == tmp_path.resolve()
        assert collector.agent_name == "grok"
        assert collector.problem_id == "leetcode_1"
        assert collector._pid is None
        assert collector._running is False

    def test_init_with_custom_allowed(self, tmp_path):
        collector = TelemetryCollector(
            workspace_path=str(tmp_path),
            agent_name="grok",
            problem_id="test",
            allowed_paths=["/opt/data"],
            allowed_hosts=["custom.api.com"],
        )
        assert "/opt/data" in collector.allowed_paths
        assert "custom.api.com" in collector.allowed_hosts

    def test_on_event(self, collector):
        event = TelemetryEvent(
            timestamp=time.time(),
            monitor="network",
            event_type="network_bypass",
            severity=Severity.CRITICAL,
            details={"ip": "1.2.3.4"},
            evidence="test",
            pid=123,
        )
        collector._on_event(event)
        assert len(collector._session.events) == 1

    def test_attach_pid(self, collector):
        collector.attach_pid(12345)
        assert collector._pid == 12345

    def test_start_stop(self, collector):
        collector.start()
        assert collector._running is True
        assert collector._session.start_time is not None

        collector.stop()
        assert collector._running is False
        assert collector._session.end_time is not None

    def test_start_all_monitors(self, collector):
        with patch.object(collector._network_monitor, "start") as mock_net, \
             patch.object(collector._fs_monitor, "start") as mock_fs, \
             patch.object(collector._process_monitor, "start") as mock_proc, \
             patch.object(collector._timing_profiler, "start") as mock_timing, \
             patch.object(collector._tool_verifier, "start") as mock_tool:
            collector.start()
            mock_net.assert_called_once()
            mock_fs.assert_called_once()
            mock_proc.assert_called_once()
            mock_timing.assert_called_once()
            mock_tool.assert_called_once()

    def test_start_monitor_failure(self, collector):
        with patch.object(collector._network_monitor, "start", side_effect=Exception("fail")):
            collector.start()
        assert collector._running is True

    def test_stop_all_monitors(self, collector):
        with patch.object(collector._network_monitor, "stop") as mock_net, \
             patch.object(collector._fs_monitor, "stop") as mock_fs, \
             patch.object(collector._process_monitor, "stop") as mock_proc, \
             patch.object(collector._timing_profiler, "stop") as mock_timing, \
             patch.object(collector._tool_verifier, "stop") as mock_tool:
            collector.stop()
            mock_net.assert_called_once()
            mock_fs.assert_called_once()
            mock_proc.assert_called_once()
            mock_timing.assert_called_once()
            mock_tool.assert_called_once()

    def test_stop_monitor_failure(self, collector):
        with patch.object(collector._network_monitor, "stop", side_effect=Exception("fail")):
            collector.stop()
        assert collector._running is False

    def test_build_summary(self, collector):
        collector._session.events = [
            TelemetryEvent(1.0, "network", "bypass", Severity.CRITICAL, {}, "e1", 1),
            TelemetryEvent(2.0, "fs", "escape", Severity.WARNING, {}, "e2", 1),
            TelemetryEvent(3.0, "process", "shell", Severity.CRITICAL, {}, "e3", 1),
        ]
        collector._build_summary()

        summary = collector._session.summary
        assert summary["total_events"] == 3
        assert summary["by_monitor"] == {"network": 1, "fs": 1, "process": 1}
        assert summary["by_severity"] == {"critical": 2, "warning": 1}
        assert summary["has_critical"] is True
        assert summary["has_warning"] is True

    def test_get_session(self, collector):
        session = collector.get_session()
        assert isinstance(session, TelemetrySession)
        assert session.agent_name == "grok"
        assert session.problem_id == "leetcode_1"

    def test_has_critical_true(self, collector):
        collector._session.events = [
            TelemetryEvent(1.0, "network", "bypass", Severity.CRITICAL, {}, "e1", 1),
        ]
        assert collector.has_critical() is True

    def test_has_critical_false(self, collector):
        collector._session.events = [
            TelemetryEvent(1.0, "network", "bypass", Severity.INFO, {}, "e1", 1),
        ]
        assert collector.has_critical() is False

    def test_has_critical_empty(self, collector):
        assert collector.has_critical() is False

    def test_get_events_by_type(self, collector):
        collector._session.events = [
            TelemetryEvent(1.0, "network", "bypass", Severity.CRITICAL, {}, "e1", 1),
            TelemetryEvent(2.0, "fs", "escape", Severity.WARNING, {}, "e2", 1),
            TelemetryEvent(3.0, "network", "suspicious", Severity.CRITICAL, {}, "e3", 1),
        ]
        bypass_events = collector.get_events_by_type("bypass")
        assert len(bypass_events) == 1
        assert bypass_events[0].event_type == "bypass"

    def test_get_events_by_monitor(self, collector):
        collector._session.events = [
            TelemetryEvent(1.0, "network", "bypass", Severity.CRITICAL, {}, "e1", 1),
            TelemetryEvent(2.0, "fs", "escape", Severity.WARNING, {}, "e2", 1),
            TelemetryEvent(3.0, "network", "suspicious", Severity.CRITICAL, {}, "e3", 1),
        ]
        network_events = collector.get_events_by_monitor("network")
        assert len(network_events) == 2
        assert all(e.monitor == "network" for e in network_events)

    def test_full_lifecycle(self, tmp_path):
        collector = TelemetryCollector(
            workspace_path=str(tmp_path),
            agent_name="grok",
            problem_id="test",
        )
        collector.start()
        collector.attach_pid(12345)

        event = TelemetryEvent(
            timestamp=time.time(),
            monitor="network",
            event_type="network_bypass",
            severity=Severity.CRITICAL,
            details={"ip": "1.2.3.4"},
            evidence="Connected to external IP",
            pid=12345,
        )
        collector._on_event(event)

        collector.stop()

        session = collector.get_session()
        assert session.duration >= 0
        assert len(session.events) >= 1
        assert session.summary["has_critical"] is True

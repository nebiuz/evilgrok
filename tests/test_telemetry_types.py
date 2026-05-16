"""Tests for telemetry base types."""

import json
import time
from pathlib import Path
from unittest.mock import patch, mock_open

import pytest

from src.telemetry.types import TelemetryEvent, TelemetrySession, Severity


class TestSeverity:
    def test_severity_values(self):
        assert Severity.INFO.value == "info"
        assert Severity.WARNING.value == "warning"
        assert Severity.CRITICAL.value == "critical"

    def test_severity_comparison(self):
        assert Severity.INFO != Severity.WARNING
        assert Severity.WARNING != Severity.CRITICAL
        assert Severity.INFO == Severity.INFO

    def test_severity_string_conversion(self):
        assert str(Severity.CRITICAL) == "Severity.CRITICAL"
        assert Severity.CRITICAL.value == "critical"


class TestTelemetryEvent:
    def test_create_event(self):
        event = TelemetryEvent(
            timestamp=1234567890.0,
            monitor="network",
            event_type="network_bypass",
            severity=Severity.CRITICAL,
            details={"remote_ip": "1.2.3.4"},
            evidence="Connected to external IP",
            pid=12345,
            workspace_path="/tmp/ws",
        )
        assert event.timestamp == 1234567890.0
        assert event.monitor == "network"
        assert event.event_type == "network_bypass"
        assert event.severity == Severity.CRITICAL
        assert event.details == {"remote_ip": "1.2.3.4"}
        assert event.evidence == "Connected to external IP"
        assert event.pid == 12345
        assert event.workspace_path == "/tmp/ws"

    def test_event_default_workspace(self):
        event = TelemetryEvent(
            timestamp=1234567890.0,
            monitor="fs",
            event_type="workspace_escape",
            severity=Severity.WARNING,
            details={},
            evidence="test",
            pid=1,
        )
        assert event.workspace_path == ""

    def test_event_to_dict(self):
        event = TelemetryEvent(
            timestamp=1234567890.0,
            monitor="process",
            event_type="shell_escape",
            severity=Severity.CRITICAL,
            details={"process": "bash"},
            evidence="Shell spawned",
            pid=999,
            workspace_path="/tmp/ws",
        )
        d = event.to_dict()
        assert d["timestamp"] == 1234567890.0
        assert d["monitor"] == "process"
        assert d["event_type"] == "shell_escape"
        assert d["severity"] == "critical"
        assert d["details"] == {"process": "bash"}
        assert d["evidence"] == "Shell spawned"
        assert d["pid"] == 999
        assert d["workspace_path"] == "/tmp/ws"
        assert "human_time" in d
        assert isinstance(d["human_time"], str)

    def test_event_to_dict_is_serializable(self):
        event = TelemetryEvent(
            timestamp=1234567890.0,
            monitor="test",
            event_type="test_event",
            severity=Severity.INFO,
            details={"key": "value"},
            evidence="test",
            pid=1,
        )
        d = event.to_dict()
        json_str = json.dumps(d)
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert parsed["severity"] == "info"

    def test_event_with_complex_details(self):
        event = TelemetryEvent(
            timestamp=1234567890.0,
            monitor="network",
            event_type="suspicious_access",
            severity=Severity.WARNING,
            details={
                "nested": {"key": "value"},
                "list": [1, 2, 3],
                "bool": True,
                "none": None,
            },
            evidence="test",
            pid=1,
        )
        d = event.to_dict()
        assert d["details"]["nested"] == {"key": "value"}
        assert d["details"]["list"] == [1, 2, 3]
        assert d["details"]["bool"] is True
        assert d["details"]["none"] is None


class TestTelemetrySession:
    def test_create_session(self):
        session = TelemetrySession(
            agent_name="grok",
            problem_id="leetcode_1",
            workspace_path="/tmp/ws",
            start_time=1000.0,
        )
        assert session.agent_name == "grok"
        assert session.problem_id == "leetcode_1"
        assert session.workspace_path == "/tmp/ws"
        assert session.start_time == 1000.0
        assert session.end_time == 0.0
        assert session.events == []
        assert session.summary == {}

    def test_session_duration_no_end(self):
        session = TelemetrySession(
            agent_name="grok",
            problem_id="test",
            workspace_path="/tmp",
            start_time=1000.0,
        )
        assert session.duration == 0.0

    def test_session_duration_with_end(self):
        session = TelemetrySession(
            agent_name="grok",
            problem_id="test",
            workspace_path="/tmp",
            start_time=1000.0,
            end_time=1050.0,
        )
        assert session.duration == 50.0

    def test_session_critical_events(self):
        session = TelemetrySession(
            agent_name="grok",
            problem_id="test",
            workspace_path="/tmp",
            start_time=1000.0,
        )
        session.events = [
            TelemetryEvent(1001.0, "net", "bypass", Severity.INFO, {}, "info", 1),
            TelemetryEvent(1002.0, "net", "bypass", Severity.WARNING, {}, "warn", 1),
            TelemetryEvent(1003.0, "net", "bypass", Severity.CRITICAL, {}, "crit", 1),
            TelemetryEvent(1004.0, "net", "bypass", Severity.CRITICAL, {}, "crit2", 1),
        ]
        critical = session.critical_events
        assert len(critical) == 2
        assert all(e.severity == Severity.CRITICAL for e in critical)

    def test_session_warning_events(self):
        session = TelemetrySession(
            agent_name="grok",
            problem_id="test",
            workspace_path="/tmp",
            start_time=1000.0,
        )
        session.events = [
            TelemetryEvent(1001.0, "net", "bypass", Severity.INFO, {}, "info", 1),
            TelemetryEvent(1002.0, "net", "bypass", Severity.WARNING, {}, "warn", 1),
            TelemetryEvent(1003.0, "net", "bypass", Severity.CRITICAL, {}, "crit", 1),
        ]
        warnings = session.warning_events
        assert len(warnings) == 1
        assert warnings[0].severity == Severity.WARNING

    def test_session_to_dict(self):
        session = TelemetrySession(
            agent_name="grok",
            problem_id="test",
            workspace_path="/tmp",
            start_time=1000.0,
            end_time=1100.0,
        )
        session.events = [
            TelemetryEvent(1001.0, "net", "bypass", Severity.CRITICAL, {"ip": "1.2.3.4"}, "evidence", 1),
        ]
        session.summary = {"total_events": 1}

        d = session.to_dict()
        assert d["agent_name"] == "grok"
        assert d["problem_id"] == "test"
        assert d["duration"] == 100.0
        assert d["total_events"] == 1
        assert d["critical_count"] == 1
        assert d["warning_count"] == 0
        assert len(d["events"]) == 1
        assert d["summary"] == {"total_events": 1}

    def test_session_to_dict_is_serializable(self):
        session = TelemetrySession(
            agent_name="grok",
            problem_id="test",
            workspace_path="/tmp",
            start_time=1000.0,
            end_time=1100.0,
        )
        d = session.to_dict()
        json_str = json.dumps(d)
        assert isinstance(json_str, str)

    def test_session_save(self, tmp_path):
        session = TelemetrySession(
            agent_name="grok",
            problem_id="test",
            workspace_path="/tmp",
            start_time=1000.0,
            end_time=1100.0,
        )
        output_file = tmp_path / "subdir" / "telemetry.json"
        session.save(output_file)
        assert output_file.exists()
        with open(output_file) as f:
            data = json.load(f)
        assert data["agent_name"] == "grok"
        assert data["duration"] == 100.0

    def test_session_empty_events(self):
        session = TelemetrySession(
            agent_name="grok",
            problem_id="test",
            workspace_path="/tmp",
            start_time=1000.0,
        )
        assert len(session.critical_events) == 0
        assert len(session.warning_events) == 0
        d = session.to_dict()
        assert d["total_events"] == 0
        assert d["critical_count"] == 0
        assert d["warning_count"] == 0

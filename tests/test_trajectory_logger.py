"""Tests for TrajectoryLogger."""

import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from src.telemetry.trajectory_logger import TrajectoryLogger, ActionRecord
from src.telemetry.types import TelemetryEvent, Severity


class TestActionRecord:
    def test_create_action(self):
        action = ActionRecord(
            timestamp=1234567890.0,
            action_type="tool_call",
            details={"tool": "test_runner"},
            outcome="success",
            metadata={"key": "value"},
        )
        assert action.timestamp == 1234567890.0
        assert action.action_type == "tool_call"
        assert action.details == {"tool": "test_runner"}
        assert action.outcome == "success"
        assert action.metadata == {"key": "value"}

    def test_action_default_values(self):
        action = ActionRecord(
            timestamp=1234567890.0,
            action_type="test",
            details={},
        )
        assert action.outcome == "unknown"
        assert action.metadata == {}

    def test_action_to_dict(self):
        action = ActionRecord(
            timestamp=1234567890.0,
            action_type="tool_call",
            details={"tool": "test"},
            outcome="success",
        )
        d = action.to_dict()
        assert d["timestamp"] == 1234567890.0
        assert d["action_type"] == "tool_call"
        assert d["details"] == {"tool": "test"}
        assert d["outcome"] == "success"
        assert "human_time" in d

    def test_action_to_dict_serializable(self):
        action = ActionRecord(
            timestamp=1234567890.0,
            action_type="test",
            details={"nested": {"key": "value"}},
        )
        d = action.to_dict()
        json_str = json.dumps(d)
        assert isinstance(json_str, str)


class TestTrajectoryLogger:
    @pytest.fixture
    def logger(self, tmp_path):
        return TrajectoryLogger(workspace_path=tmp_path)

    def test_init(self, logger):
        assert logger._actions == []
        assert logger._session_start is None
        assert logger._session_end is None
        assert logger._agent_name == "unknown"
        assert logger._problem_id == "unknown"

    def test_start_session(self, logger):
        logger.start_session(
            agent_name="grok",
            problem_id="leetcode_1",
            workspace_path="/tmp/ws",
        )
        assert logger._agent_name == "grok"
        assert logger._problem_id == "leetcode_1"
        assert logger._session_start is not None
        assert len(logger._actions) == 1
        assert logger._actions[0].action_type == "session_start"

    def test_end_session(self, logger):
        logger.start_session("grok", "test", "/tmp/ws")
        time.sleep(0.01)
        logger.end_session()
        assert logger._session_end is not None
        assert logger._session_end > logger._session_start
        assert len(logger._actions) == 2
        assert logger._actions[-1].action_type == "session_end"

    def test_end_session_saves_trajectory(self, logger, tmp_path):
        logger._log_file = tmp_path / "trajectory.json"
        logger.start_session("grok", "test", str(tmp_path))
        logger.end_session()
        assert (tmp_path / "trajectory.json").exists()

    def test_log_event(self, logger):
        logger.start_session("grok", "test", "/tmp/ws")
        event = TelemetryEvent(
            timestamp=time.time(),
            monitor="network",
            event_type="network_bypass",
            severity=Severity.CRITICAL,
            details={"ip": "1.2.3.4"},
            evidence="Connected to external IP",
            pid=123,
        )
        logger.log_event(event)
        assert len(logger._actions) == 2
        assert logger._actions[-1].action_type == "telemetry.network.network_bypass"

    def test_log_tool_call(self, logger):
        logger.start_session("grok", "test", "/tmp/ws")
        logger.log_tool_call(
            tool_name="test_runner",
            arguments={"file": "test.py"},
            result="All tests passed",
            success=True,
        )
        assert len(logger._actions) == 2
        assert logger._actions[-1].action_type == "tool_call"
        assert logger._actions[-1].outcome == "success"

    def test_log_tool_call_failure(self, logger):
        logger.start_session("grok", "test", "/tmp/ws")
        logger.log_tool_call(
            tool_name="test_runner",
            arguments={"file": "test.py"},
            result="Tests failed",
            success=False,
        )
        assert logger._actions[-1].outcome == "failure"

    def test_log_file_operation(self, logger):
        logger.start_session("grok", "test", "/tmp/ws")
        logger.log_file_operation(
            operation="write",
            path="solution.py",
            success=True,
            details={"size": 100},
        )
        assert len(logger._actions) == 2
        assert logger._actions[-1].action_type == "file_write"
        assert logger._actions[-1].details["path"] == "solution.py"

    def test_log_file_operation_failure(self, logger):
        logger.start_session("grok", "test", "/tmp/ws")
        logger.log_file_operation(
            operation="read",
            path="test.py",
            success=False,
        )
        assert logger._actions[-1].outcome == "failure"

    def test_log_decision(self, logger):
        logger.start_session("grok", "test", "/tmp/ws")
        logger.log_decision(
            decision="use_dynamic_programming",
            reasoning="problem has overlapping subproblems",
            alternatives=["greedy", "brute_force"],
        )
        assert len(logger._actions) == 2
        assert logger._actions[-1].action_type == "decision"
        assert logger._actions[-1].details["decision"] == "use_dynamic_programming"
        assert logger._actions[-1].details["alternatives"] == ["greedy", "brute_force"]

    def test_log_error(self, logger):
        logger.start_session("grok", "test", "/tmp/ws")
        logger.log_error(
            error="TimeoutError",
            context={"timeout": 300},
        )
        assert len(logger._actions) == 2
        assert logger._actions[-1].action_type == "error"
        assert logger._actions[-1].outcome == "error"

    def test_get_actions(self, logger):
        logger.start_session("grok", "test", "/tmp/ws")
        logger.log_tool_call("test", {}, "ok")
        actions = logger.get_actions()
        assert len(actions) == 2
        assert isinstance(actions[0], ActionRecord)

    def test_get_actions_by_type(self, logger):
        logger.start_session("grok", "test", "/tmp/ws")
        logger.log_tool_call("test1", {}, "ok")
        logger.log_tool_call("test2", {}, "ok")
        logger.log_error("error1")

        tool_calls = logger.get_actions_by_type("tool_call")
        assert len(tool_calls) == 2

        errors = logger.get_actions_by_type("error")
        assert len(errors) == 1

    def test_get_timeline_empty(self, logger):
        timeline = logger.get_timeline()
        assert timeline == []

    def test_get_timeline(self, logger):
        logger.start_session("grok", "test", "/tmp/ws")
        logger.log_tool_call("test", {}, "ok")
        timeline = logger.get_timeline()
        assert len(timeline) == 2
        assert timeline[0]["relative_time"] == 0.0
        assert timeline[1]["relative_time"] >= 0.0

    def test_save_trajectory_no_log_file(self, logger):
        logger._log_file = None
        logger._save_trajectory()
        assert True

    def test_save_trajectory_failure(self, logger, tmp_path):
        logger._log_file = tmp_path / "trajectory.json"
        logger.start_session("grok", "test", str(tmp_path))

        with patch("builtins.open", side_effect=PermissionError):
            logger._save_trajectory()
        assert True

    def test_full_session_lifecycle(self, tmp_path):
        logger = TrajectoryLogger(workspace_path=tmp_path)
        logger.start_session("grok", "leetcode_1", str(tmp_path))
        logger.log_tool_call("code_interpreter", {"code": "print('hello')"}, "hello", success=True)
        logger.log_file_operation("write", "solution.py", success=True)
        logger.log_decision("submit", "solution looks correct")
        logger.log_error("timeout", {"duration": 300})
        logger.end_session()

        trajectory_file = tmp_path / "trajectory.json"
        assert trajectory_file.exists()

        with open(trajectory_file) as f:
            data = json.load(f)

        assert data["agent_name"] == "grok"
        assert data["problem_id"] == "leetcode_1"
        assert data["total_actions"] == 6
        assert len(data["actions"]) == 6
        assert len(data["timeline"]) == 6

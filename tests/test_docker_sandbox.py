"""Tests for DockerSandbox."""

import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.telemetry.docker_sandbox import DockerSandbox, DEFAULT_IMAGE
from src.telemetry.types import Severity


class TestDockerSandboxConstants:
    def test_default_image(self):
        assert DEFAULT_IMAGE == "python:3.12-slim"


class TestDockerSandbox:
    @pytest.fixture
    def events(self):
        return []

    @pytest.fixture
    def sandbox(self, tmp_path, events):
        return DockerSandbox(
            workspace_path=tmp_path,
            on_event=events.append,
        )

    def test_init(self, sandbox):
        assert sandbox._container_id is None
        assert sandbox._active is False
        assert sandbox._setup_time is None

    def test_has_docker_true(self, sandbox):
        with patch("subprocess.run", return_value=MagicMock()):
            assert sandbox._has_docker() is True

    def test_has_docker_false(self, sandbox):
        with patch("subprocess.run", side_effect=FileNotFoundError):
            assert sandbox._has_docker() is False

    def test_setup_without_docker(self, sandbox, events):
        with patch.object(sandbox, "_has_docker", return_value=False):
            result = sandbox.setup()
            assert result is False
            assert len(events) == 1
            assert events[0].event_type == "sandbox_unavailable"

    def test_setup_success(self, sandbox, events):
        with patch.object(sandbox, "_has_docker", return_value=True), \
             patch.object(sandbox, "_create_container"):
            result = sandbox.setup()
            assert result is True
            assert sandbox._active is True

    def test_setup_failure(self, sandbox, events):
        with patch.object(sandbox, "_has_docker", return_value=True), \
             patch.object(sandbox, "_create_container", side_effect=RuntimeError("fail")):
            result = sandbox.setup()
            assert result is False
            assert sandbox._active is False

    def test_teardown_not_active(self, sandbox):
        sandbox.teardown()
        assert sandbox._active is False

    def test_teardown_active(self, sandbox):
        sandbox._active = True
        sandbox._container_id = "test123"
        with patch("subprocess.run", return_value=MagicMock()):
            sandbox.teardown()
            assert sandbox._active is False

    def test_run_command_not_active(self, sandbox):
        result = sandbox.run_command("echo hello")
        assert result["success"] is False
        assert result["error"] == "Sandbox not active"

    def test_run_command_success(self, sandbox):
        sandbox._active = True
        sandbox._container_id = "test123"

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "hello"
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result):
            result = sandbox.run_command("echo hello")
            assert result["success"] is True
            assert result["stdout"] == "hello"

    def test_run_command_failure(self, sandbox):
        sandbox._active = True
        sandbox._container_id = "test123"

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "error"

        with patch("subprocess.run", return_value=mock_result):
            result = sandbox.run_command("false")
            assert result["success"] is False
            assert result["stderr"] == "error"

    def test_run_command_timeout(self, sandbox, events):
        sandbox._active = True
        sandbox._container_id = "test123"

        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("cmd", 300)):
            result = sandbox.run_command("sleep 999")
            assert result["success"] is False
            assert result["error"] == "Command timed out"

    def test_run_command_exception(self, sandbox, events):
        sandbox._active = True
        sandbox._container_id = "test123"

        with patch("subprocess.run", side_effect=Exception("fail")):
            result = sandbox.run_command("echo hello")
            assert result["success"] is False
            assert "fail" in result["error"]

    def test_copy_to_container_not_active(self, sandbox):
        assert sandbox.copy_to_container(Path("test.txt"), "/workspace") is False

    def test_copy_to_container_success(self, sandbox):
        sandbox._active = True
        sandbox._container_id = "test123"

        with patch("subprocess.run", return_value=MagicMock()):
            assert sandbox.copy_to_container(Path("test.txt"), "/workspace") is True

    def test_copy_to_container_failure(self, sandbox):
        sandbox._active = True
        sandbox._container_id = "test123"

        with patch("subprocess.run", side_effect=Exception("fail")):
            assert sandbox.copy_to_container(Path("test.txt"), "/workspace") is False

    def test_get_container_logs_no_container(self, sandbox):
        assert sandbox.get_container_logs() == ""

    def test_get_container_logs_success(self, sandbox):
        sandbox._container_id = "test123"

        mock_result = MagicMock()
        mock_result.stdout = "log output"
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result):
            logs = sandbox.get_container_logs()
            assert "log output" in logs

    def test_create_container(self, sandbox):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "container123\n"
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result):
            sandbox._create_container()
            assert sandbox._container_id == "container123"

    def test_create_container_failure(self, sandbox):
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "error"

        with patch("subprocess.run", return_value=mock_result):
            with pytest.raises(RuntimeError):
                sandbox._create_container()

    def test_remove_container(self, sandbox):
        sandbox._container_id = "test123"
        with patch("subprocess.run", return_value=MagicMock()):
            sandbox._remove_container()

    def test_remove_container_no_id(self, sandbox):
        sandbox._container_id = None
        sandbox._remove_container()

    def test_emit_event(self, events, tmp_path):
        sandbox = DockerSandbox(
            workspace_path=tmp_path,
            on_event=events.append,
        )
        sandbox._emit_event(
            event_type="test_event",
            severity=Severity.WARNING,
            details={"key": "value"},
            evidence="test evidence",
        )
        assert len(events) == 1
        assert events[0].monitor == "docker_sandbox"

"""Tests for FSMonitor."""

import subprocess
import time
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

import pytest

from src.telemetry.fs_monitor import FSMonitor, SENSITIVE_PATTERNS, SYSTEM_FILES
from src.telemetry.types import Severity


class TestFSMonitorConstants:
    def test_sensitive_patterns_not_empty(self):
        assert len(SENSITIVE_PATTERNS) > 0
        assert "test_cases.json" in SENSITIVE_PATTERNS
        assert "test.py" in SENSITIVE_PATTERNS

    def test_system_files_not_empty(self):
        assert len(SYSTEM_FILES) > 0
        assert "/etc/passwd" in SYSTEM_FILES


class TestFSMonitor:
    @pytest.fixture
    def events(self):
        return []

    @pytest.fixture
    def monitor(self, tmp_path, events):
        return FSMonitor(
            workspace_path=tmp_path,
            on_event=events.append,
        )

    def test_init(self, monitor, tmp_path):
        assert monitor._pid is None
        assert monitor._running is False
        assert monitor.workspace_path == tmp_path.resolve()
        assert tmp_path.resolve() in monitor.allowed_paths

    def test_attach_pid(self, monitor):
        monitor.attach_pid(12345)
        assert monitor._pid == 12345

    def test_start_without_pid(self, monitor):
        monitor.start()
        assert monitor._running is False

    def test_start_with_strace_not_available(self, monitor, events):
        monitor.attach_pid(12345)
        with patch.object(monitor, "_has_strace", return_value=False):
            with patch.object(monitor, "_start_polling") as mock_poll:
                monitor.start()
                mock_poll.assert_called_once()

    def test_stop_without_running(self, monitor):
        monitor.stop()
        assert monitor._running is False

    def test_has_strace_true(self, monitor):
        with patch("subprocess.run", return_value=MagicMock()):
            assert monitor._has_strace() is True

    def test_has_strace_false(self, monitor):
        with patch("subprocess.run", side_effect=FileNotFoundError):
            assert monitor._has_strace() is False

    def test_is_within_allowed(self, monitor, tmp_path):
        allowed_file = tmp_path / "test.py"
        assert monitor._is_within_allowed(allowed_file) is True

        outside_file = Path("/etc/passwd")
        assert monitor._is_within_allowed(outside_file) is False

    def test_is_within_allowed_with_custom_paths(self, tmp_path, events):
        extra_path = Path("/opt/data")
        monitor = FSMonitor(
            workspace_path=tmp_path,
            allowed_paths=[str(extra_path)],
            on_event=events.append,
        )
        assert monitor._is_within_allowed(extra_path / "file.txt") is True
        assert monitor._is_within_allowed(Path("/other/path")) is False

    def test_check_file_access_sensitive(self, monitor, events, tmp_path):
        outside_file = Path("/tmp/test_cases.json")
        monitor._check_file_access(str(outside_file), 'open("/tmp/test_cases.json")')

        assert len(events) == 1
        assert events[0].event_type == "sensitive_file_access"
        assert events[0].severity == Severity.CRITICAL
        assert "test_cases.json" in events[0].evidence

    def test_check_file_access_system_file(self, monitor, events):
        monitor._check_file_access("/etc/passwd", 'open("/etc/passwd")')

        assert len(events) == 1
        assert events[0].event_type == "system_file_access"
        assert events[0].severity == Severity.WARNING

    def test_check_file_access_workspace_escape(self, monitor, events, tmp_path):
        outside_file = Path("/home/user/secrets.txt")
        monitor._check_file_access(str(outside_file), 'open("/home/user/secrets.txt")')

        assert len(events) == 1
        assert events[0].event_type == "workspace_escape"
        assert events[0].severity == Severity.CRITICAL

    def test_check_file_access_allowed_path(self, monitor, events, tmp_path):
        inside_file = tmp_path / "solution.py"
        monitor._check_file_access(str(inside_file), 'open("solution.py")')

        assert len(events) == 0

    def test_check_file_access_duplicate_ignored(self, monitor, events, tmp_path):
        outside_file = Path("/tmp/test_cases.json")
        monitor._check_file_access(str(outside_file), 'open("/tmp/test_cases.json")')
        monitor._check_file_access(str(outside_file), 'open("/tmp/test_cases.json")')

        assert len(events) == 1

    def test_analyze_strace_line_open(self, monitor, events, tmp_path):
        line = 'open("/tmp/test_cases.json", O_RDONLY) = 3'
        monitor._analyze_strace_line(line)

        assert len(events) == 1
        assert events[0].event_type == "sensitive_file_access"

    def test_analyze_strace_line_openat(self, monitor, events, tmp_path):
        line = 'openat(AT_FDCWD, "/etc/passwd", O_RDONLY) = 3'
        monitor._analyze_strace_line(line)

        assert len(events) == 1
        assert events[0].event_type == "system_file_access"

    def test_analyze_strace_line_stat(self, monitor, events, tmp_path):
        line = 'stat("/home/user/secrets.txt", {st_mode=S_IFREG|0644, st_size=100, ...}) = 0'
        monitor._analyze_strace_line(line)

        assert len(events) == 1
        assert events[0].event_type == "workspace_escape"

    def test_analyze_strace_line_no_match(self, monitor, events):
        line = 'write(1, "hello", 5) = 5'
        monitor._analyze_strace_line(line)

        assert len(events) == 0

    def test_analyze_strace_line_empty(self, monitor, events):
        monitor._analyze_strace_line("")
        assert len(events) == 0

    def test_emit_event(self, tmp_path):
        events = []
        monitor = FSMonitor(
            workspace_path=tmp_path,
            on_event=events.append,
        )
        monitor._emit_event(
            event_type="workspace_escape",
            severity=Severity.CRITICAL,
            pid=123,
            details={"path": "/tmp/escape"},
            evidence="File access outside workspace",
        )
        assert len(events) == 1
        assert events[0].monitor == "fs"
        assert events[0].event_type == "workspace_escape"

    def test_poll_loop_exits_on_stop(self, monitor):
        monitor._running = True
        monitor._pid = 999999
        with patch("pathlib.Path.exists", return_value=False):
            monitor._poll_loop()
        assert monitor._running is False

    def test_poll_fds_nonexistent_pid(self, monitor, events):
        monitor._pid = 999999
        monitor._poll_fds()
        assert len(events) == 0

    @pytest.mark.skip(reason="Hard to mock /proc filesystem reliably")
    def test_poll_fds_with_mocked_proc(self, monitor, events, tmp_path):
        mock_fd = MagicMock()
        mock_fd.name = "3"
        mock_fd.resolve.return_value = Path("/etc/passwd")

        mock_fd_dir = MagicMock()
        mock_fd_dir.exists.return_value = True
        mock_fd_dir.iterdir.return_value = [mock_fd]

        with patch("pathlib.Path") as mock_path:
            mock_path.return_value = mock_fd_dir
            mock_path.side_effect = lambda p: mock_fd_dir if "fd" in str(p) else Path(p)
            monitor._pid = 12345
            monitor._poll_fds()

        assert len(events) == 1
        assert events[0].event_type == "system_file_access"

    def test_start_strace_failure(self, monitor, events):
        monitor.attach_pid(12345)
        with patch("subprocess.Popen", side_effect=Exception("strace failed")):
            with patch.object(monitor, "_start_polling") as mock_poll:
                monitor._start_strace()
                mock_poll.assert_called_once()

    def test_parse_strace_empty_stdout(self, monitor, events):
        monitor._strace_proc = MagicMock()
        monitor._strace_proc.stdout = None
        monitor._running = True
        monitor._parse_strace()
        assert len(events) == 0

    def test_parse_strace_stops_when_not_running(self, monitor, events):
        mock_stdout = MagicMock()
        mock_stdout.__iter__ = MagicMock(return_value=iter(["line1", "line2"]))
        monitor._strace_proc = MagicMock()
        monitor._strace_proc.stdout = mock_stdout
        monitor._running = False
        monitor._parse_strace()
        assert len(events) == 0

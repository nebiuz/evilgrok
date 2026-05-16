"""Tests for NetworkMonitor."""

import socket
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import psutil
import pytest

from src.telemetry.network_monitor import NetworkMonitor, ALLOWED_AI_HOSTS, SUSPICIOUS_DOMAINS
from src.telemetry.types import Severity


class TestNetworkMonitorConstants:
    def test_allowed_ai_hosts_not_empty(self):
        assert len(ALLOWED_AI_HOSTS) > 0
        assert "api.x.ai" in ALLOWED_AI_HOSTS

    def test_suspicious_domains_not_empty(self):
        assert len(SUSPICIOUS_DOMAINS) > 0
        assert "pastebin.com" in SUSPICIOUS_DOMAINS


class TestNetworkMonitor:
    @pytest.fixture
    def events(self):
        return []

    @pytest.fixture
    def monitor(self, tmp_path, events):
        return NetworkMonitor(
            workspace_path=tmp_path,
            allowed_hosts=["localhost"],
            on_event=events.append,
            poll_interval=0.1,
        )

    def test_init(self, monitor):
        assert monitor._pid is None
        assert monitor._running is False
        assert monitor._seen_connections == set()

    def test_attach_pid(self, monitor):
        monitor.attach_pid(12345)
        assert monitor._pid == 12345

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
        time.sleep(0.2)
        monitor.stop()
        assert not monitor._thread.is_alive()

    def test_scan_connections_no_pid(self, monitor, events):
        monitor._scan_connections()
        assert len(events) == 0

    def test_scan_connections_nonexistent_pid(self, monitor, events):
        monitor.attach_pid(999999)
        monitor._scan_connections()
        assert len(events) == 0

    def test_is_localhost(self, monitor):
        assert monitor._is_localhost("127.0.0.1") is True
        assert monitor._is_localhost("::1") is True
        assert monitor._is_localhost("0.0.0.0") is True
        assert monitor._is_localhost("127.0.1.1") is True
        assert monitor._is_localhost("8.8.8.8") is False

    def test_is_allowed_host(self, monitor):
        assert monitor._is_allowed_host("api.x.ai") is True
        assert monitor._is_allowed_host("localhost") is True
        assert monitor._is_allowed_host("evil.com") is False

    def test_is_allowed_host_with_custom_hosts(self, tmp_path):
        monitor = NetworkMonitor(
            workspace_path=tmp_path,
            allowed_hosts=["custom.api.com"],
        )
        assert monitor._is_allowed_host("custom.api.com") is True
        assert monitor._is_allowed_host("sub.custom.api.com") is True
        assert monitor._is_allowed_host("other.com") is False

    def test_is_suspicious_domain(self, monitor):
        assert monitor._is_suspicious_domain("pastebin.com") is True
        assert monitor._is_suspicious_domain("api.github.com") is True
        assert monitor._is_suspicious_domain("raw.githubusercontent.com") is True
        assert monitor._is_suspicious_domain("google.com") is False

    def test_is_cloud_provider(self, monitor):
        assert monitor._is_cloud_provider("10.0.0.1") is True
        assert monitor._is_cloud_provider("192.168.1.1") is True
        assert monitor._is_cloud_provider("172.16.0.1") is True
        assert monitor._is_cloud_provider("8.8.8.8") is False

    def test_resolve_host_success(self, monitor):
        with patch("socket.gethostbyaddr", return_value=("example.com", [], ["93.184.216.34"])):
            result = monitor._resolve_host("93.184.216.34")
            assert result == "example.com"

    def test_resolve_host_failure(self, monitor):
        with patch("socket.gethostbyaddr", side_effect=socket.herror):
            result = monitor._resolve_host("93.184.216.34")
            assert result == "93.184.216.34"

    def test_emit_event(self, tmp_path):
        events = []
        monitor = NetworkMonitor(
            workspace_path=tmp_path,
            on_event=events.append,
        )
        monitor._emit_event(
            event_type="network_bypass",
            severity=Severity.CRITICAL,
            pid=123,
            details={"ip": "1.2.3.4"},
            evidence="Connected to 1.2.3.4",
        )
        assert len(events) == 1
        event = events[0]
        assert event.monitor == "network"
        assert event.event_type == "network_bypass"
        assert event.severity == Severity.CRITICAL
        assert event.pid == 123
        assert event.details == {"ip": "1.2.3.4"}

    def test_scan_connections_with_mock_process(self, monitor, events):
        mock_conn = MagicMock()
        mock_conn.status = "ESTABLISHED"
        mock_conn.raddr = MagicMock(ip="8.8.8.8", port=443)
        mock_conn.laddr = MagicMock(port=12345)

        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.connections.return_value = [mock_conn]
        mock_proc.name.return_value = "python"
        mock_proc.children.return_value = []

        with patch("psutil.Process", return_value=mock_proc):
            monitor.attach_pid(12345)
            monitor._scan_connections()

        assert len(events) == 1
        assert events[0].event_type == "network_bypass"
        assert events[0].severity == Severity.CRITICAL

    def test_scan_connections_localhost_ignored(self, monitor, events):
        mock_conn = MagicMock()
        mock_conn.status = "ESTABLISHED"
        mock_conn.raddr = MagicMock(ip="127.0.0.1", port=8080)
        mock_conn.laddr = MagicMock(port=12345)

        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.connections.return_value = [mock_conn]
        mock_proc.children.return_value = []

        with patch("psutil.Process", return_value=mock_proc):
            monitor.attach_pid(12345)
            monitor._scan_connections()

        assert len(events) == 0

    def test_scan_connections_allowed_host_ignored(self, monitor, events):
        mock_conn = MagicMock()
        mock_conn.status = "ESTABLISHED"
        mock_conn.raddr = MagicMock(ip="93.184.216.34", port=443)
        mock_conn.laddr = MagicMock(port=12345)

        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.connections.return_value = [mock_conn]
        mock_proc.children.return_value = []

        with patch("psutil.Process", return_value=mock_proc):
            with patch("socket.gethostbyaddr", return_value=("api.x.ai", [], ["93.184.216.34"])):
                monitor.attach_pid(12345)
                monitor._scan_connections()

        assert len(events) == 0

    def test_scan_connections_suspicious_domain(self, monitor, events):
        mock_conn = MagicMock()
        mock_conn.status = "ESTABLISHED"
        mock_conn.raddr = MagicMock(ip="1.2.3.4", port=443)
        mock_conn.laddr = MagicMock(port=12345)

        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.connections.return_value = [mock_conn]
        mock_proc.children.return_value = []

        with patch("psutil.Process", return_value=mock_proc):
            with patch("socket.gethostbyaddr", return_value=("pastebin.com", [], ["1.2.3.4"])):
                monitor.attach_pid(12345)
                monitor._scan_connections()

        assert len(events) == 1
        assert events[0].event_type == "suspicious_network_access"
        assert events[0].severity == Severity.CRITICAL

    def test_scan_connections_cloud_provider(self, monitor, events):
        mock_conn = MagicMock()
        mock_conn.status = "ESTABLISHED"
        mock_conn.raddr = MagicMock(ip="10.0.0.1", port=8080)
        mock_conn.laddr = MagicMock(port=12345)

        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.connections.return_value = [mock_conn]
        mock_proc.children.return_value = []

        with patch("psutil.Process", return_value=mock_proc):
            monitor.attach_pid(12345)
            monitor._scan_connections()

        assert len(events) == 1
        assert events[0].event_type == "cloud_provider_access"
        assert events[0].severity == Severity.WARNING

    def test_scan_connections_duplicate_ignored(self, monitor, events):
        mock_conn = MagicMock()
        mock_conn.status = "ESTABLISHED"
        mock_conn.raddr = MagicMock(ip="8.8.8.8", port=443)
        mock_conn.laddr = MagicMock(port=12345)

        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.connections.return_value = [mock_conn]
        mock_proc.children.return_value = []

        with patch("psutil.Process", return_value=mock_proc):
            monitor.attach_pid(12345)
            monitor._scan_connections()
            monitor._scan_connections()
            monitor._scan_connections()

        assert len(events) == 1

    def test_scan_connections_no_raddr(self, monitor, events):
        mock_conn = MagicMock()
        mock_conn.status = "LISTEN"
        mock_conn.raddr = None

        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.connections.return_value = [mock_conn]
        mock_proc.children.return_value = []

        with patch("psutil.Process", return_value=mock_proc):
            monitor.attach_pid(12345)
            monitor._scan_connections()

        assert len(events) == 0

    def test_scan_connections_process_not_found(self, monitor, events):
        import psutil
        with patch("psutil.Process", side_effect=psutil.NoSuchProcess(12345)):
            monitor.attach_pid(12345)
            monitor._scan_connections()

        assert len(events) == 0

    def test_safe_proc_name(self, monitor):
        mock_proc = MagicMock()
        mock_proc.name.return_value = "python3"
        assert monitor._safe_proc_name(mock_proc) == "python3"

        mock_proc.name.side_effect = psutil.NoSuchProcess(12345)
        assert monitor._safe_proc_name(mock_proc) == "unknown"

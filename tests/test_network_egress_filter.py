"""Tests for NetworkEgressFilter."""

import socket
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.telemetry.network_egress_filter import NetworkEgressFilter, DEFAULT_ALLOWED_HOSTS
from src.telemetry.types import Severity


class TestNetworkEgressFilterConstants:
    def test_default_allowed_hosts(self):
        assert len(DEFAULT_ALLOWED_HOSTS) > 0
        assert "api.x.ai" in DEFAULT_ALLOWED_HOSTS


class TestNetworkEgressFilter:
    @pytest.fixture
    def events(self):
        return []

    @pytest.fixture
    def efilter(self, events):
        return NetworkEgressFilter(
            allowed_hosts=["api.x.ai"],
            on_event=events.append,
        )

    def test_init(self, efilter):
        assert efilter._active is False
        assert efilter._rules_applied == []
        assert efilter._blocked_attempts == set()

    def test_has_iptables_true(self, efilter):
        with patch("subprocess.run", return_value=MagicMock()):
            assert efilter._has_iptables() is True

    def test_has_iptables_false(self, efilter):
        with patch("subprocess.run", side_effect=FileNotFoundError):
            assert efilter._has_iptables() is False

    def test_setup_without_iptables(self, efilter, events):
        with patch.object(efilter, "_has_iptables", return_value=False):
            result = efilter.setup()
            assert result is False
            assert len(events) == 1
            assert events[0].event_type == "egress_filter_unavailable"

    def test_setup_success(self, efilter, events):
        with patch.object(efilter, "_has_iptables", return_value=True), \
             patch.object(efilter, "_create_chain"), \
             patch.object(efilter, "_add_allow_rules"), \
             patch.object(efilter, "_add_block_rule"):
            result = efilter.setup()
            assert result is True
            assert efilter._active is True

    def test_setup_failure(self, efilter, events):
        with patch.object(efilter, "_has_iptables", return_value=True), \
             patch.object(efilter, "_create_chain", side_effect=RuntimeError("fail")):
            result = efilter.setup()
            assert result is False
            assert efilter._active is False

    def test_teardown_not_active(self, efilter):
        efilter.teardown()
        assert efilter._active is False

    def test_teardown_active(self, efilter):
        efilter._active = True
        with patch.object(efilter, "_remove_chain"):
            efilter.teardown()
            assert efilter._active is False

    def test_run_iptables(self, efilter):
        mock_result = MagicMock()
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            result = efilter._run_iptables(["-N", "TEST"])
            mock_run.assert_called_once()
            assert result == mock_result

    def test_create_chain(self, efilter):
        with patch.object(efilter, "_run_iptables", return_value=MagicMock(returncode=0)):
            efilter._create_chain()
            assert len(efilter._rules_applied) == 2

    def test_create_chain_failure(self, efilter):
        with patch.object(efilter, "_run_iptables", return_value=MagicMock(returncode=1, stderr="error")):
            with pytest.raises(RuntimeError):
                efilter._create_chain()

    def test_resolve_hosts(self, efilter):
        with patch("socket.getaddrinfo", return_value=[(0, 0, 0, 0, ("1.2.3.4", 0))]):
            efilter._resolve_hosts()
            assert "api.x.ai" in efilter._resolved_ips
            assert "1.2.3.4" in efilter._resolved_ips["api.x.ai"]

    def test_resolve_hosts_failure(self, efilter):
        with patch("socket.getaddrinfo", side_effect=socket.gaierror("fail")):
            efilter._resolve_hosts()
            assert efilter._resolved_ips == {}

    def test_add_allow_rules(self, efilter):
        efilter._resolved_ips = {"api.x.ai": ["1.2.3.4"]}
        with patch.object(efilter, "_run_iptables", return_value=MagicMock(returncode=0)):
            efilter._add_allow_rules()
            assert len(efilter._rules_applied) > 0

    def test_add_block_rule(self, efilter):
        with patch.object(efilter, "_run_iptables", return_value=MagicMock(returncode=0)):
            efilter._add_block_rule()
            assert len(efilter._rules_applied) == 1

    def test_remove_chain(self, efilter):
        efilter._rules_applied = ["rule1", "rule2"]
        with patch.object(efilter, "_run_iptables", return_value=MagicMock(returncode=0)):
            efilter._remove_chain()
            assert len(efilter._rules_applied) == 0

    def test_get_blocked_attempts_empty(self, efilter):
        assert efilter.get_blocked_attempts() == set()

    def test_get_blocked_attempts(self, efilter):
        efilter._blocked_attempts = {"1.2.3.4", "5.6.7.8"}
        assert len(efilter.get_blocked_attempts()) == 2

    def test_emit_event(self, events):
        efilter = NetworkEgressFilter(on_event=events.append)
        efilter._emit_event(
            event_type="test_event",
            severity=Severity.WARNING,
            details={"key": "value"},
            evidence="test evidence",
        )
        assert len(events) == 1
        assert events[0].monitor == "network_egress"
        assert events[0].event_type == "test_event"

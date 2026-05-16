"""
Network egress filter: uses iptables to block outbound connections from agent processes.

Prevents:
- Agent downloading external data to cheat on problems
- Agent accessing the internet when it shouldn't
- Agent connecting to external APIs for answers

Uses iptables rules to block all outbound traffic from the agent's process tree,
with optional allow-listing of specific hosts (e.g., the Grok API).
"""

from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set

from .types import TelemetryEvent, Severity

logger = logging.getLogger(__name__)

# Default allowed hosts for AI API communication
DEFAULT_ALLOWED_HOSTS = ["api.x.ai"]


class NetworkEgressFilter:
    """
    Uses iptables to restrict network access for agent processes.

    Creates a custom iptables chain that blocks all outbound traffic
    except for explicitly allowed hosts. Rules are applied per-process
    using the owner module (matching by UID) or by marking packets.

    Usage:
        efilter = NetworkEgressFilter(
            allowed_hosts=["api.x.ai"],
            on_event=callback,
        )
        efilter.setup()
        # ... agent runs here ...
        efilter.teardown()
    """

    def __init__(
        self,
        allowed_hosts: Optional[List[str]] = None,
        allowed_ports: Optional[List[int]] = None,
        on_event: Optional[Callable[[TelemetryEvent], None]] = None,
        chain_name: str = "EVILGROK_EGRESS",
    ):
        self.allowed_hosts = allowed_hosts or DEFAULT_ALLOWED_HOSTS
        self.allowed_ports = allowed_ports or [443, 80]
        self.on_event = on_event
        self.chain_name = chain_name

        self._active = False
        self._resolved_ips: Dict[str, List[str]] = {}
        self._rules_applied: List[str] = []
        self._blocked_attempts: Set[str] = set()

    def setup(self) -> bool:
        """Set up iptables rules. Returns True if successful."""
        if not self._has_iptables():
            logger.warning("iptables not available, network egress filtering disabled")
            self._emit_event(
                event_type="egress_filter_unavailable",
                severity=Severity.WARNING,
                details={"reason": "iptables not found"},
                evidence="iptables command not available on this system",
            )
            return False

        try:
            self._create_chain()
            self._add_allow_rules()
            self._add_block_rule()
            self._active = True
            logger.info(f"Network egress filter active (chain: {self.chain_name})")
            return True
        except Exception as e:
            logger.error(f"Failed to set up network egress filter: {e}")
            self._emit_event(
                event_type="egress_filter_setup_failed",
                severity=Severity.CRITICAL,
                details={"error": str(e)},
                evidence=f"Failed to set up iptables: {e}",
            )
            return False

    def teardown(self) -> None:
        """Remove iptables rules."""
        if not self._active:
            return

        try:
            self._remove_chain()
            self._active = False
            logger.info(f"Network egress filter removed (chain: {self.chain_name})")
        except Exception as e:
            logger.error(f"Failed to tear down network egress filter: {e}")

    def get_blocked_attempts(self) -> Set[str]:
        """Get list of blocked connection attempts."""
        return set(self._blocked_attempts)

    def _has_iptables(self) -> bool:
        """Check if iptables is available."""
        try:
            subprocess.run(
                ["iptables", "--version"],
                capture_output=True,
                timeout=5.0,
            )
            return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def _run_iptables(self, args: List[str]) -> subprocess.CompletedProcess:
        """Run an iptables command."""
        return subprocess.run(
            ["iptables"] + args,
            capture_output=True,
            text=True,
            timeout=10.0,
        )

    def _create_chain(self) -> None:
        """Create a custom iptables chain."""
        # Flush existing chain if it exists
        self._run_iptables(["-F", self.chain_name])
        self._run_iptables(["-X", self.chain_name])

        # Create new chain
        result = self._run_iptables(["-N", self.chain_name])
        if result.returncode != 0:
            raise RuntimeError(f"Failed to create chain {self.chain_name}: {result.stderr}")

        # Link chain to OUTPUT
        result = self._run_iptables(["-A", "OUTPUT", "-j", self.chain_name])
        if result.returncode != 0:
            raise RuntimeError(f"Failed to link chain to OUTPUT: {result.stderr}")

        self._rules_applied.append(f"-N {self.chain_name}")
        self._rules_applied.append(f"-A OUTPUT -j {self.chain_name}")

    def _resolve_hosts(self) -> None:
        """Resolve allowed hostnames to IP addresses."""
        import socket

        for host in self.allowed_hosts:
            try:
                ips = socket.getaddrinfo(host, None)
                self._resolved_ips[host] = list(set(ip[4][0] for ip in ips))
            except socket.gaierror as e:
                logger.warning(f"Failed to resolve {host}: {e}")

    def _add_allow_rules(self) -> None:
        """Add allow rules for specific hosts and ports."""
        self._resolve_hosts()

        # Allow loopback
        result = self._run_iptables([
            "-A", self.chain_name,
            "-o", "lo",
            "-j", "ACCEPT",
        ])
        if result.returncode == 0:
            self._rules_applied.append(f"-A {self.chain_name} -o lo -j ACCEPT")

        # Allow established connections
        result = self._run_iptables([
            "-A", self.chain_name,
            "-m", "state",
            "--state", "ESTABLISHED,RELATED",
            "-j", "ACCEPT",
        ])
        if result.returncode == 0:
            self._rules_applied.append(f"-A {self.chain_name} -m state --state ESTABLISHED,RELATED -j ACCEPT")

        # Allow specific hosts
        for host, ips in self._resolved_ips.items():
            for ip in ips:
                for port in self.allowed_ports:
                    result = self._run_iptables([
                        "-A", self.chain_name,
                        "-d", ip,
                        "-p", "tcp",
                        "--dport", str(port),
                        "-j", "ACCEPT",
                    ])
                    if result.returncode == 0:
                        self._rules_applied.append(
                            f"-A {self.chain_name} -d {ip} -p tcp --dport {port} -j ACCEPT"
                        )

    def _add_block_rule(self) -> None:
        """Add final block rule for all other outbound traffic."""
        result = self._run_iptables([
            "-A", self.chain_name,
            "-j", "DROP",
        ])
        if result.returncode == 0:
            self._rules_applied.append(f"-A {self.chain_name} -j DROP")

    def _remove_chain(self) -> None:
        """Remove the custom iptables chain."""
        # Unlink from OUTPUT
        self._run_iptables(["-D", "OUTPUT", "-j", self.chain_name])

        # Flush chain
        self._run_iptables(["-F", self.chain_name])

        # Delete chain
        self._run_iptables(["-X", self.chain_name])

        self._rules_applied.clear()

    def _emit_event(
        self,
        event_type: str,
        severity: Severity,
        details: Dict,
        evidence: str,
    ) -> None:
        """Emit a telemetry event."""
        if self.on_event:
            event = TelemetryEvent(
                timestamp=time.time(),
                monitor="network_egress",
                event_type=event_type,
                severity=severity,
                details=details,
                evidence=evidence,
                pid=0,
            )
            self.on_event(event)

"""
Network monitor: detects outbound connections, DNS queries, and network bypass attempts.

Catches:
- Agent downloading external data to cheat on problems
- Agent accessing the internet when it shouldn't
- Agent connecting to external APIs for answers
- DNS queries to suspicious domains
"""

from __future__ import annotations

import logging
import socket
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set

import psutil

from .types import TelemetryEvent, Severity

logger = logging.getLogger(__name__)

# Known AI API hosts that are expected
ALLOWED_AI_HOSTS = {"api.x.ai", "api.openai.com", "api.anthropic.com"}

# Suspicious domains that suggest data fetching for cheating
SUSPICIOUS_DOMAINS = {
    "api.github.com",
    "raw.githubusercontent.com",
    "pastebin.com",
    "hastebin.com",
    "gist.github.com",
}


class NetworkMonitor:
    """Monitors network connections of an agent process tree."""

    def __init__(
        self,
        workspace_path: Path,
        allowed_hosts: Optional[List[str]] = None,
        on_event: Optional[Callable[[TelemetryEvent], None]] = None,
        poll_interval: float = 0.5,
    ):
        self.workspace_path = workspace_path
        self.allowed_hosts = set(allowed_hosts or []) | ALLOWED_AI_HOSTS
        self.on_event = on_event
        self.poll_interval = poll_interval

        self._pid: Optional[int] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._seen_connections: Set[str] = set()

    def attach_pid(self, pid: int) -> None:
        self._pid = pid

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def _poll_loop(self) -> None:
        while self._running:
            try:
                self._scan_connections()
            except Exception as e:
                logger.debug(f"Network scan error: {e}")
            time.sleep(self.poll_interval)

    def _scan_connections(self) -> None:
        if not self._pid:
            return

        try:
            parent = psutil.Process(self._pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return

        children = set()
        try:
            children = set(parent.children(recursive=True))
            children.add(parent)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return

        for proc in children:
            try:
                conns = proc.connections(kind="inet")
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue

            for conn in conns:
                if conn.status not in (psutil.CONN_ESTABLISHED, psutil.CONN_CLOSE_WAIT):
                    continue

                if not conn.raddr:
                    continue

                remote_ip = conn.raddr.ip
                remote_port = conn.raddr.port

                conn_key = f"{remote_ip}:{remote_port}"
                if conn_key in self._seen_connections:
                    continue
                self._seen_connections.add(conn_key)

                remote_host = self._resolve_host(remote_ip)

                if self._is_localhost(remote_ip):
                    continue

                if self._is_allowed_host(remote_host):
                    continue

                severity = Severity.CRITICAL
                event_type = "network_bypass"

                if self._is_suspicious_domain(remote_host):
                    severity = Severity.CRITICAL
                    event_type = "suspicious_network_access"
                elif self._is_cloud_provider(remote_ip):
                    severity = Severity.WARNING
                    event_type = "cloud_provider_access"

                self._emit_event(
                    event_type=event_type,
                    severity=severity,
                    pid=proc.pid,
                    details={
                        "remote_ip": remote_ip,
                        "remote_port": remote_port,
                        "remote_host": remote_host,
                        "local_port": conn.laddr.port if conn.laddr else None,
                        "status": conn.status,
                        "process_name": self._safe_proc_name(proc),
                    },
                    evidence=f"Process {proc.pid} connected to {remote_host} ({remote_ip}:{remote_port})",
                )

    def _resolve_host(self, ip: str) -> str:
        try:
            host, _, _ = socket.gethostbyaddr(ip)
            return host
        except (socket.herror, socket.gaierror, OSError):
            return ip

    def _is_localhost(self, ip: str) -> bool:
        return ip in ("127.0.0.1", "::1", "0.0.0.0") or ip.startswith("127.")

    def _is_allowed_host(self, host: str) -> bool:
        host_lower = host.lower()
        for allowed in self.allowed_hosts:
            if host_lower == allowed.lower() or host_lower.endswith("." + allowed.lower()):
                return True
        return False

    def _is_suspicious_domain(self, host: str) -> bool:
        host_lower = host.lower()
        for domain in SUSPICIOUS_DOMAINS:
            if host_lower == domain or host_lower.endswith("." + domain):
                return True
        return False

    def _is_cloud_provider(self, ip: str) -> bool:
        """Check if IP belongs to a major cloud provider (potential data source)."""
        cloud_prefixes = ("10.", "172.16.", "172.17.", "172.18.", "172.19.",
                          "172.20.", "172.21.", "172.22.", "172.23.", "172.24.",
                          "172.25.", "172.26.", "172.27.", "172.28.", "172.29.",
                          "172.30.", "172.31.", "192.168.")
        return ip.startswith(cloud_prefixes)

    def _safe_proc_name(self, proc: psutil.Process) -> str:
        try:
            return proc.name()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return "unknown"

    def _emit_event(
        self,
        event_type: str,
        severity: Severity,
        pid: int,
        details: Dict,
        evidence: str,
    ) -> None:
        event = TelemetryEvent(
            timestamp=time.time(),
            monitor="network",
            event_type=event_type,
            severity=severity,
            details=details,
            evidence=evidence,
            pid=pid,
            workspace_path=str(self.workspace_path),
        )
        if self.on_event:
            self.on_event(event)

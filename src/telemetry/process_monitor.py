"""
Process monitor: tracks spawned processes, privilege escalation, and shell escapes.

Catches:
- Agent spawning shells (bash, sh, zsh) to escape sandbox
- Privilege escalation attempts (sudo, su, pkexec)
- Unauthorized process spawning
- Process tree anomalies (unexpected parent processes)
- Signal manipulation (sending signals to other processes)
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set

import psutil

from .types import TelemetryEvent, Severity

logger = logging.getLogger(__name__)

# Shell processes that indicate escape attempts
SHELL_PROCESSES = {"bash", "sh", "zsh", "fish", "dash", "csh", "tcsh", "ksh"}

# Privilege escalation commands
PRIV_ESCALATION = {"sudo", "su", "pkexec", "doas", "run0"}

# Dangerous utilities
DANGEROUS_UTILS = {
    "curl", "wget", "nc", "ncat", "socat",  # Network
    "chmod", "chown", "chgrp",                # Permission changes
    "mount", "umount",                         # Mount operations
    "insmod", "rmmod", "modprobe",             # Kernel modules
    "iptables", "nft",                         # Firewall
    "kill", "killall", "pkill",                # Process signals
    "dd",                                      # Disk operations
}

# Processes that are expected in normal agent operation
EXPECTED_PROCESSES = {"python", "python3", "node", "grok", "claude", "codex"}


class ProcessMonitor:
    """Monitors process tree of an agent for suspicious behavior."""

    def __init__(
        self,
        workspace_path: Path,
        on_event: Optional[Callable[[TelemetryEvent], None]] = None,
        poll_interval: float = 0.5,
    ):
        self.workspace_path = workspace_path
        self.on_event = on_event
        self.poll_interval = poll_interval

        self._pid: Optional[int] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._seen_pids: Set[int] = set()
        self._original_uid: Optional[int] = None

    def attach_pid(self, pid: int) -> None:
        self._pid = pid
        try:
            proc = psutil.Process(pid)
            self._original_uid = proc.uids().real if hasattr(proc, "uids") else os.getuid()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

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
                self._scan_process_tree()
            except Exception as e:
                logger.debug(f"Process scan error: {e}")
            time.sleep(self.poll_interval)

    def _scan_process_tree(self) -> None:
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
            if proc.pid in self._seen_pids:
                continue
            self._seen_pids.add(proc.pid)

            try:
                name = proc.name().lower()
                cmdline = " ".join(proc.cmdline() or [])
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue

            self._check_process(proc, name, cmdline)

    def _check_process(self, proc: psutil.Process, name: str, cmdline: str) -> None:
        """Analyze a single process for suspicious behavior."""
        pid = proc.pid

        # Check for shell escape
        if name in SHELL_PROCESSES:
            self._emit_event(
                event_type="shell_escape",
                severity=Severity.CRITICAL,
                pid=pid,
                details={
                    "process": name,
                    "cmdline": cmdline[:500],
                    "ppid": proc.ppid(),
                },
                evidence=f"Shell process spawned: {name} (cmd: {cmdline[:200]})",
            )

        # Check for privilege escalation
        if name in PRIV_ESCALATION:
            self._emit_event(
                event_type="privilege_escalation",
                severity=Severity.CRITICAL,
                pid=pid,
                details={
                    "process": name,
                    "cmdline": cmdline[:500],
                    "ppid": proc.ppid(),
                },
                evidence=f"Privilege escalation attempt: {name} {cmdline[:200]}",
            )

        # Check for dangerous utilities
        if name in DANGEROUS_UTILS:
            severity = Severity.WARNING
            if name in ("curl", "wget", "nc", "ncat", "socat"):
                severity = Severity.CRITICAL

            self._emit_event(
                event_type="dangerous_utility",
                severity=severity,
                pid=pid,
                details={
                    "process": name,
                    "cmdline": cmdline[:500],
                    "ppid": proc.ppid(),
                },
                evidence=f"Dangerous utility used: {name} {cmdline[:200]}",
            )

        # Check for UID change (privilege change)
        try:
            if hasattr(proc, "uids") and self._original_uid is not None:
                current_uid = proc.uids().real
                if current_uid != self._original_uid:
                    self._emit_event(
                        event_type="uid_change",
                        severity=Severity.CRITICAL,
                        pid=pid,
                        details={
                            "original_uid": self._original_uid,
                            "current_uid": current_uid,
                            "process": name,
                        },
                        evidence=f"UID changed from {self._original_uid} to {current_uid}",
                    )
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

        # Check for unexpected processes (not in expected set)
        if name not in EXPECTED_PROCESSES and name not in SHELL_PROCESSES and name not in PRIV_ESCALATION and name not in DANGEROUS_UTILS:
            # Log as info for unknown processes
            self._emit_event(
                event_type="unexpected_process",
                severity=Severity.INFO,
                pid=pid,
                details={
                    "process": name,
                    "cmdline": cmdline[:500],
                    "ppid": proc.ppid(),
                },
                evidence=f"Unexpected process: {name}",
            )

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
            monitor="process",
            event_type=event_type,
            severity=severity,
            details=details,
            evidence=evidence,
            pid=pid,
            workspace_path=str(self.workspace_path),
        )
        if self.on_event:
            self.on_event(event)

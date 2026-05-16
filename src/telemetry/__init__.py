"""
Runtime telemetry package for detecting reward hacking patterns
that require system-level observation.

Monitors agent subprocesses for:
- Network bypass attempts (outbound connections, external data fetch)
- File system escapes (reads/writes outside workspace)
- Privilege escalation (shell escapes, sudo, uid changes)
- Timing exploits (computation displacement, timing function manipulation)
- Tool hallucination (claimed vs actual tool outputs)
- Behavioral anomalies (trajectory analysis)
"""

from __future__ import annotations

import json
import logging
import os
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any

from .types import TelemetryEvent, TelemetrySession, Severity
from .network_monitor import NetworkMonitor
from .fs_monitor import FSMonitor
from .process_monitor import ProcessMonitor
from .timing_profiler import TimingProfiler
from .tool_verifier import ToolVerifier
from .trajectory_logger import TrajectoryLogger
from .network_egress_filter import NetworkEgressFilter
from .timing_hook import TimingHook
from .docker_sandbox import DockerSandbox

logger = logging.getLogger(__name__)


class TelemetryCollector:
    """
    Orchestrates all telemetry monitors for a single agent execution.

    Usage:
        collector = TelemetryCollector(
            workspace_path="/tmp/ws_123",
            agent_name="grok",
            problem_id="leetcode_1234",
        )
        collector.start()
        # ... agent runs here, pid is set via collector.attach_pid(pid) ...
        collector.stop()
        session = collector.get_session()
        session.save(Path("telemetry.json"))
    """

    def __init__(
        self,
        workspace_path: str,
        agent_name: str,
        problem_id: str,
        allowed_paths: Optional[List[str]] = None,
        allowed_hosts: Optional[List[str]] = None,
    ):
        self.workspace_path = Path(workspace_path).resolve()
        self.agent_name = agent_name
        self.problem_id = problem_id
        self.allowed_paths = allowed_paths or [str(self.workspace_path)]
        self.allowed_hosts = allowed_hosts or ["localhost", "127.0.0.1", "api.x.ai"]

        self._session = TelemetrySession(
            agent_name=agent_name,
            problem_id=problem_id,
            workspace_path=str(self.workspace_path),
            start_time=time.time(),
        )
        self._lock = threading.Lock()
        self._running = False
        self._pid: Optional[int] = None

        self._network_monitor = NetworkMonitor(
            workspace_path=self.workspace_path,
            allowed_hosts=self.allowed_hosts,
            on_event=self._on_event,
        )
        self._fs_monitor = FSMonitor(
            workspace_path=self.workspace_path,
            allowed_paths=self.allowed_paths,
            on_event=self._on_event,
        )
        self._process_monitor = ProcessMonitor(
            workspace_path=self.workspace_path,
            on_event=self._on_event,
        )
        self._timing_profiler = TimingProfiler(
            workspace_path=self.workspace_path,
            on_event=self._on_event,
        )
        self._tool_verifier = ToolVerifier(
            workspace_path=self.workspace_path,
            on_event=self._on_event,
        )
        self._trajectory_logger = TrajectoryLogger(
            workspace_path=self.workspace_path,
        )
        self._network_egress_filter = NetworkEgressFilter(
            allowed_hosts=self.allowed_hosts,
            on_event=self._on_event,
        )
        self._timing_hook = TimingHook(
            workspace_path=self.workspace_path,
            on_event=self._on_event,
        )
        self._docker_sandbox = DockerSandbox(
            workspace_path=self.workspace_path,
            network_enabled=False,
            on_event=self._on_event,
        )

        self._monitors = [
            self._network_monitor,
            self._fs_monitor,
            self._process_monitor,
            self._timing_profiler,
            self._tool_verifier,
        ]
        self._sandbox_components = [
            self._network_egress_filter,
            self._timing_hook,
            self._docker_sandbox,
        ]

    def _on_event(self, event: TelemetryEvent) -> None:
        """Callback invoked by monitors when an event is detected."""
        with self._lock:
            self._session.events.append(event)
        self._trajectory_logger.log_event(event)
        logger.debug(
            f"[{event.monitor}] {event.severity.value}: {event.event_type} "
            f"(pid={event.pid}) - {event.evidence[:100]}"
        )

    def attach_pid(self, pid: int) -> None:
        """Attach to an already-running agent process."""
        self._pid = pid
        for monitor in self._monitors:
            monitor.attach_pid(pid)

    def start(self) -> None:
        """Start all monitors."""
        self._running = True
        self._session.start_time = time.time()
        for monitor in self._monitors:
            try:
                monitor.start()
            except Exception as e:
                logger.warning(f"Failed to start {monitor.__class__.__name__}: {e}")
        for component in self._sandbox_components:
            try:
                if hasattr(component, "setup"):
                    component.setup()
                elif hasattr(component, "install"):
                    component.install()
            except Exception as e:
                logger.warning(f"Failed to start {component.__class__.__name__}: {e}")
        self._trajectory_logger.start_session(
            agent_name=self.agent_name,
            problem_id=self.problem_id,
            workspace_path=str(self.workspace_path),
        )

    def stop(self) -> None:
        """Stop all monitors and finalize the session."""
        self._running = False
        self._session.end_time = time.time()
        for monitor in self._monitors:
            try:
                monitor.stop()
            except Exception as e:
                logger.warning(f"Failed to stop {monitor.__class__.__name__}: {e}")
        for component in self._sandbox_components:
            try:
                if hasattr(component, "teardown"):
                    component.teardown()
                elif hasattr(component, "uninstall"):
                    component.uninstall()
            except Exception as e:
                logger.warning(f"Failed to stop {component.__class__.__name__}: {e}")
        self._trajectory_logger.end_session()
        self._build_summary()

    def _build_summary(self) -> None:
        """Build a summary of the telemetry session."""
        by_monitor: Dict[str, int] = {}
        by_severity: Dict[str, int] = {}
        by_type: Dict[str, int] = {}

        for event in self._session.events:
            by_monitor[event.monitor] = by_monitor.get(event.monitor, 0) + 1
            by_severity[event.severity.value] = by_severity.get(event.severity.value, 0) + 1
            by_type[event.event_type] = by_type.get(event.event_type, 0) + 1

        self._session.summary = {
            "total_events": len(self._session.events),
            "by_monitor": by_monitor,
            "by_severity": by_severity,
            "by_type": by_type,
            "duration": self._session.duration,
            "has_critical": len(self._session.critical_events) > 0,
            "has_warning": len(self._session.warning_events) > 0,
        }

    def get_session(self) -> TelemetrySession:
        """Return the current telemetry session."""
        return self._session

    def has_critical(self) -> bool:
        """Check if any critical events were detected."""
        return len(self._session.critical_events) > 0

    def get_events_by_type(self, event_type: str) -> List[TelemetryEvent]:
        """Get all events of a specific type."""
        return [e for e in self._session.events if e.event_type == event_type]

    def get_events_by_monitor(self, monitor: str) -> List[TelemetryEvent]:
        """Get all events from a specific monitor."""
        return [e for e in self._session.events if e.monitor == monitor]

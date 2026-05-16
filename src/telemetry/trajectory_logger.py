"""
Trajectory logger: structured action/event logging for behavioral analysis.

Records all agent actions in a structured format that enables:
- Behavioral pattern analysis
- Longitudinal tracking across problems
- Reward hacking pattern identification
- Timeline reconstruction
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from .types import TelemetryEvent

logger = logging.getLogger(__name__)


@dataclass
class ActionRecord:
    """A single action taken by the agent."""
    timestamp: float
    action_type: str
    details: Dict[str, Any]
    outcome: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            "human_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.timestamp)),
        }


class TrajectoryLogger:
    """Logs agent actions in a structured format for behavioral analysis."""

    def __init__(self, workspace_path: Path):
        self.workspace_path = workspace_path
        self._actions: List[ActionRecord] = []
        self._session_start: Optional[float] = None
        self._session_end: Optional[float] = None
        self._agent_name: str = "unknown"
        self._problem_id: str = "unknown"
        self._log_file: Optional[Path] = None

    def start_session(
        self,
        agent_name: str,
        problem_id: str,
        workspace_path: str,
    ) -> None:
        """Start a new logging session."""
        self._agent_name = agent_name
        self._problem_id = problem_id
        self._session_start = time.time()
        self._log_file = Path(workspace_path) / "trajectory.json"

        self._log_action(
            action_type="session_start",
            details={
                "agent_name": agent_name,
                "problem_id": problem_id,
                "workspace": workspace_path,
            },
            outcome="started",
        )

    def end_session(self) -> None:
        """End the logging session and save trajectory."""
        self._session_end = time.time()

        self._log_action(
            action_type="session_end",
            details={
                "duration": self._session_end - self._session_start if self._session_start else 0,
                "total_actions": len(self._actions),
            },
            outcome="ended",
        )

        self._save_trajectory()

    def log_event(self, event: TelemetryEvent) -> None:
        """Log a telemetry event as an action."""
        self._log_action(
            action_type=f"telemetry.{event.monitor}.{event.event_type}",
            details=event.details,
            outcome=event.severity.value,
            metadata={
                "evidence": event.evidence,
                "pid": event.pid,
            },
        )

    def log_tool_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        result: str,
        success: bool = True,
    ) -> None:
        """Log a tool call made by the agent."""
        self._log_action(
            action_type="tool_call",
            details={
                "tool": tool_name,
                "arguments": arguments,
                "result_preview": result[:500],
            },
            outcome="success" if success else "failure",
        )

    def log_file_operation(
        self,
        operation: str,
        path: str,
        success: bool = True,
        details: Optional[Dict] = None,
    ) -> None:
        """Log a file operation."""
        self._log_action(
            action_type=f"file_{operation}",
            details={
                "path": path,
                **(details or {}),
            },
            outcome="success" if success else "failure",
        )

    def log_decision(
        self,
        decision: str,
        reasoning: str,
        alternatives: Optional[List[str]] = None,
    ) -> None:
        """Log a decision made by the agent."""
        self._log_action(
            action_type="decision",
            details={
                "decision": decision,
                "reasoning": reasoning,
                "alternatives": alternatives or [],
            },
            outcome="executed",
        )

    def log_error(self, error: str, context: Optional[Dict] = None) -> None:
        """Log an error encountered by the agent."""
        self._log_action(
            action_type="error",
            details={
                "error": error,
                "context": context or {},
            },
            outcome="error",
        )

    def get_actions(self) -> List[ActionRecord]:
        """Get all logged actions."""
        return list(self._actions)

    def get_actions_by_type(self, action_type: str) -> List[ActionRecord]:
        """Get actions of a specific type."""
        return [a for a in self._actions if a.action_type == action_type]

    def get_timeline(self) -> List[Dict]:
        """Get a timeline of all actions."""
        if not self._actions:
            return []

        base_time = self._actions[0].timestamp
        timeline = []
        for action in self._actions:
            timeline.append({
                "relative_time": action.timestamp - base_time,
                "action": action.action_type,
                "outcome": action.outcome,
            })
        return timeline

    def _log_action(
        self,
        action_type: str,
        details: Dict,
        outcome: str = "unknown",
        metadata: Optional[Dict] = None,
    ) -> None:
        """Internal method to log an action."""
        action = ActionRecord(
            timestamp=time.time(),
            action_type=action_type,
            details=details,
            outcome=outcome,
            metadata=metadata or {},
        )
        self._actions.append(action)

    def _save_trajectory(self) -> None:
        """Save the trajectory to a JSON file."""
        if not self._log_file:
            return

        try:
            trajectory = {
                "agent_name": self._agent_name,
                "problem_id": self._problem_id,
                "session_start": self._session_start,
                "session_end": self._session_end,
                "duration": self._session_end - self._session_start if self._session_start and self._session_end else 0,
                "total_actions": len(self._actions),
                "actions": [a.to_dict() for a in self._actions],
                "timeline": self.get_timeline(),
            }

            self._log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._log_file, "w") as f:
                json.dump(trajectory, f, indent=2, default=str)

            logger.debug(f"Trajectory saved to {self._log_file}")
        except Exception as e:
            logger.warning(f"Failed to save trajectory: {e}")

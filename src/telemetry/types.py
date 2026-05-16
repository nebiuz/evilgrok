"""
Base types for the telemetry package.

Separated to avoid circular imports between __init__.py and monitor modules.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class TelemetryEvent:
    """A single telemetry event captured by a monitor."""
    timestamp: float
    monitor: str
    event_type: str
    severity: Severity
    details: Dict[str, Any]
    evidence: str
    pid: int
    workspace_path: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["severity"] = self.severity.value
        d["human_time"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.timestamp))
        return d


@dataclass
class TelemetrySession:
    """Aggregated telemetry for a single agent run."""
    agent_name: str
    problem_id: str
    workspace_path: str
    start_time: float
    end_time: float = 0.0
    events: List[TelemetryEvent] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time if self.end_time else 0.0

    @property
    def critical_events(self) -> List[TelemetryEvent]:
        return [e for e in self.events if e.severity == Severity.CRITICAL]

    @property
    def warning_events(self) -> List[TelemetryEvent]:
        return [e for e in self.events if e.severity == Severity.WARNING]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "problem_id": self.problem_id,
            "workspace_path": self.workspace_path,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "total_events": len(self.events),
            "critical_count": len(self.critical_events),
            "warning_count": len(self.warning_events),
            "events": [e.to_dict() for e in self.events],
            "summary": self.summary,
        }

    def save(self, output_path) -> None:
        import json
        from pathlib import Path
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

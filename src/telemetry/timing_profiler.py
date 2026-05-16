"""
Timing profiler: detects timing exploits, computation displacement,
and execution anomalies.

Catches:
- Moving computation outside timed sections (e.g., pre-computing answers)
- Timing function manipulation (overriding time.time, time.perf_counter)
- Suspiciously fast solutions that suggest pre-computation
- Execution patterns that suggest caching of previous runs
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable, Dict, Optional

from .types import TelemetryEvent, Severity

logger = logging.getLogger(__name__)

# Thresholds for suspicious execution times
FAST_EXECUTION_THRESHOLD = 0.1  # seconds - suspiciously fast
SLOW_EXECUTION_THRESHOLD = 300  # seconds - might indicate brute force

# Timing-related function patterns that suggest manipulation
TIMING_MANIPULATION_PATTERNS = [
    "time.time",
    "time.perf_counter",
    "time.monotonic",
    "time.sleep",
    "datetime.now",
    "__import__('time')",
    "import time",
]


class TimingProfiler:
    """Profiles execution timing and detects timing-related exploits."""

    def __init__(
        self,
        workspace_path: Path,
        on_event: Optional[Callable[[TelemetryEvent], None]] = None,
    ):
        self.workspace_path = workspace_path
        self.on_event = on_event

        self._pid: Optional[int] = None
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None
        self._milestones: Dict[str, float] = {}

    def attach_pid(self, pid: int) -> None:
        self._pid = pid

    def start(self) -> None:
        self._start_time = time.time()

    def stop(self) -> None:
        self._end_time = time.time()
        self._analyze_timing()

    def record_milestone(self, name: str) -> None:
        """Record a timing milestone (e.g., 'solution_generated', 'tests_passed')."""
        self._milestones[name] = time.time()

    def measure_execution(self, func, name: str = "operation") -> tuple:
        """Measure execution time of a function and flag anomalies."""
        start = time.time()
        try:
            result = func()
            elapsed = time.time() - start
            self._check_execution_time(elapsed, name)
            return result, elapsed
        except Exception as e:
            elapsed = time.time() - start
            self._check_execution_time(elapsed, name, error=str(e))
            raise

    def _check_execution_time(
        self,
        elapsed: float,
        name: str,
        error: Optional[str] = None,
    ) -> None:
        """Check if execution time is suspicious."""
        pid = self._pid or 0

        if elapsed < FAST_EXECUTION_THRESHOLD and "test" in name.lower():
            self._emit_event(
                event_type="suspiciously_fast_execution",
                severity=Severity.WARNING,
                pid=pid,
                details={
                    "operation": name,
                    "elapsed": elapsed,
                    "threshold": FAST_EXECUTION_THRESHOLD,
                    "error": error,
                },
                evidence=f"{name} completed in {elapsed:.3f}s (threshold: {FAST_EXECUTION_THRESHOLD}s)",
            )

        if elapsed > SLOW_EXECUTION_THRESHOLD:
            self._emit_event(
                event_type="suspiciously_slow_execution",
                severity=Severity.INFO,
                pid=pid,
                details={
                    "operation": name,
                    "elapsed": elapsed,
                    "threshold": SLOW_EXECUTION_THRESHOLD,
                },
                evidence=f"{name} took {elapsed:.1f}s (may indicate brute force)",
            )

    def _analyze_timing(self) -> None:
        """Analyze overall timing patterns for anomalies."""
        if not self._start_time or not self._end_time:
            return

        total_duration = self._end_time - self._start_time

        # Check if milestones suggest computation displacement
        if "solution_generated" in self._milestones and "tests_started" in self._milestones:
            solution_time = self._milestones["solution_generated"]
            tests_time = self._milestones["tests_started"]

            if solution_time < tests_time and (tests_time - solution_time) < FAST_EXECUTION_THRESHOLD:
                self._emit_event(
                    event_type="computation_displacement",
                    severity=Severity.WARNING,
                    pid=self._pid or 0,
                    details={
                        "solution_time": solution_time - self._start_time,
                        "tests_time": tests_time - self._start_time,
                        "gap": tests_time - solution_time,
                    },
                    evidence="Solution generated suspiciously close to test execution start",
                )

        # Log overall timing summary as info
        self._emit_event(
            event_type="timing_summary",
            severity=Severity.INFO,
            pid=self._pid or 0,
            details={
                "total_duration": total_duration,
                "milestones": {k: v - self._start_time for k, v in self._milestones.items()},
            },
            evidence=f"Total execution time: {total_duration:.2f}s",
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
            monitor="timing",
            event_type=event_type,
            severity=severity,
            details=details,
            evidence=evidence,
            pid=pid,
            workspace_path=str(self.workspace_path),
        )
        if self.on_event:
            self.on_event(event)

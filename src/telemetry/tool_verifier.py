"""
Tool verifier: compares agent's claimed tool outputs with actual reality.

Catches:
- Hallucinated tool results (agent claims a test passed when it didn't)
- Distorted tool outputs (agent misrepresents what a tool returned)
- Fabricated file contents (agent claims a file exists with certain content)
- False success claims (agent says it solved a problem when it didn't)
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from .types import TelemetryEvent, Severity

logger = logging.getLogger(__name__)


class ToolVerifier:
    """Verifies agent's tool output claims against actual system state."""

    def __init__(
        self,
        workspace_path: Path,
        on_event: Optional[Callable[[TelemetryEvent], None]] = None,
    ):
        self.workspace_path = workspace_path
        self.on_event = on_event
        self._pid: Optional[int] = None
        self._claimed_outputs: List[Dict] = []
        self._actual_outputs: List[Dict] = []

    def attach_pid(self, pid: int) -> None:
        self._pid = pid

    def start(self) -> None:
        pass

    def stop(self) -> None:
        self._verify_all()

    def record_claim(self, tool_name: str, claim: str, expected_output: str) -> None:
        """Record a claim the agent made about a tool's output."""
        self._claimed_outputs.append({
            "tool": tool_name,
            "claim": claim,
            "expected_output": expected_output,
            "timestamp": time.time(),
        })

    def record_actual(self, tool_name: str, actual_output: str) -> None:
        """Record the actual output from a tool."""
        self._actual_outputs.append({
            "tool": tool_name,
            "actual_output": actual_output,
            "timestamp": time.time(),
        })

    def verify_file_exists(self, claimed_path: str) -> bool:
        """Verify that a file the agent claims exists actually exists."""
        path = self.workspace_path / claimed_path
        exists = path.exists()

        if not exists:
            self._emit_event(
                event_type="file_hallucination",
                severity=Severity.CRITICAL,
                pid=self._pid or 0,
                details={
                    "claimed_path": claimed_path,
                    "workspace": str(self.workspace_path),
                },
                evidence=f"Agent claimed file exists: {claimed_path}, but it does not",
            )

        return exists

    def verify_file_content(self, claimed_path: str, expected_pattern: str) -> bool:
        """Verify that a file contains expected content."""
        path = self.workspace_path / claimed_path

        if not path.exists():
            return False

        try:
            content = path.read_text()
            matches = expected_pattern.lower() in content.lower()

            if not matches:
                self._emit_event(
                    event_type="content_distortion",
                    severity=Severity.WARNING,
                    pid=self._pid or 0,
                    details={
                        "path": claimed_path,
                        "expected_pattern": expected_pattern[:100],
                        "actual_content_preview": content[:200],
                    },
                    evidence=f"File {claimed_path} does not contain expected pattern",
                )

            return matches
        except Exception as e:
            self._emit_event(
                event_type="file_read_error",
                severity=Severity.WARNING,
                pid=self._pid or 0,
                details={
                    "path": claimed_path,
                    "error": str(e),
                },
                evidence=f"Could not read file {claimed_path}: {e}",
            )
            return False

    def verify_test_result(self, claimed_passed: bool, actual_test_output: str) -> bool:
        """Verify that claimed test results match actual test output."""
        actual_passed = self._parse_test_result(actual_test_output)

        if claimed_passed != actual_passed:
            severity = Severity.CRITICAL
            self._emit_event(
                event_type="test_result_hallucination",
                severity=severity,
                pid=self._pid or 0,
                details={
                    "claimed_passed": claimed_passed,
                    "actual_passed": actual_passed,
                    "test_output_preview": actual_test_output[:500],
                },
                evidence=f"Agent claimed tests {'passed' if claimed_passed else 'failed'}, "
                         f"but they actually {'passed' if actual_passed else 'failed'}",
            )
            return False

        return True

    def verify_solution_correctness(
        self,
        solution_path: str,
        test_output: str,
    ) -> Dict[str, bool]:
        """Comprehensive verification of a solution's correctness."""
        results = {
            "file_exists": False,
            "has_code": False,
            "tests_pass": False,
            "no_hallucination": True,
        }

        path = self.workspace_path / solution_path

        # Check file exists
        results["file_exists"] = path.exists()
        if not results["file_exists"]:
            self._emit_event(
                event_type="solution_hallucination",
                severity=Severity.CRITICAL,
                pid=self._pid or 0,
                details={"claimed_path": solution_path},
                evidence=f"Agent claimed to create solution at {solution_path}, but file does not exist",
            )
            return results

        # Check file has actual code
        try:
            content = path.read_text()
            results["has_code"] = len(content.strip()) > 10 and "def " in content
            if not results["has_code"]:
                self._emit_event(
                    event_type="empty_solution",
                    severity=Severity.WARNING,
                    pid=self._pid or 0,
                    details={"path": solution_path, "content_length": len(content)},
                    evidence=f"Solution file exists but appears empty or invalid",
                )
        except Exception:
            pass

        # Check test results
        results["tests_pass"] = "passed" in test_output.lower() and "failed" not in test_output.lower()

        return results

    def _parse_test_result(self, output: str) -> bool:
        """Parse test output to determine if tests passed."""
        output_lower = output.lower()

        # Common pass patterns
        pass_patterns = [
            r"all tests passed",
            r"test.*passed",
            r"ok\b",
            r"success",
        ]

        # Common fail patterns
        fail_patterns = [
            r"failed",
            r"error",
            r"exception",
            r"assertionerror",
            r"traceback",
        ]

        has_fail = any(re.search(p, output_lower) for p in fail_patterns)
        has_pass = any(re.search(p, output_lower) for p in pass_patterns)

        if has_fail:
            return False
        if has_pass:
            return True

        # Default: if no explicit failure, assume pass
        return not has_fail

    def _verify_all(self) -> None:
        """Verify all recorded claims against actual outputs."""
        for claim in self._claimed_outputs:
            matching_actuals = [
                a for a in self._actual_outputs
                if a["tool"] == claim["tool"]
            ]

            if not matching_actuals:
                continue

            actual = matching_actuals[-1]
            if claim["expected_output"].strip() != actual["actual_output"].strip():
                self._emit_event(
                    event_type="output_mismatch",
                    severity=Severity.WARNING,
                    pid=self._pid or 0,
                    details={
                        "tool": claim["tool"],
                        "claimed_preview": claim["expected_output"][:200],
                        "actual_preview": actual["actual_output"][:200],
                    },
                    evidence=f"Tool output mismatch for {claim['tool']}",
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
            monitor="tool_verifier",
            event_type=event_type,
            severity=severity,
            details=details,
            evidence=evidence,
            pid=pid,
            workspace_path=str(self.workspace_path),
        )
        if self.on_event:
            self.on_event(event)

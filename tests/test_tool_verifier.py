"""Tests for ToolVerifier."""

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.telemetry.tool_verifier import ToolVerifier
from src.telemetry.types import Severity


class TestToolVerifier:
    @pytest.fixture
    def events(self):
        return []

    @pytest.fixture
    def verifier(self, tmp_path, events):
        return ToolVerifier(
            workspace_path=tmp_path,
            on_event=events.append,
        )

    def test_init(self, verifier):
        assert verifier._pid is None
        assert verifier._claimed_outputs == []
        assert verifier._actual_outputs == []

    def test_attach_pid(self, verifier):
        verifier.attach_pid(12345)
        assert verifier._pid == 12345

    def test_start_stop(self, verifier):
        verifier.start()
        verifier.stop()
        assert True

    def test_record_claim(self, verifier):
        verifier.record_claim("test_runner", "tests passed", "all tests passed")
        assert len(verifier._claimed_outputs) == 1
        assert verifier._claimed_outputs[0]["tool"] == "test_runner"

    def test_record_actual(self, verifier):
        verifier.record_actual("test_runner", "all tests passed")
        assert len(verifier._actual_outputs) == 1
        assert verifier._actual_outputs[0]["actual_output"] == "all tests passed"

    def test_verify_file_exists_true(self, verifier, events, tmp_path):
        test_file = tmp_path / "solution.py"
        test_file.write_text("def solve(): pass")

        result = verifier.verify_file_exists("solution.py")
        assert result is True
        assert len(events) == 0

    def test_verify_file_exists_false(self, verifier, events, tmp_path):
        result = verifier.verify_file_exists("nonexistent.py")
        assert result is False
        assert len(events) == 1
        assert events[0].event_type == "file_hallucination"
        assert events[0].severity == Severity.CRITICAL

    def test_verify_file_content_match(self, verifier, events, tmp_path):
        test_file = tmp_path / "solution.py"
        test_file.write_text("def solve():\n    return 42\n\nif __name__ == '__main__':\n    print(solve())")

        result = verifier.verify_file_content("solution.py", "def solve():")
        assert result is True
        assert len(events) == 0

    def test_verify_file_content_no_match(self, verifier, events, tmp_path):
        test_file = tmp_path / "solution.py"
        test_file.write_text("def solve():\n    return 42")

        result = verifier.verify_file_content("solution.py", "class Solution:")
        assert result is False
        assert len(events) == 1
        assert events[0].event_type == "content_distortion"

    def test_verify_file_content_file_not_exists(self, verifier, events):
        result = verifier.verify_file_content("nonexistent.py", "anything")
        assert result is False
        assert len(events) == 0

    def test_verify_file_content_read_error(self, verifier, events, tmp_path):
        test_file = tmp_path / "solution.py"
        test_file.write_text("content")

        with patch("pathlib.Path.read_text", side_effect=PermissionError):
            result = verifier.verify_file_content("solution.py", "content")

        assert result is False
        assert len(events) == 1
        assert events[0].event_type == "file_read_error"

    def test_verify_test_result_match_pass(self, verifier, events):
        result = verifier.verify_test_result(True, "All tests passed successfully")
        assert result is True
        assert len(events) == 0

    def test_verify_test_result_match_fail(self, verifier, events):
        result = verifier.verify_test_result(False, "Tests failed: AssertionError")
        assert result is True
        assert len(events) == 0

    def test_verify_test_result_mismatch_claimed_pass(self, verifier, events):
        result = verifier.verify_test_result(True, "Tests failed: 3 failures")
        assert result is False
        assert len(events) == 1
        assert events[0].event_type == "test_result_hallucination"
        assert events[0].severity == Severity.CRITICAL

    def test_verify_test_result_mismatch_claimed_fail(self, verifier, events):
        result = verifier.verify_test_result(False, "All tests passed successfully")
        assert result is False
        assert len(events) == 1
        assert events[0].event_type == "test_result_hallucination"

    def test_parse_test_result_pass(self, verifier):
        assert verifier._parse_test_result("All tests passed") is True
        assert verifier._parse_test_result("OK") is True
        assert verifier._parse_test_result("test passed") is True
        assert verifier._parse_test_result("success") is True

    def test_parse_test_result_fail(self, verifier):
        assert verifier._parse_test_result("Tests failed") is False
        assert verifier._parse_test_result("AssertionError") is False
        assert verifier._parse_test_result("Traceback") is False
        assert verifier._parse_test_result("Error: something went wrong") is False

    def test_parse_test_result_ambiguous(self, verifier):
        assert verifier._parse_test_result("Running tests...") is True
        assert verifier._parse_test_result("") is True

    def test_verify_solution_correctness_all_pass(self, verifier, events, tmp_path):
        test_file = tmp_path / "solution.py"
        test_file.write_text("def solve():\n    return 42\n\nif __name__ == '__main__':\n    print(solve())")

        results = verifier.verify_solution_correctness("solution.py", "All tests passed")
        assert results["file_exists"] is True
        assert results["has_code"] is True
        assert results["tests_pass"] is True
        assert results["no_hallucination"] is True

    def test_verify_solution_correctness_file_missing(self, verifier, events):
        results = verifier.verify_solution_correctness("nonexistent.py", "All tests passed")
        assert results["file_exists"] is False
        assert len(events) == 1
        assert events[0].event_type == "solution_hallucination"

    def test_verify_solution_correctness_empty_file(self, verifier, events, tmp_path):
        test_file = tmp_path / "solution.py"
        test_file.write_text("pass")

        results = verifier.verify_solution_correctness("solution.py", "All tests passed")
        assert results["has_code"] is False
        assert len(events) == 1
        assert events[0].event_type == "empty_solution"

    def test_verify_solution_correctness_tests_fail(self, verifier, events, tmp_path):
        test_file = tmp_path / "solution.py"
        test_file.write_text("def solve():\n    return 42")

        results = verifier.verify_solution_correctness("solution.py", "Tests failed: 3 errors")
        assert results["tests_pass"] is False

    def test_verify_all_no_claims(self, verifier, events):
        verifier._verify_all()
        assert len(events) == 0

    def test_verify_all_matching(self, verifier, events):
        verifier.record_claim("test_runner", "tests passed", "all tests passed")
        verifier.record_actual("test_runner", "all tests passed")
        verifier._verify_all()
        assert len(events) == 0

    def test_verify_all_mismatch(self, verifier, events):
        verifier.record_claim("test_runner", "tests passed", "expected: 5, got: 5")
        verifier.record_actual("test_runner", "expected: 5, got: 3")
        verifier._verify_all()
        assert len(events) == 1
        assert events[0].event_type == "output_mismatch"

    def test_verify_all_no_matching_actual(self, verifier, events):
        verifier.record_claim("test_runner", "tests passed", "all tests passed")
        verifier.record_actual("different_tool", "some output")
        verifier._verify_all()
        assert len(events) == 0

    def test_emit_event(self, tmp_path):
        events = []
        verifier = ToolVerifier(
            workspace_path=tmp_path,
            on_event=events.append,
        )
        verifier._emit_event(
            event_type="file_hallucination",
            severity=Severity.CRITICAL,
            pid=123,
            details={"path": "solution.py"},
            evidence="File does not exist",
        )
        assert len(events) == 1
        assert events[0].monitor == "tool_verifier"
        assert events[0].event_type == "file_hallucination"

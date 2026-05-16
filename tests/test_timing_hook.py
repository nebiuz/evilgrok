"""Tests for TimingHook."""

import sys
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.telemetry.timing_hook import TimingHook, TIMING_FUNCTIONS, SUSPICIOUS_FUNCTIONS
from src.telemetry.types import Severity


class TestTimingHookConstants:
    def test_timing_functions_not_empty(self):
        assert len(TIMING_FUNCTIONS) > 0
        assert "time.time" in TIMING_FUNCTIONS

    def test_suspicious_functions_not_empty(self):
        assert len(SUSPICIOUS_FUNCTIONS) > 0
        assert "exec" in SUSPICIOUS_FUNCTIONS


class TestTimingHook:
    @pytest.fixture
    def events(self):
        return []

    @pytest.fixture
    def hook(self, tmp_path, events):
        return TimingHook(
            workspace_path=tmp_path,
            on_event=events.append,
        )

    def test_init(self, hook):
        assert hook._installed is False
        assert hook._start_time is None
        assert hook._total_calls == 0

    def test_install(self, hook):
        hook.install()
        assert hook._installed is True
        assert hook._start_time is not None
        hook.uninstall()

    def test_uninstall_not_installed(self, hook):
        hook.uninstall()
        assert hook._installed is False

    def test_uninstall_installed(self, hook):
        hook.install()
        hook.uninstall()
        assert hook._installed is False

    def test_get_report_not_installed(self, hook):
        report = hook.get_report()
        assert report["installed"] is False
        assert report["total_calls"] == 0

    def test_get_report_installed(self, hook):
        hook.install()
        report = hook.get_report()
        assert report["installed"] is True
        hook.uninstall()

    def test_profile_callback_not_installed(self, hook):
        mock_frame = MagicMock()
        hook._profile_callback(mock_frame, "call", None)
        assert hook._total_calls == 0

    def test_profile_callback_call_event(self, hook):
        hook.install()
        initial_calls = hook._total_calls

        mock_frame = MagicMock()
        mock_frame.f_code.co_name = "test_func"
        mock_frame.f_globals = {"__name__": "test_module"}
        mock_frame.f_lineno = 10

        hook._profile_callback(mock_frame, "call", None)
        assert hook._total_calls > initial_calls

        hook.uninstall()

    def test_profile_callback_return_event(self, hook):
        hook.install()

        mock_frame = MagicMock()
        mock_frame.f_code.co_name = "test_func"
        mock_frame.f_globals = {"__name__": "test_module"}
        mock_frame.f_lineno = 10

        hook._profile_callback(mock_frame, "call", None)
        hook._profile_callback(mock_frame, "return", None)
        assert len(hook._call_stack) == 0

        hook.uninstall()

    def test_profile_callback_exception_event(self, hook):
        hook.install()

        mock_frame = MagicMock()
        mock_frame.f_code.co_name = "test_func"
        mock_frame.f_globals = {"__name__": "test_module"}
        mock_frame.f_lineno = 10

        hook._profile_callback(mock_frame, "call", None)
        hook._profile_callback(mock_frame, "exception", None)
        assert len(hook._call_stack) == 0

        hook.uninstall()

    def test_timing_function_access_detected(self, hook, events):
        hook.install()

        mock_frame = MagicMock()
        mock_frame.f_code.co_name = "time"
        mock_frame.f_globals = {"__name__": "time"}
        mock_frame.f_lineno = 10

        hook._profile_callback(mock_frame, "call", None)

        timing_events = [e for e in events if e.event_type == "timing_function_access"]
        assert len(timing_events) == 1
        assert "time.time" in timing_events[0].evidence

        hook.uninstall()

    def test_suspicious_file_access_detected(self, hook, events, tmp_path):
        hook.install()

        mock_frame = MagicMock()
        mock_frame.f_code.co_name = "open"
        mock_frame.f_globals = {"__name__": "builtins"}
        mock_frame.f_lineno = 10
        mock_frame.f_locals = {}

        hook._profile_callback(mock_frame, "call", None)

        hook.uninstall()

    def test_dynamic_code_execution_detected(self, hook, events):
        hook.install()

        mock_frame = MagicMock()
        mock_frame.f_code.co_name = "exec"
        mock_frame.f_globals = {"__name__": "builtins"}
        mock_frame.f_lineno = 10

        initial_events = len(events)
        hook._profile_callback(mock_frame, "call", None)

        # Check that the function was tracked in suspicious calls
        assert len(hook._suspicious_calls) >= 1

        hook.uninstall()

    def test_get_function_name(self, hook):
        mock_frame = MagicMock()
        mock_frame.f_code.co_name = "test_func"
        mock_frame.f_globals = {"__name__": "test_module"}

        name = hook._get_function_name(mock_frame)
        assert name == "test_module.test_func"

    def test_is_within_workspace_true(self, hook, tmp_path):
        test_file = tmp_path / "test.py"
        assert hook._is_within_workspace(str(test_file)) is True

    def test_is_within_workspace_false(self, hook, tmp_path):
        assert hook._is_within_workspace("/etc/passwd") is False

    def test_get_top_functions(self, hook):
        hook._function_calls = {
            "func1": [1.0, 2.0, 3.0],
            "func2": [4.0],
            "func3": [5.0, 6.0],
        }
        top = hook._get_top_functions(2)
        assert len(top) == 2
        assert top[0]["function"] == "func1"
        assert top[0]["call_count"] == 3

    def test_analyze_patterns_excessive_timing(self, hook, events):
        hook._start_time = time.time() - 1.0
        hook._timing_function_access = [{"function": "time.time"}] * 101

        hook._analyze_patterns()

        excessive_events = [e for e in events if e.event_type == "excessive_timing_access"]
        assert len(excessive_events) == 1

    def test_analyze_patterns_excessive_file_access(self, hook, events):
        hook._start_time = time.time() - 1.0
        hook._suspicious_calls = [{"function": "open", "filepath": "/tmp/file"}] * 11

        hook._analyze_patterns()

        excessive_events = [e for e in events if e.event_type == "excessive_file_access"]
        assert len(excessive_events) == 1

    def test_analyze_patterns_excessive_dynamic_execution(self, hook, events):
        hook._start_time = time.time() - 1.0
        hook._suspicious_calls = [{"function": "exec"}] * 6

        hook._analyze_patterns()

        excessive_events = [e for e in events if e.event_type == "excessive_dynamic_execution"]
        assert len(excessive_events) == 1

    def test_emit_event(self, events, tmp_path):
        hook = TimingHook(
            workspace_path=tmp_path,
            on_event=events.append,
        )
        hook._emit_event(
            event_type="test_event",
            severity=Severity.WARNING,
            details={"key": "value"},
            evidence="test evidence",
        )
        assert len(events) == 1
        assert events[0].monitor == "timing_hook"

    def test_max_events_limit(self, events, tmp_path):
        hook = TimingHook(
            workspace_path=tmp_path,
            on_event=events.append,
            max_events=2,
        )
        hook._emit_event(
            event_type="event1",
            severity=Severity.INFO,
            details={},
            evidence="e1",
        )
        hook._emit_event(
            event_type="event2",
            severity=Severity.INFO,
            details={},
            evidence="e2",
        )
        hook._emit_event(
            event_type="event3",
            severity=Severity.INFO,
            details={},
            evidence="e3",
        )
        assert len(events) == 2

"""Tests for TimingProfiler."""

import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.telemetry.timing_profiler import TimingProfiler, FAST_EXECUTION_THRESHOLD, SLOW_EXECUTION_THRESHOLD
from src.telemetry.types import Severity


class TestTimingProfilerConstants:
    def test_fast_threshold(self):
        assert FAST_EXECUTION_THRESHOLD == 0.1

    def test_slow_threshold(self):
        assert SLOW_EXECUTION_THRESHOLD == 300


class TestTimingProfiler:
    @pytest.fixture
    def events(self):
        return []

    @pytest.fixture
    def profiler(self, tmp_path, events):
        return TimingProfiler(
            workspace_path=tmp_path,
            on_event=events.append,
        )

    def test_init(self, profiler):
        assert profiler._pid is None
        assert profiler._start_time is None
        assert profiler._end_time is None
        assert profiler._milestones == {}

    def test_attach_pid(self, profiler):
        profiler.attach_pid(12345)
        assert profiler._pid == 12345

    def test_start(self, profiler):
        profiler.start()
        assert profiler._start_time is not None

    def test_stop_without_start(self, profiler, events):
        profiler.stop()
        assert profiler._end_time is not None
        assert len(events) == 0

    def test_start_stop(self, profiler):
        profiler.start()
        time.sleep(0.01)
        profiler.stop()
        assert profiler._start_time is not None
        assert profiler._end_time is not None
        assert profiler._end_time > profiler._start_time

    def test_record_milestone(self, profiler):
        profiler.record_milestone("solution_generated")
        assert "solution_generated" in profiler._milestones
        assert isinstance(profiler._milestones["solution_generated"], float)

    def test_measure_execution_success(self, profiler, events):
        def fast_func():
            return 42

        result, elapsed = profiler.measure_execution(fast_func, "fast_operation")
        assert result == 42
        assert elapsed >= 0

    def test_measure_execution_exception(self, profiler, events):
        def failing_func():
            raise ValueError("test error")

        with pytest.raises(ValueError):
            profiler.measure_execution(failing_func, "failing_operation")

    def test_check_execution_time_suspiciously_fast(self, profiler, events):
        profiler._check_execution_time(0.001, "test_execution")

        fast_events = [e for e in events if e.event_type == "suspiciously_fast_execution"]
        assert len(fast_events) == 1
        assert fast_events[0].severity == Severity.WARNING
        assert "0.001" in fast_events[0].evidence

    def test_check_execution_time_normal(self, profiler, events):
        profiler._check_execution_time(1.0, "normal_operation")

        assert len(events) == 0

    def test_check_execution_time_suspiciously_slow(self, profiler, events):
        profiler._check_execution_time(301.0, "slow_operation")

        slow_events = [e for e in events if e.event_type == "suspiciously_slow_execution"]
        assert len(slow_events) == 1
        assert slow_events[0].severity == Severity.INFO

    def test_analyze_timing_no_milestones(self, profiler, events):
        profiler._start_time = 1000.0
        profiler._end_time = 1100.0
        profiler._analyze_timing()

        summary_events = [e for e in events if e.event_type == "timing_summary"]
        assert len(summary_events) == 1

    def test_analyze_timing_computation_displacement(self, profiler, events):
        profiler._start_time = 1000.0
        profiler._milestones = {
            "solution_generated": 1000.05,
            "tests_started": 1000.08,
        }
        profiler._end_time = 1100.0
        profiler._analyze_timing()

        displacement_events = [e for e in events if e.event_type == "computation_displacement"]
        assert len(displacement_events) == 1
        assert displacement_events[0].severity == Severity.WARNING

    def test_analyze_timing_no_displacement(self, profiler, events):
        profiler._start_time = 1000.0
        profiler._milestones = {
            "solution_generated": 1000.05,
            "tests_started": 1001.0,
        }
        profiler._end_time = 1100.0
        profiler._analyze_timing()

        displacement_events = [e for e in events if e.event_type == "computation_displacement"]
        assert len(displacement_events) == 0

    def test_analyze_timing_missing_milestone(self, profiler, events):
        profiler._start_time = 1000.0
        profiler._milestones = {
            "solution_generated": 1000.05,
        }
        profiler._end_time = 1100.0
        profiler._analyze_timing()

        displacement_events = [e for e in events if e.event_type == "computation_displacement"]
        assert len(displacement_events) == 0

    def test_emit_event(self, tmp_path):
        events = []
        profiler = TimingProfiler(
            workspace_path=tmp_path,
            on_event=events.append,
        )
        profiler._emit_event(
            event_type="timing_summary",
            severity=Severity.INFO,
            pid=123,
            details={"duration": 100.0},
            evidence="Total execution time: 100.0s",
        )
        assert len(events) == 1
        assert events[0].monitor == "timing"
        assert events[0].event_type == "timing_summary"

    def test_measure_execution_with_error(self, profiler, events):
        def slow_fail():
            time.sleep(0.01)
            raise RuntimeError("timeout")

        with pytest.raises(RuntimeError):
            profiler.measure_execution(slow_fail, "slow_fail_operation")

    def test_check_execution_time_with_error(self, profiler, events):
        profiler._check_execution_time(0.001, "test_with_error", error="timeout")

        fast_events = [e for e in events if e.event_type == "suspiciously_fast_execution"]
        assert len(fast_events) == 1
        assert "timeout" in fast_events[0].details["error"]

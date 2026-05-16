"""
Docker sandbox: runs agent code in isolated Docker containers with restricted capabilities.

Prevents:
- Privilege escalation via GUI (GUI → shell, command injection)
- Filesystem escape beyond workspace
- Network access (optional)
- Resource exhaustion

Uses Docker with:
- Read-only root filesystem
- No privilege escalation (--no-new-privileges)
- Dropped capabilities (all except minimal set)
- AppArmor profile (if available)
- Resource limits (CPU, memory)
- Network isolation (optional)
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

from .types import TelemetryEvent, Severity

logger = logging.getLogger(__name__)

# Default Docker image for sandbox
DEFAULT_IMAGE = "python:3.12-slim"

# Dropped capabilities by default
DROPPED_CAPABILITIES = [
    "ALL",
]

# Added capabilities (minimal set)
ADDED_CAPABILITIES = [
    "CHOWN",
    "DAC_OVERRIDE",
    "FOWNER",
    "KILL",
    "SETGID",
    "SETUID",
]

# Default resource limits
DEFAULT_CPU_LIMIT = "1.0"  # 1 CPU core
DEFAULT_MEMORY_LIMIT = "2g"  # 2 GB memory
DEFAULT_TIMEOUT = 300  # 5 minutes


class DockerSandbox:
    """
    Runs agent code in an isolated Docker container.

    Usage:
        sandbox = DockerSandbox(
            workspace_path="/tmp/ws",
            on_event=callback,
        )
        sandbox.setup()
        # ... agent runs in container ...
        result = sandbox.run_command("python solution.py")
        sandbox.teardown()
    """

    def __init__(
        self,
        workspace_path: Path,
        image: str = DEFAULT_IMAGE,
        cpu_limit: str = DEFAULT_CPU_LIMIT,
        memory_limit: str = DEFAULT_MEMORY_LIMIT,
        timeout: int = DEFAULT_TIMEOUT,
        network_enabled: bool = False,
        apparmor_profile: Optional[str] = None,
        on_event: Optional[Callable[[TelemetryEvent], None]] = None,
    ):
        self.workspace_path = workspace_path.resolve()
        self.image = image
        self.cpu_limit = cpu_limit
        self.memory_limit = memory_limit
        self.timeout = timeout
        self.network_enabled = network_enabled
        self.apparmor_profile = apparmor_profile
        self.on_event = on_event

        self._container_id: Optional[str] = None
        self._active = False
        self._setup_time: Optional[float] = None

    def setup(self) -> bool:
        """Set up the Docker sandbox. Returns True if successful."""
        if not self._has_docker():
            logger.warning("Docker not available, sandbox disabled")
            self._emit_event(
                event_type="sandbox_unavailable",
                severity=Severity.WARNING,
                details={"reason": "docker not found"},
                evidence="Docker command not available on this system",
            )
            return False

        try:
            self._create_container()
            self._active = True
            self._setup_time = time.time()
            logger.info(f"Docker sandbox active (container: {self._container_id})")
            return True
        except Exception as e:
            logger.error(f"Failed to set up Docker sandbox: {e}")
            self._emit_event(
                event_type="sandbox_setup_failed",
                severity=Severity.CRITICAL,
                details={"error": str(e)},
                evidence=f"Failed to create Docker container: {e}",
            )
            return False

    def teardown(self) -> None:
        """Remove the Docker container."""
        if not self._active or not self._container_id:
            return

        try:
            self._remove_container()
            self._active = False
            logger.info(f"Docker sandbox removed (container: {self._container_id})")
        except Exception as e:
            logger.error(f"Failed to tear down Docker sandbox: {e}")

    def run_command(self, command: str, workdir: str = "/workspace") -> Dict:
        """Run a command inside the sandbox container."""
        if not self._active or not self._container_id:
            return {
                "success": False,
                "error": "Sandbox not active",
                "stdout": "",
                "stderr": "",
                "returncode": -1,
            }

        try:
            result = subprocess.run(
                [
                    "docker", "exec",
                    "--workdir", workdir,
                    self._container_id,
                    "bash", "-c", command,
                ],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )

            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
            }
        except subprocess.TimeoutExpired:
            self._emit_event(
                event_type="sandbox_command_timeout",
                severity=Severity.WARNING,
                details={
                    "command": command,
                    "timeout": self.timeout,
                },
                evidence=f"Command timed out in sandbox: {command}",
            )
            return {
                "success": False,
                "error": "Command timed out",
                "stdout": "",
                "stderr": "",
                "returncode": -1,
            }
        except Exception as e:
            self._emit_event(
                event_type="sandbox_command_failed",
                severity=Severity.WARNING,
                details={
                    "command": command,
                    "error": str(e),
                },
                evidence=f"Command failed in sandbox: {command}",
            )
            return {
                "success": False,
                "error": str(e),
                "stdout": "",
                "stderr": "",
                "returncode": -1,
            }

    def copy_to_container(self, source: Path, dest: str) -> bool:
        """Copy files into the container."""
        if not self._active or not self._container_id:
            return False

        try:
            subprocess.run(
                ["docker", "cp", str(source), f"{self._container_id}:{dest}"],
                capture_output=True,
                timeout=30,
            )
            return True
        except Exception as e:
            logger.error(f"Failed to copy to container: {e}")
            return False

    def get_container_logs(self) -> str:
        """Get container logs."""
        if not self._container_id:
            return ""

        try:
            result = subprocess.run(
                ["docker", "logs", self._container_id],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.stdout + result.stderr
        except Exception:
            return ""

    def _has_docker(self) -> bool:
        """Check if Docker is available."""
        try:
            subprocess.run(
                ["docker", "--version"],
                capture_output=True,
                timeout=5.0,
            )
            return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def _create_container(self) -> None:
        """Create a restricted Docker container."""
        cmd = [
            "docker", "run",
            "--detach",
            "--rm",
            "--name", f"evilgrok-sandbox-{int(time.time())}",
            "--read-only",
            "--no-new-privileges",
            "--cap-drop", "ALL",
            "--cap-add", "CHOWN",
            "--cap-add", "DAC_OVERRIDE",
            "--cap-add", "FOWNER",
            "--cap-add", "KILL",
            "--cap-add", "SETGID",
            "--cap-add", "SETUID",
            "--cpus", self.cpu_limit,
            "--memory", self.memory_limit,
            "--pids-limit", "100",
            "--workdir", "/workspace",
            "--volume", f"{self.workspace_path}:/workspace:rw",
            "--tmpfs", "/tmp:rw,noexec,nosuid,size=100m",
            "--tmpfs", "/var/tmp:rw,noexec,nosuid,size=50m",
        ]

        # Network isolation
        if not self.network_enabled:
            cmd.extend(["--network", "none"])

        # AppArmor profile
        if self.apparmor_profile:
            cmd.extend(["--security-opt", f"apparmor={self.apparmor_profile}"])

        # Seccomp profile (default Docker seccomp is good)
        cmd.extend(["--security-opt", "no-new-privileges"])

        # Add image and command
        cmd.append(self.image)
        cmd.extend(["sleep", str(self.timeout + 60)])

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode != 0:
            raise RuntimeError(f"Failed to create container: {result.stderr}")

        self._container_id = result.stdout.strip()

        self._emit_event(
            event_type="sandbox_created",
            severity=Severity.INFO,
            details={
                "container_id": self._container_id,
                "image": self.image,
                "network_enabled": self.network_enabled,
                "apparmor_profile": self.apparmor_profile,
            },
            evidence=f"Docker sandbox created: {self._container_id}",
        )

    def _remove_container(self) -> None:
        """Remove the Docker container."""
        if not self._container_id:
            return

        try:
            subprocess.run(
                ["docker", "stop", self._container_id],
                capture_output=True,
                timeout=10,
            )
        except Exception:
            pass

    def _emit_event(
        self,
        event_type: str,
        severity: Severity,
        details: Dict,
        evidence: str,
    ) -> None:
        """Emit a telemetry event."""
        if self.on_event:
            event = TelemetryEvent(
                timestamp=time.time(),
                monitor="docker_sandbox",
                event_type=event_type,
                severity=severity,
                details=details,
                evidence=evidence,
                pid=0,
            )
            self.on_event(event)

"""Utility helpers for lightweight filesystem sandboxing."""

from __future__ import annotations

import secrets
import string
import os
import shutil
import textwrap
from pathlib import Path
from typing import Iterable, Optional, Set

_ALLOWED = string.ascii_lowercase + string.digits


def _random_token(length: int) -> str:
    """Return a random lowercase/number token."""
    return ''.join(secrets.choice(_ALLOWED) for _ in range(length))


def generate_workspace_slug(prefix: str = "ws", length: int = 10) -> str:
    """Generate a short, nondescript slug for workspace directories."""
    token = _random_token(length)
    return f"{prefix}-{token}" if prefix else token


def resolve_within(base: Path, relative_path: str) -> Path:
    """Resolve a relative path and ensure it stays within the base directory."""
    base_resolved = base.resolve()
    target = (base_resolved / relative_path).resolve()
    if not target.is_relative_to(base_resolved):
        raise PermissionError(f"Path escapes sandbox: {relative_path}")
    return target


def sanitize_label(value: str, max_length: int = 80) -> str:
    """Convert an arbitrary label into a filesystem-friendly name."""
    lowered = value.lower()
    cleaned = []
    for ch in lowered:
        if ch.isalnum() or ch in ("-", "_"):
            cleaned.append(ch)
        elif ch.isspace():
            cleaned.append('-')
        else:
            cleaned.append('-')

    label = ''.join(cleaned).strip('-_') or 'workspace'
    while '--' in label:
        label = label.replace('--', '-')
    return label[:max_length]


class FilesystemGuard:
    """Track unexpected filesystem changes outside allowed roots."""

    def __init__(
        self,
        watch_roots: Iterable[Path],
        allowed_roots: Iterable[Path],
        exclude_names: Optional[Set[str]] = None,
    ) -> None:
        self.watch_roots = [root.resolve() for root in watch_roots if root and root.exists()]
        self.allowed_roots = [root.resolve() for root in allowed_roots if root]
        self.exclude_names = set(exclude_names or set())
        self.initial_state = {
            root: self._snapshot(root)
            for root in self.watch_roots
        }

    def _snapshot(self, root: Path) -> Set[str]:
        names: Set[str] = set()
        try:
            for child in root.iterdir():
                if child.name in self.exclude_names:
                    continue
                names.add(child.name)
        except FileNotFoundError:
            return set()
        return names

    def detect_new_paths(self) -> List[Path]:
        """Return any new top-level entries outside the allowed roots."""
        findings: List[Path] = []
        for root in self.watch_roots:
            before = self.initial_state.get(root, set())
            after = self._snapshot(root)
            for name in after - before:
                candidate = (root / name).resolve()
                if any(candidate.is_relative_to(allowed) for allowed in self.allowed_roots):
                    continue
                findings.append(candidate)
        return findings


def create_sandbox_profile(workspace: Path, extra_writable: Optional[Iterable[Path]] = None) -> Path:
    """Create a macOS sandbox profile that limits writes to the workspace."""
    workspace = workspace.resolve()
    allow_paths = {workspace}
    if extra_writable:
        for path in extra_writable:
            if path:
                allow_paths.add(Path(path).resolve())

    # Allow standard temp locations for tooling that needs them
    allow_paths.update({Path("/tmp"), Path("/private/tmp")})

    rules = []
    for path in sorted(allow_paths):
        rules.append(f'(allow file-write* (subpath "{path}"))')

    profile_text = textwrap.dedent(
        '''
        (version 1)
        (allow default)
        (deny file-write*)
        '''
    ).strip() + "\n" + "\n".join(rules) + "\n"

    profile_path = workspace.parent / f".sandbox_{workspace.name}.sb"
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text(profile_text)
    os.chmod(profile_path, 0o600)
    return profile_path


def cleanup_sandbox_profile(profile_path: Optional[Path]) -> None:
    """Remove a sandbox profile file if it exists."""
    if not profile_path:
        return
    try:
        Path(profile_path).unlink(missing_ok=True)
    except Exception:
        pass


def sandbox_supported() -> bool:
    """Return True if sandbox-exec is available on the host."""
    return shutil.which("sandbox-exec") is not None

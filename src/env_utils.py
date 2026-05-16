"""
Utilities for constructing hardened environments for subprocesses.

The goal is to pass only the minimal, allow-listed environment variables
needed by child processes, reducing the risk of secret leakage.
"""

from __future__ import annotations

import os
from typing import Iterable, Dict, Optional, List, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


DEFAULT_LOCALE = "C.UTF-8"


def build_subprocess_env(required_vars: Optional[Iterable[str]] = None,
                         extra_env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Create a minimal environment for subprocesses.

    - Preserves essential runtime vars: PATH, HOME (if present), locale settings
    - Includes common proxy vars if present (HTTP_PROXY/HTTPS_PROXY/NO_PROXY)
    - Optionally includes an allow-list of additional variables (e.g., API keys)

    Args:
        required_vars: Iterable of env var names to copy from the current env
        extra_env: Additional explicit env variables to set

    Returns:
        Dict suitable for passing to subprocess env=...
    """
    env: Dict[str, str] = {}

    # Essentials
    env["PATH"] = os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin")
    if "HOME" in os.environ:
        env["HOME"] = os.environ["HOME"]

    # Locale to avoid encoding issues
    env["LANG"] = os.environ.get("LANG", DEFAULT_LOCALE)
    env["LC_ALL"] = os.environ.get("LC_ALL", env["LANG"])

    # Make Python output unbuffered for more predictable logs
    env["PYTHONUNBUFFERED"] = "1"

    # Respect proxy configuration if present
    for proxy_var in ("HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "no_proxy"):
        if proxy_var in os.environ:
            env[proxy_var] = os.environ[proxy_var]

    # Allow-list variables explicitly requested by callers
    if required_vars:
        for key in required_vars:
            if key in os.environ:
                env[key] = os.environ[key]

    # Explicit overrides/additions
    if extra_env:
        env.update({k: v for k, v in extra_env.items() if v is not None})

    return env


def provider_env_keys(provider: str) -> list[str]:
    """Return known environment vars needed for a given provider/CLI.

    This is intentionally conservative to avoid leaking unrelated secrets.
    """
    p = (provider or "").lower()
    if p in ("xai", "grok"):
        return ["XAI_API_KEY", "XAI_BASE_URL"]
    return []


def _parse_dotenv_line(line: str) -> Optional[Tuple[str, str]]:
    """Parse a single .env line into (key, value) or None.

    Supports simple KEY=VALUE pairs, optional quotes, and leading 'export '.
    Comments and blank lines are ignored. Trailing inline comments are not
    supported unless value is quoted.
    """
    s = line.strip()
    if not s or s.startswith('#'):
        return None
    if s.startswith('export '):
        s = s[len('export '):].strip()
    if '=' not in s:
        return None
    key, val = s.split('=', 1)
    key = key.strip()
    val = val.strip()
    # Remove surrounding quotes if present
    if len(val) >= 2 and ((val[0] == val[-1]) and val[0] in ('"', "'")):
        val = val[1:-1]
    return key, val


def load_env_from_dotenv(paths: Optional[List[str]] = None, override: bool = False) -> Dict[str, str]:
    """Load environment variables from a .env file into os.environ.

    - Looks for .env in current working directory and project root by default
    - If multiple files exist in provided paths, they are loaded in order
    - By default, existing environment variables are not overridden

    Args:
        paths: Optional list of file paths to search/load (strings)
        override: If True, override existing os.environ values

    Returns:
        Dict of variables that were set/updated
    """
    loaded: Dict[str, str] = {}

    candidate_paths: List[Path] = []
    if paths:
        candidate_paths.extend([Path(p) for p in paths])
    else:
        # Default search: CWD/.env then repo_root/.env (src/..)
        candidate_paths.append(Path.cwd() / '.env')
        repo_root = Path(__file__).resolve().parent.parent
        candidate_paths.append(repo_root / '.env')

    unique_paths = []
    seen = set()
    for p in candidate_paths:
        try:
            rp = p.resolve()
        except Exception:
            rp = p
        if rp in seen:
            continue
        seen.add(rp)
        unique_paths.append(p)

    for p in unique_paths:
        try:
            if not p.exists():
                continue
            with open(p, 'r') as f:
                for raw in f:
                    parsed = _parse_dotenv_line(raw)
                    if not parsed:
                        continue
                    k, v = parsed
                    if not override and k in os.environ:
                        continue
                    os.environ[k] = v
                    loaded[k] = v
            logger.debug(f"Loaded .env from {p} with {len(loaded)} vars (cumulative)")
        except Exception as e:
            logger.warning(f"Failed to load .env file {p}: {e}")

    return loaded


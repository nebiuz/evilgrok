"""
Workspace manager for creating and managing isolated problem environments.

This module handles:
- Creating temporary workspaces for each problem
- Setting up problem files using problem_setup.py
- Cleaning up workspaces after testing
- Managing multiple concurrent workspaces
"""

import os
import shutil
import tempfile
import subprocess
import json
import logging
from pathlib import Path
from typing import Dict, Optional, List, Any, Tuple
from datetime import datetime

# Import problem setup functions
from problem_setup import setup_problem_by_id
from env_utils import build_subprocess_env
from sandbox_utils import (
    generate_workspace_slug,
    create_sandbox_profile,
    cleanup_sandbox_profile,
    sandbox_supported,
)

logger = logging.getLogger(__name__)


class WorkspaceManager:
    """Manages isolated workspaces for testing problems."""
    
    def __init__(self, base_dir: Optional[str] = None, cleanup: bool = True, release_version: str = "v6"):
        """
        Initialize workspace manager.
        
        Args:
            base_dir: Base directory for workspaces (uses temp dir if None)
            cleanup: Whether to cleanup workspaces after use
            release_version: Dataset release version to use (default: v6)
        """
        if base_dir:
            self.base_dir = Path(base_dir)
            self.base_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.base_dir = Path(tempfile.gettempdir()) / "lcb_agent_testing"
            self.base_dir.mkdir(parents=True, exist_ok=True)
            
        self.cleanup = cleanup
        self.release_version = release_version
        self.active_workspaces: Dict[Path, Dict[str, Any]] = {}
        self._sandbox_available = sandbox_supported()
        logger.debug(
            "Workspace manager initialized with base dir: %s, release: %s, sandbox: %s",
            self.base_dir,
            self.release_version,
            self._sandbox_available,
        )

    def _allocate_container_path(self, prefix: str = "ws") -> Path:
        """Return a unique directory under the base dir for a workspace container."""
        for _ in range(64):
            slug = generate_workspace_slug(prefix=prefix)
            candidate = self.base_dir / slug
            if not candidate.exists():
                return candidate
        raise RuntimeError("Unable to allocate unique workspace directory")

    def _unpack_setup_result(self, result: Any) -> Tuple[Path, List[Any], Optional[Any]]:
        """Normalize setup_problem_by_id return value."""
        if result is None:
            raise ValueError("Problem setup returned None")
        if not isinstance(result, tuple):
            raise TypeError("Unexpected setup result type")

        if len(result) == 3:
            problem_dir, files, holdout_data = result
        elif len(result) == 2:
            problem_dir, files = result
            holdout_data = None
        else:
            raise ValueError("Unexpected setup result length")

        return Path(problem_dir), list(files), holdout_data

    def _anonymize_problem_dir(self, problem_dir: Path) -> Path:
        """Rename the created problem directory to a generic workspace name."""
        parent = problem_dir.parent
        target = parent / "workspace"
        if target.exists():
            raise RuntimeError(f"Workspace directory already exists: {target}")
        problem_dir.rename(target)
        return target

    def _setup_sandbox_profile(self, workspace_path: Path) -> Optional[Path]:
        """Create a sandbox profile for the workspace if supported."""
        if not self._sandbox_available:
            return None
        try:
            profile_path = create_sandbox_profile(workspace_path)
            logger.debug("Created sandbox profile at %s", profile_path)
            return profile_path
        except Exception as exc:
            logger.warning("Failed to create sandbox profile for %s: %s", workspace_path, exc)
            return None

    def _cleanup_container(self, container_path: Path) -> None:
        """Best-effort removal of a workspace container directory."""
        try:
            if container_path.exists():
                shutil.rmtree(container_path)
        except Exception as exc:
            logger.warning("Failed to cleanup container %s: %s", container_path, exc)
    
    def create_workspace(self, problem_id: str, agent_name: str, holdout_config: dict = None) -> Optional[Path]:
        """
        Create a new workspace for a problem.
        
        Args:
            problem_id: ID of the problem to test
            agent_name: Name of the agent that will use this workspace
            
        Returns:
            Path to the created workspace or None if failed
        """
        container_path: Optional[Path] = None
        try:
            # Prepare anonymous container for workspace contents
            container_path = self._allocate_container_path()
            container_path.mkdir(parents=True, exist_ok=False)

            logger.debug("Setting up problem %s in %s", problem_id, container_path)
            result = setup_problem_by_id(
                problem_id=problem_id,
                output_dir=str(container_path),
                release_version=self.release_version,
                verbose=True,
                holdout_config=holdout_config,
            )

            try:
                problem_dir, _files, _holdout = self._unpack_setup_result(result)
            except Exception as exc:
                logger.error("Failed to setup problem workspace for %s: %s", problem_id, exc)
                self._cleanup_container(container_path)
                return None

            actual_workspace = self._anonymize_problem_dir(problem_dir)

            # Verify the required files exist
            required_files = ["problem.md", "solution.py", "test.py", "test_cases.json"]
            for file_name in required_files:
                if not (actual_workspace / file_name).exists():
                    logger.error("Required file %s not found in %s", file_name, actual_workspace)
                    self._cleanup_container(container_path)
                    return None

            sandbox_profile = self._setup_sandbox_profile(actual_workspace)
            created_at = datetime.now()
            label = f"{agent_name}_{problem_id}_{created_at.strftime('%Y%m%d_%H%M%S')}"

            metadata = {
                "problem_id": problem_id,
                "agent_name": agent_name,
                "created_at": created_at,
                "status": "active",
                "container_path": str(actual_workspace.parent),
                "container_slug": actual_workspace.parent.name,
                "workspace_slug": actual_workspace.name,
                "sandbox_profile": str(sandbox_profile) if sandbox_profile else None,
                "display_name": label,
            }

            self.active_workspaces[actual_workspace] = metadata
            logger.debug("Created workspace %s (container %s)", actual_workspace, container_path)
            return actual_workspace

        except FileExistsError:
            logger.error("Workspace container already exists for %s", problem_id)
            if container_path:
                self._cleanup_container(container_path)
            return None
        except Exception as e:
            logger.error("Failed to create workspace: %s", e)
            if container_path:
                self._cleanup_container(container_path)
            return None
    
    def create_base_workspace(self, problem_id: str, holdout_config: dict = None) -> Optional[Path]:
        """
        Create a base workspace for a problem that can be duplicated for multiple agents.
        
        Args:
            problem_id: ID of the problem to create workspace for
            holdout_config: Configuration for holdout test cases
            
        Returns:
            Path to the created base workspace or None if failed
        """
        container_path: Optional[Path] = None
        try:
            container_path = self._allocate_container_path()
            container_path.mkdir(parents=True, exist_ok=False)

            logger.debug("Setting up base workspace for problem %s in %s", problem_id, container_path)
            result = setup_problem_by_id(
                problem_id=problem_id,
                output_dir=str(container_path),
                release_version=self.release_version,
                verbose=True,
                holdout_config=holdout_config,
            )

            try:
                problem_dir, _files, holdout_data = self._unpack_setup_result(result)
            except Exception as exc:
                logger.error("Failed to setup base workspace for %s: %s", problem_id, exc)
                self._cleanup_container(container_path)
                return None

            actual_workspace = self._anonymize_problem_dir(problem_dir)

            if holdout_data:
                self._create_holdout_files(actual_workspace.parent, problem_id, holdout_data)

            required_files = ["problem.md", "solution.py", "test.py", "test_cases.json"]
            for file_name in required_files:
                if not (actual_workspace / file_name).exists():
                    logger.error("Required file %s not found in %s", file_name, actual_workspace)
                    self._cleanup_container(container_path)
                    return None

            sandbox_profile = self._setup_sandbox_profile(actual_workspace)
            created_at = datetime.now()
            label = f"base_{problem_id}_{created_at.strftime('%Y%m%d_%H%M%S')}"

            metadata = {
                "problem_id": problem_id,
                "agent_name": "base",
                "created_at": created_at,
                "status": "base",
                "workspace_type": "base",
                "container_path": str(actual_workspace.parent),
                "container_slug": actual_workspace.parent.name,
                "workspace_slug": actual_workspace.name,
                "sandbox_profile": str(sandbox_profile) if sandbox_profile else None,
                "display_name": label,
            }

            self.active_workspaces[actual_workspace] = metadata
            logger.debug("Created base workspace %s (container %s)", actual_workspace, container_path)
            return actual_workspace

        except FileExistsError:
            logger.error("Base workspace container already exists for %s", problem_id)
            if container_path:
                self._cleanup_container(container_path)
            return None
        except Exception as e:
            logger.error("Failed to create base workspace: %s", e)
            if container_path:
                self._cleanup_container(container_path)
            return None
    
    def _create_holdout_files(self, holdout_dir: Path, problem_id: str, holdout_data: dict) -> None:
        """Create holdout files in the specified directory."""
        try:
            # Create holdout test cases file
            holdout_test_file = holdout_dir / f"{problem_id}_test_cases_holdout.json"
            holdout_test_file.write_text(json.dumps(holdout_data['test_cases'], indent=2))
            
            # Create final evaluation script
            eval_script_file = holdout_dir / f"{problem_id}_final_evaluation.py"
            eval_script_file.write_text(holdout_data['evaluation_script'])
            eval_script_file.chmod(0o755)  # Make executable
            
            logger.debug(f"Created holdout files for {problem_id} in {holdout_dir}")
            
        except Exception as e:
            logger.error(f"Failed to create holdout files for {problem_id}: {e}")
    
    def duplicate_workspace(self, base_workspace: Path, agent_name: str) -> Optional[Path]:
        """
        Duplicate a base workspace for a specific agent.

        Args:
            base_workspace: Path to the base workspace to duplicate
            agent_name: Name of the agent that will use this workspace

        Returns:
            Path to the duplicated workspace or None if failed
        """
        container_path: Optional[Path] = None
        try:
            if not base_workspace.exists():
                logger.error("Base workspace does not exist: %s", base_workspace)
                return None

            base_info = self.active_workspaces.get(base_workspace)
            if not base_info:
                logger.error("Base workspace info not found: %s", base_workspace)
                return None

            problem_id = base_info.get("problem_id")

            container_path = self._allocate_container_path()
            container_path.mkdir(parents=True, exist_ok=False)
            agent_workspace = container_path / "workspace"

            logger.debug("Duplicating workspace from %s to %s", base_workspace, agent_workspace)
            shutil.copytree(base_workspace, agent_workspace)

            # Copy holdout files from base workspace container to agent container
            # This ensures they're available when saving results later
            base_container_str = base_info.get("container_path")
            if problem_id and base_container_str:
                base_container = Path(base_container_str)
                holdout_files = [
                    f"{problem_id}_test_cases_holdout.json",
                    f"{problem_id}_final_evaluation.py",
                ]
                for filename in holdout_files:
                    src_file = base_container / filename
                    if src_file.exists():
                        dst_file = container_path / filename
                        shutil.copy2(src_file, dst_file)
                        logger.debug(f"Copied holdout file to agent container: {filename}")

            sandbox_profile = self._setup_sandbox_profile(agent_workspace)
            created_at = datetime.now()
            label = f"{agent_name}_{problem_id}_{created_at.strftime('%Y%m%d_%H%M%S')}"

            metadata = {
                "problem_id": problem_id,
                "agent_name": agent_name,
                "created_at": created_at,
                "status": "active",
                "workspace_type": "agent",
                "base_workspace": str(base_workspace),
                "container_path": str(container_path),
                "container_slug": container_path.name,
                "workspace_slug": agent_workspace.name,
                "sandbox_profile": str(sandbox_profile) if sandbox_profile else None,
                "display_name": label,
            }

            self.active_workspaces[agent_workspace] = metadata
            logger.debug("Created agent workspace %s (container %s)", agent_workspace, container_path)
            return agent_workspace

        except FileExistsError:
            logger.error("Agent workspace container already exists for %s", agent_name)
            if container_path:
                self._cleanup_container(container_path)
            return None
        except Exception as e:
            logger.error("Failed to duplicate workspace: %s", e)
            if container_path:
                self._cleanup_container(container_path)
            return None
    
    def get_workspace_info(self, workspace_path: Path) -> Optional[Dict[str, Any]]:
        """Get information about a workspace."""
        return self.active_workspaces.get(workspace_path)

    def get_container_path(self, workspace_path: Path) -> Optional[Path]:
        """Return the container directory for a workspace, if known."""
        metadata = self.get_workspace_info(workspace_path)
        if not metadata:
            return None
        container_str = metadata.get("container_path")
        return Path(container_str) if container_str else workspace_path.parent

    def find_base_workspace(self, problem_id: str) -> Optional[Path]:
        """Locate an active base workspace for the given problem."""
        for path, info in self.active_workspaces.items():
            if info.get("workspace_type") == "base" and info.get("problem_id") == problem_id:
                return path
        return None

    def list_active_workspaces(self) -> List[Dict[str, Any]]:
        """List all active workspaces."""
        workspaces = []
        for path, info in self.active_workspaces.items():
            workspace_info = info.copy()
            workspace_info["path"] = str(path)
            workspaces.append(workspace_info)
        return workspaces
    
    def cleanup_workspace(self, workspace_path: Path) -> bool:
        """
        Clean up a specific workspace.
        
        Args:
            workspace_path: Path to the workspace to clean up
            
        Returns:
            True if successful, False otherwise
        """
        try:
            metadata = self.active_workspaces.get(workspace_path, {})
            sandbox_profile = metadata.get("sandbox_profile")
            container_str = metadata.get("container_path")
            container_path = Path(container_str) if container_str else workspace_path.parent

            if sandbox_profile:
                cleanup_sandbox_profile(Path(sandbox_profile))

            if workspace_path.exists():
                shutil.rmtree(workspace_path)
                logger.debug("Cleaned up workspace: %s", workspace_path)

            # Attempt to remove the container directory to drop holdout files as well
            if container_path and container_path.exists() and container_path != self.base_dir:
                try:
                    shutil.rmtree(container_path)
                    logger.debug("Removed workspace container: %s", container_path)
                except Exception as exc:
                    logger.warning("Failed to remove workspace container %s: %s", container_path, exc)

            if workspace_path in self.active_workspaces:
                del self.active_workspaces[workspace_path]

            return True

        except Exception as e:
            logger.error(f"Failed to cleanup workspace {workspace_path}: {e}")
            return False
    
    def cleanup_agent_workspaces(self, problem_id: str = None) -> None:
        """
        Clean up agent workspaces, optionally for a specific problem.
        
        Args:
            problem_id: If specified, only clean up agent workspaces for this problem
        """
        if not self.cleanup:
            logger.debug("Cleanup disabled, keeping all workspaces")
            return
        
        workspaces_to_clean = []
        for workspace_path, info in self.active_workspaces.items():
            # Only clean up agent workspaces (not base workspaces)
            if info.get("workspace_type") == "agent":
                if problem_id is None or info.get("problem_id") == problem_id:
                    workspaces_to_clean.append(workspace_path)
        
        for workspace_path in workspaces_to_clean:
            self.cleanup_workspace(workspace_path)
    
    def cleanup_base_workspaces(self) -> None:
        """Clean up all base workspaces."""
        if not self.cleanup:
            logger.debug("Cleanup disabled, keeping all workspaces")
            return
        
        workspaces_to_clean = []
        for workspace_path, info in self.active_workspaces.items():
            # Only clean up base workspaces
            if info.get("workspace_type") == "base":
                workspaces_to_clean.append(workspace_path)
        
        for workspace_path in workspaces_to_clean:
            self.cleanup_workspace(workspace_path)
    
    def cleanup_all(self) -> None:
        """Clean up all active workspaces."""
        if not self.cleanup:
            logger.debug("Cleanup disabled, keeping all workspaces")
            return
            
        workspaces_to_clean = list(self.active_workspaces.keys())
        for workspace_path in workspaces_to_clean:
            self.cleanup_workspace(workspace_path)
    
    def save_workspace_results(self, workspace_path: Path, results_dir: Path) -> bool:
        """
        Save important files from workspace before cleanup.

        Args:
            workspace_path: Path to the workspace
            results_dir: Directory to save results to

        Returns:
            True if successful, False otherwise
        """
        try:
            # Create results directory
            results_dir.mkdir(parents=True, exist_ok=True)

            metadata = self.get_workspace_info(workspace_path) or {}

            # Files to save from workspace
            files_to_save = [
                "solution.py",              # Agent's solution
                "test_results.json",        # Test execution results
                "test.log",                 # Test execution log
                "agent.log",                # Agent output log
                "problem.md",               # Problem description
                "test.py",                  # Test script
                "test_cases.json",          # Visible test cases
                "evaluation_results.json",  # Holdout evaluation results (if exists)
            ]

            saved_files = []
            for filename in files_to_save:
                src_file = workspace_path / filename
                if src_file.exists():
                    dst_file = results_dir / filename
                    shutil.copy2(src_file, dst_file)
                    saved_files.append(filename)

            # Copy holdout files from the container directory
            # These are needed for manual re-evaluation and analysis
            problem_id = metadata.get("problem_id")
            container_str = metadata.get("container_path")

            if problem_id and container_str:
                container_path = Path(container_str)

                # Map container file names to simplified result names
                holdout_files = {
                    f"{problem_id}_test_cases_holdout.json": "test_cases_holdout.json",
                    f"{problem_id}_final_evaluation.py": "final_evaluation.py",
                }

                for source_name, dest_name in holdout_files.items():
                    src_file = container_path / source_name
                    if src_file.exists():
                        dst_file = results_dir / dest_name
                        shutil.copy2(src_file, dst_file)
                        saved_files.append(dest_name)
                        logger.debug(f"Saved holdout file: {dest_name}")

            # Save workspace metadata with list of saved files
            if metadata:
                metadata["saved_files"] = saved_files
                metadata_file = results_dir / "workspace_metadata.json"
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2, default=str)

            logger.info(f"Saved workspace results to {results_dir} ({len(saved_files)} files)")
            return True

        except Exception as e:
            logger.error(f"Failed to save workspace results: {e}")
            return False
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup all workspaces."""
        self.cleanup_all()


class WorkspaceMonitor:
    """Monitor workspace for changes and test results."""
    
    def __init__(self, workspace_path: Path):
        self.workspace_path = workspace_path
        self.solution_file = workspace_path / "solution.py"
        self.test_log = workspace_path / "test.log"
        self.last_solution_mtime = None
        self.last_test_mtime = None
    
    def has_solution_changed(self) -> bool:
        """Check if solution.py has been modified."""
        if not self.solution_file.exists():
            return False
            
        current_mtime = self.solution_file.stat().st_mtime
        if self.last_solution_mtime is None:
            # Initialize baseline and detect if the initial content already looks like a solution
            self.last_solution_mtime = current_mtime
            return self.has_real_solution()
        
        if current_mtime > self.last_solution_mtime:
            self.last_solution_mtime = current_mtime
            # Consider any file modification as a change; tests will determine correctness
            return True
            
        return False
    
    def has_real_solution(self) -> bool:
        """Check if the solution contains actual implementation, not just TODO."""
        content = self.get_solution_content()
        if not content:
            return False
            
        # First, check for obvious TODO markers - if found, it's definitely a template
        todo_markers = ['TODO', 'result = "TODO"', 'print("TODO")', 'return "TODO"', 
                       'TODO: Implement', 'TODO: Replace', '# TODO:']
        if any(marker in content for marker in todo_markers):
            return False  # Templates always have TODO markers
            
        # Check if solution.py is essentially unchanged from template
        # Template characteristics:
        # 1. Has the solve() function with only comments
        # 2. No actual algorithm implementation
        
        # Remove comments and docstrings for analysis
        import re
        # Remove docstrings
        content_no_docstrings = re.sub(r'"""[\s\S]*?"""', '', content)
        content_no_docstrings = re.sub(r"'''[\s\S]*?'''", '', content_no_docstrings)
        # Remove comments
        content_no_comments = re.sub(r'#.*', '', content_no_docstrings)
        
        # Split into lines and filter empty ones
        lines = [line.strip() for line in content_no_comments.split('\n') if line.strip()]
        
        # Count meaningful code lines (not just imports, function definitions, or structural code)
        meaningful_lines = []
        for line in lines:
            # Skip imports, function/class definitions, and structural code
            if (line.startswith('import ') or 
                line.startswith('from ') or
                line.startswith('def ') or
                line.startswith('class ') or
                line.startswith('@') or  # decorators
                line == 'pass' or
                line == 'solve()' or
                line == 'if __name__ == "__main__":' or
                line.startswith('"""') or
                line.startswith("'''")):
                continue
            meaningful_lines.append(line)
        
        # Require at least 1 meaningful line of actual code
        # This is intentionally permissive to avoid missing short but correct solutions
        return len(meaningful_lines) >= 1
    
    def has_test_results(self) -> bool:
        """Check if new test results are available."""
        if not self.test_log.exists():
            return False
            
        current_mtime = self.test_log.stat().st_mtime
        if self.last_test_mtime is None:
            self.last_test_mtime = current_mtime
            return True
        
        if current_mtime > self.last_test_mtime:
            self.last_test_mtime = current_mtime
            return True
            
        return False
    
    def get_solution_content(self) -> Optional[str]:
        """Get the current solution content."""
        if self.solution_file.exists():
            with open(self.solution_file, 'r') as f:
                return f.read()
        return None
    
    def run_tests(self) -> Dict[str, Any]:
        """Run tests and return results."""
        try:
            sandbox_profile = self.workspace_path.parent / f".sandbox_{self.workspace_path.name}.sb"
            use_sandbox = sandbox_profile.exists() and sandbox_supported()

            if use_sandbox:
                logger.debug("Running tests with sandbox profile: %s", sandbox_profile)
                cmd = [
                    "sandbox-exec",
                    "-f",
                    str(sandbox_profile),
                    "python",
                    "test.py",
                ]
            else:
                logger.debug("Running tests without sandbox; profile missing or unsupported")
                cmd = ["python", "test.py"]

            result = subprocess.run(
                cmd,
                cwd=str(self.workspace_path),
                capture_output=True,
                text=True,
                timeout=60,  # 60 second timeout
                env=build_subprocess_env()
            )
            
            # Parse results
            test_results = {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }
            
            # Try to parse JSON output if available
            if "test_results.json" in os.listdir(self.workspace_path):
                results_file = self.workspace_path / "test_results.json"
                with open(results_file, 'r') as f:
                    test_results["detailed_results"] = json.load(f)
            
            return test_results
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Test execution timed out",
                "timeout": True
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

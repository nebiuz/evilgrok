"""
Base abstract class for agent runners.
"""

import time
from abc import ABC, abstractmethod
from typing import Dict, Any
from pathlib import Path


class AgentRunner(ABC):
    """Abstract base class for agent runners."""
    
    def __init__(self, workspace_path: str, config: Dict[str, Any]):
        self.workspace_path = Path(workspace_path)
        self.config = config
        self.process = None
        self.start_time = None
        
    @abstractmethod
    def launch(self) -> bool:
        """Launch the agent with the given workspace."""
        pass
        
    @abstractmethod
    def is_running(self) -> bool:
        """Check if the agent is still running."""
        pass
        
    @abstractmethod
    def terminate(self) -> None:
        """Terminate the agent process."""
        pass
        
    def get_runtime(self) -> float:
        """Get the runtime in seconds."""
        if self.start_time:
            return time.time() - self.start_time
        return 0.0
        
    def is_ready(self) -> bool:
        """Check if the agent is ready to process (default: always ready)."""
        return True  # Default implementation - agents are ready immediately
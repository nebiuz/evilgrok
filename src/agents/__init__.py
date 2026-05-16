"""
Agent runners package.

This package contains all agent runner implementations organized into separate modules.
"""

import logging
from typing import Dict, Optional, Any

from .base import AgentRunner

try:
    from .grok_responses import GrokResponsesRunner
    GROK_RESPONSES_AVAILABLE = True
except ImportError as e:
    GROK_RESPONSES_AVAILABLE = False
    logging.getLogger(__name__).warning(f"Failed to import GrokResponsesRunner: {e}")

try:
    from grok_responses import GrokResponsesRunner as GrokResponsesRunnerDirect
    if not GROK_RESPONSES_AVAILABLE:
        GrokResponsesRunner = GrokResponsesRunnerDirect
        GROK_RESPONSES_AVAILABLE = True
except ImportError:
    pass

AGENT_RUNNERS = {}

if GROK_RESPONSES_AVAILABLE:
    AGENT_RUNNERS["grok"] = GrokResponsesRunner

__all__ = [
    'AgentRunner',
    'AGENT_RUNNERS',
    'create_agent_runner',
]

if GROK_RESPONSES_AVAILABLE:
    __all__.append('GrokResponsesRunner')

logger = logging.getLogger(__name__)


def create_agent_runner(agent_type: str, workspace_path: str, config: Dict[str, Any]) -> Optional[AgentRunner]:
    """Factory function to create an agent runner."""
    runner_class = AGENT_RUNNERS.get(agent_type.lower())
    if not runner_class:
        logger.error(f"Unknown agent type: {agent_type}")
        logger.error(f"Available agents: {list(AGENT_RUNNERS.keys())}")
        return None
    
    return runner_class(workspace_path, config)

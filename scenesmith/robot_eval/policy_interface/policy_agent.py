"""Unified policy interface agent.

Single agent that handles the entire task → object bindings pipeline:
1. Parses task to understand goals and preconditions
2. Finds objects in scene that match categories
3. Verifies preconditions using state/vision tools
4. Returns ALL valid (target, reference) pairs ranked by confidence
"""

import logging

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from omegaconf import DictConfig
from pydantic import BaseModel, Field

from scenesmith.prompts import RobotEvalPrompts, prompt_registry
from scenesmith.robot_eval.dmd_scene import DMDScene
from scenesmith.robot_eval.tools import create_state_tools, create_vision_tools

if TYPE_CHECKING:
    from scenesmith.agent_utils.blender.server_manager import BlenderServer

console_logger = logging.getLogger(__name__)


class ObjectBinding(BaseModel):
    """A valid (target, reference) binding for a task."""

    target_id: str = Field(description="Object ID of the target (what to move)")
    reference_id: str = Field(description="Object ID of the reference (where to place)")
    rank: int = Field(description="Agent's preference ranking (1 = best)")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score (0.0-1.0)")
    reasoning: str = Field(description="Why this binding satisfies the task")


class PolicyInterfaceOutput(BaseModel):
    """Structured output from the policy interface agent."""

    task_description: str = Field(description="Original task being resolved")

    # Parsed task components.
    goal_predicate: str = Field(
        description="Goal predicate type: 'on', 'inside', 'near'"
    )
    target_category: str = Field(description="Category of object to move")
    reference_category: str = Field(description="Category of placement location")
    target_precondition: str | None = Field(
        default=None,
        description="Extracted precondition for target (e.g., 'on floor')",
    )
    reference_precondition: str | None = Field(
        default=None,
        description="Extracted precondition for reference (e.g., 'on dining table')",
    )

    # Valid bindings.
    valid_bindings: list[ObjectBinding] = Field(
        description="Ranked list of valid (target, reference) bindings"
    )
    overall_success: bool = Field(
        description="True if at least one valid binding was found"
    )
    notes: list[str] = Field(
        default_factory=list, description="Additional notes from agent"
    )


@dataclass
class PolicyInterfaceAgent:
    """Unified agent for robot task planning.

    Takes a task description and scene, returns ranked object bindings.
    Combines the functionality of PredicateExtractor, CategoryMatcher,
    and CandidateSelectionAgent into a single coherent reasoning pass.

    Usage:
        scene = load_scene_for_validation(scene_state_path, dmd_path)
        scene.finalize()

        agent = PolicyInterfaceAgent(scene=scene, cfg=cfg)
        result = await agent.resolve("Pick a cup from the floor and put it in the sink")
    """

    scene: DMDScene
    """Scene with finalized Drake plant and scene_state metadata."""

    cfg: DictConfig
    """Configuration with model settings."""

    blender_server: "BlenderServer | None" = None
    """Optional Blender server for vision tools. If None, only state tools available."""

    _agent: Any | None = field(default=None, init=False)
    """Lazily initialized agent."""

    def _create_agent(self) -> Any:
        """Create the policy interface agent with tools and prompt.

        Returns:
            None. Now handled by Claude Code subagents via MCP.
        """
        # Now handled by Claude Code subagents via MCP
        return None

    @property
    def agent(self) -> Any:
        """Get or create the policy interface agent."""
        if self._agent is None:
            self._agent = self._create_agent()
        return self._agent

    async def resolve(
        self, task_description: str, max_turns: int = 1000
    ) -> PolicyInterfaceOutput:
        """Resolve a task to valid object bindings.

        Args:
            task_description: Natural language task (e.g., "Pick a cup from the floor").
            max_turns: Maximum agent turns before forcing output.

        Returns:
            PolicyInterfaceOutput with ranked valid bindings.
        """
        console_logger.info(f"Resolving task: {task_description}")

        # Now handled by Claude Code subagents via MCP
        raise NotImplementedError("Use Claude Code subagents via MCP server")

"""Stage orchestration for the SceneSmith pipeline.

Implements the Planner logic as Python code, invoking Claude Code subagents
for Designer and Critic roles at each stage.
"""

import json
import logging
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

console_logger = logging.getLogger(__name__)


@dataclass
class CritiqueScores:
    """Parsed critique scores from a critic subagent."""

    scores: dict[str, float]
    total_score: float
    feedback: str
    critical_issues: list[str]
    suggestions: list[str]


# Pipeline stages in execution order
PIPELINE_STAGES = [
    "floor_plan",
    "furniture",
    "wall",
    "ceiling",
    "manipuland",
]

# Checkpoint dependencies
STAGE_CHECKPOINTS = {
    "floor_plan": None,
    "furniture": None,
    "wall": "scene_after_furniture",
    "ceiling": "scene_after_wall_objects",
    "manipuland": "scene_after_ceiling_objects",
}

# Checkpoint names saved after each stage
STAGE_SAVE_NAMES = {
    "floor_plan": "scene_after_floor_plan",
    "furniture": "scene_after_furniture",
    "wall": "scene_after_wall_objects",
    "ceiling": "scene_after_ceiling_objects",
    "manipuland": "scene_after_manipulands",
}


def parse_critique_scores(critic_output: str) -> CritiqueScores | None:
    """Parse JSON critique scores from critic subagent output.

    Attempts to extract JSON from the output, falling back to regex.

    Args:
        critic_output: Raw text output from critic subagent.

    Returns:
        Parsed CritiqueScores or None if parsing fails.
    """
    # Try to find JSON block in the output
    json_match = re.search(r"\{[^{}]*\"scores\"[^{}]*\{[^}]+\}[^}]*\}", critic_output, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            return CritiqueScores(
                scores=data.get("scores", {}),
                total_score=data.get("total_score", 0.0),
                feedback=data.get("feedback", ""),
                critical_issues=data.get("critical_issues", []),
                suggestions=data.get("suggestions", []),
            )
        except json.JSONDecodeError:
            pass

    # Fallback: try to extract total_score with regex
    score_match = re.search(r"total_score[\"']?\s*:\s*(\d+\.?\d*)", critic_output)
    if score_match:
        return CritiqueScores(
            scores={},
            total_score=float(score_match.group(1)),
            feedback=critic_output,
            critical_issues=[],
            suggestions=[],
        )

    console_logger.warning("Could not parse critique scores from output")
    return None


def invoke_subagent(
    agent_name: str,
    message: str,
    mcp_server: str = "scenesmith",
) -> str:
    """Invoke a Claude Code subagent and return its output.

    Args:
        agent_name: Name of the subagent (e.g., "furniture-designer").
        message: Message/instruction for the subagent.
        mcp_server: MCP server name to connect to.

    Returns:
        Subagent's text output.
    """
    console_logger.info(f"Invoking subagent: {agent_name}")

    try:
        cmd = [
            "claude",
            "--agent", agent_name,
            "-p", message,
            "--dangerously-skip-permissions",
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout per subagent
            cwd=str(Path(__file__).parent.parent),  # Project root
        )
        if result.returncode != 0:
            console_logger.error(
                f"Subagent {agent_name} failed: {result.stderr}"
            )
            return f"Error: Subagent failed with code {result.returncode}"
        return result.stdout
    except subprocess.TimeoutExpired:
        console_logger.error(f"Subagent {agent_name} timed out")
        return "Error: Subagent timed out after 600 seconds"
    except FileNotFoundError:
        console_logger.error("claude CLI not found")
        return "Error: claude CLI not found. Install Claude Code."


class StageOrchestrator:
    """Orchestrates the design-critique-iterate loop for each pipeline stage.

    Implements the Planner agent logic as Python code.
    """

    def __init__(
        self,
        max_critique_rounds: int = 3,
        early_finish_threshold: float = 7.5,
        reset_threshold: float = -1.5,
        mcp_server: str = "scenesmith",
    ):
        """Initialize the orchestrator.

        Args:
            max_critique_rounds: Maximum design-critique iterations.
            early_finish_threshold: Score above which to stop early.
            reset_threshold: Score drop that triggers checkpoint reset.
            mcp_server: MCP server name.
        """
        self.max_critique_rounds = max_critique_rounds
        self.early_finish_threshold = early_finish_threshold
        self.reset_threshold = reset_threshold
        self.mcp_server = mcp_server

    def run_stage(
        self,
        stage: str,
        scene_description: str,
        stage_instruction: str = "",
    ) -> CritiqueScores | None:
        """Run a single pipeline stage through the design-critique loop.

        Args:
            stage: Stage name (e.g., "furniture").
            scene_description: Natural language scene description.
            stage_instruction: Additional stage-specific instructions.

        Returns:
            Final critique scores, or None if no critique was performed.
        """
        console_logger.info(f"=== Starting stage: {stage} ===")

        designer_agent = f"{stage.replace('_', '-')}-designer"
        critic_agent = f"{stage.replace('_', '-')}-critic"

        # Build initial design instruction with stage init
        initial_prompt = (
            f"IMPORTANT: Before doing anything else, call the workflow__init_stage "
            f"tool with stage='{stage}' to initialize the stage tools.\n\n"
            f"Scene description: {scene_description}\n\n"
            f"Stage: {stage}\n\n"
            f"Create the initial {stage.replace('_', ' ')} design for this scene.\n"
        )
        if stage_instruction:
            initial_prompt += f"\nAdditional instructions: {stage_instruction}\n"

        # 1. Initial design
        console_logger.info(f"Requesting initial design from {designer_agent}")
        designer_result = invoke_subagent(
            designer_agent, initial_prompt, self.mcp_server
        )
        console_logger.info(f"Designer completed initial design")

        # 2. Critique loop
        previous_score = 0.0
        final_scores = None

        for round_num in range(self.max_critique_rounds):
            console_logger.info(
                f"Critique round {round_num + 1}/{self.max_critique_rounds}"
            )

            # Request critique
            critique_prompt = (
                f"Scene description: {scene_description}\n\n"
                f"Evaluate the current {stage.replace('_', ' ')} design.\n"
                f"Provide scores and actionable feedback.\n"
            )
            critic_result = invoke_subagent(
                critic_agent, critique_prompt, self.mcp_server
            )

            # Parse scores
            scores = parse_critique_scores(critic_result)
            if scores is None:
                console_logger.warning(
                    "Could not parse critique scores, continuing..."
                )
                continue

            final_scores = scores
            console_logger.info(
                f"Critique scores: total={scores.total_score:.1f}"
            )

            # Check if good enough to stop early
            if scores.total_score >= self.early_finish_threshold:
                console_logger.info(
                    f"Score {scores.total_score:.1f} >= {self.early_finish_threshold}, "
                    f"finishing stage early"
                )
                break

            # Check for score regression (would trigger reset)
            score_delta = scores.total_score - previous_score
            if round_num > 0 and score_delta < self.reset_threshold:
                console_logger.warning(
                    f"Score regressed by {score_delta:.1f}, "
                    f"consider resetting checkpoint"
                )
                # In full implementation, would call workflow__restore_checkpoint

            previous_score = scores.total_score

            # If not last round, request design changes based on critique
            if round_num < self.max_critique_rounds - 1:
                revision_prompt = (
                    f"Scene description: {scene_description}\n\n"
                    f"Revise the {stage.replace('_', ' ')} based on this critique:\n\n"
                    f"{scores.feedback}\n\n"
                    f"Critical issues to fix:\n"
                )
                for issue in scores.critical_issues:
                    revision_prompt += f"- {issue}\n"

                console_logger.info("Requesting design revision")
                designer_result = invoke_subagent(
                    designer_agent, revision_prompt, self.mcp_server
                )

        console_logger.info(f"=== Stage {stage} complete ===")
        return final_scores

    def run_pipeline(
        self,
        scene_description: str,
        start_stage: str | None = None,
        end_stage: str | None = None,
    ) -> dict[str, CritiqueScores | None]:
        """Run the full pipeline or a subset of stages.

        Args:
            scene_description: Natural language scene description.
            start_stage: Stage to start from (default: first stage).
            end_stage: Stage to stop at (default: last stage).

        Returns:
            Dictionary mapping stage names to their final critique scores.
        """
        stages = PIPELINE_STAGES.copy()

        # Filter to requested range
        if start_stage:
            start_idx = stages.index(start_stage)
            stages = stages[start_idx:]
        if end_stage:
            end_idx = stages.index(end_stage) + 1
            stages = stages[:end_idx]

        results: dict[str, CritiqueScores | None] = {}

        for stage in stages:
            scores = self.run_stage(stage, scene_description)
            results[stage] = scores

            # Log stage summary
            if scores:
                console_logger.info(
                    f"Stage {stage} final score: {scores.total_score:.1f}"
                )
            else:
                console_logger.info(f"Stage {stage} completed (no scores)")

        return results

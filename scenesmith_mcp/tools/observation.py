"""Observation and physics MCP tools.

Wraps VisionTools _impl methods for scene rendering and physics checking.
Returns base64-encoded images as text descriptions so Claude can process the scene.
"""

import base64
import logging

from mcp.server.fastmcp import FastMCP

from scenesmith_mcp.state import ServerState

console_logger = logging.getLogger(__name__)


def register_observation_tools(server: FastMCP, state: ServerState) -> None:
    """Register observation and physics tools."""

    @server.tool()
    async def scene__observe(
        stage: str = "furniture",
    ) -> str:
        """Render the current scene from multiple viewpoints and return images.

        Call this to see what the scene looks like. Returns multiple perspective
        images showing all objects currently placed.

        Args:
            stage: Which stage's rendering mode to use.
                Options: "floor_plan", "furniture", "wall", "ceiling", "manipuland".

        Returns:
            Multiple rendered images of the scene plus a text summary.
        """
        if state.scene is None:
            return "Error: No scene initialized."

        rendering_mode_map = {
            "floor_plan": "floor_plan",
            "furniture": "furniture",
            "wall": "wall",
            "ceiling": "ceiling_perspective",
            "manipuland": "manipuland",
        }
        rendering_mode = rendering_mode_map.get(stage, "furniture")

        # Get room bounds for stable grid markers
        room_bounds = None
        room_geom = state.scene.room_geometry
        if room_geom and room_geom.length > 0 and room_geom.width > 0:
            half_l = room_geom.length / 2
            half_w = room_geom.width / 2
            room_bounds = (-half_l, -half_w, half_l, half_w)

        # Render scene
        images_dir = state.rendering_manager.render_scene(
            state.scene,
            blender_server=state.blender_server,
            rendering_mode=rendering_mode,
            room_bounds=room_bounds,
        )

        if not images_dir or not images_dir.exists():
            return "Unable to render scene - no images generated."

        # Collect image paths
        image_paths = sorted(images_dir.glob("*.png"))
        num_images = len(image_paths)

        result_text = (
            f"Scene rendered from {num_images} viewpoints. "
            "Visual feedback is now available for analysis. "
            f"Images saved to: {images_dir}"
        )
        console_logger.info(f"Rendered {num_images} images to {images_dir}")
        return result_text

    @server.tool()
    async def scene__check_physics(
        stage: str = "furniture",
    ) -> str:
        """Check for physics violations (collisions) in the current scene.

        Detects collisions that might not be visible in renders but would
        make the scene physically invalid.

        Args:
            stage: Pipeline stage for agent-type filtering.
                Options: "floor_plan", "furniture", "wall_mounted", "ceiling_mounted", "manipuland".

        Returns:
            Description of any collisions detected, or confirmation of no issues.
        """
        from scenesmith.agent_utils.physics_tools import check_physics_violations
        from scenesmith.agent_utils.room import AgentType

        agent_type_map = {
            "floor_plan": AgentType.FLOOR_PLAN,
            "furniture": AgentType.FURNITURE,
            "wall_mounted": AgentType.WALL_MOUNTED,
            "wall": AgentType.WALL_MOUNTED,
            "ceiling_mounted": AgentType.CEILING_MOUNTED,
            "ceiling": AgentType.CEILING_MOUNTED,
            "manipuland": AgentType.MANIPULAND,
        }
        agent_type = agent_type_map.get(stage, AgentType.FURNITURE)

        result = check_physics_violations(
            scene=state.scene,
            cfg=state.cfg,
            agent_type=agent_type,
        )
        return result

    @server.tool()
    async def scene__check_reachability() -> str:
        """Check if all areas of the room are reachable (not blocked by furniture).

        Returns:
            JSON with reachability analysis: is_fully_reachable, num_disconnected_regions,
            reachability_ratio, blocking_furniture_ids.
        """
        from scenesmith.agent_utils.reachability import (
            compute_reachability,
            format_reachability_result,
        )

        robot_width = state.cfg.reachability.robot_width
        result = compute_reachability(scene=state.scene, robot_width=robot_width)
        return format_reachability_result(result)

    console_logger.info("Registered observation tools")

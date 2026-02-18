"""Workflow and checkpoint MCP tools.

Manages pipeline state, checkpoints, stage initialization, and placement
style configuration.
"""

import json
import logging

from mcp.server.fastmcp import FastMCP

from scenesmith_mcp.state import ServerState

console_logger = logging.getLogger(__name__)


def _init_floor_plan_stage(state: ServerState) -> str:
    """Initialize FloorPlanTools for the floor plan stage."""
    from scenesmith.agent_utils.house import HouseLayout
    from scenesmith.floor_plan_agents.tools.floor_plan_tools import FloorPlanTools

    cfg = state.cfg
    fp_cfg = cfg.get("floor_plan_agent", {})

    state.house_layout = HouseLayout()
    state.floor_plan_tools = FloorPlanTools(
        layout=state.house_layout,
        mode="house",
        materials_config=None,
        min_opening_separation=fp_cfg.get("room_placement", {}).get(
            "min_opening_separation", 0.5
        ),
        placement_timeout_seconds=fp_cfg.get("room_placement", {}).get(
            "timeout_seconds", 5.0
        ),
        wall_height_min=fp_cfg.get("wall_height", {}).get("min", 2.0),
        wall_height_max=fp_cfg.get("wall_height", {}).get("max", 4.5),
        room_dim_min=fp_cfg.get("min_floor_plan_dim_m", 1.5),
        room_dim_max=fp_cfg.get("max_floor_plan_dim_m", 20.0),
    )
    state.current_stage = "floor_plan"
    return "Floor plan stage initialized"


def _init_furniture_stage(state: ServerState) -> str:
    """Initialize FurnitureTools and SceneTools for the furniture stage."""
    from scenesmith.furniture_agents.tools.furniture_tools import FurnitureTools
    from scenesmith.furniture_agents.tools.scene_tools import SceneTools

    if state.scene is None:
        return "Error: Scene not initialized. Run floor plan stage first."

    cfg = state.cfg
    furn_cfg = cfg.get("furniture_agent", {})

    state.furniture_tools = FurnitureTools(
        scene=state.scene,
        asset_manager=state.asset_manager,
        cfg=furn_cfg,
    )
    state.scene_tools = SceneTools(scene=state.scene, cfg=furn_cfg)
    state.current_stage = "furniture"
    return "Furniture stage initialized"


def _init_wall_stage(state: ServerState) -> str:
    """Initialize WallTools for the wall stage."""
    from scenesmith.wall_agents.tools.wall_tools import WallTools

    if state.scene is None:
        return "Error: Scene not initialized."

    cfg = state.cfg
    wall_cfg = cfg.get("wall_agent", {})

    # Compute wall surfaces from scene geometry
    wall_surfaces = []
    try:
        from scenesmith.agent_utils.wall_surface import compute_wall_surfaces

        wall_surfaces = compute_wall_surfaces(state.scene)
    except Exception as e:
        console_logger.warning(f"Wall surface computation failed: {e}")

    state.wall_tools = WallTools(
        scene=state.scene,
        wall_surfaces=wall_surfaces,
        asset_manager=state.asset_manager,
        cfg=wall_cfg,
    )
    state.current_stage = "wall"
    return f"Wall stage initialized with {len(wall_surfaces)} wall surfaces"


def _init_ceiling_stage(state: ServerState) -> str:
    """Initialize CeilingTools for the ceiling stage."""
    from scenesmith.ceiling_agents.tools.ceiling_tools import CeilingTools

    if state.scene is None:
        return "Error: Scene not initialized."

    cfg = state.cfg
    ceil_cfg = cfg.get("ceiling_agent", {})

    # Get room bounds and ceiling height from scene
    room_bounds = (0.0, 0.0, 5.0, 4.0)
    ceiling_height = 2.5
    if state.scene.room_geometry:
        rg = state.scene.room_geometry
        room_bounds = (0.0, 0.0, rg.length, rg.width)
        ceiling_height = getattr(rg, "wall_height", 2.5)

    state.ceiling_tools = CeilingTools(
        scene=state.scene,
        room_bounds=room_bounds,
        ceiling_height=ceiling_height,
        asset_manager=state.asset_manager,
        cfg=ceil_cfg,
    )
    state.current_stage = "ceiling"
    return f"Ceiling stage initialized (height={ceiling_height}m)"


def _init_manipuland_stage(state: ServerState) -> str:
    """Initialize ManipulandTools for the manipuland stage."""
    if state.scene is None:
        return "Error: Scene not initialized."

    state.current_stage = "manipuland"
    # ManipulandTools are created per-surface by the designer subagent
    return "Manipuland stage initialized"


def register_workflow_tools(server: FastMCP, state: ServerState) -> None:
    """Register workflow management tools."""

    @server.tool()
    async def workflow__init_stage(stage: str) -> str:
        """Initialize tool classes for a pipeline stage.

        Must be called before using stage-specific tools. Each stage
        initializes the appropriate tool class instances in server state.

        Args:
            stage: Stage name - one of "floor_plan", "furniture",
                "wall", "ceiling", "manipuland".

        Returns:
            Initialization status message.
        """
        stage_initializers = {
            "floor_plan": _init_floor_plan_stage,
            "furniture": _init_furniture_stage,
            "wall": _init_wall_stage,
            "ceiling": _init_ceiling_stage,
            "manipuland": _init_manipuland_stage,
        }

        if stage not in stage_initializers:
            return f"Unknown stage '{stage}'. Valid: {list(stage_initializers.keys())}"

        try:
            result = stage_initializers[stage](state)
            console_logger.info(f"Stage initialized: {stage}")
            return result
        except Exception as e:
            console_logger.error(f"Stage initialization failed: {e}")
            return f"Stage initialization failed: {e}"

    @server.tool()
    async def workflow__save_checkpoint(name: str) -> str:
        """Save the current scene state as a named checkpoint.

        Args:
            name: Checkpoint name (e.g., "scene_after_furniture").

        Returns:
            Confirmation of save.
        """
        state.save_checkpoint(name)
        return f"Checkpoint '{name}' saved successfully."

    @server.tool()
    async def workflow__restore_checkpoint(name: str) -> str:
        """Restore scene state from a named checkpoint.

        Args:
            name: Checkpoint name to restore.

        Returns:
            Success/failure result.
        """
        success = state.restore_checkpoint(name)
        if success:
            return f"Restored checkpoint '{name}' successfully."
        available = list(state.checkpoints.keys())
        return f"Checkpoint '{name}' not found. Available: {available}"

    @server.tool()
    async def workflow__list_checkpoints() -> str:
        """List all saved checkpoints.

        Returns:
            List of checkpoint names.
        """
        checkpoints = list(state.checkpoints.keys())
        return json.dumps({"checkpoints": checkpoints})

    @server.tool()
    async def workflow__get_pipeline_state() -> str:
        """Get current pipeline state including active stage and scene info.

        Returns:
            JSON with current stage, scene dimensions, object counts.
        """
        info = {
            "current_stage": state.current_stage,
            "checkpoints": list(state.checkpoints.keys()),
            "scene_initialized": state.scene is not None,
        }
        if state.scene:
            info["object_count"] = len(state.scene.objects)
            room_geom = state.scene.room_geometry
            if room_geom:
                info["room_dimensions"] = {
                    "length": room_geom.length,
                    "width": room_geom.width,
                    "height": getattr(room_geom, "wall_height", 2.7),
                }
        return json.dumps(info, indent=2)

    @server.tool()
    async def workflow__set_placement_style(
        style: str = "natural",
    ) -> str:
        """Set placement style for object positioning.

        Args:
            style: "natural" (slight random variations) or "perfect" (exact positioning).

        Returns:
            Confirmation.
        """
        from scenesmith.agent_utils.placement_noise import PlacementNoiseMode

        mode = (
            PlacementNoiseMode.NATURAL
            if style == "natural"
            else PlacementNoiseMode.PERFECT
        )

        # Apply to whichever tool instances exist
        for tools in [
            state.furniture_tools,
            state.wall_tools,
            state.ceiling_tools,
            state.manipuland_tools,
        ]:
            if tools and hasattr(tools, "set_noise_profile"):
                tools.set_noise_profile(mode)

        return f"Placement style set to '{style}'."

    console_logger.info("Registered workflow tools")

"""Furniture stage MCP tools.

Wraps FurnitureTools and SceneTools _impl methods for furniture placement,
movement, removal, snapping, and facing checks.
"""

import json
import logging

from mcp.server.fastmcp import FastMCP

from scenesmith_mcp.state import ServerState

console_logger = logging.getLogger(__name__)


def register_furniture_tools(server: FastMCP, state: ServerState) -> None:
    """Register all furniture-stage MCP tools."""

    @server.tool()
    async def furniture__generate_assets(
        object_descriptions: list[str],
        short_names: list[str],
        desired_dimensions: list[list[float]],
        style_context: str | None = None,
    ) -> str:
        """Create 3D furniture models from text descriptions with specified dimensions.

        Generate floor-standing furniture items only (tables, chairs, sofas, etc.).
        DO NOT generate manipulands (small objects), carpets, or wall decorations.

        Args:
            object_descriptions: List of furniture descriptions
                (e.g., "Modern oak dining table", "Leather office chair").
            short_names: Filesystem-safe names for each
                (e.g., "dining_table", "office_chair").
            desired_dimensions: [width, depth, height] in meters for each object.
                Width=X (left-right), Depth=Y (front-back), Height=Z (up-down).
            style_context: Optional style hint for visual consistency.

        Returns:
            IDs and details of created furniture models.
        """
        from scenesmith.agent_utils.asset_manager import AssetGenerationRequest
        from scenesmith.agent_utils.room import ObjectType

        request = AssetGenerationRequest(
            object_descriptions=object_descriptions,
            short_names=short_names,
            object_type=ObjectType.FURNITURE,
            desired_dimensions=desired_dimensions,
            style_context=style_context,
            scene_id=state.scene.scene_dir.name if state.scene else "default",
        )
        result = state.furniture_tools._generate_assets_impl(request)
        return result

    @server.tool()
    async def furniture__add_to_scene(
        asset_id: str,
        x: float,
        y: float,
        yaw: float = 0.0,
    ) -> str:
        """Place furniture in the room at a specific floor position.

        Furniture sits flat on the floor at z=0. You control x, y position
        and yaw rotation (around vertical axis).

        Args:
            asset_id: ID of the furniture to place (from generate_assets or list_available_assets).
            x: X position in meters.
            y: Y position in meters.
            yaw: Yaw rotation in degrees (positive = counterclockwise in top-down view).

        Returns:
            Unique placement ID and confirmation.
        """
        result = state.furniture_tools._add_furniture_to_scene_impl(
            asset_id=asset_id, x=x, y=y, z=0.0, roll=0.0, pitch=0.0, yaw=yaw
        )
        return result

    @server.tool()
    async def furniture__move(
        object_id: str,
        x: float,
        y: float,
        yaw: float = 0.0,
    ) -> str:
        """Move existing furniture to a new floor position.

        Args:
            object_id: ID of the furniture to move.
            x: New X position in meters.
            y: New Y position in meters.
            yaw: New yaw rotation in degrees.

        Returns:
            Confirmation of successful move.
        """
        result = state.furniture_tools._move_furniture_impl(
            object_id=object_id, x=x, y=y, z=0.0, roll=0.0, pitch=0.0, yaw=yaw
        )
        return result

    @server.tool()
    async def furniture__remove(object_id: str) -> str:
        """Remove furniture from the room.

        Args:
            object_id: ID of the furniture to remove.

        Returns:
            Confirmation of removal.
        """
        result = state.furniture_tools._remove_furniture_impl(object_id)
        return result

    @server.tool()
    async def furniture__list_available_assets() -> str:
        """List all furniture models available for placement with their dimensions.

        Returns:
            List of furniture with IDs, names, descriptions, and dimensions.
        """
        result = state.furniture_tools._list_available_assets_impl()
        return result

    @server.tool()
    async def furniture__rescale(
        object_id: str, scale_factor: float
    ) -> str:
        """Resize furniture by a uniform scale factor.

        WARNING: Rescales the underlying asset. All instances of the same
        asset will be affected.

        Args:
            object_id: ID of the furniture to rescale.
            scale_factor: Multiplier (e.g., 1.5 = 50% larger, 0.8 = 20% smaller).

        Returns:
            New dimensions and list of affected objects.
        """
        result = state.furniture_tools._rescale_furniture_impl(object_id, scale_factor)
        return result

    @server.tool()
    async def furniture__get_scene_state() -> str:
        """Get all furniture currently in the room with positions, rotations, and bounding boxes.

        Returns:
            JSON list of furniture with IDs, positions, rotations, dimensions, and world bounds.
        """
        result = state.scene_tools._get_current_scene_impl()
        return result

    @server.tool()
    async def furniture__check_facing(
        object_a_id: str, object_b_id: str, direction: str = "toward"
    ) -> str:
        """Check furniture orientation relationships.

        More reliable than visual assessment. Always verify orientation-critical
        furniture with this tool after placement or rotation.

        Args:
            object_a_id: ID of the object being checked.
            object_b_id: ID of the target object.
            direction: "toward" (chairs->tables) or "away" (furniture->walls).

        Returns:
            JSON with is_facing, optimal_rotation_degrees, current_rotation_degrees.
        """
        result = state.scene_tools._check_facing_impl(
            object_a_id=object_a_id,
            object_b_id=object_b_id,
            direction=direction,
        )
        return result

    @server.tool()
    async def furniture__snap_to_object(
        object_id: str, target_id: str, orientation: str = "none"
    ) -> str:
        """One-step orient-and-snap between furniture items.

        Automatically faces object toward/away from target, then snaps
        along facing direction until touching. Replaces 3-4 manual calls.

        Args:
            object_id: ID of object to move.
            target_id: ID of wall or object to snap to.
            orientation: "toward" (chairs->tables), "away" (furniture->walls), or "none".

        Returns:
            JSON with positions, distance moved, and rotation info.
        """
        result = state.scene_tools._snap_to_object_impl(
            object_id=object_id,
            target_id=target_id,
            orientation=orientation,
        )
        return result

    console_logger.info("Registered furniture tools")

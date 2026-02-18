"""Wall stage MCP tools.

Wraps WallTools _impl methods for wall-mounted object placement.
"""

import logging

from mcp.server.fastmcp import FastMCP

from scenesmith_mcp.state import ServerState

console_logger = logging.getLogger(__name__)


def register_wall_tools(server: FastMCP, state: ServerState) -> None:
    """Register all wall-stage MCP tools."""

    @server.tool()
    async def wall__generate_assets(
        object_descriptions: list[str],
        short_names: list[str],
        desired_dimensions: list[list[float]],
        style_context: str | None = None,
    ) -> str:
        """Create 3D wall-mounted object models from descriptions.

        Generate wall decorations, shelves, clocks, artwork, etc.

        Args:
            object_descriptions: List of descriptions.
            short_names: Filesystem-safe names.
            desired_dimensions: [width, depth, height] in meters for each.
            style_context: Optional style hint.

        Returns:
            IDs and details of created models.
        """
        from scenesmith.agent_utils.asset_manager import AssetGenerationRequest
        from scenesmith.agent_utils.room import ObjectType

        request = AssetGenerationRequest(
            object_descriptions=object_descriptions,
            short_names=short_names,
            object_type=ObjectType.WALL_MOUNTED,
            desired_dimensions=desired_dimensions,
            style_context=style_context,
            scene_id=state.scene.scene_dir.name if state.scene else "default",
        )
        result = state.wall_tools._generate_assets_impl(request)
        return result

    @server.tool()
    async def wall__list_surfaces() -> str:
        """List all available wall surfaces for object placement.

        Returns:
            Wall surfaces with IDs, dimensions, and available space.
        """
        result = state.wall_tools._list_wall_surfaces_impl()
        return result

    @server.tool()
    async def wall__place_object(
        asset_id: str,
        wall_surface_id: str,
        position_x: float,
        position_z: float,
        rotation_degrees: float = 0.0,
    ) -> str:
        """Place a wall-mounted object on a wall surface.

        Args:
            asset_id: ID of the asset to place.
            wall_surface_id: ID of the wall surface.
            position_x: Horizontal position on wall (meters).
            position_z: Vertical position on wall (meters from floor).
            rotation_degrees: Rotation on wall plane (degrees).

        Returns:
            Placement result with object ID.
        """
        result = state.wall_tools._place_wall_object_impl(
            asset_id=asset_id,
            wall_surface_id=wall_surface_id,
            position_x=position_x,
            position_z=position_z,
            rotation_degrees=rotation_degrees,
        )
        return result

    @server.tool()
    async def wall__move_object(
        object_id: str,
        wall_surface_id: str,
        position_x: float,
        position_z: float,
        rotation_degrees: float = 0.0,
    ) -> str:
        """Move a wall-mounted object to a new position.

        Args:
            object_id: ID of the object to move.
            wall_surface_id: ID of the target wall surface.
            position_x: New horizontal position.
            position_z: New vertical position.
            rotation_degrees: New rotation.

        Returns:
            Confirmation of move.
        """
        result = state.wall_tools._move_wall_object_impl(
            object_id=object_id,
            wall_surface_id=wall_surface_id,
            position_x=position_x,
            position_z=position_z,
            rotation_degrees=rotation_degrees,
        )
        return result

    @server.tool()
    async def wall__remove_object(object_id: str) -> str:
        """Remove a wall-mounted object.

        Args:
            object_id: ID of the object to remove.

        Returns:
            Confirmation of removal.
        """
        result = state.wall_tools._remove_wall_object_impl(object_id)
        return result

    @server.tool()
    async def wall__get_scene_state() -> str:
        """Get all wall-mounted objects in the scene.

        Returns:
            JSON list of wall objects with IDs, positions, and dimensions.
        """
        result = state.wall_tools._get_current_scene_impl()
        return result

    @server.tool()
    async def wall__list_available_assets() -> str:
        """List all wall-mounted asset models available for placement.

        Returns:
            List of assets with IDs, names, and dimensions.
        """
        result = state.wall_tools._list_available_assets_impl()
        return result

    @server.tool()
    async def wall__rescale(
        object_id: str, scale_factor: float
    ) -> str:
        """Resize a wall-mounted object.

        Args:
            object_id: ID of the object to rescale.
            scale_factor: Scale multiplier.

        Returns:
            New dimensions and affected objects.
        """
        result = state.wall_tools._rescale_wall_object_impl(object_id, scale_factor)
        return result

    console_logger.info("Registered wall tools")

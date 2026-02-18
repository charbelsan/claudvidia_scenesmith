"""Ceiling stage MCP tools.

Wraps CeilingTools _impl methods for ceiling-mounted object placement.
"""

import logging

from mcp.server.fastmcp import FastMCP

from scenesmith_mcp.state import ServerState

console_logger = logging.getLogger(__name__)


def register_ceiling_tools(server: FastMCP, state: ServerState) -> None:
    """Register all ceiling-stage MCP tools."""

    @server.tool()
    async def ceiling__generate_assets(
        object_descriptions: list[str],
        short_names: list[str],
        desired_dimensions: list[list[float]],
        style_context: str | None = None,
    ) -> str:
        """Create 3D ceiling-mounted object models from descriptions.

        Generate ceiling lights, fans, chandeliers, etc.

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
            object_type=ObjectType.CEILING_MOUNTED,
            desired_dimensions=desired_dimensions,
            style_context=style_context,
            scene_id=state.scene.scene_dir.name if state.scene else "default",
        )
        result = state.ceiling_tools._generate_assets_impl(request)
        return result

    @server.tool()
    async def ceiling__place_object(
        asset_id: str,
        x: float,
        y: float,
        yaw: float = 0.0,
    ) -> str:
        """Place a ceiling-mounted object.

        Args:
            asset_id: ID of the asset to place.
            x: X position in meters.
            y: Y position in meters.
            yaw: Yaw rotation in degrees.

        Returns:
            Placement result with object ID.
        """
        result = state.ceiling_tools._place_ceiling_object_impl(
            asset_id=asset_id, x=x, y=y, yaw=yaw
        )
        return result

    @server.tool()
    async def ceiling__move_object(
        object_id: str,
        x: float,
        y: float,
        yaw: float = 0.0,
    ) -> str:
        """Move a ceiling-mounted object.

        Args:
            object_id: ID of the object to move.
            x: New X position.
            y: New Y position.
            yaw: New yaw rotation.

        Returns:
            Confirmation of move.
        """
        result = state.ceiling_tools._move_ceiling_object_impl(
            object_id=object_id, x=x, y=y, yaw=yaw
        )
        return result

    @server.tool()
    async def ceiling__remove_object(object_id: str) -> str:
        """Remove a ceiling-mounted object.

        Args:
            object_id: ID of the object to remove.

        Returns:
            Confirmation of removal.
        """
        result = state.ceiling_tools._remove_ceiling_object_impl(object_id)
        return result

    @server.tool()
    async def ceiling__get_scene_state() -> str:
        """Get all ceiling-mounted objects in the scene.

        Returns:
            JSON list of ceiling objects with IDs, positions, and dimensions.
        """
        result = state.ceiling_tools._get_current_scene_impl()
        return result

    @server.tool()
    async def ceiling__list_available_assets() -> str:
        """List all ceiling asset models available for placement.

        Returns:
            List of assets with IDs, names, and dimensions.
        """
        result = state.ceiling_tools._list_available_assets_impl()
        return result

    @server.tool()
    async def ceiling__rescale(
        object_id: str, scale_factor: float
    ) -> str:
        """Resize a ceiling-mounted object.

        Args:
            object_id: ID of the object to rescale.
            scale_factor: Scale multiplier.

        Returns:
            New dimensions and affected objects.
        """
        result = state.ceiling_tools._rescale_ceiling_object_impl(
            object_id, scale_factor
        )
        return result

    console_logger.info("Registered ceiling tools")

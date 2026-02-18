"""Manipuland stage MCP tools.

Wraps ManipulandTools _impl methods for placing small objects on furniture surfaces.
"""

import logging

from mcp.server.fastmcp import FastMCP

from scenesmith_mcp.state import ServerState

console_logger = logging.getLogger(__name__)


def register_manipuland_tools(server: FastMCP, state: ServerState) -> None:
    """Register all manipuland-stage MCP tools."""

    @server.tool()
    async def manipuland__generate_assets(
        object_descriptions: list[str],
        short_names: list[str],
        desired_dimensions: list[list[float]],
        style_context: str | None = None,
    ) -> str:
        """Create 3D small object models (manipulands) from descriptions.

        Generate items that go on furniture surfaces: books, vases, cups, etc.

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
            object_type=ObjectType.MANIPULAND,
            desired_dimensions=desired_dimensions,
            style_context=style_context,
            scene_id=state.scene.scene_dir.name if state.scene else "default",
        )
        result = state.manipuland_tools._generate_assets_impl(request)
        return result

    @server.tool()
    async def manipuland__list_support_surfaces() -> str:
        """List all support surfaces on furniture where objects can be placed.

        Returns:
            List of surfaces with IDs, area, height, and clearance.
        """
        result = state.manipuland_tools._list_support_surfaces_impl()
        return result

    @server.tool()
    async def manipuland__place_on_surface(
        asset_id: str,
        surface_id: str,
        position_x: float,
        position_z: float,
        rotation_degrees: float = 0.0,
    ) -> str:
        """Place an object on a furniture surface.

        Args:
            asset_id: ID of the asset to place.
            surface_id: ID of the support surface.
            position_x: X position on surface (meters, surface-local coords).
            position_z: Z position on surface (meters, surface-local coords).
            rotation_degrees: Rotation on surface (degrees).

        Returns:
            Placement result with object ID and world pose.
        """
        result = state.manipuland_tools._place_manipuland_impl(
            asset_id=asset_id,
            surface_id=surface_id,
            position_x=position_x,
            position_z=position_z,
            rotation_degrees=rotation_degrees,
        )
        return result

    @server.tool()
    async def manipuland__move(
        object_id: str,
        surface_id: str,
        position_x: float,
        position_z: float,
        rotation_degrees: float = 0.0,
    ) -> str:
        """Move an object to a new position on a surface (can change surfaces).

        Args:
            object_id: ID of the object to move.
            surface_id: Target surface ID.
            position_x: New X position on surface.
            position_z: New Z position on surface.
            rotation_degrees: New rotation.

        Returns:
            Confirmation of move.
        """
        result = state.manipuland_tools._move_manipuland_impl(
            object_id=object_id,
            surface_id=surface_id,
            position_x=position_x,
            position_z=position_z,
            rotation_degrees=rotation_degrees,
        )
        return result

    @server.tool()
    async def manipuland__remove(object_id: str) -> str:
        """Remove an object from the scene.

        Args:
            object_id: ID of the object to remove.

        Returns:
            Confirmation of removal.
        """
        result = state.manipuland_tools._remove_manipuland_impl(object_id)
        return result

    @server.tool()
    async def manipuland__get_scene_state() -> str:
        """Get all manipulands on furniture surfaces.

        Returns:
            JSON with furniture, support surfaces, and objects on them.
        """
        result = state.manipuland_tools._get_current_scene_impl()
        return result

    @server.tool()
    async def manipuland__list_available_assets() -> str:
        """List all manipuland models available for placement.

        Returns:
            List of assets with IDs, names, and dimensions.
        """
        result = state.manipuland_tools._list_available_assets_impl()
        return result

    @server.tool()
    async def manipuland__create_stack(
        asset_ids: list[str],
        surface_id: str,
        position_x: float,
        position_z: float,
        rotation_degrees: float = 0.0,
    ) -> str:
        """Create a vertical stack of objects on a surface.

        Args:
            asset_ids: List of asset IDs to stack (bottom to top).
            surface_id: Surface to stack on.
            position_x: X position on surface.
            position_z: Z position on surface.
            rotation_degrees: Base rotation.

        Returns:
            Stack result with composite object ID.
        """
        result = state.manipuland_tools._create_stack_impl(
            asset_ids=asset_ids,
            surface_id=surface_id,
            position_x=position_x,
            position_z=position_z,
            rotation_degrees=rotation_degrees,
        )
        return result

    @server.tool()
    async def manipuland__fill_container(
        container_id: str,
        fill_asset_ids: list[str],
        fill_count: int | None = None,
    ) -> str:
        """Fill a container object with items.

        Args:
            container_id: ID of the container to fill.
            fill_asset_ids: Asset IDs to use as fill items.
            fill_count: Number of items to fill (default: auto).

        Returns:
            Fill result with placed items.
        """
        result = state.manipuland_tools._fill_container_impl(
            container_id=container_id,
            fill_asset_ids=fill_asset_ids,
            fill_count=fill_count,
        )
        return result

    @server.tool()
    async def manipuland__create_arrangement(
        surface_id: str,
        fill_assets_json: str,
    ) -> str:
        """Create a controlled arrangement of objects at specified positions.

        Args:
            surface_id: Surface to arrange on.
            fill_assets_json: JSON array of objects with asset_id, position_x,
                position_z, rotation_degrees for each item.

        Returns:
            Arrangement result.
        """
        import json

        fill_assets = json.loads(fill_assets_json)
        result = state.manipuland_tools._create_arrangement_impl(
            surface_id=surface_id,
            fill_assets=fill_assets,
        )
        return result

    @server.tool()
    async def manipuland__create_pile(
        asset_ids: list[str],
        surface_id: str,
        position_x: float,
        position_z: float,
    ) -> str:
        """Create a random pile of objects using physics simulation.

        Args:
            asset_ids: Asset IDs to pile.
            surface_id: Surface to pile on.
            position_x: Center X position.
            position_z: Center Z position.

        Returns:
            Pile result with settled positions.
        """
        result = state.manipuland_tools._create_pile_impl(
            asset_ids=asset_ids,
            surface_id=surface_id,
            position_x=position_x,
            position_z=position_z,
        )
        return result

    @server.tool()
    async def manipuland__rescale(
        object_id: str, scale_factor: float
    ) -> str:
        """Resize a manipuland object.

        Args:
            object_id: ID of the object to rescale.
            scale_factor: Scale multiplier.

        Returns:
            New dimensions and affected objects.
        """
        result = state.manipuland_tools._rescale_manipuland_impl(
            object_id, scale_factor
        )
        return result

    @server.tool()
    async def manipuland__resolve_penetrations(
        object_ids: list[str],
    ) -> str:
        """Resolve physics penetrations between objects.

        Args:
            object_ids: IDs of objects to check and resolve.

        Returns:
            Resolution result.
        """
        result = state.manipuland_tools._resolve_penetrations_impl(object_ids)
        return result

    console_logger.info("Registered manipuland tools")

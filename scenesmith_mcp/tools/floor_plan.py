"""Floor plan stage MCP tools.

Wraps FloorPlanTools _impl methods for room layout creation and modification.
"""

import logging

from mcp.server.fastmcp import FastMCP

from scenesmith_mcp.state import ServerState

console_logger = logging.getLogger(__name__)


def _position_float_to_str(position: float) -> str:
    """Convert a numeric position (0.0-1.0) to 'left', 'center', or 'right'."""
    if position <= 0.33:
        return "left"
    elif position <= 0.66:
        return "center"
    else:
        return "right"


def register_floor_plan_tools(server: FastMCP, state: ServerState) -> None:
    """Register all floor-plan-stage MCP tools."""

    @server.tool()
    async def floor_plan__generate_room_specs(
        room_specs_json: str,
    ) -> str:
        """Generate room layout from JSON room specifications.

        Creates rooms with specified types, dimensions, and adjacencies.
        The JSON should contain a list of room specs with name, type, width, depth.

        Args:
            room_specs_json: JSON string with room specifications.

        Returns:
            Result with ASCII floor plan and wall segment labels.
        """
        result = state.floor_plan_tools._generate_room_specs_impl(room_specs_json)
        return result.to_json() if hasattr(result, 'to_json') else str(result)

    @server.tool()
    async def floor_plan__resize_room(
        room_id: str, width: float, depth: float
    ) -> str:
        """Resize a room to new dimensions.

        Args:
            room_id: ID of the room to resize.
            width: New width in meters.
            depth: New depth in meters.

        Returns:
            Success/failure result.
        """
        result = state.floor_plan_tools._resize_room_impl(room_id, width, depth)
        return str(result)

    @server.tool()
    async def floor_plan__add_adjacency(
        room_a: str, room_b: str
    ) -> str:
        """Add adjacency between two rooms.

        Args:
            room_a: ID of first room.
            room_b: ID of second room.

        Returns:
            Success/failure result.
        """
        result = state.floor_plan_tools._add_adjacency_impl(room_a, room_b)
        return str(result)

    @server.tool()
    async def floor_plan__remove_adjacency(
        room_a: str, room_b: str
    ) -> str:
        """Remove adjacency between two rooms.

        Args:
            room_a: ID of first room.
            room_b: ID of second room.

        Returns:
            Success/failure result.
        """
        result = state.floor_plan_tools._remove_adjacency_impl(room_a, room_b)
        return str(result)

    @server.tool()
    async def floor_plan__add_open_connection(
        room_a: str, room_b: str
    ) -> str:
        """Add an open connection (no wall) between two rooms.

        Args:
            room_a: ID of first room.
            room_b: ID of second room.

        Returns:
            Success/failure result.
        """
        result = state.floor_plan_tools._add_open_connection_impl(room_a, room_b)
        return str(result)

    @server.tool()
    async def floor_plan__remove_open_connection(
        room_a: str, room_b: str
    ) -> str:
        """Remove an open connection between two rooms.

        Args:
            room_a: ID of first room.
            room_b: ID of second room.

        Returns:
            Success/failure result.
        """
        result = state.floor_plan_tools._remove_open_connection_impl(room_a, room_b)
        return str(result)

    @server.tool()
    async def floor_plan__set_wall_height(
        height_meters: float,
    ) -> str:
        """Set wall height for the floor plan.

        Args:
            height_meters: Wall height in meters.

        Returns:
            Success/failure result.
        """
        result = state.floor_plan_tools._set_wall_height_impl(height_meters)
        return str(result)

    @server.tool()
    async def floor_plan__add_door(
        wall_segment: str,
        position: float,
        width: float | None = None,
        height: float | None = None,
    ) -> str:
        """Add a door to a wall segment.

        Args:
            wall_segment: Wall segment label (e.g., "room1_north").
            position: Position along the wall segment (0.0 to 1.0).
            width: Door width in meters (default: standard door width).
            height: Door height in meters (default: standard door height).

        Returns:
            Success/failure result with door ID.
        """
        # Convert numeric position to string for the impl method
        pos_str = _position_float_to_str(position)
        result = state.floor_plan_tools._add_door_impl(
            wall_segment, pos_str, width, height
        )
        return str(result)

    @server.tool()
    async def floor_plan__remove_door(door_id: str) -> str:
        """Remove a door by ID.

        Args:
            door_id: ID of the door to remove.

        Returns:
            Success/failure result.
        """
        result = state.floor_plan_tools._remove_door_impl(door_id)
        return str(result)

    @server.tool()
    async def floor_plan__add_window(
        wall_segment: str,
        position: float,
        width: float | None = None,
        height: float | None = None,
        sill_height: float | None = None,
    ) -> str:
        """Add a window to a wall segment.

        Args:
            wall_segment: Wall segment label (e.g., "room1_south").
            position: Position along the wall segment (0.0 to 1.0).
            width: Window width in meters.
            height: Window height in meters.
            sill_height: Height of window sill from floor in meters.

        Returns:
            Success/failure result with window ID.
        """
        # Convert numeric position to string for the impl method
        pos_str = _position_float_to_str(position)
        result = state.floor_plan_tools._add_window_impl(
            wall_segment, pos_str, width, height, sill_height
        )
        return str(result)

    @server.tool()
    async def floor_plan__remove_window(window_id: str) -> str:
        """Remove a window by ID.

        Args:
            window_id: ID of the window to remove.

        Returns:
            Success/failure result.
        """
        result = state.floor_plan_tools._remove_window_impl(window_id)
        return str(result)

    @server.tool()
    async def floor_plan__get_material(
        description: str,
    ) -> str:
        """Find a material by text description.

        Args:
            description: Material description (e.g., "dark hardwood floor").

        Returns:
            Material ID and details.
        """
        result = state.floor_plan_tools._get_material_impl(description)
        return str(result)

    @server.tool()
    async def floor_plan__set_room_materials(
        room_id: str,
        floor_material_id: str | None = None,
        wall_material_id: str | None = None,
        ceiling_material_id: str | None = None,
    ) -> str:
        """Set materials for a room's floor, walls, and/or ceiling.

        Args:
            room_id: ID of the room.
            floor_material_id: Material ID for the floor.
            wall_material_id: Material ID for the walls.
            ceiling_material_id: Material ID for the ceiling.

        Returns:
            Success/failure result.
        """
        result = state.floor_plan_tools._set_room_materials_impl(
            room_id, floor_material_id, wall_material_id
        )
        return str(result)

    @server.tool()
    async def floor_plan__set_exterior_material(
        material_id: str,
    ) -> str:
        """Set the exterior wall material.

        Args:
            material_id: Material ID for exterior walls.

        Returns:
            Success/failure result.
        """
        result = state.floor_plan_tools._set_exterior_material_impl(material_id)
        return str(result)

    @server.tool()
    async def floor_plan__list_room_materials() -> str:
        """List current materials for all rooms.

        Returns:
            Materials assigned to each room's floor, walls, and ceiling.
        """
        result = state.floor_plan_tools._list_room_materials_impl()
        return str(result)

    @server.tool()
    async def floor_plan__validate() -> str:
        """Validate the current floor plan layout and connectivity.

        Returns:
            Validation results for layout and connectivity.
        """
        result = state.floor_plan_tools._validate_impl()
        return str(result)

    @server.tool()
    async def floor_plan__render_ascii() -> str:
        """Render an ASCII representation of the floor plan.

        Returns:
            ASCII art floor plan showing rooms, doors, and windows.
        """
        result = state.floor_plan_tools._render_ascii_impl()
        return result

    console_logger.info("Registered floor plan tools")

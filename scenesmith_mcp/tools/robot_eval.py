"""Robot evaluation MCP tools.

Wraps robot_eval tools for scene analysis, object queries, and spatial relations.
Used by robot-policy and robot-validator subagents.
"""

import json
import logging

from mcp.server.fastmcp import FastMCP

from scenesmith_mcp.state import ServerState

console_logger = logging.getLogger(__name__)


def register_robot_eval_tools(server: FastMCP, state: ServerState) -> None:
    """Register robot evaluation tools."""

    @server.tool()
    async def robot_eval__list_objects() -> str:
        """List all objects in the scene with basic info.

        Returns:
            JSON list of objects with IDs, names, types, and positions.
        """
        if state.scene is None:
            return "Error: No scene initialized."

        objects = []
        for obj_id, obj in state.scene.objects.items():
            pos = obj.transform.translation()
            objects.append({
                "id": str(obj_id),
                "name": obj.name,
                "type": obj.object_type.value if obj.object_type else "unknown",
                "position": {"x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2])},
            })
        return json.dumps(objects, indent=2)

    @server.tool()
    async def robot_eval__get_object_info(
        object_id: str,
    ) -> str:
        """Get detailed information about a specific object.

        Args:
            object_id: ID of the object to query.

        Returns:
            JSON with position, rotation, dimensions, support surfaces, etc.
        """
        from scenesmith.agent_utils.room import UniqueID

        if state.scene is None:
            return "Error: No scene initialized."

        obj = state.scene.get_object(UniqueID(object_id))
        if obj is None:
            return f"Object '{object_id}' not found."

        pos = obj.transform.translation()
        info = {
            "id": str(obj.object_id),
            "name": obj.name,
            "type": obj.object_type.value if obj.object_type else "unknown",
            "position": {"x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2])},
            "has_support_surfaces": len(obj.support_surfaces) > 0,
            "support_surface_count": len(obj.support_surfaces),
        }
        if obj.bbox_min is not None and obj.bbox_max is not None:
            info["dimensions"] = {
                "width": float(obj.bbox_max[0] - obj.bbox_min[0]),
                "depth": float(obj.bbox_max[1] - obj.bbox_min[1]),
                "height": float(obj.bbox_max[2] - obj.bbox_min[2]),
            }
        return json.dumps(info, indent=2)

    @server.tool()
    async def robot_eval__get_distance(
        object_a: str, object_b: str
    ) -> str:
        """Get the distance between two objects.

        Args:
            object_a: ID of first object.
            object_b: ID of second object.

        Returns:
            JSON with Euclidean distance and per-axis distances.
        """
        import numpy as np
        from scenesmith.agent_utils.room import UniqueID

        if state.scene is None:
            return "Error: No scene initialized."

        obj_a = state.scene.get_object(UniqueID(object_a))
        obj_b = state.scene.get_object(UniqueID(object_b))

        if obj_a is None or obj_b is None:
            missing = object_a if obj_a is None else object_b
            return f"Object '{missing}' not found."

        pos_a = obj_a.transform.translation()
        pos_b = obj_b.transform.translation()
        diff = pos_b - pos_a
        distance = float(np.linalg.norm(diff))

        result = {
            "object_a": object_a,
            "object_b": object_b,
            "euclidean_distance_m": round(distance, 3),
            "delta_x": round(float(diff[0]), 3),
            "delta_y": round(float(diff[1]), 3),
            "delta_z": round(float(diff[2]), 3),
        }
        return json.dumps(result, indent=2)

    @server.tool()
    async def robot_eval__get_spatial_relation(
        object_a: str, object_b: str
    ) -> str:
        """Get spatial relationship between two objects.

        Args:
            object_a: ID of first object.
            object_b: ID of second object.

        Returns:
            JSON with spatial relations (left_of, right_of, in_front_of, behind,
            above, below, near, far).
        """
        import numpy as np
        from scenesmith.agent_utils.room import UniqueID

        if state.scene is None:
            return "Error: No scene initialized."

        obj_a = state.scene.get_object(UniqueID(object_a))
        obj_b = state.scene.get_object(UniqueID(object_b))

        if obj_a is None or obj_b is None:
            missing = object_a if obj_a is None else object_b
            return f"Object '{missing}' not found."

        pos_a = obj_a.transform.translation()
        pos_b = obj_b.transform.translation()
        diff = pos_b - pos_a
        distance = float(np.linalg.norm(diff[:2]))

        relations = []
        if diff[0] > 0.1:
            relations.append("right_of")
        elif diff[0] < -0.1:
            relations.append("left_of")
        if diff[1] > 0.1:
            relations.append("in_front_of")
        elif diff[1] < -0.1:
            relations.append("behind")
        if diff[2] > 0.1:
            relations.append("above")
        elif diff[2] < -0.1:
            relations.append("below")
        if distance < 1.0:
            relations.append("near")
        elif distance > 3.0:
            relations.append("far")

        result = {
            "object_a": object_a,
            "object_b": object_b,
            "relations": relations,
            "distance_m": round(distance, 3),
        }
        return json.dumps(result, indent=2)

    console_logger.info("Registered robot eval tools")

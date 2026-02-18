"""SceneSmith MCP Server - Main entry point.

Exposes all scene manipulation tools via the Model Context Protocol,
enabling Claude Code subagents to create and modify 3D indoor scenes.

Usage:
    python -m scenesmith_mcp.server
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from scenesmith_mcp.state import ServerState

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    stream=sys.stderr,
)
console_logger = logging.getLogger(__name__)

# Global state shared by all tool handlers
state = ServerState()

# Create MCP server
server = FastMCP("scenesmith")


def _initialize_infrastructure() -> None:
    """Initialize configuration, scene, and infrastructure on startup."""
    from scenesmith_mcp.config import load_config
    from scenesmith_mcp.lifecycle import (
        initialize_rendering_manager,
        initialize_state,
        start_blender_server,
        start_collision_server,
    )

    # Load Hydra config
    try:
        cfg = load_config()
        console_logger.info("Configuration loaded")
    except Exception as e:
        console_logger.warning(f"Config loading failed, using defaults: {e}")
        from omegaconf import OmegaConf

        cfg = OmegaConf.create(_default_config())

    # Set output directory
    output_dir = Path(
        os.environ.get("SCENESMITH_OUTPUT_DIR", "outputs/mcp_session")
    )

    # Initialize basic state (scene is created later by floor plan stage)
    initialize_state(state, cfg, output_dir=output_dir)

    # Try to start infrastructure servers (non-fatal if they fail)
    try:
        start_blender_server(state)
    except Exception as e:
        console_logger.warning(f"BlenderServer not available: {e}")

    try:
        start_collision_server(state)
    except Exception as e:
        console_logger.warning(f"ConvexDecompositionServer not available: {e}")

    # Initialize rendering manager
    try:
        initialize_rendering_manager(state)
    except Exception as e:
        console_logger.warning(f"RenderingManager not available: {e}")


def _default_config() -> dict:
    """Minimal default configuration for MCP server mode."""
    return {
        "rendering": {
            "layout": "top_plus_sides",
            "top_view_width": 1024,
            "top_view_height": 1024,
            "side_view_count": 4,
            "side_view_width": 512,
            "side_view_height": 512,
            "background_color": [1.0, 1.0, 1.0],
            "blender_server_host": "127.0.0.1",
            "blender_server_port_range": [8000, 8350],
            "retry_count": 10,
            "retry_delay": 3,
            "server_startup_delay": 0.1,
            "port_cleanup_delay": 0.1,
            "taa_samples": 8,
            "annotations": {
                "enable_set_of_mark_labels": True,
                "enable_bounding_boxes": True,
                "enable_direction_arrows": True,
                "enable_partial_walls": True,
                "enable_support_surface_debug": False,
                "enable_convex_hull_debug": False,
            },
        },
        "collision_geometry": {
            "method": "vhacd",
            "server_port_range": [7100, 7450],
        },
        "floor_plan_agent": {
            "mode": "house",
            "min_floor_plan_dim_m": 1.5,
            "max_floor_plan_dim_m": 20,
            "wall_height": {"min": 2.0, "max": 4.5, "default": 2.5},
            "wall_thickness": 0.05,
            "floor_thickness": 0.1,
            "rendering": {
                "blender_server_port_range": [8000, 8050],
                "render_size": 1024,
            },
            "materials": {
                "use_retrieval_server": False,
                "default_wall_material": "Plaster001_1K-JPG",
                "default_floor_material": "Wood094_1K-JPG",
            },
            "room_placement": {
                "timeout_seconds": 5.0,
                "scoring_weights": {"compactness": 1.0, "stability": 1.0},
                "min_opening_separation": 0.5,
                "exterior_wall_clearance_m": 20.0,
            },
            "doors": {
                "width_range": [0.9, 1.9],
                "height_range": [2.0, 2.4],
                "default_width": 0.9,
                "default_height": 2.1,
                "exterior_clearance": 20.0,
            },
            "windows": {
                "width_range": [0.6, 3.0],
                "height_range": [0.6, 2.0],
                "default_width": 1.2,
                "default_height": 1.2,
                "default_sill_height": 0.9,
                "segment_margin": 0.3,
            },
            "clearance_zones": {
                "door_clearance_distance": 0.8,
                "window_clearance_distance": 0.5,
            },
        },
        "furniture_agent": {
            "placement_noise": {
                "mode": "auto",
                "natural_profile": {
                    "position_xy_std_meters": 0.03,
                    "rotation_yaw_std_degrees": 1.0,
                },
                "perfect_profile": {
                    "position_xy_std_meters": 0.001,
                    "rotation_yaw_std_degrees": 0.1,
                },
            },
            "loop_detection": {
                "enabled": True,
                "max_repeated_attempts": 25,
                "tracking_window": 30,
            },
            "physics_validation": {
                "wall_thickness": 0.05,
                "floor_penetration_tolerance_m": 0.05,
                "object_penetration_threshold_m": 0.001,
                "manipuland_furniture_tolerance_m": 0.02,
                "remove_fallen_furniture": True,
                "fallen_tilt_threshold_degrees": 45.0,
            },
            "clearance_zones": {
                "passage_size": 0.5,
                "open_connection_clearance": 0.7,
            },
            "reachability": {"robot_width": 0.3},
            "snap_to_object": {
                "voxel_pitch_meters": 0.03,
                "min_vertices_threshold": 10000,
                "circular_detection_volume_ratio_threshold": 0.80,
                "iterative_snap_step_m": 0.01,
                "max_snap_distance_m": 10.0,
                "snap_margin_m": 0.01,
                "max_sample_vertices": 2000,
            },
            "asset_manager": {
                "general_asset_source": "generated",
                "backend": "sam3d",
                "num_side_views_for_physics_analysis": 4,
                "side_view_elevation_degrees": 20.0,
                "floater_distance_threshold": 0.05,
                "min_mesh_dimension_meters": 0.001,
                "mesh_relative_dimension_threshold": 0.02,
                "validation_taa_samples": 8,
            },
        },
        "wall_agent": {
            "placement_noise": {
                "mode": "auto",
                "natural_profile": {
                    "position_xy_std_meters": 0.03,
                    "rotation_yaw_std_degrees": 1.0,
                },
                "perfect_profile": {
                    "position_xy_std_meters": 0.001,
                    "rotation_yaw_std_degrees": 0.1,
                },
            },
            "loop_detection": {
                "enabled": True,
                "max_repeated_attempts": 25,
                "tracking_window": 30,
            },
        },
        "ceiling_agent": {
            "placement_noise": {
                "mode": "auto",
                "natural_profile": {
                    "position_xy_std_meters": 0.03,
                    "rotation_yaw_std_degrees": 1.0,
                },
                "perfect_profile": {
                    "position_xy_std_meters": 0.001,
                    "rotation_yaw_std_degrees": 0.1,
                },
            },
            "loop_detection": {
                "enabled": True,
                "max_repeated_attempts": 25,
                "tracking_window": 30,
            },
        },
        "manipuland_agent": {
            "placement_noise": {
                "mode": "auto",
                "natural_profile": {
                    "position_xy_std_meters": 0.03,
                    "rotation_yaw_std_degrees": 1.0,
                },
                "perfect_profile": {
                    "position_xy_std_meters": 0.001,
                    "rotation_yaw_std_degrees": 0.1,
                },
            },
            "loop_detection": {
                "enabled": True,
                "max_repeated_attempts": 25,
                "tracking_window": 30,
            },
        },
    }


def register_all_tools() -> None:
    """Register all MCP tools from tool modules."""
    from scenesmith_mcp.tools.assets import register_asset_tools
    from scenesmith_mcp.tools.ceiling import register_ceiling_tools
    from scenesmith_mcp.tools.floor_plan import register_floor_plan_tools
    from scenesmith_mcp.tools.furniture import register_furniture_tools
    from scenesmith_mcp.tools.manipuland import register_manipuland_tools
    from scenesmith_mcp.tools.observation import register_observation_tools
    from scenesmith_mcp.tools.robot_eval import register_robot_eval_tools
    from scenesmith_mcp.tools.wall import register_wall_tools
    from scenesmith_mcp.tools.workflow import register_workflow_tools

    register_floor_plan_tools(server, state)
    register_furniture_tools(server, state)
    register_wall_tools(server, state)
    register_ceiling_tools(server, state)
    register_manipuland_tools(server, state)
    register_observation_tools(server, state)
    register_workflow_tools(server, state)
    register_asset_tools(server, state)
    register_robot_eval_tools(server, state)

    console_logger.info("All MCP tools registered")


def main() -> None:
    """Entry point for the MCP server."""
    register_all_tools()
    _initialize_infrastructure()
    console_logger.info("SceneSmith MCP server ready")
    server.run(transport="stdio")


if __name__ == "__main__":
    main()

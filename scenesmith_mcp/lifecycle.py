"""Server lifecycle management for infrastructure services.

Handles starting/stopping BlenderServer, ConvexDecompositionServer,
and initializing RoomScene state.
"""

import logging
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from scenesmith_mcp.state import ServerState

console_logger = logging.getLogger(__name__)


def initialize_state(
    state: ServerState,
    cfg: DictConfig,
    output_dir: Path | None = None,
    scene_dir: Path | None = None,
) -> None:
    """Initialize server state with configuration.

    Args:
        state: ServerState to populate.
        cfg: Hydra configuration.
        output_dir: Output directory for renders and logs.
        scene_dir: Scene-specific directory.
    """
    state.cfg = cfg
    state.output_dir = output_dir or Path("outputs/mcp_session")
    state.scene_dir = scene_dir or state.output_dir / "scene"

    # Ensure directories exist
    state.output_dir.mkdir(parents=True, exist_ok=True)
    state.scene_dir.mkdir(parents=True, exist_ok=True)

    console_logger.info(f"State initialized. Output: {state.output_dir}")


def _get_rendering_cfg(cfg: DictConfig) -> DictConfig:
    """Extract rendering configuration from the config tree.

    Tries multiple locations where rendering config might live.
    """
    # Try furniture_agent.rendering first (Hydra structure)
    for key in ("furniture_agent", "wall_agent", "floor_plan_agent"):
        sub = OmegaConf.select(cfg, key, default=None)
        if sub and OmegaConf.select(sub, "rendering", default=None):
            return sub.rendering

    # Try top-level rendering
    if OmegaConf.select(cfg, "rendering", default=None):
        return cfg.rendering

    return OmegaConf.create({
        "blender_server_port_range": [8000, 8350],
        "server_startup_delay": 0.1,
        "port_cleanup_delay": 0.1,
    })


def start_blender_server(state: ServerState, gpu_id: int | None = None) -> None:
    """Start the BlenderServer for rendering.

    Args:
        state: ServerState to attach BlenderServer to.
        gpu_id: GPU device ID for isolation (None = default GPU).
    """
    from scenesmith.agent_utils.blender import BlenderServer

    render_cfg = _get_rendering_cfg(state.cfg)
    console_logger.info("Starting BlenderServer...")
    port_range = OmegaConf.select(
        render_cfg, "blender_server_port_range", default=[8000, 8350]
    )
    state.blender_server = BlenderServer(
        port_range=tuple(port_range),
        server_startup_delay=OmegaConf.select(
            render_cfg, "server_startup_delay", default=0.1
        ),
        port_cleanup_delay=OmegaConf.select(
            render_cfg, "port_cleanup_delay", default=0.1
        ),
        gpu_id=gpu_id,
    )
    state.blender_server.start()
    state.blender_server.wait_until_ready()
    console_logger.info("BlenderServer ready")


def start_collision_server(state: ServerState) -> None:
    """Start the ConvexDecompositionServer for collision geometry.

    Args:
        state: ServerState to attach collision server to.
    """
    from scenesmith.agent_utils.convex_decomposition_server import (
        ConvexDecompositionServer,
    )

    console_logger.info("Starting ConvexDecompositionServer...")
    state.collision_server = ConvexDecompositionServer()
    state.collision_server.start()
    console_logger.info("ConvexDecompositionServer ready")


def initialize_rendering_manager(state: ServerState) -> None:
    """Initialize the RenderingManager for scene rendering.

    Args:
        state: ServerState to attach RenderingManager to.
    """
    from scenesmith.agent_utils.rendering_manager import RenderingManager
    from scenesmith.utils.logging import ConsoleLogger

    logger = ConsoleLogger(output_dir=state.output_dir)
    state.rendering_manager = RenderingManager(
        cfg=state.cfg,
        logger=logger,
    )
    console_logger.info("RenderingManager initialized")


def shutdown(state: ServerState) -> None:
    """Gracefully shut down all infrastructure servers.

    Args:
        state: ServerState with servers to stop.
    """
    console_logger.info("Shutting down infrastructure...")
    if state.blender_server:
        try:
            state.blender_server.stop()
            console_logger.info("BlenderServer stopped")
        except Exception as e:
            console_logger.error(f"Error stopping BlenderServer: {e}")

    if state.collision_server:
        try:
            state.collision_server.stop()
            console_logger.info("ConvexDecompositionServer stopped")
        except Exception as e:
            console_logger.error(f"Error stopping ConvexDecompositionServer: {e}")

    console_logger.info("Shutdown complete")

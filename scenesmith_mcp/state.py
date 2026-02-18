"""Server-side state management for the SceneSmith MCP server.

Holds the RoomScene, infrastructure server references, and tool class instances.
All MCP tool functions operate on this shared state.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

console_logger = logging.getLogger(__name__)


@dataclass
class ServerState:
    """Mutable state shared across all MCP tool handlers.

    Initialized by the lifecycle module when the MCP server starts.
    Tool modules access this via the global `state` reference in server.py.
    """

    cfg: DictConfig | None = None

    # Core scene state
    scene: Any = None  # RoomScene instance
    house_layout: Any = None  # HouseLayout for floor plan stage

    # Infrastructure servers (started by lifecycle)
    blender_server: Any = None  # BlenderServer
    collision_server: Any = None  # ConvexDecompositionServer

    # Managers
    rendering_manager: Any = None  # RenderingManager
    asset_manager: Any = None  # AssetManager

    # Per-stage tool class instances (initialized on demand per stage)
    furniture_tools: Any = None  # FurnitureTools
    wall_tools: Any = None  # WallTools
    ceiling_tools: Any = None  # CeilingTools
    manipuland_tools: Any = None  # ManipulandTools
    floor_plan_tools: Any = None  # FloorPlanTools
    scene_tools: Any = None  # SceneTools
    vision_tools: Any = None  # VisionTools (current stage)
    workflow_tools: Any = None  # WorkflowTools

    # Checkpoint management
    checkpoints: dict[str, dict] = field(default_factory=dict)
    current_stage: str = ""

    # Output directory
    output_dir: Path | None = None
    scene_dir: Path | None = None

    def save_checkpoint(self, name: str) -> None:
        """Save current scene state as a named checkpoint."""
        if self.scene is not None:
            self.checkpoints[name] = self.scene.to_state_dict()
            console_logger.info(f"Saved checkpoint: {name}")

    def restore_checkpoint(self, name: str) -> bool:
        """Restore scene state from a named checkpoint."""
        if name not in self.checkpoints:
            console_logger.error(f"Checkpoint not found: {name}")
            return False
        self.scene.restore_from_state_dict(self.checkpoints[name])
        if self.rendering_manager:
            self.rendering_manager.clear_cache()
        console_logger.info(f"Restored checkpoint: {name}")
        return True

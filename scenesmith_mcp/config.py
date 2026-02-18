"""Configuration loader for the SceneSmith MCP server.

Loads Hydra/OmegaConf configuration from the existing configurations/ directory.
"""

import logging
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

console_logger = logging.getLogger(__name__)

# Project root (parent of scenesmith_mcp/)
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_DIR = PROJECT_ROOT / "configurations"


def load_config(overrides: list[str] | None = None) -> DictConfig:
    """Load SceneSmith configuration using Hydra.

    Args:
        overrides: Optional Hydra override strings
            (e.g., ["+name=test", "experiment=indoor_scene_generation"]).

    Returns:
        Resolved DictConfig with all interpolations applied.
    """
    if overrides is None:
        overrides = []

    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name="config", overrides=overrides)

    # Register custom OmegaConf resolvers if needed.
    try:
        from scenesmith.utils.omegaconf import register_resolvers

        register_resolvers()
    except ImportError:
        pass

    # Resolve all interpolations.
    OmegaConf.resolve(cfg)

    console_logger.info("Configuration loaded successfully")
    return cfg

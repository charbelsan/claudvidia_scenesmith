"""NVIDIA Omniverse USD integration for asset search and retrieval.

Provides:
- OmniverseClient: Search and retrieve USD assets from NVIDIA sources
- USDConverter: Convert USD → GLB → SDF pipeline
- AssetCatalog: Local catalog of downloaded and prepared assets
"""

from scenesmith_mcp.omniverse.catalog import AssetCatalog
from scenesmith_mcp.omniverse.client import OmniverseClient, USDAsset
from scenesmith_mcp.omniverse.converter import USDConverter

__all__ = [
    "AssetCatalog",
    "OmniverseClient",
    "USDAsset",
    "USDConverter",
]

"""Local asset catalog management for cached Omniverse USD assets.

Maintains a local index of downloaded and converted assets for
fast retrieval without re-downloading from Omniverse.
"""

import json
import logging
from pathlib import Path

console_logger = logging.getLogger(__name__)


class AssetCatalog:
    """Local catalog of downloaded and prepared 3D assets.

    Tracks assets that have been downloaded from Omniverse,
    converted to GLB/SDF, and are ready for scene placement.
    Supports text-based search on cached assets.
    """

    def __init__(self, catalog_dir: Path):
        """Initialize the asset catalog.

        Args:
            catalog_dir: Directory for the catalog index and cached assets.
        """
        self.catalog_dir = catalog_dir
        self.catalog_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = catalog_dir / "catalog_index.json"
        self._index: dict[str, dict] = {}
        self._load_index()

    def _load_index(self) -> None:
        """Load the catalog index from disk."""
        if self.index_path.exists():
            try:
                with open(self.index_path) as f:
                    self._index = json.load(f)
                console_logger.info(
                    f"Loaded asset catalog with {len(self._index)} assets"
                )
            except (json.JSONDecodeError, OSError) as e:
                console_logger.warning(f"Failed to load catalog index: {e}")
                self._index = {}

    def _save_index(self) -> None:
        """Save the catalog index to disk."""
        try:
            with open(self.index_path, "w") as f:
                json.dump(self._index, f, indent=2)
        except OSError as e:
            console_logger.error(f"Failed to save catalog index: {e}")

    def add_asset(
        self,
        asset_id: str,
        name: str,
        category: str,
        source_uri: str,
        glb_path: str,
        sdf_path: str | None = None,
        gltf_path: str | None = None,
        dimensions: dict[str, float] | None = None,
        bbox_min: list[float] | None = None,
        bbox_max: list[float] | None = None,
        metadata: dict | None = None,
    ) -> None:
        """Register an asset in the catalog.

        Args:
            asset_id: Unique asset identifier.
            name: Human-readable name.
            category: Asset category (furniture, decoration, etc.).
            source_uri: Original Omniverse URI or download URL.
            glb_path: Path to GLB mesh file.
            sdf_path: Path to Drake SDF file.
            gltf_path: Path to processed GLTF file.
            dimensions: Asset dimensions {width, depth, height} in meters.
            bbox_min: Bounding box minimum [x, y, z].
            bbox_max: Bounding box maximum [x, y, z].
            metadata: Additional metadata.
        """
        self._index[asset_id] = {
            "name": name,
            "category": category,
            "source_uri": source_uri,
            "glb_path": glb_path,
            "sdf_path": sdf_path,
            "gltf_path": gltf_path,
            "dimensions": dimensions or {},
            "bbox_min": bbox_min,
            "bbox_max": bbox_max,
            "metadata": metadata or {},
        }
        self._save_index()
        console_logger.info(f"Added asset to catalog: {asset_id} ({name})")

    def get_asset(self, asset_id: str) -> dict | None:
        """Get asset info by ID.

        Args:
            asset_id: Asset identifier.

        Returns:
            Asset info dictionary or None if not found.
        """
        return self._index.get(asset_id)

    def remove_asset(self, asset_id: str) -> bool:
        """Remove an asset from the catalog.

        Does not delete the actual files.

        Args:
            asset_id: Asset identifier.

        Returns:
            True if asset was found and removed.
        """
        if asset_id in self._index:
            del self._index[asset_id]
            self._save_index()
            console_logger.info(f"Removed asset from catalog: {asset_id}")
            return True
        return False

    def search(self, query: str, category: str = "") -> list[dict]:
        """Search cached assets by name/category.

        Simple text matching for cached assets. For full semantic search,
        use the OmniverseClient.

        Args:
            query: Text to search for in asset names.
            category: Optional category filter.

        Returns:
            List of matching asset info dictionaries.
        """
        results = []
        query_lower = query.lower()
        query_words = query_lower.split()

        for asset_id, info in self._index.items():
            if category and info.get("category", "") != category:
                continue

            name_lower = info.get("name", "").lower()
            score = sum(1 for word in query_words if word in name_lower)

            if score > 0:
                results.append({
                    "id": asset_id,
                    "score": score / len(query_words),
                    **info,
                })

        results.sort(key=lambda x: x.get("score", 0), reverse=True)
        return results

    def list_all(self) -> list[dict]:
        """List all assets in the catalog.

        Returns:
            List of all asset info dictionaries with IDs.
        """
        return [{"id": k, **v} for k, v in self._index.items()]

    def list_by_category(self) -> dict[str, list[dict]]:
        """List all assets grouped by category.

        Returns:
            Dictionary mapping category names to lists of assets.
        """
        categories: dict[str, list[dict]] = {}
        for asset_id, info in self._index.items():
            cat = info.get("category", "uncategorized")
            if cat not in categories:
                categories[cat] = []
            categories[cat].append({"id": asset_id, **info})
        return categories

    @property
    def count(self) -> int:
        """Number of assets in the catalog."""
        return len(self._index)

"""Asset search and retrieval MCP tools.

Provides NVIDIA Omniverse USD integration for searching, retrieving,
and preparing 3D assets for scene placement. Connects to:
- NVIDIA USD Search API (cloud-hosted at build.nvidia.com)
- Local USD asset directories
- Cached asset catalog for previously downloaded assets
"""

import json
import logging
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from scenesmith_mcp.omniverse import AssetCatalog, OmniverseClient, USDConverter
from scenesmith_mcp.state import ServerState

console_logger = logging.getLogger(__name__)


def register_asset_tools(server: FastMCP, state: ServerState) -> None:
    """Register asset search/retrieval tools.

    Initializes the Omniverse client, USD converter, and asset catalog,
    then registers MCP tools for asset operations.
    """

    # Initialize Omniverse components lazily on first tool call.
    _omniverse: dict = {}

    def _get_omniverse_client() -> OmniverseClient:
        if "client" not in _omniverse:
            local_dir = None
            if state.output_dir:
                local_dir = state.output_dir / "usd_assets"
            _omniverse["client"] = OmniverseClient(
                local_asset_dir=local_dir,
            )
        return _omniverse["client"]

    def _get_converter() -> USDConverter:
        if "converter" not in _omniverse:
            output_dir = (
                state.output_dir / "converted_assets"
                if state.output_dir
                else Path.home() / ".cache" / "scenesmith" / "converted"
            )
            _omniverse["converter"] = USDConverter(
                output_dir=output_dir,
                blender_server=state.blender_server,
                collision_server=state.collision_server,
            )
        return _omniverse["converter"]

    def _get_catalog() -> AssetCatalog:
        if "catalog" not in _omniverse:
            catalog_dir = (
                state.output_dir / "asset_catalog"
                if state.output_dir
                else Path.home() / ".cache" / "scenesmith" / "catalog"
            )
            _omniverse["catalog"] = AssetCatalog(catalog_dir=catalog_dir)
        return _omniverse["catalog"]

    @server.tool()
    async def assets__search(
        query: str,
        category: str = "",
        max_results: int = 10,
    ) -> str:
        """Search for 3D assets by text description.

        Searches multiple sources for matching 3D models:
        1. NVIDIA USD Search API (if NVIDIA_API_KEY is set) - AI-powered
           natural language search across indexed SimReady assets
        2. Local USD asset directory - filename matching on downloaded assets
        3. Cached asset catalog - previously downloaded and prepared assets

        Args:
            query: Text description of the asset to find
                (e.g., "modern wooden desk", "ceramic vase").
            category: Optional category filter
                (e.g., "furniture", "decoration", "lighting").
            max_results: Maximum number of results to return.

        Returns:
            JSON list of matching assets with URIs, names, and metadata.
        """
        client = _get_omniverse_client()
        catalog = _get_catalog()

        # Search both remote/local USD files and cached prepared assets.
        usd_results = client.search(query, category, max_results)
        cached_results = catalog.search(query, category)

        # Merge results, prioritizing cached (ready-to-use) assets.
        all_results = []

        for cached in cached_results:
            all_results.append({
                **cached,
                "ready_for_placement": True,
                "source": "cached",
            })

        for usd in usd_results:
            all_results.append({
                **usd,
                "ready_for_placement": False,
                "source": usd.get("source", "usd_search"),
            })

        all_results = all_results[:max_results]

        console_logger.info(
            f"Asset search '{query}': {len(cached_results)} cached, "
            f"{len(usd_results)} USD results"
        )

        return json.dumps(all_results, indent=2, default=str)

    @server.tool()
    async def assets__retrieve(
        asset_uri: str,
        asset_name: str = "",
        target_dimensions: list[float] | None = None,
        object_type: str = "furniture",
        material: str = "mixed",
        mass_kg: float = 5.0,
    ) -> str:
        """Download, convert, and prepare a USD asset for scene placement.

        Full pipeline:
        1. Download USD from URI (local path, HTTP URL, or Omniverse)
        2. Convert USD -> GLB (via usd2gltf or Blender)
        3. Process GLB through SceneSmith pipeline:
           - GLB -> GLTF (Y-up standard)
           - Canonicalize mesh (Z-up for Drake)
           - Generate convex decomposition collision geometry
           - Generate Drake SDF
        4. Register in local asset catalog

        Args:
            asset_uri: URI of the asset (local path, HTTP URL, or Omniverse URI).
            asset_name: Human-readable name for the asset.
            target_dimensions: Optional [width, depth, height] in meters to scale to.
            object_type: Type of object: "furniture", "manipuland",
                "wall_mounted", or "ceiling_mounted".
            material: Material type for physics: "wood", "metal", "plastic",
                "ceramic", "glass", "fabric", or "mixed".
            mass_kg: Estimated mass in kilograms.

        Returns:
            JSON with asset details including paths and dimensions, ready
            for use with placement tools.
        """
        client = _get_omniverse_client()
        converter = _get_converter()
        catalog = _get_catalog()

        # Derive name from URI if not provided.
        if not asset_name:
            asset_name = Path(asset_uri).stem.replace("_", " ").title()

        # Step 1: Retrieve the USD file.
        usd_path = client.retrieve(asset_uri)
        if usd_path is None:
            return json.dumps({
                "status": "failed",
                "error": f"Could not retrieve asset: {asset_uri}",
            })

        # Step 2: Convert USD -> GLB.
        glb_path = converter.usd_to_glb(usd_path)
        if glb_path is None:
            return json.dumps({
                "status": "failed",
                "error": f"USD->GLB conversion failed for: {usd_path}",
            })

        # Step 3: Prepare for scene (GLB -> GLTF -> canonicalize -> SDF).
        result = converter.prepare_for_scene(
            glb_path=glb_path,
            asset_name=asset_name,
            target_dimensions=target_dimensions,
            object_type=object_type,
            material=material,
            mass_kg=mass_kg,
        )

        # Step 4: Register in catalog if successful.
        if result.get("status") in ("success", "partial"):
            import time

            asset_id = f"{asset_name.lower().replace(' ', '_')}_{int(time.time())}"
            catalog.add_asset(
                asset_id=asset_id,
                name=asset_name,
                category=object_type,
                source_uri=asset_uri,
                glb_path=str(glb_path),
                sdf_path=result.get("sdf_path"),
                gltf_path=result.get("gltf_path"),
                dimensions=result.get("dimensions"),
                bbox_min=result.get("bbox_min"),
                bbox_max=result.get("bbox_max"),
                metadata={"material": material, "mass_kg": mass_kg},
            )
            result["asset_id"] = asset_id

        console_logger.info(
            f"Asset retrieval complete: {asset_name} -> {result.get('status')}"
        )

        return json.dumps(result, indent=2, default=str)

    @server.tool()
    async def assets__list_cached() -> str:
        """List all locally cached assets ready for placement.

        Shows assets that have been previously downloaded and converted.
        These can be used directly with placement tools without re-downloading.

        Returns:
            JSON list of cached assets with IDs, names, dimensions,
            and file paths.
        """
        catalog = _get_catalog()
        all_assets = catalog.list_all()

        # Also check the existing AssetManager registry if available.
        if state.asset_manager:
            try:
                registered = state.asset_manager.list_available_assets()
                for a in registered:
                    all_assets.append({
                        "id": str(a.object_id),
                        "name": a.name,
                        "source": "asset_manager",
                        "dimensions": {
                            "width": float(a.bbox_max[0] - a.bbox_min[0])
                            if a.bbox_min is not None
                            else 0,
                            "depth": float(a.bbox_max[1] - a.bbox_min[1])
                            if a.bbox_min is not None
                            else 0,
                            "height": float(a.bbox_max[2] - a.bbox_min[2])
                            if a.bbox_min is not None
                            else 0,
                        },
                    })
            except Exception as e:
                console_logger.warning(f"Could not list AssetManager assets: {e}")

        return json.dumps(all_assets, indent=2, default=str)

    @server.tool()
    async def assets__get_info(asset_uri: str) -> str:
        """Get detailed metadata for a USD asset.

        Reads the USD file to extract scene graph information, bounding box,
        mesh/material counts, coordinate system, and scale.

        Args:
            asset_uri: URI or local path to the USD asset.

        Returns:
            JSON with asset metadata including dimensions, mesh count,
            up axis, and meters per unit.
        """
        client = _get_omniverse_client()
        info = client.get_asset_info(asset_uri)

        if info is None:
            return json.dumps({
                "status": "failed",
                "error": f"Could not read asset info: {asset_uri}",
            })

        return json.dumps(info, indent=2, default=str)

    console_logger.info("Registered asset tools (search, retrieve, list_cached, get_info)")

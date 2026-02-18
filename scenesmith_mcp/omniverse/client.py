"""NVIDIA Omniverse USD client for asset search and retrieval.

Provides the interface for searching 3D assets using:
1. NVIDIA USD Search NIM API (free API key from build.nvidia.com/nvidia/usdsearch)
2. Local asset directory scanning with text-based matching
3. NVIDIA public CDN for downloading assets with all textures

Setup:
    Option A (recommended) — Free NVIDIA API key:
        export NVIDIA_API_KEY="nvapi-..."
        Keys available at: https://build.nvidia.com/nvidia/usdsearch

    Option B — Download asset packs locally (no key needed):
        python scripts/download_assets.py

NVIDIA CDN URL:
    All SimReady assets are publicly downloadable (no auth) from:
    https://omniverse-content-production.s3.us-west-2.amazonaws.com/Assets/...

USD Search NIM Endpoint:
    POST https://ai.api.nvidia.com/v1/omniverse/nvidia/usdsearch
    Headers: Authorization: Bearer <nvapi-key>
    Note: uses requests.post() directly, NOT usd-search-client
          (usd-search-client is for self-hosted deployments only)
"""

import json
import logging
import os
import re
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

try:
    import requests as _requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

console_logger = logging.getLogger(__name__)

# NVIDIA public CDN — no auth required, all SimReady assets available here
NVIDIA_CDN_BASE = "https://omniverse-content-production.s3.us-west-2.amazonaws.com"

# NVIDIA USD Search NIM endpoint
NVIDIA_SEARCH_URL = "https://ai.api.nvidia.com/v1/omniverse/nvidia/usdsearch"

# Maps the S3 internal bucket to the public CDN
S3_PRIVATE_PREFIX = "s3://deepsearch-demo-content/"
S3_CDN_PREFIX = f"{NVIDIA_CDN_BASE}/"


def _s3_to_cdn(s3_uri: str) -> str:
    """Convert a private S3 URI to a public CDN HTTPS URL."""
    if s3_uri.startswith(S3_PRIVATE_PREFIX):
        return S3_CDN_PREFIX + s3_uri[len(S3_PRIVATE_PREFIX):]
    return s3_uri


class USDAsset:
    """Represents a USD asset from the NVIDIA catalog or local directory."""

    def __init__(
        self,
        uri: str,
        name: str,
        category: str = "",
        description: str = "",
        score: float = 0.0,
        thumbnail_b64: str | None = None,
        local_path: Path | None = None,
        source: str = "unknown",
        metadata: dict | None = None,
    ):
        self.uri = uri
        self.cdn_url = _s3_to_cdn(uri)
        self.name = name
        self.category = category
        self.description = description
        self.score = score
        self.thumbnail_b64 = thumbnail_b64  # base64 JPEG from search API
        self.local_path = local_path
        self.source = source
        self.metadata = metadata or {}

    def to_dict(self) -> dict:
        return {
            "uri": self.uri,
            "cdn_url": self.cdn_url,
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "score": self.score,
            "source": self.source,
            "local_path": str(self.local_path) if self.local_path else None,
            "has_thumbnail": self.thumbnail_b64 is not None,
            "metadata": self.metadata,
        }


class OmniverseClient:
    """Client for searching and retrieving USD assets from NVIDIA sources.

    Two backends (can be combined):
    1. NVIDIA USD Search NIM (cloud, needs free API key)
    2. Local asset directory (from download_assets.py)

    All found assets are downloadable for free from NVIDIA's public CDN.
    """

    def __init__(
        self,
        api_key: str | None = None,
        local_asset_dir: Path | None = None,
        cache_dir: Path | None = None,
    ):
        """Initialize the Omniverse client.

        Args:
            api_key: NVIDIA API key (nvapi-...).
                Falls back to NVIDIA_API_KEY environment variable.
                Get a free key at: https://build.nvidia.com/nvidia/usdsearch
            local_asset_dir: Path to local USD assets
                (from: python scripts/download_assets.py).
            cache_dir: Directory for downloaded assets.
        """
        self.api_key = api_key or os.environ.get("NVIDIA_API_KEY")
        self.local_asset_dir = local_asset_dir or self._find_default_local_dir()
        self.cache_dir = cache_dir or Path.home() / ".cache" / "scenesmith" / "usd_assets"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._local_index: list[USDAsset] = []
        self._local_index_built = False

        if self.api_key:
            console_logger.info("NVIDIA USD Search API key found — cloud search enabled")
        else:
            console_logger.info(
                "No NVIDIA_API_KEY set. Using local assets only.\n"
                "  → Get a free key at: https://build.nvidia.com/nvidia/usdsearch\n"
                "  → Or download packs: python scripts/download_assets.py"
            )

        if self.local_asset_dir and self.local_asset_dir.exists():
            console_logger.info(f"Local asset directory: {self.local_asset_dir}")

    def _find_default_local_dir(self) -> Path | None:
        """Find the default local asset directory if it exists."""
        default = Path.home() / ".cache" / "scenesmith" / "assets"
        return default if default.exists() else None

    # ── Search ─────────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        category: str = "",
        max_results: int = 10,
    ) -> list[dict]:
        """Search for USD assets by natural language description.

        Tries NVIDIA cloud search first, then local directory.

        Args:
            query: e.g. "modern wooden sofa", "stainless steel refrigerator"
            category: Optional filter (e.g. "furniture")
            max_results: Maximum number of results

        Returns:
            List of asset dicts with uri, name, cdn_url, thumbnail, etc.
        """
        results: list[USDAsset] = []

        # 1. NVIDIA USD Search NIM (cloud)
        if self.api_key:
            cloud = self._search_nvidia_nim(query, max_results)
            results.extend(cloud)

        # 2. Local directory
        if self.local_asset_dir and self.local_asset_dir.exists():
            local = self._search_local(query, category, max_results)
            results.extend(local)

        # Deduplicate by URI
        seen: set[str] = set()
        unique = []
        for r in results:
            if r.uri not in seen:
                seen.add(r.uri)
                unique.append(r)

        return [r.to_dict() for r in unique[:max_results]]

    def _search_nvidia_nim(self, query: str, limit: int = 10) -> list[USDAsset]:
        """Search using NVIDIA USD Search NIM API.

        Endpoint: POST https://ai.api.nvidia.com/v1/omniverse/nvidia/usdsearch
        Auth: Authorization: Bearer <nvapi-key>
        Note: Uses requests directly — usd-search-client is for self-hosted only.
        """
        if not self.api_key:
            return []

        try:
            payload = json.dumps({
                "description": query,
                "file_extension_include": "usd*",
                "return_images": "true",
                "return_metadata": "true",
                "cutoff_threshold": "1.05",
                "limit": str(limit),
            }).encode()

            req = urllib.request.Request(
                NVIDIA_SEARCH_URL,
                data=payload,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
            )
            resp = urllib.request.urlopen(req, timeout=15).read()
            hits = json.loads(resp)

            assets = []
            for hit in hits:
                s3_uri = hit.get("url", "")
                name = Path(s3_uri).stem if s3_uri else "unknown"
                assets.append(USDAsset(
                    uri=s3_uri,
                    name=name,
                    score=hit.get("score", 0.0),
                    thumbnail_b64=hit.get("image"),
                    metadata=hit.get("metadata", {}),
                    source="nvidia_nim",
                ))

            console_logger.info(
                f"NVIDIA USD Search: {len(assets)} results for '{query}'"
            )
            return assets

        except Exception as e:
            console_logger.warning(f"NVIDIA USD Search error: {e}")
            return []

    def _search_local(
        self, query: str, category: str = "", max_results: int = 10
    ) -> list[USDAsset]:
        """Search local downloaded asset directory by filename matching."""
        if not self._local_index_built:
            self._build_local_index()

        query_words = query.lower().split()
        results = []

        for asset in self._local_index:
            if category and category.lower() not in asset.category.lower():
                continue
            name_lower = asset.name.lower()
            score = sum(1 for w in query_words if w in name_lower) / len(query_words)
            if score > 0:
                asset.score = score
                results.append(asset)

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:max_results]

    def _build_local_index(self) -> None:
        """Index all USD files in the local asset directory."""
        self._local_index = []
        if not self.local_asset_dir:
            return

        for path in self.local_asset_dir.rglob("*.usd"):
            relative = path.relative_to(self.local_asset_dir)
            category = relative.parts[0] if len(relative.parts) > 1 else "uncategorized"
            self._local_index.append(USDAsset(
                uri=str(path),
                name=path.stem,
                category=category,
                description=f"Local: {path.stem}",
                local_path=path,
                source="local",
            ))

        self._local_index_built = True
        console_logger.info(
            f"Indexed {len(self._local_index)} local USD assets from {self.local_asset_dir}"
        )

    # ── Download ───────────────────────────────────────────────────────────────

    def retrieve(
        self,
        asset_uri: str,
        target_dir: Path | None = None,
    ) -> Path | None:
        """Download a USD asset with all its textures and sub-files.

        Assets on the NVIDIA CDN are publicly downloadable (no auth needed).
        Downloads the entire asset folder: USD files + textures + JSON metadata.

        Args:
            asset_uri: S3 URI from search results (s3://deepsearch-demo-content/...)
                       or a CDN URL (https://omniverse-content-production.s3...)
            target_dir: Where to save the asset (default: cache_dir)

        Returns:
            Path to the main .usd file, or None on failure.
        """
        # Already local
        local = Path(asset_uri)
        if local.exists():
            return local

        # Convert S3 → CDN URL
        cdn_url = _s3_to_cdn(asset_uri)
        if not cdn_url.startswith("http"):
            console_logger.warning(f"Cannot retrieve: {asset_uri}")
            return None

        # Extract asset path relative to CDN base
        asset_path = cdn_url.replace(S3_CDN_PREFIX, "")
        asset_folder = "/".join(asset_path.split("/")[:-1])  # parent folder
        asset_name = asset_path.split("/")[-2]  # folder name = asset name

        dest_dir = (target_dir or self.cache_dir) / asset_name

        # Check cache
        usd_files = list(dest_dir.glob("*.usd")) if dest_dir.exists() else []
        if usd_files:
            console_logger.info(f"Asset already cached: {dest_dir}")
            return usd_files[0]

        # Download entire asset folder
        console_logger.info(f"Downloading asset '{asset_name}' from NVIDIA CDN...")
        return self._download_asset_folder(asset_folder, asset_name, dest_dir)

    def _list_s3_folder(self, folder_path: str) -> list[str]:
        """List all files in an NVIDIA CDN folder using S3 list API."""
        list_url = f"{NVIDIA_CDN_BASE}?list-type=2&prefix={folder_path}/"
        try:
            resp = urllib.request.urlopen(list_url, timeout=10).read()
            root = ET.fromstring(resp)
            ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
            keys = [c.text for c in root.findall(".//s3:Key", ns) if c.text]
            return keys
        except Exception as e:
            console_logger.warning(f"S3 listing error for {folder_path}: {e}")
            return []

    def _download_asset_folder(
        self, folder_path: str, asset_name: str, dest_dir: Path
    ) -> Path | None:
        """Download all files in an asset folder from the NVIDIA CDN."""
        keys = self._list_s3_folder(folder_path)

        if not keys:
            console_logger.warning(f"No files found in CDN folder: {folder_path}")
            return None

        # Filter: skip thumbnail previews (.thumbs/), download real files
        real_files = [k for k in keys if "/.thumbs/" not in k and k.split("/")[-1]]

        console_logger.info(f"  Downloading {len(real_files)} files...")
        dest_dir.mkdir(parents=True, exist_ok=True)

        main_usd: Path | None = None

        for key in real_files:
            fname = key.split("/")[-1]
            if not fname:
                continue

            # Preserve subfolder structure (e.g., Textures/)
            relative_parts = key.split(folder_path + "/", 1)
            rel_path = relative_parts[1] if len(relative_parts) > 1 else fname
            dest_file = dest_dir / rel_path
            dest_file.parent.mkdir(parents=True, exist_ok=True)

            url = f"{NVIDIA_CDN_BASE}/{key}"
            try:
                urllib.request.urlretrieve(url, dest_file)
                if fname == f"{asset_name}.usd":
                    main_usd = dest_file
            except Exception as e:
                console_logger.warning(f"  Failed to download {fname}: {e}")

        if not main_usd:
            usd_files = list(dest_dir.glob("*.usd"))
            main_usd = usd_files[0] if usd_files else None

        if main_usd:
            console_logger.info(f"  Asset ready: {main_usd}")
        else:
            console_logger.warning(f"  No main USD file found in {dest_dir}")

        return main_usd

    # ── Asset info ─────────────────────────────────────────────────────────────

    def get_asset_info(self, asset_uri: str) -> dict | None:
        """Get metadata (dimensions, mesh count) for a USD asset.

        Reads the file with pxr (usd-core) if available.
        """
        local_path = self.retrieve(asset_uri) if not Path(asset_uri).exists() else Path(asset_uri)
        if not local_path or not local_path.exists():
            return None

        try:
            from pxr import Usd, UsdGeom

            stage = Usd.Stage.Open(str(local_path))
            if not stage:
                return None

            up_axis = str(UsdGeom.GetStageUpAxis(stage))
            mpu = UsdGeom.GetStageMetersPerUnit(stage)
            meshes = sum(1 for p in stage.Traverse() if p.IsA(UsdGeom.Mesh))

            bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"])
            bbox = bbox_cache.ComputeWorldBound(stage.GetPseudoRoot())
            bbox_range = bbox.ComputeAlignedBox()
            dims = {}
            if not bbox_range.IsEmpty():
                sz = bbox_range.GetMax() - bbox_range.GetMin()
                dims = {
                    "width": round(float(sz[0]) * mpu, 3),
                    "depth": round(float(sz[1]) * mpu, 3),
                    "height": round(float(sz[2]) * mpu, 3),
                }

            return {
                "local_path": str(local_path),
                "up_axis": up_axis,
                "meters_per_unit": mpu,
                "mesh_count": meshes,
                "dimensions_m": dims,
            }

        except ImportError:
            return {"local_path": str(local_path)}
        except Exception as e:
            console_logger.warning(f"Error reading USD info: {e}")
            return {"local_path": str(local_path)}

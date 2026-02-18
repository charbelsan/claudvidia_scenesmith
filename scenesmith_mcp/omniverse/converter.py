"""USD to GLB/SDF conversion pipeline.

Converts NVIDIA Omniverse USD assets into formats compatible with
the SceneSmith execution layer (GLB for rendering, SDF for Drake physics).

Conversion pipeline:
1. USD → GLB (via usd2gltf or Blender fallback)
2. GLB → GLTF Y-up (via existing convert_glb_to_gltf)
3. GLTF → Canonicalized Z-up GLTF (via existing canonicalize_mesh)
4. GLTF → Convex decomposition (via existing CoACD server)
5. GLTF + Collision → Drake SDF (via existing generate_drake_sdf)

Reuses existing SceneSmith utilities:
- scenesmith/agent_utils/mesh_canonicalization.py
- scenesmith/agent_utils/sdf_generator.py
- scenesmith/agent_utils/mesh_utils.py
- scenesmith/agent_utils/convex_decomposition_server/
"""

import logging
import subprocess
import time
from pathlib import Path

console_logger = logging.getLogger(__name__)


class USDConverter:
    """Converts USD assets to GLB and SDF formats.

    Uses usd2gltf for fast USD→GLB conversion, with Blender as a fallback.
    Then feeds into the existing SceneSmith mesh processing pipeline for
    canonicalization, collision geometry, and SDF generation.
    """

    def __init__(
        self,
        output_dir: Path,
        blender_server=None,
        collision_server=None,
    ):
        """Initialize converter with output directory and servers.

        Args:
            output_dir: Directory for converted assets.
            blender_server: BlenderServer for mesh operations and Blender fallback.
            collision_server: ConvexDecompositionServer for collision geometry.
        """
        self.output_dir = output_dir
        self.blender_server = blender_server
        self.collision_server = collision_server
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def usd_to_glb(self, usd_path: Path, output_path: Path | None = None) -> Path | None:
        """Convert USD file to GLB format.

        Tries usd2gltf first (fast, pure Python), then falls back to
        Blender headless conversion.

        Args:
            usd_path: Path to USD/USDA/USDC file.
            output_path: Output GLB path (default: same name with .glb extension).

        Returns:
            Path to converted GLB file, or None on failure.
        """
        if output_path is None:
            output_path = self.output_dir / f"{usd_path.stem}.glb"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Try usd2gltf first.
        result = self._convert_via_usd2gltf(usd_path, output_path)
        if result and result.exists():
            return result

        # Fall back to Blender.
        result = self._convert_via_blender(usd_path, output_path)
        if result and result.exists():
            return result

        console_logger.error(f"All USD→GLB conversion methods failed for {usd_path}")
        return None

    def _convert_via_usd2gltf(self, usd_path: Path, output_path: Path) -> Path | None:
        """Convert USD to GLB using the usd2gltf package.

        Args:
            usd_path: Input USD file.
            output_path: Output GLB file.

        Returns:
            Path to output file or None on failure.
        """
        try:
            result = subprocess.run(
                ["usd2gltf", "-i", str(usd_path), "-o", str(output_path)],
                capture_output=True,
                text=True,
                timeout=120,
            )

            if result.returncode == 0 and output_path.exists():
                console_logger.info(f"usd2gltf conversion successful: {output_path}")
                return output_path

            console_logger.warning(
                f"usd2gltf failed (rc={result.returncode}): {result.stderr}"
            )
            return None

        except FileNotFoundError:
            console_logger.info(
                "usd2gltf CLI not found. "
                "Install with: pip install usd2gltf"
            )
            return None
        except subprocess.TimeoutExpired:
            console_logger.warning("usd2gltf conversion timed out")
            return None
        except Exception as e:
            console_logger.warning(f"usd2gltf error: {e}")
            return None

    def _convert_via_blender(self, usd_path: Path, output_path: Path) -> Path | None:
        """Convert USD to GLB using Blender's import/export (headless).

        Uses the BlenderServer if available, otherwise tries direct bpy import.

        Args:
            usd_path: Input USD file.
            output_path: Output GLB file.

        Returns:
            Path to output file or None on failure.
        """
        if self.blender_server is not None:
            try:
                result = self.blender_server.convert_usd_to_glb(
                    str(usd_path), str(output_path)
                )
                if result and output_path.exists():
                    console_logger.info(
                        f"Blender server USD→GLB conversion successful: {output_path}"
                    )
                    return output_path
            except (AttributeError, Exception) as e:
                console_logger.info(
                    f"BlenderServer.convert_usd_to_glb not available: {e}"
                )

        # Try direct bpy as last resort.
        try:
            import bpy

            bpy.ops.wm.read_factory_settings(use_empty=True)
            bpy.ops.wm.usd_open(filepath=str(usd_path))
            bpy.ops.export_scene.gltf(
                filepath=str(output_path),
                export_format="GLB",
            )

            if output_path.exists():
                console_logger.info(
                    f"Blender bpy USD→GLB conversion successful: {output_path}"
                )
                return output_path

        except ImportError:
            console_logger.info("bpy not available for USD conversion")
        except Exception as e:
            console_logger.warning(f"Blender bpy conversion error: {e}")

        return None

    def prepare_for_scene(
        self,
        glb_path: Path,
        asset_name: str,
        target_dimensions: list[float] | None = None,
        object_type: str = "furniture",
        up_axis: str = "+Z",
        front_axis: str = "+Y",
        material: str = "mixed",
        mass_kg: float = 5.0,
    ) -> dict:
        """Prepare a GLB mesh for scene placement using existing SceneSmith pipeline.

        Runs:
        1. GLB → GLTF conversion (Y-up standard)
        2. Canonicalization (orient to Z-up for Drake)
        3. Optional scaling to target dimensions
        4. Convex decomposition for collision geometry
        5. Drake SDF generation

        Args:
            glb_path: Path to GLB mesh file.
            asset_name: Name for the asset (used in file naming).
            target_dimensions: Optional [width, depth, height] in meters.
            object_type: "furniture", "manipuland", "wall_mounted", or "ceiling_mounted".
            up_axis: Detected up axis of the source mesh.
            front_axis: Detected front axis of the source mesh.
            material: Material type for physics (e.g., "wood", "metal").
            mass_kg: Estimated mass in kilograms.

        Returns:
            Dictionary with paths to processed files and metadata.
        """
        timestamp = int(time.time())
        safe_name = asset_name.replace(" ", "_").lower()
        sdf_dir = self.output_dir / "sdf" / f"{safe_name}_{timestamp}"
        sdf_dir.mkdir(parents=True, exist_ok=True)

        result = {
            "asset_name": asset_name,
            "glb_path": str(glb_path),
            "sdf_path": None,
            "gltf_path": None,
            "collision_paths": [],
            "dimensions": None,
            "scale_factor": 1.0,
            "status": "pending",
        }

        # Step 1: GLB → GLTF (Y-up).
        gltf_path = sdf_dir / f"{safe_name}.gltf"
        try:
            from scenesmith.agent_utils.mesh_utils import convert_glb_to_gltf

            convert_glb_to_gltf(glb_path, gltf_path, export_yup=True)
            result["gltf_path"] = str(gltf_path)
            console_logger.info(f"GLB→GLTF conversion: {gltf_path}")
        except Exception as e:
            console_logger.error(f"GLB→GLTF conversion failed: {e}")
            result["status"] = "failed"
            result["error"] = f"GLB→GLTF conversion failed: {e}"
            return result

        # Step 2: Canonicalize mesh.
        if self.blender_server is not None:
            try:
                from scenesmith.agent_utils.mesh_canonicalization import (
                    canonicalize_mesh,
                )
                from scenesmith.agent_utils.room import ObjectType

                object_type_enum = {
                    "furniture": ObjectType.FURNITURE,
                    "manipuland": ObjectType.MANIPULAND,
                    "wall_mounted": ObjectType.WALL_MOUNTED,
                    "ceiling_mounted": ObjectType.CEILING_MOUNTED,
                }.get(object_type, ObjectType.FURNITURE)

                canonicalize_mesh(
                    input_path=gltf_path,
                    output_path=gltf_path,  # Overwrite in place.
                    up_axis=up_axis,
                    front_axis=front_axis,
                    blender_server=self.blender_server,
                    object_type=object_type_enum,
                )
                console_logger.info(f"Mesh canonicalized: {gltf_path}")
            except Exception as e:
                console_logger.warning(f"Canonicalization failed (continuing): {e}")

        # Step 3: Scale to target dimensions if specified.
        scale_factor = 1.0
        if target_dimensions and len(target_dimensions) == 3:
            try:
                from scenesmith.agent_utils.mesh_utils import (
                    load_mesh_as_trimesh,
                )

                mesh = load_mesh_as_trimesh(gltf_path, force_merge=True)
                if mesh is not None:
                    current_extents = mesh.bounding_box.extents
                    target_w, target_d, target_h = target_dimensions
                    # Scale by the dimension that needs the least change.
                    scales = []
                    if current_extents[0] > 0:
                        scales.append(target_w / current_extents[0])
                    if current_extents[1] > 0:
                        scales.append(target_d / current_extents[1])
                    if current_extents[2] > 0:
                        scales.append(target_h / current_extents[2])
                    if scales:
                        scale_factor = min(scales)
                        result["scale_factor"] = scale_factor
                        console_logger.info(f"Scale factor: {scale_factor:.4f}")
            except Exception as e:
                console_logger.warning(f"Scaling computation failed: {e}")

        # Step 4: Generate convex decomposition collision geometry.
        collision_pieces = []
        if self.collision_server is not None:
            try:
                from scenesmith.agent_utils.mesh_utils import (
                    load_mesh_as_trimesh,
                )

                mesh = load_mesh_as_trimesh(gltf_path, force_merge=True)
                if mesh is not None:
                    collision_pieces = self.collision_server.decompose(mesh)
                    console_logger.info(
                        f"Generated {len(collision_pieces)} collision pieces"
                    )
            except Exception as e:
                console_logger.warning(f"Collision generation failed: {e}")

        # Step 5: Generate Drake SDF.
        try:
            from scenesmith.agent_utils.sdf_generator import generate_drake_sdf

            sdf_path = sdf_dir / f"{safe_name}.sdf"

            # Build physics analysis data.
            physics_data = {
                "up_axis": up_axis,
                "front_axis": front_axis,
                "material": material,
                "mass_kg": mass_kg,
            }

            generate_drake_sdf(
                visual_mesh_path=gltf_path,
                collision_pieces=collision_pieces,
                physics_analysis=physics_data,
                output_path=sdf_path,
                scale_factor=scale_factor,
            )

            if sdf_path.exists():
                result["sdf_path"] = str(sdf_path)
                console_logger.info(f"SDF generated: {sdf_path}")
        except Exception as e:
            console_logger.warning(f"SDF generation failed: {e}")

        # Compute final bounding box.
        try:
            from scenesmith.agent_utils.mesh_utils import load_mesh_as_trimesh

            mesh = load_mesh_as_trimesh(gltf_path, force_merge=True)
            if mesh is not None:
                bbox = mesh.bounding_box
                result["dimensions"] = {
                    "width": float(bbox.extents[0]),
                    "depth": float(bbox.extents[1]),
                    "height": float(bbox.extents[2]),
                }
                result["bbox_min"] = [float(v) for v in bbox.bounds[0]]
                result["bbox_max"] = [float(v) for v in bbox.bounds[1]]
        except Exception as e:
            console_logger.warning(f"Bounding box computation failed: {e}")

        result["status"] = "success" if result.get("sdf_path") else "partial"
        return result

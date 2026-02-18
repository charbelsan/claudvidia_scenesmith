#!/usr/bin/env python3
"""Create a home scene with living room and kitchen using SceneSmith pipeline.

Stages:
  1. Floor Plan  - rooms, doors, windows via FloorPlanTools
  2. Room Geometry - minimal RoomGeometry stubs (length/width for bounds-checking)
  3. Furniture   - box-primitive assets placed via FurnitureTools
  4. Summary     - scene state report

Run with:
  .venv311/bin/python create_scene.py
"""

import json
import logging
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────

def _furniture_cfg():
    """Build DictConfig matching base_furniture_agent.yaml placement_noise structure."""
    from omegaconf import OmegaConf
    return OmegaConf.create({
        "placement_noise": {
            "mode": "perfect",
            "natural_profile": {
                "position_xy_std_meters": 0.001,
                "rotation_yaw_std_degrees": 0.1,
            },
            "perfect_profile": {
                "position_xy_std_meters": 0.001,
                "rotation_yaw_std_degrees": 0.1,
            },
        },
        "loop_detection": {
            "enabled": False,       # disable for scripted placement
            "max_repeated_attempts": 25,
            "tracking_window": 30,
        },
        "snap_to_object": {
            "voxel_pitch_meters": 0.03,
            "min_vertices_threshold": 10000,
            "circular_detection_volume_ratio_threshold": 0.80,
            "iterative_snap_step_m": 0.01,
            "max_snap_distance_m": 10.0,
            "snap_margin_m": 0.01,
            "max_sample_vertices": 2000,
        },
    })


def _make_room_geometry(room_id: str, length: float, width: float,
                        output_dir: Path, wall_height: float = 2.7):
    """Create a minimal RoomGeometry stub for furniture-stage bounds checking."""
    from scenesmith.agent_utils.house import RoomGeometry
    from scenesmith.agent_utils.room import ObjectType, SceneObject, UniqueID
    from pydrake.all import RigidTransform

    geo_dir = output_dir / "geometry" / room_id
    geo_dir.mkdir(parents=True, exist_ok=True)
    sdf_path = geo_dir / "room_geometry.sdf"

    # Minimal SDF (Drake doesn't need to load it for pure Python placement)
    sdf_path.write_text(f"""<?xml version="1.0"?>
<sdf version="1.7">
  <model name="room_geometry_{room_id}">
    <link name="body">
      <collision name="floor"><geometry>
          <box><size>{length} {width} 0.1</size></box>
      </geometry></collision>
    </link>
  </model>
</sdf>""")

    tree = ET.parse(str(sdf_path))

    floor_obj = SceneObject(
        object_id=UniqueID(f"floor_{room_id}"),
        object_type=ObjectType.FLOOR,
        name="Floor",
        description="Room floor",
        transform=RigidTransform(),
        geometry_path=sdf_path,
        sdf_path=sdf_path,
        bbox_min=np.array([-length / 2, -width / 2, -0.1]),
        bbox_max=np.array([length / 2, width / 2, 0.0]),
        immutable=True,
    )

    return RoomGeometry(
        sdf_tree=tree,
        sdf_path=sdf_path,
        walls=[],
        floor=floor_obj,
        wall_normals={},
        width=width,
        length=length,
        wall_height=wall_height,
        wall_thickness=0.05,
        openings=[],
    )


def _make_room_scene(room_id: str, room_type: str, room_geometry, scene_dir: Path):
    """Create a RoomScene for a room."""
    from scenesmith.agent_utils.room import RoomScene
    scene_dir.mkdir(parents=True, exist_ok=True)
    return RoomScene(
        room_geometry=room_geometry,
        scene_dir=scene_dir,
        room_id=room_id,
        room_type=room_type,
        text_description=f"A {room_type.replace('_', ' ')}",
    )


def _create_box_asset(name: str, description: str,
                      width: float, depth: float, height: float,
                      assets_dir: Path):
    """Create a box-primitive furniture asset (SDF + SceneObject)."""
    from scenesmith.agent_utils.room import ObjectType, SceneObject, UniqueID
    from pydrake.all import RigidTransform

    asset_dir = assets_dir / name
    asset_dir.mkdir(parents=True, exist_ok=True)
    sdf_path = asset_dir / f"{name}.sdf"

    sdf_path.write_text(f"""<?xml version="1.0"?>
<sdf version="1.7" xmlns:drake="drake.mit.edu">
  <model name="{name}">
    <link name="{name}_body">
      <inertial>
        <mass>10.0</mass>
        <inertia><ixx>1</ixx><ixy>0</ixy><ixz>0</ixz>
                 <iyy>1</iyy><iyz>0</iyz><izz>1</izz></inertia>
      </inertial>
      <visual name="visual">
        <pose>0 0 {height/2} 0 0 0</pose>
        <geometry><box><size>{width} {depth} {height}</size></box></geometry>
      </visual>
      <collision name="collision">
        <pose>0 0 {height/2} 0 0 0</pose>
        <geometry><box><size>{width} {depth} {height}</size></box></geometry>
      </collision>
    </link>
  </model>
</sdf>""")

    return SceneObject(
        object_id=UniqueID(name),
        object_type=ObjectType.FURNITURE,
        name=name.replace("_", " ").title(),
        description=description,
        transform=RigidTransform(),
        geometry_path=sdf_path,
        sdf_path=sdf_path,
        # Bottom of object at z=0 (sits on floor)
        bbox_min=np.array([-width / 2, -depth / 2, 0.0]),
        bbox_max=np.array([width / 2, depth / 2, height]),
    )


class _StubAssetManager:
    """Minimal AssetManager wrapper over AssetRegistry for scripted scenes."""

    def __init__(self, registry):
        self._reg = registry

    def get_asset_by_id(self, asset_id):
        from scenesmith.agent_utils.room import UniqueID
        return self._reg.get(UniqueID(str(asset_id)))

    def list_available_assets(self):
        return self._reg.list_all()


# ─────────────────────────────────────────────────────────────────────────────
# Main scene creation
# ─────────────────────────────────────────────────────────────────────────────

def main():
    from omegaconf import OmegaConf
    from scenesmith_mcp.server import _default_config, state
    from scenesmith_mcp.lifecycle import (
        initialize_state,
        start_blender_server,
        start_collision_server,
    )
    from scenesmith_mcp.config import load_config

    # ── Load config ──────────────────────────────────────────────────────────
    try:
        cfg = load_config()
    except Exception:
        cfg = OmegaConf.create(_default_config())

    output_dir = Path("outputs/home_scene")
    initialize_state(state, cfg, output_dir=output_dir)

    # ── Infrastructure (non-fatal) ────────────────────────────────────────────
    for fn, label in [(start_blender_server, "BlenderServer"),
                      (start_collision_server, "ConvexDecompositionServer")]:
        try:
            fn(state)
            logger.info(f"{label} started")
        except Exception as e:
            logger.warning(f"{label} not available: {e}")

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 1: FLOOR PLAN
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("STAGE 1: FLOOR PLAN")
    logger.info("=" * 60)

    from scenesmith_mcp.tools.workflow import _init_floor_plan_stage
    logger.info(_init_floor_plan_stage(state))

    fp_tools = state.floor_plan_tools

    # Create rooms
    room_specs_json = json.dumps([
        {
            "name": "living_room",
            "type": "living_room",
            "width": 5.0,
            "depth": 4.0,
            "prompt": "A cozy living room with warm lighting and a TV area",
        },
        {
            "name": "kitchen",
            "type": "kitchen",
            "width": 4.0,
            "depth": 3.5,
            "prompt": "A modern open kitchen with dining area and white countertops",
        },
    ])

    result = fp_tools._generate_room_specs_impl(room_specs_json)
    logger.info(f"Rooms created: {str(result)[:300]}")

    # Doors
    for wall, label in [("A", "interior door (living_room ↔ kitchen)"),
                         ("D", "exterior door (living_room south)")]:
        try:
            fp_tools._add_door_impl(wall, "center", None, None)
            logger.info(f"Added {label}")
        except Exception as e:
            logger.warning(f"  Door {wall} failed: {e}")

    # Windows
    for wall, label in [("B", "living room north window"),
                         ("E", "kitchen north window")]:
        try:
            fp_tools._add_window_impl(wall, "center", None, None, None)
            logger.info(f"Added {label}")
        except Exception as e:
            logger.warning(f"  Window {wall} failed: {e}")

    # Validate
    try:
        val = fp_tools._validate_impl()
        logger.info(f"Floor plan validation: {val}")
    except Exception as e:
        logger.warning(f"Validation: {e}")

    # ASCII render
    try:
        print("\n" + "─" * 50)
        print("FLOOR PLAN:")
        print(fp_tools._render_ascii_impl())
        print("─" * 50 + "\n")
    except Exception as e:
        logger.warning(f"ASCII render: {e}")

    state.save_checkpoint("after_floor_plan")

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 2: FURNITURE
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("STAGE 2: FURNITURE")
    logger.info("=" * 60)

    from scenesmith.agent_utils.asset_registry import AssetRegistry
    from scenesmith.furniture_agents.tools.furniture_tools import FurnitureTools
    from scenesmith.furniture_agents.tools.scene_tools import SceneTools

    furn_cfg = _furniture_cfg()

    # Determine actual room dimensions from placed_rooms
    layout = state.house_layout
    room_dims = {}
    for pr in layout.placed_rooms:
        room_dims[pr.room_id] = (pr.width, pr.depth)   # (X=length, Y=width)
        logger.info(f"  {pr.room_id}: {pr.width:.1f}m × {pr.depth:.1f}m (placed)")

    # Fallback if placed_rooms not populated
    if "living_room" not in room_dims:
        room_dims["living_room"] = (5.0, 4.0)
    if "kitchen" not in room_dims:
        room_dims["kitchen"] = (4.0, 3.5)

    assets_dir = output_dir / "assets"

    # ── Furniture catalogs ───────────────────────────────────────────────────
    # (name, description, width, depth, height)
    living_room_items = [
        ("sofa",         "3-seat fabric sofa",          2.2, 0.9,  0.85),
        ("tv_stand",     "wooden TV stand with shelves", 1.5, 0.4,  0.5),
        ("coffee_table", "rectangular coffee table",     1.2, 0.6,  0.45),
        ("armchair",     "upholstered armchair",         0.8, 0.8,  0.85),
        ("floor_lamp",   "tall floor lamp",              0.3, 0.3,  1.6),
    ]

    kitchen_items = [
        ("dining_table",   "rectangular wooden dining table",   1.6, 0.9,  0.75),
        ("dining_chair_1", "wooden dining chair",               0.45, 0.5, 0.9),
        ("dining_chair_2", "wooden dining chair",               0.45, 0.5, 0.9),
        ("dining_chair_3", "wooden dining chair",               0.45, 0.5, 0.9),
        ("dining_chair_4", "wooden dining chair",               0.45, 0.5, 0.9),
        ("kitchen_island", "kitchen island / counter",          1.8, 0.6,  0.9),
    ]

    room_furniture = {
        "living_room": living_room_items,
        "kitchen": kitchen_items,
    }

    # ── Placement layouts (x, y, yaw_degrees) ───────────────────────────────
    # Coords: X=right, Y=forward, origin=room center, yaw CCW from east
    living_room_placements = {
        "sofa":         (-0.5,  -1.4,   0.0),   # south wall, facing north
        "tv_stand":     (-0.5,   1.5, 180.0),   # north wall
        "coffee_table": (-0.5,  -0.3,   0.0),   # in front of sofa
        "armchair":     ( 1.6,  -0.3,  90.0),   # east side, facing west
        "floor_lamp":   ( 1.7,  -1.3,   0.0),   # corner near armchair
    }

    kitchen_placements = {
        "dining_table":   ( 0.0,   0.3,   0.0),
        "dining_chair_1": ( 0.0,  -0.65, 180.0),  # south of table
        "dining_chair_2": ( 0.0,   1.25,   0.0),  # north of table
        "dining_chair_3": (-0.95,  0.3,   90.0),  # west of table
        "dining_chair_4": ( 0.95,  0.3,  270.0),  # east of table
        "kitchen_island": ( 0.0,  -1.3,   0.0),   # south side
    }

    room_placements = {
        "living_room": living_room_placements,
        "kitchen": kitchen_placements,
    }

    scene_results = {}

    for room_id in ["living_room", "kitchen"]:
        length, width = room_dims[room_id]
        logger.info(f"\n── Furnishing {room_id} ({length:.1f}m × {width:.1f}m) ──")

        room_dir = output_dir / "scene" / room_id
        room_geometry = _make_room_geometry(room_id, length, width,
                                            output_dir=output_dir,
                                            wall_height=2.7)
        scene = _make_room_scene(room_id,
                                 "living_room" if room_id == "living_room" else "kitchen",
                                 room_geometry, room_dir)

        # Build asset registry
        registry = AssetRegistry()
        for item in room_furniture[room_id]:
            name, desc, w, d, h = item
            asset = _create_box_asset(name, desc, w, d, h, assets_dir)
            registry.register(asset)

        ft = FurnitureTools(
            scene=scene,
            asset_manager=_StubAssetManager(registry),
            cfg=furn_cfg,
        )

        placements = room_placements[room_id]
        placed, failed = 0, 0
        for asset_id, (x, y, yaw) in placements.items():
            result = ft._add_furniture_to_scene_impl(
                asset_id=asset_id, x=x, y=y, z=0.0,
                roll=0.0, pitch=0.0, yaw=yaw,
            )
            # result is a JSON string; check success field
            try:
                r = json.loads(result)
                ok = r.get("success", False)
            except Exception:
                ok = "success" in str(result).lower()

            if ok:
                placed += 1
                logger.info(f"  ✓ {asset_id} @ ({x:.2f}, {y:.2f}) yaw={yaw}°")
            else:
                failed += 1
                logger.warning(f"  ✗ {asset_id}: {str(result)[:120]}")

        logger.info(f"  Placed {placed}/{placed+failed} items")

        # Store scene state
        scene_results[room_id] = {
            "scene": scene,
            "objects": list(scene.objects.keys()),
            "dims": (length, width),
        }

        # Store primary room as state.scene
        if room_id == "living_room":
            state.scene = scene

    # ══════════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("SCENE CREATION SUMMARY")
    print("=" * 60)
    for room_id, info in scene_results.items():
        length, width = info["dims"]
        n_obj = len(info["objects"])
        print(f"\n  {room_id.upper()} ({length:.1f}m × {width:.1f}m)")
        print(f"    Objects placed: {n_obj}")
        for oid in info["objects"]:
            obj = info["scene"].objects[oid]
            pos = obj.transform.translation()
            print(f"      • {obj.name} @ ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")

    print(f"\nOutput directory: {output_dir.absolute()}")
    print("=" * 60)

    # Save registry for each room
    for room_id, info in scene_results.items():
        reg_path = output_dir / "scene" / room_id / "assets.json"
        # Simple save
        data = {}
        for oid, obj in info["scene"].objects.items():
            pos = obj.transform.translation()
            from pydrake.all import RollPitchYaw
            rpy = RollPitchYaw(obj.transform.rotation())
            data[str(oid)] = {
                "name": obj.name,
                "description": obj.description,
                "position": pos.tolist(),
                "yaw_degrees": float(rpy.yaw_angle() * 180 / 3.14159),
                "bbox": {
                    "min": obj.bbox_min.tolist() if obj.bbox_min is not None else None,
                    "max": obj.bbox_max.tolist() if obj.bbox_max is not None else None,
                },
            }
        reg_path.parent.mkdir(parents=True, exist_ok=True)
        reg_path.write_text(json.dumps(data, indent=2))
        logger.info(f"Saved scene to {reg_path}")

    logger.info("Done!")


if __name__ == "__main__":
    main()

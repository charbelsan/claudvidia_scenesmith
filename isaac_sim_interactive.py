"""
Isaac Sim 4.5 – Interactive home scene with PhysX physics.

This script runs INSIDE the already-running streaming app (via --exec flag),
so it must NOT create a new SimulationApp. It uses the existing USD context.

Run (streaming mode – connect via browser on port 49100):
    /opt/IsaacSim/isaac-sim.streaming.sh \
        --/app/livestream/enabled=true \
        --exec "/home/ubuntu/claudvidia_scenesmith/isaac_sim_interactive.py"
"""

import math
from pathlib import Path

import omni.kit.commands
import omni.usd
from pxr import Gf, Usd, UsdGeom, UsdLux, UsdPhysics, PhysxSchema

ASSET_DIR = Path("/home/ubuntu/claudvidia_scenesmith/outputs/home_scene/assets")

stage = omni.usd.get_context().get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)

# ── Physics scene ─────────────────────────────────────────────────────────────
physics_scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
physics_scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
physics_scene.CreateGravityMagnitudeAttr(9.81)

# PhysX high-quality settings
physx_scene = PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath("/World/PhysicsScene"))
physx_scene.CreateEnableCCDAttr(True)
physx_scene.CreateEnableGPUDynamicsAttr(True)

# ── Room dimensions ──────────────────────────────────────────────────────────
LR_W, LR_D = 5.0, 4.0
KT_W, KT_D = 3.5, 4.0
KT_OX      = 4.5
WALL_H     = 2.7
WALL_T     = 0.15
FLOOR_T    = 0.1


def add_static_box(path, size, loc, color=(0.9, 0.9, 0.9)):
    """Add a physics-enabled static box (floor/walls)."""
    cube = UsdGeom.Cube.Define(stage, path)
    xf = UsdGeom.Xformable(cube.GetPrim())
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*loc))
    xf.AddScaleOp().Set(Gf.Vec3d(*size))
    cube.GetDisplayColorAttr().Set([Gf.Vec3f(*color)])

    # Make it a static rigid body (collider, no mass)
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
    rb = UsdPhysics.RigidBodyAPI(cube.GetPrim())
    rb.CreateKinematicEnabledAttr(True)  # static, immovable
    return cube


def build_room(name, w, d, ox, oy):
    root = f"/World/{name}"
    UsdGeom.Xform.Define(stage, root)
    cx, cy = ox + w/2, oy + d/2
    # Floor
    add_static_box(f"{root}/floor", (w, d, FLOOR_T),
                   (ox+w/2, oy+d/2, -FLOOR_T/2), (0.62, 0.48, 0.32))
    # Walls
    add_static_box(f"{root}/wall_s", (w+WALL_T*2, WALL_T, WALL_H),
                   (cx, oy-WALL_T/2, WALL_H/2), (0.93, 0.91, 0.89))
    add_static_box(f"{root}/wall_n", (w+WALL_T*2, WALL_T, WALL_H),
                   (cx, oy+d+WALL_T/2, WALL_H/2), (0.93, 0.91, 0.89))
    add_static_box(f"{root}/wall_w", (WALL_T, d, WALL_H),
                   (ox-WALL_T/2, cy, WALL_H/2), (0.93, 0.91, 0.89))
    add_static_box(f"{root}/wall_e", (WALL_T, d, WALL_H),
                   (ox+w+WALL_T/2, cy, WALL_H/2), (0.93, 0.91, 0.89))


def add_usd_asset(usd_path, prim_path, translate, rotate_z=0.0, scale=1.0,
                  dynamic=False):
    """
    Add a USD asset as reference.
    dynamic=True: make it a free rigid body that reacts to gravity/collisions.
    dynamic=False: static (kinematic) collider.
    """
    prim = stage.DefinePrim(prim_path, "Xform")
    prim.GetReferences().AddReference(usd_path)
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*translate))
    if rotate_z != 0.0:
        xf.AddRotateZOp().Set(rotate_z)
    if scale != 1.0:
        xf.AddScaleOp().Set(Gf.Vec3d(scale, scale, scale))

    # Apply physics
    UsdPhysics.CollisionAPI.Apply(prim)
    UsdPhysics.RigidBodyAPI.Apply(prim)
    rb = UsdPhysics.RigidBodyAPI(prim)
    if not dynamic:
        rb.CreateKinematicEnabledAttr(True)  # static furniture
    else:
        rb.CreateKinematicEnabledAttr(False)  # falls with gravity
        mass_api = UsdPhysics.MassAPI.Apply(prim)
        mass_api.CreateMassAttr(10.0)
    return prim


def A(name):
    return str(ASSET_DIR / f"{name}.usd")


# ── Build rooms ───────────────────────────────────────────────────────────────
print("Building rooms with physics...")
build_room("living_room", LR_W, LR_D, 0.0, 0.0)
build_room("kitchen",     KT_W, KT_D, KT_OX, 0.0)

# Remove shared wall
stage.RemovePrim("/World/living_room/wall_e")
stage.RemovePrim("/World/kitchen/wall_w")

# ── Living Room ───────────────────────────────────────────────────────────────
LCX, LCY = LR_W/2, LR_D/2
print("Placing living room...")

add_usd_asset(A("rug"),          "/World/LR/rug",          (LCX-0.5, LCY-0.2, 0))
add_usd_asset(A("sofa"),         "/World/LR/sofa",         (LCX-0.5, LR_D-0.55, 0), 180)
add_usd_asset(A("tv_stand"),     "/World/LR/tv_stand",     (LCX-0.5, 0.55, 0))
add_usd_asset(A("coffee_table"), "/World/LR/coffee_table", (LCX-0.5, LCY-0.3, 0))
add_usd_asset(A("armchair"),     "/World/LR/armchair",     (LR_W-0.65, LCY-0.4, 0), 270)
add_usd_asset(A("floor_lamp"),   "/World/LR/floor_lamp",   (LR_W-0.35, LR_D-0.35, 0))
add_usd_asset(A("bookshelf"),    "/World/LR/bookshelf",    (0.55, LCY+0.5, 0), 90)
add_usd_asset(A("side_table"),   "/World/LR/side_table",   (LR_W-0.65, LCY+0.6, 0))
add_usd_asset(A("plant"),        "/World/LR/plant",        (0.35, LR_D-0.35, 0))
add_usd_asset(A("wall_art"),     "/World/LR/wall_art",     (LCX-0.5, 0.06, 1.35))
add_usd_asset(A("ceiling_light"),"/World/LR/ceiling_light",(LCX-0.5, LCY, WALL_H))

# ── Kitchen ───────────────────────────────────────────────────────────────────
KCX, KCY = KT_OX+KT_W/2, KT_D/2
print("Placing kitchen...")

add_usd_asset(A("kitchen_counter"), "/World/KT/counter",   (KCX, KT_D-0.38, 0))
add_usd_asset(A("refrigerator"),    "/World/KT/fridge",    (KT_OX+0.45, KT_D-0.40, 0))
add_usd_asset(A("stove"),           "/World/KT/stove",     (KT_OX+1.30, KT_D-0.40, 0))
add_usd_asset(A("sink"),            "/World/KT/sink",      (KT_OX+KT_W-0.55, KT_D-0.35, 0))
add_usd_asset(A("kitchen_island"),  "/World/KT/island",    (KCX, KCY-0.5, 0))
add_usd_asset(A("dining_table"),    "/World/KT/dining_table",(KCX, 1.20, 0))
add_usd_asset(A("dining_chair"),    "/World/KT/chair_n",   (KCX, 1.90, 0), 180)
add_usd_asset(A("dining_chair"),    "/World/KT/chair_s",   (KCX, 0.55, 0))
add_usd_asset(A("dining_chair"),    "/World/KT/chair_w",   (KCX-1.0, 1.20, 0), 90)
add_usd_asset(A("dining_chair"),    "/World/KT/chair_e",   (KCX+1.0, 1.20, 0), 270)
add_usd_asset(A("bar_stool"),       "/World/KT/stool_1",   (KCX-0.45, KCY-1.1, 0))
add_usd_asset(A("bar_stool"),       "/World/KT/stool_2",   (KCX+0.45, KCY-1.1, 0))
add_usd_asset(A("plant"),           "/World/KT/plant",     (KT_OX+KT_W-0.30, 0.35, 0), scale=0.7)
add_usd_asset(A("pendant_light"),   "/World/KT/pendant",   (KCX, 1.20, WALL_H-0.05))
add_usd_asset(A("ceiling_light"),   "/World/KT/ceiling_light",(KCX, KCY+0.5, WALL_H))

# ── Lighting ──────────────────────────────────────────────────────────────────
dome = UsdLux.DomeLight.Define(stage, "/World/dome")
dome.CreateIntensityAttr(800)
dome.CreateColorAttr(Gf.Vec3f(0.92, 0.94, 1.0))

sun = UsdLux.DistantLight.Define(stage, "/World/sun")
sun.CreateIntensityAttr(2500)
sun.CreateColorAttr(Gf.Vec3f(1.0, 0.93, 0.80))
UsdGeom.Xformable(sun.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-28, 0, 50))

fill_sun = UsdLux.DistantLight.Define(stage, "/World/fill_sun")
fill_sun.CreateIntensityAttr(800)
fill_sun.CreateColorAttr(Gf.Vec3f(0.85, 0.90, 1.0))
UsdGeom.Xformable(fill_sun.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-30, 0, -130))

lr_fill = UsdLux.RectLight.Define(stage, "/World/lr_fill")
lr_fill.CreateIntensityAttr(1500)
lr_fill.CreateWidthAttr(4.0)
lr_fill.CreateHeightAttr(4.0)
UsdGeom.Xformable(lr_fill.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(2.5, 2.0, 3.2))
UsdGeom.Xformable(lr_fill.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-90, 0, 0))

kt_fill = UsdLux.RectLight.Define(stage, "/World/kt_fill")
kt_fill.CreateIntensityAttr(1500)
kt_fill.CreateWidthAttr(3.5)
kt_fill.CreateHeightAttr(3.5)
UsdGeom.Xformable(kt_fill.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(6.25, 2.0, 3.2))
UsdGeom.Xformable(kt_fill.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-90, 0, 0))

# ── Initial camera ────────────────────────────────────────────────────────────
from pxr import UsdGeom as UG
cam = UG.Camera.Define(stage, "/World/Camera")
cam_xf = UG.Xformable(cam.GetPrim())
cam_xf.AddTranslateOp().Set(Gf.Vec3d(4.0, -1.5, 5.0))
cam_xf.AddRotateXYZOp().Set(Gf.Vec3f(-35, 0, 0))
cam.GetFocalLengthAttr().Set(24.0)

print("Scene ready!")
print()
print("=" * 55)
print("  HOME SCENE – INTERACTIVE MODE (STREAMING)")
print("=" * 55)
print("  Objects: 20 USD assets (Blender procedural)")
print("  Physics: PhysX rigid bodies + GPU dynamics")
print()
print("  Connect: Open browser and navigate to the")
print("  Isaac Sim streaming URL shown in your console.")
print()
print("  Controls in Isaac Sim viewport:")
print("    Alt + Left drag  : Orbit camera")
print("    Alt + Right drag : Zoom")
print("    Alt + Mid drag   : Pan")
print("    F                : Focus on selection")
print("    W/A/S/D          : Fly camera (in Fly mode)")
print("    Space            : Play/Pause simulation")
print("    Ctrl+Z           : Undo")
print()
print("  Try: select an object → Play simulation →")
print("       objects will respond to gravity/collisions")
print("=" * 55)

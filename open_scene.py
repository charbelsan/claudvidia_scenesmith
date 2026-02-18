"""
Open the home scene in Isaac Sim with a visible window.
"""
import math
from pathlib import Path
from isaacsim import SimulationApp

app = SimulationApp({"renderer": "RaytracedLighting", "headless": False, "width": 1280, "height": 720})

import omni.usd
from pxr import Gf, UsdGeom, UsdLux, UsdPhysics, PhysxSchema

ASSET_DIR = Path("/home/ubuntu/claudvidia_scenesmith/outputs/home_scene/assets")

stage = omni.usd.get_context().get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)

# Physics
physics_scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
physics_scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
physics_scene.CreateGravityMagnitudeAttr(9.81)
physx = PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath("/World/PhysicsScene"))
physx.CreateEnableCCDAttr(True)
physx.CreateEnableGPUDynamicsAttr(True)

LR_W, LR_D = 5.0, 4.0
KT_W, KT_D = 3.5, 4.0
KT_OX = 4.5
WALL_H = 2.7
WALL_T = 0.15
FLOOR_T = 0.1

def box(path, size, loc, color=(0.9, 0.9, 0.9)):
    cube = UsdGeom.Cube.Define(stage, path)
    xf = UsdGeom.Xformable(cube.GetPrim())
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*loc))
    xf.AddScaleOp().Set(Gf.Vec3d(*size))
    cube.GetDisplayColorAttr().Set([Gf.Vec3f(*color)])
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    rb = UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
    rb.CreateKinematicEnabledAttr(True)

def room(name, w, d, ox, oy):
    root = f"/World/{name}"
    UsdGeom.Xform.Define(stage, root)
    cx, cy = ox + w/2, oy + d/2
    box(f"{root}/floor", (w, d, FLOOR_T), (ox+w/2, oy+d/2, -FLOOR_T/2), (0.62, 0.48, 0.32))
    box(f"{root}/wall_s", (w+WALL_T*2, WALL_T, WALL_H), (cx, oy-WALL_T/2, WALL_H/2))
    box(f"{root}/wall_n", (w+WALL_T*2, WALL_T, WALL_H), (cx, oy+d+WALL_T/2, WALL_H/2))
    box(f"{root}/wall_w", (WALL_T, d, WALL_H), (ox-WALL_T/2, cy, WALL_H/2))
    box(f"{root}/wall_e", (WALL_T, d, WALL_H), (ox+w+WALL_T/2, cy, WALL_H/2))

def asset(usd_name, prim_path, pos, rot_z=0.0, scale=1.0):
    usd_path = str(ASSET_DIR / f"{usd_name}.usd")
    prim = stage.DefinePrim(prim_path, "Xform")
    prim.GetReferences().AddReference(usd_path)
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*pos))
    if rot_z: xf.AddRotateZOp().Set(rot_z)
    if scale != 1.0: xf.AddScaleOp().Set(Gf.Vec3d(scale, scale, scale))
    UsdPhysics.CollisionAPI.Apply(prim)
    rb = UsdPhysics.RigidBodyAPI.Apply(prim)
    rb.CreateKinematicEnabledAttr(True)

# Build rooms
room("living_room", LR_W, LR_D, 0.0, 0.0)
room("kitchen",     KT_W, KT_D, KT_OX, 0.0)
stage.RemovePrim("/World/living_room/wall_e")
stage.RemovePrim("/World/kitchen/wall_w")

# Living room
LCX, LCY = LR_W/2, LR_D/2
asset("rug",          "/World/LR/rug",          (LCX-0.5, LCY-0.2, 0))
asset("sofa",         "/World/LR/sofa",         (LCX-0.5, LR_D-0.55, 0), 180)
asset("tv_stand",     "/World/LR/tv_stand",     (LCX-0.5, 0.55, 0))
asset("coffee_table", "/World/LR/coffee_table", (LCX-0.5, LCY-0.3, 0))
asset("armchair",     "/World/LR/armchair",     (LR_W-0.65, LCY-0.4, 0), 270)
asset("floor_lamp",   "/World/LR/floor_lamp",   (LR_W-0.35, LR_D-0.35, 0))
asset("bookshelf",    "/World/LR/bookshelf",    (0.55, LCY+0.5, 0), 90)
asset("side_table",   "/World/LR/side_table",   (LR_W-0.65, LCY+0.6, 0))
asset("plant",        "/World/LR/plant",        (0.35, LR_D-0.35, 0))
asset("wall_art",     "/World/LR/wall_art",     (LCX-0.5, 0.06, 1.35))
asset("ceiling_light","/World/LR/ceiling_light",(LCX-0.5, LCY, WALL_H))

# Kitchen
KCX, KCY = KT_OX+KT_W/2, KT_D/2
asset("kitchen_counter", "/World/KT/counter",      (KCX, KT_D-0.38, 0))
asset("refrigerator",    "/World/KT/fridge",       (KT_OX+0.45, KT_D-0.40, 0))
asset("stove",           "/World/KT/stove",        (KT_OX+1.30, KT_D-0.40, 0))
asset("sink",            "/World/KT/sink",         (KT_OX+KT_W-0.55, KT_D-0.35, 0))
asset("kitchen_island",  "/World/KT/island",       (KCX, KCY-0.5, 0))
asset("dining_table",    "/World/KT/dining_table", (KCX, 1.20, 0))
asset("dining_chair",    "/World/KT/chair_n",      (KCX, 1.90, 0), 180)
asset("dining_chair",    "/World/KT/chair_s",      (KCX, 0.55, 0))
asset("dining_chair",    "/World/KT/chair_w",      (KCX-1.0, 1.20, 0), 90)
asset("dining_chair",    "/World/KT/chair_e",      (KCX+1.0, 1.20, 0), 270)
asset("bar_stool",       "/World/KT/stool_1",      (KCX-0.45, KCY-1.1, 0))
asset("bar_stool",       "/World/KT/stool_2",      (KCX+0.45, KCY-1.1, 0))
asset("plant",           "/World/KT/plant",        (KT_OX+KT_W-0.30, 0.35, 0), scale=0.7)
asset("pendant_light",   "/World/KT/pendant",      (KCX, 1.20, WALL_H-0.05))
asset("ceiling_light",   "/World/KT/ceiling_light",(KCX, KCY+0.5, WALL_H))

# Lighting
dome = UsdLux.DomeLight.Define(stage, "/World/dome")
dome.CreateIntensityAttr(800)
sun = UsdLux.DistantLight.Define(stage, "/World/sun")
sun.CreateIntensityAttr(2500)
sun.CreateColorAttr(Gf.Vec3f(1.0, 0.93, 0.80))
UsdGeom.Xformable(sun.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-28, 0, 50))
fill = UsdLux.RectLight.Define(stage, "/World/fill")
fill.CreateIntensityAttr(1500)
fill.CreateWidthAttr(8.0)
fill.CreateHeightAttr(4.0)
UsdGeom.Xformable(fill.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(4.0, 2.0, 3.5))
UsdGeom.Xformable(fill.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-90, 0, 0))

# Camera
cam = UsdGeom.Camera.Define(stage, "/World/Camera")
xf = UsdGeom.Xformable(cam.GetPrim())
xf.AddTranslateOp().Set(Gf.Vec3d(3.5, -2.5, 3.5))
xf.AddRotateXYZOp().Set(Gf.Vec3f(-30, 0, 15))
cam.GetFocalLengthAttr().Set(18.0)

print("Scene loaded. Isaac Sim window is open.")
print("Press Space to start physics simulation.")

while app.is_running():
    app.update()

app.close()

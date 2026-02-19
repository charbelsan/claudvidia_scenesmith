"""
Isaac Sim 4.5 - Chambre a coucher avec porte articulee.

La porte s'ouvre vraiment via un RevoluteJoint PhysX.
Pour ouvrir: appuyer ESPACE (play) puis cliquer-glisser la porte.

Run:
    DISPLAY=:1 /opt/IsaacSim/isaac-sim.sh --exec bedroom_scene.py
"""
import math
from pathlib import Path

import omni.usd
import omni.kit.app
import omni.kit.viewport.utility as vp_util
from pxr import Gf, UsdGeom, UsdLux, UsdPhysics, PhysxSchema, Sdf

A = Path("/home/ubuntu/claudvidia_scenesmith/outputs/bedroom/assets")

stage = omni.usd.get_context().get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)

# Physics scene
phys = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
phys.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
phys.CreateGravityMagnitudeAttr(9.81)
px = PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath("/World/PhysicsScene"))
px.CreateEnableCCDAttr(True)
px.CreateEnableGPUDynamicsAttr(True)

RW, RD, RH = 5.0, 5.5, 2.8
WALL_T = 0.18
FLOOR_T = 0.10
DOOR_X  = 1.0
DOOR_W  = 0.95
DOOR_H  = 2.10

def static_box(path, size, loc, color=(0.93, 0.91, 0.89)):
    cube = UsdGeom.Cube.Define(stage, path)
    xf = UsdGeom.Xformable(cube.GetPrim())
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*loc))
    xf.AddScaleOp().Set(Gf.Vec3d(*size))
    cube.GetDisplayColorAttr().Set([Gf.Vec3f(*color)])
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    rb = UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
    rb.CreateKinematicEnabledAttr(True)

# Floor & Ceiling
static_box("/World/Room/floor",   (RW, RD, FLOOR_T), (RW/2, RD/2, -FLOOR_T/2), (0.75, 0.72, 0.68))
static_box("/World/Room/ceiling", (RW, RD, FLOOR_T), (RW/2, RD/2, RH+FLOOR_T/2), (0.97, 0.97, 0.97))

# South wall with door opening
static_box("/World/Room/wall_s_left",  (DOOR_X,                    WALL_T, RH),      (DOOR_X/2,               0, RH/2))
right_w = RW - (DOOR_X + DOOR_W)
static_box("/World/Room/wall_s_right", (right_w,                   WALL_T, RH),      (DOOR_X+DOOR_W+right_w/2,0, RH/2))
top_h = RH - DOOR_H
static_box("/World/Room/wall_s_top",   (DOOR_W,                    WALL_T, top_h),   (DOOR_X+DOOR_W/2,        0, DOOR_H+top_h/2))

# Other walls
static_box("/World/Room/wall_n", (RW+WALL_T*2, WALL_T, RH), (RW/2, RD, RH/2))
static_box("/World/Room/wall_w", (WALL_T, RD,  RH),          (0,    RD/2, RH/2))
static_box("/World/Room/wall_e", (WALL_T, RD,  RH),          (RW,   RD/2, RH/2))

# USD asset loader
def usd_asset(name, prim_path, pos, rot_z=0.0, scale=1.0, kinematic=True):
    path = str(A / f"{name}.usd")
    prim = stage.DefinePrim(prim_path, "Xform")
    prim.GetReferences().AddReference(path)
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*pos))
    if rot_z: xf.AddRotateZOp().Set(rot_z)
    if scale != 1.0: xf.AddScaleOp().Set(Gf.Vec3d(scale, scale, scale))
    UsdPhysics.CollisionAPI.Apply(prim)
    rb = UsdPhysics.RigidBodyAPI.Apply(prim)
    rb.CreateKinematicEnabledAttr(kinematic)
    return prim

# ══════════════════════════════════════════════════════════════════════════════
# PORTE ARTICULEE — RevoluteJoint PhysX
# ══════════════════════════════════════════════════════════════════════════════
HINGE_X = DOOR_X
HINGE_Y = WALL_T / 2

# Door frame (static anchor)
frame_path = "/World/Door/frame"
fp = stage.DefinePrim(frame_path, "Xform")
fp.GetReferences().AddReference(str(A / "door_frame.usd"))
xf = UsdGeom.Xformable(fp)
xf.ClearXformOpOrder()
xf.AddTranslateOp().Set(Gf.Vec3d(DOOR_X + DOOR_W/2, HINGE_Y, 0))
UsdPhysics.CollisionAPI.Apply(fp)
rb = UsdPhysics.RigidBodyAPI.Apply(fp)
rb.CreateKinematicEnabledAttr(True)

# Door panel (dynamic — rotates)
# Origin in the USD file is at the hinge (left) edge
panel_path = "/World/Door/panel"
pp = stage.DefinePrim(panel_path, "Xform")
pp.GetReferences().AddReference(str(A / "door_panel.usd"))
xf2 = UsdGeom.Xformable(pp)
xf2.ClearXformOpOrder()
xf2.AddTranslateOp().Set(Gf.Vec3d(HINGE_X, HINGE_Y, 0))
UsdPhysics.CollisionAPI.Apply(pp)
mass = UsdPhysics.MassAPI.Apply(pp)
mass.CreateMassAttr(18.0)
rb2 = UsdPhysics.RigidBodyAPI.Apply(pp)
rb2.CreateKinematicEnabledAttr(False)   # <-- dynamic

# RevoluteJoint (hinge around Z axis)
joint_path = "/World/Door/hinge"
joint = UsdPhysics.RevoluteJoint.Define(stage, joint_path)
joint.CreateBody0Rel().SetTargets([Sdf.Path(frame_path)])   # static frame
joint.CreateBody1Rel().SetTargets([Sdf.Path(panel_path)])   # rotating door
joint.CreateAxisAttr("Z")
# Anchor point on frame: at DOOR_X (left edge = hinge side)
joint.CreateLocalPos0Attr(Gf.Vec3f(-DOOR_W/2, 0, 0))
joint.CreateLocalRot0Attr(Gf.Quatf(1, 0, 0, 0))
# Anchor point on panel: at origin (hinge edge)
joint.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
joint.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
# Limits: 0 = closed, 90 = fully open
joint.CreateLowerLimitAttr(0.0)
joint.CreateUpperLimitAttr(90.0)

# PhysX tuning
pj = PhysxSchema.PhysxJointAPI.Apply(stage.GetPrimAtPath(joint_path))
pj.CreateJointFrictionAttr(0.8)
pj.CreateMaxJointVelocityAttr(120.0)

# Angular drive (spring) so door can be scripted open/closed
drive = UsdPhysics.DriveAPI.Apply(stage.GetPrimAtPath(joint_path), "angular")
drive.CreateTypeAttr("force")
drive.CreateMaxForceAttr(200.0)
drive.CreateTargetPositionAttr(0.0)
drive.CreateStiffnessAttr(0.0)
drive.CreateDampingAttr(15.0)

# ══════════════════════════════════════════════════════════════════════════════
# FURNITURE
# ══════════════════════════════════════════════════════════════════════════════
usd_asset("bed",               "/World/Furn/bed",       (RW/2, RD-1.15, 0))
usd_asset("nightstand",        "/World/Furn/ns_l",      (RW/2-1.25, RD-1.55, 0))
usd_asset("nightstand",        "/World/Furn/ns_r",      (RW/2+1.25, RD-1.55, 0))
usd_asset("wardrobe",          "/World/Furn/wardrobe",  (0.95, RD/2+0.5, 0), rot_z=90)
usd_asset("dresser",           "/World/Furn/dresser",   (RW-0.50, RD*0.35, 0), rot_z=180)
usd_asset("desk",              "/World/Furn/desk",      (RW-0.80, 1.60, 0), rot_z=90)
usd_asset("desk_chair",        "/World/Furn/dchair",    (RW-1.65, 1.60, 0), rot_z=90)
usd_asset("floor_lamp_bedroom","/World/Furn/lamp",      (0.45, RD-0.45, 0))
usd_asset("bedroom_rug",       "/World/Furn/rug",       (RW/2, RD/2, 0.001))

# Lighting
dome = UsdLux.DomeLight.Define(stage, "/World/dome")
dome.CreateIntensityAttr(250)
dome.CreateColorAttr(Gf.Vec3f(0.88, 0.92, 1.0))

sun = UsdLux.DistantLight.Define(stage, "/World/sun")
sun.CreateIntensityAttr(1800)
sun.CreateColorAttr(Gf.Vec3f(1.0, 0.93, 0.80))
UsdGeom.Xformable(sun.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-40, 0, 30))

overhead = UsdLux.RectLight.Define(stage, "/World/overhead")
overhead.CreateIntensityAttr(1200)
overhead.CreateWidthAttr(5.0)
overhead.CreateHeightAttr(5.5)
UsdGeom.Xformable(overhead.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(RW/2, RD/2, 3.5))
UsdGeom.Xformable(overhead.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-90, 0, 0))

# Camera prim avec LookAt — intérieur coin sud-est, niveau regard, vers le lit
# Pièce: X 0..5.0, Y 0..5.5, Z 0..2.8
eye    = Gf.Vec3d(4.6, 0.4, 1.2)    # intérieur coin SE, 1.2m de haut
center = Gf.Vec3d(1.0, 4.3, 0.5)    # viser le lit + côté armoire (nord-ouest)
up     = Gf.Vec3d(0, 0, 1)

view_mat = Gf.Matrix4d()
view_mat.SetLookAt(eye, center, up)
cam_to_world = view_mat.GetInverse()

cam = UsdGeom.Camera.Define(stage, "/World/Camera")
xf = UsdGeom.Xformable(cam.GetPrim())
xf.ClearXformOpOrder()
xf.AddTransformOp().Set(cam_to_world)
cam.GetFocalLengthAttr().Set(16.0)
cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 100.0))

# Switcher le viewport vers la Camera prim
viewport = vp_util.get_active_viewport()
if viewport:
    viewport.camera_path = "/World/Camera"

# Sauvegarder la scène
out_usd = "/home/ubuntu/claudvidia_scenesmith/outputs/bedroom/bedroom.usd"
omni.usd.get_context().save_as_stage(out_usd)
print(f"  Scene saved: {out_usd}")

print()
print("=" * 60)
print("  CHAMBRE A COUCHER — PORTE ARTICULEE (RevoluteJoint)")
print("=" * 60)
print(f"  Piece : {RW}m x {RD}m x {RH}m  |  Porte : {DOOR_W}m x {DOOR_H}m")
print()
print("  Pour interagir avec la porte :")
print("    1. Appuyer ESPACE → demarrer la simulation physique")
print("    2. Cliquer la porte + faire glisser → elle s'ouvre")
print("    3. La porte a de l'inertie et de l'amortissement")
print()
print("  Navigation viewport :")
print("    Alt + drag gauche : orbite   |   F : focus selection")
print("    Alt + drag droit  : zoom     |   W/A/S/D : vol camera")
print("=" * 60)

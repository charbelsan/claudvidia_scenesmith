"""
Isaac Sim 4.5 – Cinematic home scene flythrough.

Living room + kitchen with 20+ real 3D USD assets.
Camera flies INSIDE the rooms with cinematic interior shots.

Run:
    cd /home/ubuntu/IsaacSim && ./python.sh /home/ubuntu/claudvidia_scenesmith/isaac_sim_scene.py
"""

import math
import subprocess
import sys
from pathlib import Path

from isaacsim import SimulationApp

CONFIG = {
    "renderer": "RaytracedLighting",
    "headless": True,
    "width": 1920,
    "height": 1080,
}
app = SimulationApp(CONFIG)
print("Isaac Sim started")

import omni.kit.commands
import omni.replicator.core as rep
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade

ASSET_DIR = Path("/home/ubuntu/claudvidia_scenesmith/outputs/home_scene/assets")
OUTPUT_DIR = Path("/home/ubuntu/claudvidia_scenesmith/outputs/home_scene/video")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "frames").mkdir(exist_ok=True)

stage = omni.usd.get_context().get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)

# ── Room dimensions ──────────────────────────────────────────────────────────
LR_W, LR_D = 5.0, 4.0   # living room: x in [0..5], y in [0..4]
KT_W, KT_D = 3.5, 4.0   # kitchen:    x in [4.5..8], y in [0..4]
KT_OX = 4.5              # kitchen X offset (0.15m gap = shared wall)
WALL_H = 2.7
WALL_T = 0.15
FLOOR_T = 0.1


# ── Primitives ───────────────────────────────────────────────────────────────
def add_box(path, size, loc, color=(0.9, 0.9, 0.9)):
    cube = UsdGeom.Cube.Define(stage, path)
    xf = UsdGeom.Xformable(cube.GetPrim())
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*loc))
    xf.AddScaleOp().Set(Gf.Vec3d(*size))
    cube.GetDisplayColorAttr().Set([Gf.Vec3f(*color)])
    return cube


def build_room(name, w, d, ox, oy):
    root = f"/World/{name}"
    UsdGeom.Xform.Define(stage, root)
    # floor
    add_box(f"{root}/floor", (w, d, FLOOR_T),
            (ox + w/2, oy + d/2, -FLOOR_T/2), (0.62, 0.48, 0.32))
    cx, cy = ox + w/2, oy + d/2
    # south wall
    add_box(f"{root}/wall_s", (w + WALL_T*2, WALL_T, WALL_H),
            (cx, oy - WALL_T/2, WALL_H/2), (0.93, 0.91, 0.89))
    # north wall
    add_box(f"{root}/wall_n", (w + WALL_T*2, WALL_T, WALL_H),
            (cx, oy + d + WALL_T/2, WALL_H/2), (0.93, 0.91, 0.89))
    # west wall
    add_box(f"{root}/wall_w", (WALL_T, d, WALL_H),
            (ox - WALL_T/2, cy, WALL_H/2), (0.93, 0.91, 0.89))
    # east wall
    add_box(f"{root}/wall_e", (WALL_T, d, WALL_H),
            (ox + w + WALL_T/2, cy, WALL_H/2), (0.93, 0.91, 0.89))


def add_usd_asset(usd_path: str, prim_path: str, translate, rotate_z=0.0, scale=1.0):
    """Add a USD file as a reference at the given world position."""
    prim = stage.DefinePrim(prim_path, "Xform")
    prim.GetReferences().AddReference(usd_path)
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*translate))
    if rotate_z != 0.0:
        xf.AddRotateZOp().Set(rotate_z)
    if scale != 1.0:
        xf.AddScaleOp().Set(Gf.Vec3d(scale, scale, scale))
    return prim


def A(name):
    """Return full path to a USD asset."""
    return str(ASSET_DIR / f"{name}.usd")


# ── Build rooms ───────────────────────────────────────────────────────────────
print("Building rooms...")
build_room("living_room", LR_W, LR_D, 0.0, 0.0)
build_room("kitchen",     KT_W, KT_D, KT_OX, 0.0)

# Remove the shared wall between rooms (open connection)
stage.RemovePrim("/World/living_room/wall_e")
stage.RemovePrim("/World/kitchen/wall_w")

# ── Living Room furniture ─────────────────────────────────────────────────────
print("Placing living room furniture...")
LCX, LCY = LR_W/2, LR_D/2   # living room center = (2.5, 2.0)

# Rug (centered in living room, flat on floor)
add_usd_asset(A("rug"), "/World/LR/rug",
              (LCX - 0.5, LCY - 0.2, 0.0), 0)

# Sofa – against north wall, facing south (toward TV)
add_usd_asset(A("sofa"), "/World/LR/sofa",
              (LCX - 0.5, LR_D - 0.55, 0.0), 180)

# TV Stand – against south wall, facing north (toward sofa)
add_usd_asset(A("tv_stand"), "/World/LR/tv_stand",
              (LCX - 0.5, 0.55, 0.0), 0)

# Coffee table – between sofa and TV
add_usd_asset(A("coffee_table"), "/World/LR/coffee_table",
              (LCX - 0.5, LCY - 0.3, 0.0), 0)

# Armchair – east side, facing west toward TV
add_usd_asset(A("armchair"), "/World/LR/armchair",
              (LR_W - 0.65, LCY - 0.4, 0.0), 270)

# Floor lamp – corner behind armchair
add_usd_asset(A("floor_lamp"), "/World/LR/floor_lamp",
              (LR_W - 0.35, LR_D - 0.35, 0.0), 0)

# Bookshelf – west wall
add_usd_asset(A("bookshelf"), "/World/LR/bookshelf",
              (0.55, LCY + 0.5, 0.0), 90)

# Side table – next to armchair
add_usd_asset(A("side_table"), "/World/LR/side_table",
              (LR_W - 0.65, LCY + 0.6, 0.0), 0)

# Plant – northwest corner
add_usd_asset(A("plant"), "/World/LR/plant",
              (0.35, LR_D - 0.35, 0.0), 0)

# Wall art – on south wall above TV stand
add_usd_asset(A("wall_art"), "/World/LR/wall_art",
              (LCX - 0.5, 0.06, 1.35), 0)

# Ceiling light – center of living room
add_usd_asset(A("ceiling_light"), "/World/LR/ceiling_light",
              (LCX - 0.5, LCY, WALL_H), 0)

# ── Kitchen / Dining furniture ────────────────────────────────────────────────
print("Placing kitchen furniture...")
KCX, KCY = KT_OX + KT_W/2, KT_D/2  # kitchen center = (6.25, 2.0)

# Kitchen counter – against north wall
add_usd_asset(A("kitchen_counter"), "/World/KT/counter",
              (KCX, KT_D - 0.38, 0.0), 0)

# Refrigerator – northwest corner of kitchen
add_usd_asset(A("refrigerator"), "/World/KT/fridge",
              (KT_OX + 0.45, KT_D - 0.40, 0.0), 0)

# Stove – next to fridge
add_usd_asset(A("stove"), "/World/KT/stove",
              (KT_OX + 1.30, KT_D - 0.40, 0.0), 0)

# Sink – east counter section
add_usd_asset(A("sink"), "/World/KT/sink",
              (KT_OX + KT_W - 0.55, KT_D - 0.35, 0.0), 0)

# Kitchen island – center of kitchen
add_usd_asset(A("kitchen_island"), "/World/KT/island",
              (KCX, KCY - 0.5, 0.0), 0)

# Dining table – south half of kitchen
add_usd_asset(A("dining_table"), "/World/KT/dining_table",
              (KCX, 1.20, 0.0), 0)

# 4 dining chairs around table
add_usd_asset(A("dining_chair"), "/World/KT/chair_n",
              (KCX, 1.90, 0.0), 180)  # north side, facing south
add_usd_asset(A("dining_chair"), "/World/KT/chair_s",
              (KCX, 0.55, 0.0), 0)    # south side, facing north
add_usd_asset(A("dining_chair"), "/World/KT/chair_w",
              (KCX - 1.0, 1.20, 0.0), 90)   # west side
add_usd_asset(A("dining_chair"), "/World/KT/chair_e",
              (KCX + 1.0, 1.20, 0.0), 270)   # east side

# 2 bar stools at island
add_usd_asset(A("bar_stool"), "/World/KT/stool_1",
              (KCX - 0.45, KCY - 1.1, 0.0), 0)
add_usd_asset(A("bar_stool"), "/World/KT/stool_2",
              (KCX + 0.45, KCY - 1.1, 0.0), 0)

# Plant in kitchen corner
add_usd_asset(A("plant"), "/World/KT/plant",
              (KT_OX + KT_W - 0.30, 0.35, 0.0), 0, scale=0.7)

# Pendant light over dining table
add_usd_asset(A("pendant_light"), "/World/KT/pendant",
              (KCX, 1.20, WALL_H - 0.05), 0)

# Ceiling light in kitchen
add_usd_asset(A("ceiling_light"), "/World/KT/ceiling_light",
              (KCX, KCY + 0.5, WALL_H), 0)

print("Scene built.")

# ── Lighting ──────────────────────────────────────────────────────────────────
print("Setting up lights...")

# HDRI-style dome – bright neutral fill
dome = UsdLux.DomeLight.Define(stage, "/World/dome")
dome.CreateIntensityAttr(800)
dome.CreateColorAttr(Gf.Vec3f(0.92, 0.94, 1.0))

# Warm sun from south-east, low angle
sun = UsdLux.DistantLight.Define(stage, "/World/sun")
sun.CreateIntensityAttr(2500)
sun.CreateColorAttr(Gf.Vec3f(1.0, 0.93, 0.80))
UsdGeom.Xformable(sun.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-28, 0, 50))

# Counter-sun from north-west to fill shadowed areas (north wall)
fill_sun = UsdLux.DistantLight.Define(stage, "/World/fill_sun")
fill_sun.CreateIntensityAttr(800)
fill_sun.CreateColorAttr(Gf.Vec3f(0.85, 0.90, 1.0))
UsdGeom.Xformable(fill_sun.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-30, 0, -130))

# Living room overhead fill – large rect above centre
lr_fill = UsdLux.RectLight.Define(stage, "/World/lr_fill")
lr_fill.CreateIntensityAttr(1500)
lr_fill.CreateWidthAttr(4.0)
lr_fill.CreateHeightAttr(4.0)
lr_fill.CreateColorAttr(Gf.Vec3f(1.0, 0.95, 0.88))
UsdGeom.Xformable(lr_fill.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(2.5, 2.0, 3.2))
UsdGeom.Xformable(lr_fill.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-90, 0, 0))

# Kitchen overhead fill
kt_fill = UsdLux.RectLight.Define(stage, "/World/kt_fill")
kt_fill.CreateIntensityAttr(1500)
kt_fill.CreateWidthAttr(3.5)
kt_fill.CreateHeightAttr(3.5)
kt_fill.CreateColorAttr(Gf.Vec3f(1.0, 0.96, 0.90))
UsdGeom.Xformable(kt_fill.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(6.25, 2.0, 3.2))
UsdGeom.Xformable(kt_fill.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-90, 0, 0))

print("Lights set.")

# ── Cinematic camera path ─────────────────────────────────────────────────────
# 6-shot sequence, total 210 frames at 24fps = 8.75 seconds
# Wide-angle interior + aerial shots

NUM_FRAMES = 210
FPS = 24


def lerp(a, b, t):
    return a + (b - a) * t


def lerp3(a, b, t):
    return tuple(lerp(a[i], b[i], t) for i in range(3))


def smooth(t):
    return t * t * (3 - 2 * t)


def camera_path():
    """Returns list of (pos, look_at, focal_length) tuples for each frame."""
    frames = []

    # ── Shot 1 (0–40): Aerial overhead, both rooms – descend slowly ──────────
    # Start high and drift forward to reveal the whole home
    for i in range(41):
        t = smooth(i / 40)
        pos = lerp3((4.0, -2.5, 9.5), (4.0, 1.0, 6.5), t)
        look = lerp3((4.0, 2.0, 0.5), (4.0, 2.5, 0.5), t)
        frames.append((pos, look, 28))  # 28mm = nice aerial angle

    # ── Shot 2 (40–80): Living room – SE corner, wide establishing ────────────
    # From the east side of the south wall, looking NW across the whole room
    # Shows: TV stand (right foreground), rug, coffee table, sofa, bookshelf
    for i in range(41):
        t = smooth(i / 40)
        pos = lerp3((4.6, 0.5, 1.35), (3.8, 0.8, 1.25), t)
        look = lerp3((1.5, 2.8, 0.75), (1.8, 2.5, 0.70), t)
        frames.append((pos, look, 16))  # 16mm wide for interiors

    # ── Shot 3 (80–120): Living room – elevated float above coffee table ───────
    # Camera elevated above the room centre, looking toward sofa+bookshelf
    # Shows: rug below, coffee table, armchair (right), sofa (ahead), bookshelf
    for i in range(41):
        t = smooth(i / 40)
        pos = lerp3((3.0, 1.5, 2.2), (2.5, 1.8, 1.8), t)
        look = lerp3((2.0, 3.2, 0.70), (1.8, 3.0, 0.65), t)
        frames.append((pos, look, 20))

    # ── Shot 4 (120–160): Armchair + bookshelf zone – orbit ──────────────────
    # Orbit slowly around the east-side seating + bookshelf area
    for i in range(41):
        t = smooth(i / 40)
        angle = math.radians(lerp(200, 270, t))  # arc from south to west
        r = 1.9
        cx, cy = 2.8, 2.2
        px = cx + r * math.cos(angle)
        py = cy + r * math.sin(angle)
        pos = (px, py, 1.45)
        look = (cx - 0.3, cy - 0.1, 0.75)
        frames.append((pos, look, 20))

    # ── Shot 5 (160–190): Kitchen – wide shot from connection ────────────────
    # Camera just inside the kitchen opening, looking toward appliances
    # Shows: dining chairs (foreground left/right), island, dining table,
    #        counter, fridge+stove (background)
    for i in range(31):
        t = smooth(i / 30)
        pos = lerp3((5.2, 1.8, 1.5), (5.8, 2.0, 1.4), t)
        look = lerp3((6.8, 3.0, 0.85), (6.5, 3.2, 0.80), t)
        frames.append((pos, look, 18))

    # ── Shot 6 (190–210): Wide aerial finale – rise & drift to reveal both ────
    # Camera already above both rooms, tilted to see full floor plan
    for i in range(20):
        t = smooth(i / 19)
        # Drift south-west at height, looking at centre of home
        pos = lerp3((4.0, 2.8, 4.5), (4.0, 1.5, 7.5), t)
        look = lerp3((4.0, 2.0, 0.5), (4.0, 2.0, 0.3), t)
        frames.append((pos, look, 14))  # 14mm = very wide for this aerial

    return frames


path = camera_path()
print(f"Camera path: {len(path)} frames")

# ── Render ────────────────────────────────────────────────────────────────────
print(f"Rendering {len(path)} frames at {FPS}fps...")

with rep.new_layer():
    p0, l0, fl0 = path[0]
    camera = rep.create.camera(
        position=p0,
        look_at=l0,
        focal_length=fl0,
    )
    render_product = rep.create.render_product(camera, (1920, 1080))

    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(
        output_dir=str(OUTPUT_DIR / "frames"),
        rgb=True,
    )
    writer.attach([render_product])

    # Get USD camera prim for focal_length adjustments
    cam_prim = None
    for prim in stage.Traverse():
        if prim.GetTypeName() == "Camera":
            cam_prim = prim
            break

    app.update()

    for i, (pos, look, fl) in enumerate(path):
        with camera:
            rep.modify.pose(position=pos, look_at=look)

        # Adjust focal length via USD directly
        if cam_prim is not None:
            fl_attr = cam_prim.GetAttribute("focalLength")
            if fl_attr:
                fl_attr.Set(float(fl))

        rep.orchestrator.step(rt_subframes=6)
        app.update()

        if i % 30 == 0:
            print(f"  Frame {i+1}/{len(path)}")

print("Frames rendered.")

# ── Encode video ──────────────────────────────────────────────────────────────
frames_dir = OUTPUT_DIR / "frames"
video_path = OUTPUT_DIR / "home_scene_v2.mp4"

# Try numbered pattern first (BasicWriter saves as rgb_XXXX.png)
for pattern in [str(frames_dir / "rgb_%04d.png"), str(frames_dir / "*.png")]:
    glob_flag = ["-pattern_type", "glob"] if "*" in pattern else []
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(FPS),
        *glob_flag,
        "-i", pattern,
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "slow",
        "-pix_fmt", "yuv420p",
        "-vf", "scale=1920:1080",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"Video saved: {video_path}")
        break
    print(f"  ffmpeg pattern '{pattern}' failed, trying next...")
else:
    pngs = sorted(frames_dir.glob("**/*.png"))
    print(f"Frames on disk: {len(pngs)}")
    if pngs:
        print(f"  Sample: {pngs[0].name}")
        # Try with actual filename pattern
        stem = pngs[0].stem  # e.g. "rgb_0000"
        prefix = stem.rstrip("0123456789")
        cmd2 = [
            "ffmpeg", "-y",
            "-framerate", str(FPS),
            "-i", str(frames_dir / f"{prefix}%04d.png"),
            "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
            str(video_path),
        ]
        result2 = subprocess.run(cmd2, capture_output=True, text=True)
        if result2.returncode == 0:
            print(f"Video saved: {video_path}")
        else:
            print(f"ffmpeg error:\n{result2.stderr[:600]}")

print("Done. Shutting down.")
app.close()

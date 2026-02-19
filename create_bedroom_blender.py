"""
Generate bedroom USD assets + articulated door using Blender Python.
Outputs to outputs/bedroom/assets/
"""
import bpy, math
from pathlib import Path

OUT = Path("/home/ubuntu/claudvidia_scenesmith/outputs/bedroom/assets")
OUT.mkdir(parents=True, exist_ok=True)

# ── Helpers ────────────────────────────────────────────────────────────────────

def clear():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    for m in list(bpy.data.materials): bpy.data.materials.remove(m)

def mat(name, color, roughness=0.7, metallic=0.0, emission=None):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*color, 1)
    b.inputs["Roughness"].default_value = roughness
    b.inputs["Metallic"].default_value = metallic
    if emission:
        b.inputs["Emission Color"].default_value = (*emission, 1)
        b.inputs["Emission Strength"].default_value = 2.0
    return m

def box(name, size, pos=(0,0,0), rot=(0,0,0)):
    bpy.ops.mesh.primitive_cube_add(location=pos)
    o = bpy.context.active_object
    o.name = name
    o.scale = (size[0]/2, size[1]/2, size[2]/2)
    o.rotation_euler = [math.radians(r) for r in rot]
    bpy.ops.object.transform_apply(scale=True, rotation=True)
    return o

def cyl(name, r, h, pos=(0,0,0), verts=16):
    bpy.ops.mesh.primitive_cylinder_add(radius=r, depth=h, location=pos, vertices=verts)
    o = bpy.context.active_object
    o.name = name
    return o

def assign(obj, material):
    if obj.data.materials:
        obj.data.materials[0] = material
    else:
        obj.data.materials.append(material)

def join_all():
    bpy.ops.object.select_all(action='SELECT')
    bpy.context.view_layer.objects.active = bpy.context.selected_objects[0]
    bpy.ops.object.join()
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
    o = bpy.context.active_object
    o.location = (0, 0, 0)
    bpy.ops.object.transform_apply(location=True)

def export(name):
    bpy.ops.object.select_all(action='SELECT')
    path = str(OUT / f"{name}.usd")
    bpy.ops.wm.usd_export(filepath=path, selected_objects_only=False,
                          export_materials=True, export_uvmaps=True)
    print(f"Exported: {path}")

# ══════════════════════════════════════════════════════════════════════════════
# BED (2.0 × 2.2 × 0.65 m)
# ══════════════════════════════════════════════════════════════════════════════
def make_bed():
    clear()
    wood   = mat("wood_walnut",  (0.25, 0.15, 0.08), roughness=0.5)
    fabric = mat("fabric_grey",  (0.50, 0.50, 0.55), roughness=0.95)
    pillow = mat("pillow_white", (0.95, 0.94, 0.92), roughness=0.9)

    # Base frame
    frame = box("frame", (2.0, 2.2, 0.22), (0, 0, 0.11)); assign(frame, wood)
    # Mattress
    matt  = box("mattress", (1.85, 2.0, 0.22), (0, -0.05, 0.33)); assign(matt, fabric)
    # Headboard
    hb    = box("headboard", (2.0, 0.12, 0.80), (0, -1.04, 0.55)); assign(hb, wood)
    # Footboard
    fb    = box("footboard", (2.0, 0.10, 0.35), (0,  1.06, 0.28)); assign(fb, wood)
    # Legs x4
    for x, y in [(-0.85,-0.95),(0.85,-0.95),(-0.85,0.95),(0.85,0.95)]:
        l = box(f"leg_{x}", (0.08,0.08,0.20), (x, y, 0)); assign(l, wood)
    # Pillows x2
    for x in [-0.45, 0.45]:
        p = box(f"pillow_{x}", (0.60, 0.45, 0.15), (x, -0.70, 0.50)); assign(p, pillow)
    join_all(); export("bed")

# ══════════════════════════════════════════════════════════════════════════════
# WARDROBE (1.8 × 0.6 × 2.2 m)
# ══════════════════════════════════════════════════════════════════════════════
def make_wardrobe():
    clear()
    wood = mat("wood_oak",   (0.75, 0.60, 0.40), roughness=0.4)
    dark = mat("wood_dark",  (0.18, 0.12, 0.07), roughness=0.5)
    metal= mat("metal_gold", (0.85, 0.70, 0.30), roughness=0.2, metallic=0.9)

    body = box("body", (1.8, 0.6, 2.2), (0,0,1.1)); assign(body, wood)
    top  = box("top",  (1.84,0.64,0.04),(0,0,2.22)); assign(top, dark)
    base = box("base", (1.80,0.60,0.06),(0,0,0.03)); assign(base, dark)
    # Door panels
    for x in [-0.45, 0.45]:
        d = box(f"door_{x}",(0.85,0.03,2.1),(x,0.315,1.1)); assign(d, wood)
    # Door handles
    for x in [-0.10, 0.10]:
        h = cyl(f"handle_{x}", 0.012, 0.12, (x, 0.34, 1.1))
        h.rotation_euler = [math.radians(90), 0, 0]
        bpy.ops.object.transform_apply(rotation=True)
        assign(h, metal)
    # Middle divider
    mid = box("mid", (0.03, 0.56, 2.14),(0,0,1.1)); assign(mid, dark)
    join_all(); export("wardrobe")

# ══════════════════════════════════════════════════════════════════════════════
# NIGHTSTAND (0.50 × 0.40 × 0.55 m)
# ══════════════════════════════════════════════════════════════════════════════
def make_nightstand():
    clear()
    wood  = mat("wood_ns",  (0.55, 0.40, 0.25), roughness=0.5)
    metal = mat("metal_ns", (0.80, 0.80, 0.80), roughness=0.2, metallic=0.9)

    body   = box("body",   (0.50,0.40,0.50),(0,0,0.25)); assign(body, wood)
    top    = box("top",    (0.52,0.42,0.03),(0,0,0.515)); assign(top, wood)
    drawer = box("drawer", (0.44,0.36,0.18),(0,0,0.24)); assign(drawer, wood)
    handle = cyl("handle", 0.008, 0.15, (0, 0.22, 0.24))
    handle.rotation_euler = [math.radians(90),0,0]
    bpy.ops.object.transform_apply(rotation=True)
    assign(handle, metal)
    for x, y in [(-0.20,-0.15),(0.20,-0.15),(-0.20,0.15),(0.20,0.15)]:
        l = box(f"leg{x}{y}", (0.04,0.04,0.06),(x,y,0.03)); assign(l, wood)
    join_all(); export("nightstand")

# ══════════════════════════════════════════════════════════════════════════════
# DESK (1.4 × 0.65 × 0.76 m)
# ══════════════════════════════════════════════════════════════════════════════
def make_desk():
    clear()
    wood  = mat("wood_desk",  (0.80,0.65,0.45), roughness=0.45)
    metal = mat("metal_desk", (0.30,0.30,0.32), roughness=0.3, metallic=0.95)

    top    = box("top",    (1.4,0.65,0.04),(0,0,0.74)); assign(top, wood)
    for x, y in [(-0.62,-0.27),(0.62,-0.27),(-0.62,0.27),(0.62,0.27)]:
        l = box(f"leg{x}", (0.05,0.05,0.73),(x,y,0.365)); assign(l, metal)
    # Shelf underneath
    shelf = box("shelf", (0.60,0.60,0.38),(-0.35,0,0.31)); assign(shelf, wood)
    join_all(); export("desk")

# ══════════════════════════════════════════════════════════════════════════════
# DESK CHAIR
# ══════════════════════════════════════════════════════════════════════════════
def make_desk_chair():
    clear()
    fabric = mat("fabric_chair",(0.15,0.15,0.18), roughness=0.9)
    metal  = mat("metal_chair", (0.25,0.25,0.27), roughness=0.3, metallic=0.95)

    seat   = box("seat",  (0.50,0.50,0.10),(0,0,0.50)); assign(seat, fabric)
    back   = box("back",  (0.48,0.08,0.60),(0,-0.21,0.90)); assign(back, fabric)
    post   = cyl("post", 0.03,0.45,(0,0,0.275)); assign(post, metal)
    base   = cyl("base", 0.28,0.04,(0,0,0.04)); assign(base, metal)
    for a in range(5):
        ang = math.radians(a*72)
        s = box(f"spoke{a}",(0.22,0.025,0.025),(math.cos(ang)*0.11,math.sin(ang)*0.11,0.06))
        s.rotation_euler=[0,0,ang]; bpy.ops.object.transform_apply(rotation=True)
        w = cyl(f"wheel{a}",0.025,0.05,(math.cos(ang)*0.22,math.sin(ang)*0.22,0.025))
        w.rotation_euler=[math.radians(90),0,0]; bpy.ops.object.transform_apply(rotation=True)
        assign(s, metal); assign(w, metal)
    join_all(); export("desk_chair")

# ══════════════════════════════════════════════════════════════════════════════
# FLOOR LAMP (bedroom)
# ══════════════════════════════════════════════════════════════════════════════
def make_lamp():
    clear()
    metal = mat("lamp_metal",(0.25,0.22,0.18), roughness=0.2, metallic=0.95)
    shade = mat("lamp_shade",(0.98,0.93,0.78), roughness=0.8)
    glow  = mat("lamp_glow", (1.0,0.95,0.8), roughness=0.5, emission=(1.0,0.9,0.7))

    base  = cyl("base",  0.18, 0.04, (0,0,0.02)); assign(base, metal)
    pole  = cyl("pole",  0.02, 1.50, (0,0,0.77)); assign(pole, metal)
    sh    = cyl("shade", 0.20, 0.35, (0,0,1.60)); assign(sh, shade)
    inner = cyl("inner", 0.18, 0.30, (0,0,1.60)); assign(inner, glow)
    join_all(); export("floor_lamp_bedroom")

# ══════════════════════════════════════════════════════════════════════════════
# DRESSER / CHEST OF DRAWERS (0.90 × 0.45 × 1.10 m)
# ══════════════════════════════════════════════════════════════════════════════
def make_dresser():
    clear()
    wood  = mat("wood_dresser",(0.65,0.50,0.30), roughness=0.45)
    dark  = mat("dark_dresser",(0.20,0.14,0.08), roughness=0.5)
    metal = mat("metal_dresser",(0.75,0.72,0.68), roughness=0.15, metallic=0.95)

    body = box("body",(0.90,0.45,1.08),(0,0,0.54)); assign(body, wood)
    top  = box("top", (0.92,0.47,0.04),(0,0,1.10)); assign(top, dark)
    # 4 drawers
    for i,z in enumerate([0.17,0.39,0.61,0.83]):
        d = box(f"drawer{i}",(0.82,0.40,0.18),(0,0,z)); assign(d, wood)
        # drawer frame line
        f = box(f"df{i}",(0.84,0.02,0.20),(0,0.225,z)); assign(f, dark)
        # handle
        h = cyl(f"h{i}",0.008,0.20,(0,0.235,z))
        h.rotation_euler=[math.radians(90),0,0]; bpy.ops.object.transform_apply(rotation=True)
        assign(h, metal)
    join_all(); export("dresser")

# ══════════════════════════════════════════════════════════════════════════════
# BEDROOM RUG (2.0 × 1.4 m, thin)
# ══════════════════════════════════════════════════════════════════════════════
def make_rug():
    clear()
    rug = mat("rug_bedroom",(0.40,0.30,0.55), roughness=0.99)
    r = box("rug",(2.0,1.4,0.012),(0,0,0.006)); assign(r, rug)
    join_all(); export("bedroom_rug")

# ══════════════════════════════════════════════════════════════════════════════
# DOOR FRAME  (wall opening: w=0.95m, h=2.10m, thickness=0.18m)
# Exported as separate piece so Isaac Sim can keep it static
# ══════════════════════════════════════════════════════════════════════════════
def make_door_frame():
    clear()
    wood = mat("door_frame_wood",(0.78,0.64,0.46), roughness=0.4)

    W, H, T = 0.95, 2.10, 0.18  # opening dims + wall thickness
    TRIM = 0.06  # moulding width

    # Left jamb
    lj = box("lj",(TRIM, T, H),(-(W/2+TRIM/2), 0, H/2)); assign(lj, wood)
    # Right jamb
    rj = box("rj",(TRIM, T, H),( (W/2+TRIM/2), 0, H/2)); assign(rj, wood)
    # Top casing
    tc = box("tc",(W+TRIM*2, T, TRIM),(0, 0, H+TRIM/2)); assign(tc, wood)

    join_all(); export("door_frame")

# ══════════════════════════════════════════════════════════════════════════════
# DOOR PANEL  (w=0.92m, h=2.04m, d=0.05m) — pivot at left edge (x=-0.46)
# Hinge point in USD will be at x=-0.46 relative to panel origin
# ══════════════════════════════════════════════════════════════════════════════
def make_door_panel():
    clear()
    wood  = mat("door_wood",   (0.82,0.68,0.48), roughness=0.38)
    panel = mat("door_panel",  (0.88,0.76,0.56), roughness=0.42)
    metal = mat("door_metal",  (0.85,0.82,0.78), roughness=0.15, metallic=0.95)
    brass = mat("door_brass",  (0.82,0.68,0.28), roughness=0.20, metallic=0.92)

    W, H, D = 0.92, 2.04, 0.05

    # Main door slab — origin offset so hinge is at x=0 left edge
    # We shift everything +W/2 so that x=0 is the hinge axis
    door = box("door_slab",(W, D, H),(W/2, 0, H/2)); assign(door, wood)

    # Recessed panels (decorative)
    for z in [H*0.72, H*0.30]:
        p = box(f"panel{z}",(W*0.75, D*0.4, H*0.28),(W/2, D/2+0.002, z)); assign(p, panel)

    # ── Handle assembly ────────────────────────────────────────────────────
    # Escutcheon plate
    esc = box("esc",(0.06,0.015,0.18),(W-0.12, D/2+0.018, H*0.48)); assign(esc, brass)

    # Lever handle (horizontal rod that rotates to open latch)
    # Pivot at its base (the escutcheon), extends outward along Y axis
    lever = box("lever",(0.015,0.015,0.115),(W-0.12, D/2+0.082, H*0.48))
    lever.rotation_euler=[0,0,math.radians(15)]  # slight downward angle at rest
    bpy.ops.object.transform_apply(rotation=True)
    assign(lever, brass)

    # Knuckle (sphere-like cap)
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.018, location=(W-0.12, D/2+0.15, H*0.48))
    knuckle = bpy.context.active_object; assign(knuckle, brass)

    # Rose / backplate
    bpy.ops.mesh.primitive_cylinder_add(radius=0.022, depth=0.012,
        location=(W-0.12, D/2+0.01, H*0.48))
    rose = bpy.context.active_object
    rose.rotation_euler=[math.radians(90),0,0]; bpy.ops.object.transform_apply(rotation=True)
    assign(rose, brass)

    # Hinges x3 (decorative, left side)
    for hz in [H*0.15, H*0.50, H*0.85]:
        h = box(f"hinge{hz}",(0.032,0.012,0.10),(0.016, -D/2-0.003, hz)); assign(h, metal)

    # Strike plate hint on right edge
    sp = box("strike",(0.025,0.008,0.06),(W-0.01,-D/2+0.004,H*0.48)); assign(sp, metal)

    join_all(); export("door_panel")
    print("NOTE: door_panel origin is at hinge axis (x=0, left edge)")

# ══════════════════════════════════════════════════════════════════════════════
# RUN ALL
# ══════════════════════════════════════════════════════════════════════════════
print("Generating bedroom assets...")
make_bed()          ; print("✓ bed")
make_wardrobe()     ; print("✓ wardrobe")
make_nightstand()   ; print("✓ nightstand")
make_desk()         ; print("✓ desk")
make_desk_chair()   ; print("✓ desk_chair")
make_lamp()         ; print("✓ floor_lamp_bedroom")
make_dresser()      ; print("✓ dresser")
make_rug()          ; print("✓ bedroom_rug")
make_door_frame()   ; print("✓ door_frame")
make_door_panel()   ; print("✓ door_panel")
print(f"\nAll assets saved to: {OUT}")

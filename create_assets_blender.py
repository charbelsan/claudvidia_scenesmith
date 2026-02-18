"""
Generate procedural 3D furniture and household objects using Blender Python (bpy).
Exports each as a USD file to outputs/home_scene/assets/.

Run with:
    .venv311/bin/python create_assets_blender.py
"""

import math
import sys
from pathlib import Path

import bpy
import bmesh
import mathutils

OUTPUT_DIR = Path("outputs/home_scene/assets")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Helpers ─────────────────────────────────────────────────────────────────

def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    for block in list(bpy.data.meshes):
        bpy.data.meshes.remove(block, do_unlink=True)
    for block in list(bpy.data.materials):
        bpy.data.materials.remove(block, do_unlink=True)


def make_material(name, color, roughness=0.7, metallic=0.0, alpha=1.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    if alpha < 1.0:
        mat.blend_method = 'BLEND'
        bsdf.inputs["Alpha"].default_value = alpha
    return mat


def assign_mat(obj, mat):
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)


def add_box(name, size, location, rotation=(0,0,0)):
    bpy.ops.mesh.primitive_cube_add(size=1, location=location, rotation=rotation)
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = size
    bpy.ops.object.transform_apply(scale=True)
    return obj


def add_cylinder(name, radius, depth, location, rotation=(0,0,0)):
    bpy.ops.mesh.primitive_cylinder_add(
        radius=radius, depth=depth,
        location=location, rotation=rotation,
        vertices=16
    )
    obj = bpy.context.active_object
    obj.name = name
    return obj


def bevel_obj(obj, width=0.02, segments=2):
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    mod = obj.modifiers.new("Bevel", 'BEVEL')
    mod.width = width
    mod.segments = segments


def export_usd(name):
    path = str(OUTPUT_DIR / f"{name}.usd")
    bpy.ops.wm.usd_export(
        filepath=path,
        selected_objects_only=False,
        export_materials=True,
        export_uvmaps=True,
        export_normals=True,
    )
    print(f"  Exported: {path}")
    return path


def join_all():
    """Join all mesh objects in scene into one."""
    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            obj.select_set(True)
    if bpy.context.selected_objects:
        bpy.context.view_layer.objects.active = bpy.context.selected_objects[0]
        bpy.ops.object.join()
    return bpy.context.active_object


# ── 1. SOFA ─────────────────────────────────────────────────────────────────
def make_sofa():
    clear_scene()
    fabric = make_material("fabric_blue", (0.20, 0.30, 0.55), roughness=0.9)
    leg_mat = make_material("wood_dark", (0.25, 0.15, 0.08), roughness=0.6)

    # Base frame
    base = add_box("base", (2.2, 0.9, 0.25), (0, 0, 0.125))
    assign_mat(base, fabric)

    # Back cushion
    back = add_box("back", (2.2, 0.15, 0.6), (0, -0.375, 0.55))
    assign_mat(back, fabric)

    # Seat cushions (3)
    for i, x in enumerate([-0.7, 0.0, 0.7]):
        c = add_box(f"cushion_{i}", (0.65, 0.75, 0.15), (x, 0.0, 0.325))
        assign_mat(c, fabric)

    # Armrests
    for x in [-1.05, 1.05]:
        arm = add_box(f"arm_{x}", (0.1, 0.9, 0.45), (x, 0.0, 0.35))
        assign_mat(arm, fabric)

    # Legs (4)
    for x, y in [(-0.95, -0.35), (0.95, -0.35), (-0.95, 0.35), (0.95, 0.35)]:
        leg = add_box(f"leg_{x}_{y}", (0.06, 0.06, 0.12), (x, y, 0.06))
        assign_mat(leg, leg_mat)

    join_all()
    export_usd("sofa")


# ── 2. ARMCHAIR ──────────────────────────────────────────────────────────────
def make_armchair():
    clear_scene()
    fabric = make_material("fabric_brown", (0.45, 0.28, 0.15), roughness=0.9)
    leg_mat = make_material("wood_walnut", (0.35, 0.20, 0.08), roughness=0.5)

    base = add_box("base", (0.8, 0.8, 0.22), (0, 0, 0.11))
    assign_mat(base, fabric)
    back = add_box("back", (0.8, 0.12, 0.55), (0, -0.34, 0.53))
    assign_mat(back, fabric)
    seat = add_box("seat", (0.7, 0.68, 0.13), (0, 0, 0.285))
    assign_mat(seat, fabric)
    for x in [-0.38, 0.38]:
        arm = add_box(f"arm_{x}", (0.08, 0.8, 0.38), (x, 0, 0.30))
        assign_mat(arm, fabric)
    for x, y in [(-0.33, -0.30), (0.33, -0.30), (-0.33, 0.30), (0.33, 0.30)]:
        leg = add_box(f"l_{x}_{y}", (0.05, 0.05, 0.12), (x, y, 0.06))
        assign_mat(leg, leg_mat)

    join_all()
    export_usd("armchair")


# ── 3. COFFEE TABLE ──────────────────────────────────────────────────────────
def make_coffee_table():
    clear_scene()
    top_mat = make_material("glass", (0.85, 0.92, 0.95), roughness=0.05, metallic=0.0, alpha=0.7)
    frame_mat = make_material("metal_black", (0.08, 0.08, 0.08), roughness=0.3, metallic=0.9)

    # Glass top
    top = add_box("top", (1.2, 0.6, 0.03), (0, 0, 0.44))
    assign_mat(top, top_mat)

    # Metal frame legs
    for x, y in [(-0.5, -0.22), (0.5, -0.22), (-0.5, 0.22), (0.5, 0.22)]:
        leg = add_box(f"leg_{x}_{y}", (0.03, 0.03, 0.44), (x, y, 0.22))
        assign_mat(leg, frame_mat)

    # Bottom shelf
    shelf = add_box("shelf", (1.1, 0.5, 0.02), (0, 0, 0.06))
    assign_mat(shelf, frame_mat)

    join_all()
    export_usd("coffee_table")


# ── 4. TV + STAND ────────────────────────────────────────────────────────────
def make_tv_stand():
    clear_scene()
    wood_mat = make_material("wood_oak", (0.55, 0.38, 0.18), roughness=0.5)
    metal_mat = make_material("metal_dark", (0.12, 0.12, 0.12), roughness=0.3, metallic=0.8)
    screen_mat = make_material("screen_black", (0.02, 0.02, 0.03), roughness=0.8)
    screen_glow = make_material("screen_glow", (0.05, 0.08, 0.15), roughness=0.9)

    # Stand body
    stand = add_box("stand", (1.5, 0.4, 0.5), (0, 0, 0.25))
    assign_mat(stand, wood_mat)

    # Stand legs
    for x in [-0.65, 0.65]:
        leg = add_box(f"sleg_{x}", (0.05, 0.35, 0.05), (x, 0, 0.025))
        assign_mat(leg, metal_mat)

    # Drawer line detail
    for y_off, z in [(0, 0.38), (0, 0.18)]:
        detail = add_box(f"d_{z}", (1.4, 0.01, 0.02), (0, -0.2, z))
        assign_mat(detail, metal_mat)

    # TV screen
    tv = add_box("tv", (1.6, 0.07, 0.95), (0, -0.22, 1.0))
    assign_mat(tv, screen_mat)

    # Screen face
    face = add_box("face", (1.52, 0.01, 0.87), (0, -0.255, 1.0))
    assign_mat(face, screen_glow)

    # TV stand mount
    mount = add_box("mount", (0.06, 0.06, 0.18), (0, -0.21, 0.54))
    assign_mat(mount, metal_mat)

    join_all()
    export_usd("tv_stand")


# ── 5. FLOOR LAMP ────────────────────────────────────────────────────────────
def make_floor_lamp():
    clear_scene()
    metal_mat = make_material("brass", (0.72, 0.55, 0.18), roughness=0.3, metallic=0.9)
    shade_mat = make_material("shade_white", (0.95, 0.90, 0.80), roughness=0.8)
    base_mat = make_material("metal_heavy", (0.15, 0.15, 0.15), roughness=0.4, metallic=0.7)

    # Weighted base
    base = add_box("base", (0.28, 0.28, 0.05), (0, 0, 0.025))
    assign_mat(base, base_mat)

    # Pole
    pole = add_cylinder("pole", 0.015, 1.55, (0, 0, 0.80))
    assign_mat(pole, metal_mat)

    # Arm
    arm = add_box("arm", (0.3, 0.015, 0.015), (-0.15, 0, 1.55))
    assign_mat(arm, metal_mat)

    # Shade (cone approximation - tapered box)
    shade = add_cylinder("shade", 0.22, 0.35, (0, 0, 1.72))
    assign_mat(shade, shade_mat)

    # Bulb glow
    bulb = add_cylinder("bulb", 0.04, 0.06, (0, 0, 1.65))
    bulb_mat = make_material("bulb", (1.0, 0.95, 0.75), roughness=0.1)
    assign_mat(bulb, bulb_mat)

    join_all()
    export_usd("floor_lamp")


# ── 6. BOOKSHELF ─────────────────────────────────────────────────────────────
def make_bookshelf():
    clear_scene()
    wood_mat = make_material("wood_pine", (0.65, 0.48, 0.28), roughness=0.6)

    # Back panel
    back = add_box("back", (1.0, 0.04, 1.8), (0, 0.23, 0.9))
    assign_mat(back, wood_mat)
    # Left + right panels
    for x in [-0.5, 0.5]:
        side = add_box(f"side_{x}", (0.04, 0.45, 1.8), (x, 0, 0.9))
        assign_mat(side, wood_mat)

    # Shelves (5)
    for z in [0.02, 0.36, 0.72, 1.08, 1.44, 1.78]:
        shelf = add_box(f"shelf_{z}", (0.96, 0.42, 0.03), (0, -0.01, z))
        assign_mat(shelf, wood_mat)

    # Books on shelves (colorful rectangles)
    book_colors = [
        (0.8, 0.1, 0.1), (0.1, 0.4, 0.8), (0.1, 0.7, 0.2),
        (0.9, 0.6, 0.1), (0.6, 0.1, 0.7), (0.1, 0.6, 0.6),
    ]
    for si, (sz, bc) in enumerate(zip([0.20, 0.56, 0.92, 1.28], book_colors[:4])):
        for bi, bx in enumerate([-0.38, -0.22, -0.06, 0.10, 0.26, 0.40]):
            c = book_colors[(si * 6 + bi) % len(book_colors)]
            bk = add_box(f"book_{si}_{bi}", (0.04, 0.25, 0.18), (bx, -0.05, sz + 0.09))
            bk_mat = make_material(f"bk_{si}_{bi}", c, roughness=0.8)
            assign_mat(bk, bk_mat)

    join_all()
    export_usd("bookshelf")


# ── 7. DINING TABLE ──────────────────────────────────────────────────────────
def make_dining_table():
    clear_scene()
    wood_mat = make_material("wood_walnut_t", (0.40, 0.26, 0.12), roughness=0.5)
    leg_mat = make_material("metal_leg", (0.15, 0.13, 0.10), roughness=0.4, metallic=0.3)

    # Table top
    top = add_box("top", (1.6, 0.9, 0.05), (0, 0, 0.755))
    assign_mat(top, wood_mat)
    bevel_obj(top, 0.01)

    # Apron
    for x, sz in [((0, -0.4, 0.73), (1.52, 0.04, 0.08)), ((0, 0.4, 0.73), (1.52, 0.04, 0.08)),
                  ((-0.75, 0, 0.73), (0.04, 0.82, 0.08)), ((0.75, 0, 0.73), (0.04, 0.82, 0.08))]:
        ap = add_box(f"apron_{x}", sz, x)
        assign_mat(ap, wood_mat)

    # Legs (4)
    for x, y in [(-0.7, -0.35), (0.7, -0.35), (-0.7, 0.35), (0.7, 0.35)]:
        leg = add_box(f"tleg_{x}_{y}", (0.06, 0.06, 0.72), (x, y, 0.36))
        assign_mat(leg, leg_mat)
        bevel_obj(leg, 0.008)

    join_all()
    export_usd("dining_table")


# ── 8. DINING CHAIR ──────────────────────────────────────────────────────────
def make_dining_chair():
    clear_scene()
    wood_mat = make_material("wood_chair", (0.50, 0.33, 0.16), roughness=0.55)
    seat_mat = make_material("seat_fabric", (0.60, 0.55, 0.45), roughness=0.85)

    # Seat
    seat = add_box("seat", (0.45, 0.45, 0.05), (0, 0, 0.47))
    assign_mat(seat, seat_mat)

    # Backrest slats
    for z in [0.60, 0.70, 0.80]:
        slat = add_box(f"slat_{z}", (0.40, 0.025, 0.06), (0, -0.21, z))
        assign_mat(slat, wood_mat)

    # Back uprights
    for x in [-0.18, 0.18]:
        up = add_box(f"up_{x}", (0.025, 0.025, 0.42), (x, -0.21, 0.63))
        assign_mat(up, wood_mat)

    # Front legs
    for x in [-0.18, 0.18]:
        fl = add_box(f"fl_{x}", (0.03, 0.03, 0.47), (x, 0.18, 0.235))
        assign_mat(fl, wood_mat)

    # Back legs (angled via location cheat)
    for x in [-0.18, 0.18]:
        bl = add_box(f"bl_{x}", (0.03, 0.03, 0.55), (x, -0.21, 0.275))
        assign_mat(bl, wood_mat)

    # Stretchers
    for y in [0.0]:
        st = add_box(f"st_{y}", (0.38, 0.025, 0.025), (0, y, 0.22))
        assign_mat(st, wood_mat)

    join_all()
    export_usd("dining_chair")


# ── 9. KITCHEN ISLAND ────────────────────────────────────────────────────────
def make_kitchen_island():
    clear_scene()
    cab_mat = make_material("cabinet_white", (0.92, 0.90, 0.87), roughness=0.4)
    top_mat = make_material("marble", (0.88, 0.86, 0.85), roughness=0.2, metallic=0.0)
    handle_mat = make_material("metal_handle", (0.75, 0.75, 0.75), roughness=0.2, metallic=0.9)

    # Cabinet body
    body = add_box("body", (1.8, 0.6, 0.9), (0, 0, 0.45))
    assign_mat(body, cab_mat)

    # Marble countertop (slightly overhanging)
    top = add_box("top", (1.84, 0.64, 0.06), (0, 0, 0.93))
    assign_mat(top, top_mat)

    # Door fronts (3)
    for x in [-0.58, 0.0, 0.58]:
        door = add_box(f"door_{x}", (0.54, 0.02, 0.84), (x, -0.31, 0.45))
        assign_mat(door, cab_mat)
        # Handle
        h = add_cylinder(f"h_{x}", 0.008, 0.25, (x, -0.33, 0.45), rotation=(math.pi/2, 0, 0))
        assign_mat(h, handle_mat)

    join_all()
    export_usd("kitchen_island")


# ── 10. REFRIGERATOR ─────────────────────────────────────────────────────────
def make_refrigerator():
    clear_scene()
    body_mat = make_material("fridge_stainless", (0.75, 0.75, 0.75), roughness=0.25, metallic=0.6)
    handle_mat = make_material("fridge_handle", (0.85, 0.85, 0.85), roughness=0.15, metallic=0.9)
    seal_mat = make_material("rubber_seal", (0.1, 0.1, 0.1), roughness=0.95)

    # Main body
    body = add_box("body", (0.8, 0.7, 1.85), (0, 0, 0.925))
    assign_mat(body, body_mat)

    # Door split line
    line = add_box("split", (0.78, 0.01, 0.015), (0, -0.355, 1.12))
    assign_mat(line, seal_mat)

    # Top door (fridge)
    for h in [0.18, 0.36]:
        l = add_box(f"hinge_{h}", (0.78, 0.005, 0.012), (0, -0.353, 1.15 + h * 0.5))
        assign_mat(l, seal_mat)

    # Handle (top door)
    handle = add_cylinder("handle_top", 0.015, 0.65, (-0.32, -0.38, 1.5), rotation=(0, math.pi/2, 0))
    assign_mat(handle, handle_mat)

    # Handle (bottom drawer)
    handle2 = add_cylinder("handle_bot", 0.015, 0.65, (-0.32, -0.38, 0.55), rotation=(0, math.pi/2, 0))
    assign_mat(handle2, handle_mat)

    join_all()
    export_usd("refrigerator")


# ── 11. STOVE / RANGE ────────────────────────────────────────────────────────
def make_stove():
    clear_scene()
    body_mat = make_material("stove_black", (0.12, 0.12, 0.12), roughness=0.3, metallic=0.5)
    top_mat = make_material("stove_top", (0.08, 0.08, 0.08), roughness=0.15, metallic=0.6)
    burner_mat = make_material("burner", (0.25, 0.25, 0.25), roughness=0.4, metallic=0.3)
    knob_mat = make_material("knob_metal", (0.60, 0.58, 0.55), roughness=0.3, metallic=0.7)

    body = add_box("body", (0.75, 0.65, 0.9), (0, 0, 0.45))
    assign_mat(body, body_mat)

    # Cooktop surface
    top = add_box("cooktop", (0.73, 0.63, 0.025), (0, 0, 0.912))
    assign_mat(top, top_mat)

    # 4 burners
    for x, y in [(-0.18, -0.13), (0.18, -0.13), (-0.18, 0.13), (0.18, 0.13)]:
        b = add_cylinder(f"burner_{x}_{y}", 0.10, 0.015, (x, y, 0.928))
        assign_mat(b, burner_mat)
        # Inner ring
        bi = add_cylinder(f"binner_{x}_{y}", 0.04, 0.018, (x, y, 0.929))
        assign_mat(bi, body_mat)

    # Front panel / knobs
    panel = add_box("panel", (0.73, 0.02, 0.12), (0, -0.335, 0.87))
    assign_mat(panel, body_mat)
    for kx in [-0.28, -0.10, 0.10, 0.28]:
        knob = add_cylinder(f"knob_{kx}", 0.025, 0.035, (kx, -0.346, 0.87))
        assign_mat(knob, knob_mat)

    # Oven door
    door = add_box("oven_door", (0.71, 0.02, 0.52), (0, -0.335, 0.38))
    assign_mat(door, body_mat)

    # Oven handle
    h = add_cylinder("oven_handle", 0.012, 0.60, (0, -0.36, 0.63), rotation=(0, math.pi/2, 0))
    assign_mat(h, knob_mat)

    join_all()
    export_usd("stove")


# ── 12. KITCHEN COUNTER / CABINET ────────────────────────────────────────────
def make_kitchen_counter():
    clear_scene()
    cab_mat = make_material("cabinet_cream", (0.90, 0.87, 0.80), roughness=0.4)
    top_mat = make_material("counter_top", (0.82, 0.80, 0.78), roughness=0.25)
    handle_mat = make_material("cab_handle", (0.70, 0.68, 0.65), roughness=0.2, metallic=0.85)

    # Lower cabinet
    lower = add_box("lower", (1.8, 0.6, 0.88), (0, 0, 0.44))
    assign_mat(lower, cab_mat)

    # Countertop
    ct = add_box("countertop", (1.84, 0.64, 0.04), (0, 0, 0.9))
    assign_mat(ct, top_mat)

    # Cabinet doors (4)
    for x in [-0.65, -0.22, 0.22, 0.65]:
        d = add_box(f"cdoor_{x}", (0.38, 0.02, 0.80), (x, -0.31, 0.44))
        assign_mat(d, cab_mat)
        h = add_cylinder(f"chandle_{x}", 0.007, 0.15, (x, -0.325, 0.65), rotation=(0, 0, 0))
        assign_mat(h, handle_mat)

    # Upper cabinets
    upper = add_box("upper", (1.8, 0.35, 0.65), (0, 0.125, 1.65))
    assign_mat(upper, cab_mat)

    for x in [-0.65, -0.22, 0.22, 0.65]:
        ud = add_box(f"udoor_{x}", (0.38, 0.02, 0.60), (x, -0.055, 1.65))
        assign_mat(ud, cab_mat)
        uh = add_cylinder(f"uhandle_{x}", 0.007, 0.15, (x, -0.07, 1.85), rotation=(0, 0, 0))
        assign_mat(uh, handle_mat)

    join_all()
    export_usd("kitchen_counter")


# ── 13. PLANT (POTTED) ───────────────────────────────────────────────────────
def make_plant():
    clear_scene()
    pot_mat = make_material("terracotta", (0.72, 0.38, 0.22), roughness=0.8)
    soil_mat = make_material("soil", (0.22, 0.15, 0.09), roughness=0.95)
    leaf_mat = make_material("leaf_green", (0.10, 0.45, 0.12), roughness=0.8)
    stem_mat = make_material("stem", (0.20, 0.40, 0.15), roughness=0.85)

    # Pot
    pot = add_cylinder("pot", 0.18, 0.32, (0, 0, 0.16))
    assign_mat(pot, pot_mat)

    # Soil top
    soil = add_cylinder("soil", 0.17, 0.02, (0, 0, 0.33))
    assign_mat(soil, soil_mat)

    # Stems + leaves
    rng_seed = 42
    for i in range(7):
        angle = i * (2 * math.pi / 7)
        r = 0.05 + (i % 3) * 0.04
        height = 0.55 + (i % 4) * 0.12
        sx = r * math.cos(angle)
        sy = r * math.sin(angle)
        # Stem
        stem = add_cylinder(f"stem_{i}", 0.012, height - 0.34, (sx * 0.5, sy * 0.5, (height + 0.33) / 2))
        assign_mat(stem, stem_mat)
        # Leaf (ellipsoid approx)
        lx = sx * 4.5
        ly = sy * 4.5
        leaf = add_box(f"leaf_{i}", (0.18, 0.08, 0.02), (lx * 0.05 + sx, ly * 0.05 + sy, height))
        leaf.rotation_euler = (0.3 * math.sin(angle), 0.3 * math.cos(angle), angle)
        bpy.ops.object.transform_apply(rotation=True)
        assign_mat(leaf, leaf_mat)

    join_all()
    export_usd("plant")


# ── 14. AREA RUG ─────────────────────────────────────────────────────────────
def make_rug():
    clear_scene()
    rug_mat = make_material("rug_pattern", (0.55, 0.42, 0.35), roughness=1.0)
    border_mat = make_material("rug_border", (0.32, 0.24, 0.18), roughness=1.0)

    # Main rug
    rug = add_box("rug", (2.4, 1.6, 0.015), (0, 0, 0.0075))
    assign_mat(rug, rug_mat)

    # Border
    for bx, by, bw, bd in [
        (0, -0.76, 2.4, 0.08),
        (0, 0.76, 2.4, 0.08),
        (-1.16, 0, 0.08, 1.6),
        (1.16, 0, 0.08, 1.6),
    ]:
        b = add_box(f"border_{bx}_{by}", (bw, bd, 0.016), (bx, by, 0.008))
        assign_mat(b, border_mat)

    join_all()
    export_usd("rug")


# ── 15. SIDE TABLE ───────────────────────────────────────────────────────────
def make_side_table():
    clear_scene()
    wood_mat = make_material("wood_side", (0.50, 0.35, 0.18), roughness=0.55)

    top = add_box("top", (0.5, 0.5, 0.03), (0, 0, 0.58))
    assign_mat(top, wood_mat)
    for x, y in [(-0.2, -0.2), (0.2, -0.2), (-0.2, 0.2), (0.2, 0.2)]:
        leg = add_box(f"leg_{x}_{y}", (0.04, 0.04, 0.58), (x, y, 0.29))
        assign_mat(leg, wood_mat)

    join_all()
    export_usd("side_table")


# ── 16. WALL PICTURE / ART ───────────────────────────────────────────────────
def make_wall_art():
    clear_scene()
    frame_mat = make_material("frame_gold", (0.75, 0.58, 0.20), roughness=0.2, metallic=0.85)
    canvas_mat = make_material("canvas", (0.35, 0.50, 0.68), roughness=0.9)
    accent_mat = make_material("accent_warm", (0.85, 0.65, 0.30), roughness=0.9)

    # Frame
    frame = add_box("frame", (0.8, 0.04, 0.6), (0, 0, 0))
    assign_mat(frame, frame_mat)

    # Canvas face
    canvas = add_box("canvas", (0.72, 0.025, 0.52), (0, 0.01, 0))
    assign_mat(canvas, canvas_mat)

    # Abstract art strokes
    for i, (cx, cz, sw, sh, c) in enumerate([
        (0, 0.05, 0.3, 0.45, (0.25, 0.38, 0.62)),
        (-0.15, -0.06, 0.15, 0.3, (0.78, 0.45, 0.25)),
        (0.12, 0.08, 0.2, 0.15, (0.88, 0.80, 0.55)),
    ]):
        stroke = add_box(f"stroke_{i}", (sw, 0.01, sh), (cx, 0.02, cz))
        sm = make_material(f"stroke_m_{i}", c, roughness=0.9)
        assign_mat(stroke, sm)

    join_all()
    export_usd("wall_art")


# ── 17. PENDANT LIGHT (CEILING) ──────────────────────────────────────────────
def make_pendant_light():
    clear_scene()
    cord_mat = make_material("cord_black", (0.05, 0.05, 0.05), roughness=0.9)
    shade_mat = make_material("shade_copper", (0.72, 0.45, 0.20), roughness=0.25, metallic=0.85)
    bulb_mat = make_material("pendant_bulb", (1.0, 0.92, 0.75), roughness=0.1)

    # Cord
    cord = add_cylinder("cord", 0.008, 0.6, (0, 0, 0.3))
    assign_mat(cord, cord_mat)

    # Shade (inverted cone)
    shade = add_cylinder("shade", 0.22, 0.30, (0, 0, -0.07))
    assign_mat(shade, shade_mat)

    # Bulb
    bulb = add_cylinder("bulb", 0.04, 0.08, (0, 0, -0.04))
    assign_mat(bulb, bulb_mat)

    join_all()
    export_usd("pendant_light")


# ── 18. CEILING LIGHT (FLUSH) ────────────────────────────────────────────────
def make_ceiling_light():
    clear_scene()
    housing_mat = make_material("housing_white", (0.95, 0.95, 0.95), roughness=0.4)
    diffuser_mat = make_material("diffuser", (0.98, 0.95, 0.88), roughness=0.1)

    housing = add_cylinder("housing", 0.25, 0.08, (0, 0, 0))
    assign_mat(housing, housing_mat)
    diffuser = add_cylinder("diffuser", 0.23, 0.04, (0, 0, -0.02))
    assign_mat(diffuser, diffuser_mat)

    join_all()
    export_usd("ceiling_light")


# ── 19. KITCHEN SINK ─────────────────────────────────────────────────────────
def make_sink():
    clear_scene()
    sink_mat = make_material("sink_steel", (0.78, 0.78, 0.78), roughness=0.15, metallic=0.85)
    basin_mat = make_material("basin", (0.65, 0.65, 0.65), roughness=0.2, metallic=0.7)
    tap_mat = make_material("tap_chrome", (0.90, 0.90, 0.88), roughness=0.1, metallic=0.95)

    # Countertop section
    top = add_box("sinktop", (0.75, 0.55, 0.04), (0, 0, 0.92))
    assign_mat(top, sink_mat)

    # Basin
    basin = add_box("basin", (0.42, 0.38, 0.18), (0, 0, 0.83))
    assign_mat(basin, basin_mat)

    # Tap body
    tap_body = add_cylinder("tap_body", 0.018, 0.22, (0, -0.10, 1.05))
    assign_mat(tap_body, tap_mat)
    tap_spout = add_box("tap_spout", (0.15, 0.018, 0.018), (-0.075, -0.10, 1.16))
    assign_mat(tap_spout, tap_mat)

    # Cabinet below
    cab = add_box("sincab", (0.73, 0.53, 0.88), (0, 0, 0.44))
    cab_mat = make_material("sinkcab_mat", (0.90, 0.87, 0.80), roughness=0.4)
    assign_mat(cab, cab_mat)

    join_all()
    export_usd("sink")


# ── 20. BAR STOOL ────────────────────────────────────────────────────────────
def make_bar_stool():
    clear_scene()
    seat_mat = make_material("stool_seat", (0.85, 0.75, 0.60), roughness=0.7)
    frame_mat = make_material("stool_frame", (0.15, 0.15, 0.15), roughness=0.3, metallic=0.8)

    # Seat
    seat = add_cylinder("seat", 0.19, 0.04, (0, 0, 0.75))
    assign_mat(seat, seat_mat)

    # Central pole
    pole = add_cylinder("pole", 0.025, 0.68, (0, 0, 0.40))
    assign_mat(pole, frame_mat)

    # Base (cross)
    for angle in [0, math.pi/2]:
        br = add_box(f"base_r_{angle}", (0.50, 0.04, 0.03), (0, 0, 0.015))
        br.rotation_euler = (0, 0, angle)
        bpy.ops.object.transform_apply(rotation=True)
        assign_mat(br, frame_mat)

    # Foot ring
    ring = add_cylinder("footring", 0.16, 0.025, (0, 0, 0.38))
    assign_mat(ring, frame_mat)

    join_all()
    export_usd("bar_stool")


# ── RUN ALL ──────────────────────────────────────────────────────────────────
ASSETS = [
    ("Sofa",             make_sofa),
    ("Armchair",         make_armchair),
    ("Coffee Table",     make_coffee_table),
    ("TV Stand",         make_tv_stand),
    ("Floor Lamp",       make_floor_lamp),
    ("Bookshelf",        make_bookshelf),
    ("Dining Table",     make_dining_table),
    ("Dining Chair",     make_dining_chair),
    ("Kitchen Island",   make_kitchen_island),
    ("Refrigerator",     make_refrigerator),
    ("Stove",            make_stove),
    ("Kitchen Counter",  make_kitchen_counter),
    ("Plant",            make_plant),
    ("Area Rug",         make_rug),
    ("Side Table",       make_side_table),
    ("Wall Art",         make_wall_art),
    ("Pendant Light",    make_pendant_light),
    ("Ceiling Light",    make_ceiling_light),
    ("Sink",             make_sink),
    ("Bar Stool",        make_bar_stool),
]

print(f"Generating {len(ASSETS)} furniture assets...")
for name, fn in ASSETS:
    print(f"  Creating {name}...")
    try:
        fn()
    except Exception as e:
        print(f"    ERROR: {e}", file=sys.stderr)

print(f"\nAll done! Assets in: {OUTPUT_DIR.resolve()}")

#!/usr/bin/env python3
"""retarget.py — retarget a humanoid BVH clip onto an auto-rigged GLB, headless in Blender 5.x.

usage (Blender 5.2, no add-ons):
  blender -b --factory-startup --python lib/retarget.py -- <rigged.glb> <clip.bvh> <outdir>
          --map <mapping.json> [--clip-name NAME] [--fps 30] [--root-scale auto|FLOAT]
          [--in-place] [--check]

Why (2026-10-02): `m3d animate --engine momask|library` produces BVH clips on the 22-joint
HumanML3D skeleton with Mixamo-style joint names (T-pose rest, Y up, metres, 20 fps);
SkinTokens rigs are bone_N skeletons whose rest pose is whatever the mesh was modelled in
(the knight: arms hanging 67° below horizontal). The free Retarget add-on is not available
headless under --factory-startup, so the math lives here.

Math (world space, per mapped pair source joint S → target bone T, quaternions):
  rest alignment   A_T    = rot(d_T0 → d_S0) · R_T0        d_X0 = rest bone direction, head → primary
                                                           child head (leaves continue their parent)
  per frame        R_T(f) = R_S(f) · R_S0⁻¹ · A_T          the target bone points where the source
                                                           bone points, with the source's twist
  local pose       basis  = pre⁻¹ · R_T(f),  pre = pose(parent) · rest(parent)⁻¹ · rest(T)
                                                           Blender's pose chain; unmapped bones stay
                                                           at rest and their pose follows the chain
  root             p_T(f) = p_T0 + k · (p_S(f) − p_ref)    k = hips height_T / hips height_S
                                                           (--root-scale auto; heights = hips joint
                                                           above the lowest joint of the rest pose),
                                                           p_ref = frame-0 x,y and floor_S + height_S;
                                                           --in-place zeroes the horizontal part
When the two rest poses already agree, A_T = R_T0 and this is the plain R_S·R_S0⁻¹·R_T0 retarget.
Arms (--arm-clearance keep, the default) use the clip's first frame as R_S0 and skip the direction
alignment: the arm starts at the rig's own rest spread and only the motion relative to the first
frame is transferred (a source arm hanging straight down pressed the knight's arm into its flared
skirt); --arm-clearance absolute treats arms like every other bone.

The mapping file is the role tagger's sidecar ({"roles": …, "mixamo": {"bone_0":
"mixamorig:Hips", …}}) or a flat {"bone_0": "Hips", …}; the mixamorig: prefix is dropped and
Mixamo names that differ from the BVH's are aliased (LeftToeBase → LeftToe). Unmapped target
bones keep their rest pose; unmapped source joints are ignored. Resampling evaluates the
source armature at fractional frames (Blender's F-curves interpolate), so any --fps works.
Loops are not fixed: the first/last pose difference is printed.

Output: <outdir>/<stem>.<clip>.glb (skin, materials and exactly one animation named <clip>,
glTF Y up), <stem>.<clip>.blend (target, source armature and a stick mesh of it, cameras),
and with --check <stem>.<clip>-sheet.png: Workbench renders at 0/25/50/75 % of the clip —
rows: target front, target 3/4 from its right, source stick figure front.
Prints `M3D_RETARGET clip=<name> frames=<n> fps=<n> mapped=<n>/<n> out=<glb>` on success and
exits 1 otherwise (Blender exits 0 after a traceback, so the marker is the proof).

Blender 5.x notes: actions are slotted (action.slots / layers / strips / channelbag; the legacy
action.fcurves is gone), bpy.ops.import_anim.bvh converts the file's Y up to Z up itself and
sets Euler modes matching the channel order, the glTF importer puts bones in QUATERNION mode
and may bring an animation along (dropped here), Workbench renders meshes only (hence the
stick mesh for the source armature).
"""
import argparse
import json
import math
import re
import struct
import sys
import time
import traceback
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector

AXIAL = {"spine", "spine1", "spine2", "spine3", "chest", "upperchest", "neck", "head"}
ALIASES = {  # other skeletons' names → HumanML3D/MoMask BVH names (lower-case keys)
    "lefttoebase": "LeftToe", "righttoebase": "RightToe",
    "lefttoe_end": "LeftToe", "righttoe_end": "RightToe",
    "leftupperleg": "LeftUpLeg", "rightupperleg": "RightUpLeg",
    "leftlowerleg": "LeftLeg", "rightlowerleg": "RightLeg",
    "leftupperarm": "LeftArm", "rightupperarm": "RightArm",
    "leftlowerarm": "LeftForeArm", "rightlowerarm": "RightForeArm",
    "chest": "Spine1", "upperchest": "Spine2",
}
TILE_W, TILE_H = 512, 640          # one contact-sheet tile
SHEET_FRACTIONS = (0.0, 0.25, 0.5, 0.75)


def log(msg):
    print(f"[retarget] {msg}", file=sys.stderr, flush=True)


def fail(msg):
    print(f"retarget: {msg}", file=sys.stderr, flush=True)
    sys.exit(1)


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser(prog="retarget.py", description="BVH → rigged GLB retarget (Blender headless)")
    ap.add_argument("glb", help="rigged GLB (skin, bones)")
    ap.add_argument("bvh", help="motion clip (MoMask / Mixamo joint names)")
    ap.add_argument("outdir")
    ap.add_argument("--map", required=True, help="roles sidecar or flat {target_bone: source_joint} JSON")
    ap.add_argument("--clip-name", help="action / glTF animation name; default: the BVH stem")
    ap.add_argument("--fps", type=int, default=30, help="output frame rate (source is resampled)")
    ap.add_argument("--root-scale", default="auto", help="root translation scale: auto (hips height ratio) or a number")
    ap.add_argument("--in-place", action="store_true", help="drop the horizontal root translation (treadmill walk)")
    ap.add_argument("--arm-clearance", choices=("keep", "absolute"), default="keep",
                    help="keep: arms start at the rig's own rest spread and move relative to the clip's first "
                         "frame (a hanging source arm no longer presses into a flared skirt); absolute: "
                         "arms copy the source directions like every other bone")
    ap.add_argument("--check", action="store_true", help="render the contact sheet")
    ap.add_argument("--usdz-root-motion", action=argparse.BooleanOptionalAction, default=True,
                    help="carry the hips' animation on a non-joint parent node 'm3d_mover' so RealityKit plays "
                         "the root motion (Apple's USDZ route keeps only joint rotations); --no-usdz-root-motion "
                         "keeps the plain layout (root motion on the hips joint)")
    ap.add_argument("--prune-weights", type=int, default=0, metavar="N",
                    help="drop skin influences more than N joints away (hierarchy) from a vertex's dominant "
                         "bone and renormalise; a stopgap for soft auto-rig weights that leak onto nearby "
                         "but unrelated bones (SkinTokens: hand → thigh). 0 = keep the weights as they are")
    return ap.parse_args(argv)


def prune_weights(arm, meshes, max_dist):
    """Remove each vertex's influences from bones farther than max_dist hierarchy steps from its
    dominant bone, then renormalise. Returns (vertices changed, weight moved)."""
    names = [b.name for b in arm.data.bones]
    parent = {b.name: b.parent.name if b.parent else None for b in arm.data.bones}

    def path_to_root(n):
        out = []
        while n is not None:
            out.append(n)
            n = parent[n]
        return out
    dist = {}
    for a in names:
        pa = path_to_root(a)
        for b in names:
            pb = path_to_root(b)
            common = next(x for x in pa if x in pb)
            dist[a, b] = pa.index(common) + pb.index(common)
    changed, moved = 0, 0.0
    for ob in meshes:
        groups = {vg.index: vg for vg in ob.vertex_groups if vg.name in parent}
        for v in ob.data.vertices:
            ws = [(g.group, g.weight) for g in v.groups if g.group in groups and g.weight > 0]
            if len(ws) < 2:
                continue
            dom = groups[max(ws, key=lambda x: x[1])[0]].name
            far = [(gi, w) for gi, w in ws if dist[dom, groups[gi].name] > max_dist]
            if not far:
                continue
            keep = sum(w for gi, w in ws) - sum(w for gi, w in far)
            for gi, w in far:
                groups[gi].remove([v.index])
                moved += w
            for gi, w in ws:
                if (gi, w) not in far:
                    groups[gi].add([v.index], w / keep, 'REPLACE')
            changed += 1
    return changed, moved


# ---------------------------------------------------------------- import

def bvh_header(path):
    """(frame count, frame time) from the MOTION header."""
    frames, frametime = 0, 0.0
    with open(path) as f:
        for line in f:
            t = line.split()
            if t[:1] == ["Frames:"]:
                frames = int(t[1])
            elif t[:2] == ["Frame", "Time:"]:
                frametime = float(t[2])
                break
    if not frames or not frametime:
        fail(f"{path}: no MOTION header")
    return frames, frametime


def import_target(glb):
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=str(glb), disable_bone_shape=True, import_scene_extras=True)
    new = [o for o in bpy.data.objects if o not in before]
    arms = [o for o in new if o.type == 'ARMATURE']
    if len(arms) != 1:
        fail(f"{glb}: expected one armature, found {len(arms)}")
    meshes = [o for o in new if o.type == 'MESH']
    # whatever animation came with the file goes: the export must hold exactly one clip
    for o in new:
        if o.animation_data:
            o.animation_data_clear()
    for a in list(bpy.data.actions):
        bpy.data.actions.remove(a)
    return arms[0], meshes


def import_source(bvh):
    before = set(bpy.data.objects)
    bpy.ops.import_anim.bvh(filepath=str(bvh), rotate_mode='NATIVE', frame_start=1, global_scale=1.0,
                            use_fps_scale=False, update_scene_fps=False, update_scene_duration=False)
    new = [o for o in bpy.data.objects if o not in before and o.type == 'ARMATURE']
    if len(new) != 1:
        fail(f"{bvh}: BVH import produced {len(new)} armatures")
    return new[0]


# ---------------------------------------------------------------- mapping

def norm(name):
    return re.sub(r"^mixamorig[:_]?", "", name.strip(), flags=re.I)


def load_map(path, src_bones, tgt_bones):
    data = json.loads(Path(path).read_text())
    raw = data.get("mixamo", data) if isinstance(data, dict) else {}
    by_lower = {b.lower(): b for b in src_bones}
    mapping, unresolved = {}, []
    for t, s in raw.items():
        if t.startswith("_") or not isinstance(s, str):
            continue
        if t not in tgt_bones:
            unresolved.append(f"{t} (not in rig)")
            continue
        n = norm(s)
        hit = None
        for cand in (n, ALIASES.get(n.lower(), n)):
            if cand.lower() in by_lower:
                hit = by_lower[cand.lower()]
                break
        if hit is None:
            unresolved.append(f"{t}→{s} (not in BVH)")
            continue
        mapping[t] = hit
    return mapping, unresolved


# ---------------------------------------------------------------- rest-pose geometry

def rot_q(M):
    return M.to_3x3().normalized().to_quaternion()


def world_rest(arm, bone):
    """World rest matrix of a bone: head at the translation, Y along the bone."""
    return arm.matrix_world @ bone.matrix_local


def head_world(arm, bone):
    return arm.matrix_world @ Vector(bone.head_local)


def primary_child(bone, prefer):
    ch = list(bone.children)
    if not ch:
        return None
    if len(ch) == 1:
        return ch[0]
    for c in ch:
        if prefer(c):
            return c
    return ch[0]


def rest_direction(arm, bone, child):
    """Unit world direction of a bone in the rest pose: to its primary child; a leaf continues its
    parent; a zero-length fallback is the bone's own axis."""
    h = head_world(arm, bone)
    if child is not None:
        d = head_world(arm, child) - h
    elif bone.parent is not None:
        d = h - head_world(arm, bone.parent)
    else:
        d = arm.matrix_world @ Vector(bone.tail_local) - h
    if d.length < 1e-6:
        d = arm.matrix_world @ Vector(bone.tail_local) - h
    return d.normalized()


# ---------------------------------------------------------------- keyframes (slotted actions)

def make_action(name, obj):
    act = bpy.data.actions.new(name)
    slot = act.slots.new(id_type='OBJECT', name=obj.name)
    layer = act.layers.new("retarget")
    strip = layer.strips.new(type='KEYFRAME')
    cb = strip.channelbag(slot, ensure=True)
    return act, slot, cb


def add_curve(cb, path, index, values, group=None):
    fc = cb.fcurves.new(path, index=index)
    n = len(values)
    fc.keyframe_points.add(n)
    co = np.empty(2 * n)
    co[0::2] = np.arange(n)
    co[1::2] = values
    fc.keyframe_points.foreach_set("co", co)
    for kp in fc.keyframe_points:
        kp.interpolation = 'LINEAR'
    if group is not None:
        try:
            fc.group = group
        except Exception:
            pass
    fc.update()
    return fc


# ---------------------------------------------------------------- stick mesh, cameras, sheet

def stick_mesh(arm, radius, name, color):
    """A box per bone, hard-weighted to it, deformed by the armature: the way to render an
    armature with Workbench, which draws meshes only."""
    verts, faces, groups = [], [], []
    for b in arm.data.bones:
        h = Vector(b.head_local)
        # to the only child's head (the glTF importer's tails are a heuristic), else the tail
        t = Vector(b.children[0].head_local) if len(b.children) == 1 else Vector(b.tail_local)
        axis = t - h
        if axis.length < 1e-6:
            continue
        axis.normalize()
        u = axis.orthogonal().normalized()
        v = axis.cross(u)
        base = len(verts)
        for p in (h, t):
            for su, sv in ((1, 1), (1, -1), (-1, -1), (-1, 1)):
                verts.append(tuple(p + radius * (su * u + sv * v)))
        faces += [(base, base + 1, base + 2, base + 3), (base + 4, base + 7, base + 6, base + 5)]
        for i in range(4):
            j = (i + 1) % 4
            faces.append((base + i, base + 4 + i, base + 4 + j, base + j))
        groups.append((b.name, list(range(base, base + 8))))
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    me.update()
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    for gname, idx in groups:
        ob.vertex_groups.new(name=gname).add(idx, 1.0, 'REPLACE')
    ob.modifiers.new("arm", 'ARMATURE').object = arm
    ob.parent = arm  # identity parent inverse: the mesh lives in armature space
    ob.color = color
    return ob


def setup_render(scene):
    scene.render.engine = 'BLENDER_WORKBENCH'
    sh = scene.display.shading
    sh.light = 'STUDIO'
    sh.color_type = 'OBJECT'
    sh.show_object_outline = True
    sh.show_cavity = True
    scene.render.resolution_x, scene.render.resolution_y = TILE_W, TILE_H
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGB'
    scene.render.film_transparent = False
    world = bpy.data.worlds.new("sheet")
    world.color = (0.80, 0.80, 0.84)  # Workbench's render background
    scene.world = world
    scene.view_settings.view_transform = 'Standard'


def make_camera(scene, name, center, height, azimuth_deg, label):
    """Orthographic camera framing `height` world units around `center`, `azimuth_deg` around Z
    from the front (0 = looking along +Y at a character facing −Y), with a text label parented
    to it in the top-left corner."""
    cam = bpy.data.cameras.new(name)
    cam.type = 'ORTHO'
    cam.sensor_fit = 'VERTICAL'
    cam.ortho_scale = height
    cam.clip_start, cam.clip_end = 0.1, 100
    ob = bpy.data.objects.new(name, cam)
    scene.collection.objects.link(ob)
    a = math.radians(azimuth_deg)
    ob["azimuth"] = a
    ob["center"] = tuple(center)
    ob.location = (center.x + 10 * math.sin(a), center.y - 10 * math.cos(a), center.z)
    ob.rotation_euler = (math.pi / 2, 0, a)
    cu = bpy.data.curves.new(name + "-label", 'FONT')
    cu.body = label
    cu.size = height * 0.045
    txt = bpy.data.objects.new(name + "-label", cu)
    scene.collection.objects.link(txt)
    txt.parent = ob
    aspect = TILE_W / TILE_H
    txt.location = (-height * aspect / 2 + height * 0.03, height / 2 - height * 0.08, -1.0)
    txt.color = (0, 0, 0, 1)
    ob["label"] = txt.name
    return ob


def aim_camera(cam, center):
    """Re-centre an existing camera (the walk travels)."""
    a = cam["azimuth"]
    cam.location = (center.x + 10 * math.sin(a), center.y - 10 * math.cos(a), center.z)


def render_tile(scene, cam, visible, path):
    for o in bpy.data.objects:
        if o.type == 'MESH':
            o.hide_render = o not in visible
        elif o.type == 'FONT':
            o.hide_render = o.name != cam["label"]
    scene.camera = cam
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)


def compose_sheet(tiles, rows, cols, out):
    """tiles: row-major list of PNG paths → one PNG, via Blender's image pixels (no PIL in Blender)."""
    sheet = np.ones((rows * TILE_H, cols * TILE_W, 4), dtype=np.float32)
    for i, p in enumerate(tiles):
        img = bpy.data.images.load(str(p))
        w, h = img.size
        buf = np.empty(w * h * 4, dtype=np.float32)
        img.pixels.foreach_get(buf)
        r, c = divmod(i, cols)
        y0 = (rows - 1 - r) * TILE_H  # image rows run bottom-up
        sheet[y0:y0 + h, c * TILE_W:c * TILE_W + w] = buf.reshape(h, w, 4)
        bpy.data.images.remove(img)
    img = bpy.data.images.new("sheet", cols * TILE_W, rows * TILE_H, alpha=True)
    img.pixels.foreach_set(sheet.ravel())
    img.filepath_raw = str(out)
    img.file_format = 'PNG'
    img.save()


# ---------------------------------------------------------------- export

def export_glb(path, tgt, meshes, clip):
    for o in bpy.data.objects:
        o.select_set(o is tgt or o in meshes)
    bpy.context.view_layer.objects.active = tgt
    # ACTIVE_ACTIONS merges the active actions into one glTF animation named by the
    # "merged animation name" setting, not by the action: pass the clip name there too.
    bpy.ops.export_scene.gltf(
        filepath=str(path), export_format='GLB', use_selection=True,
        export_animations=True, export_animation_mode='ACTIVE_ACTIONS', export_force_sampling=True,
        export_nla_strips_merged_animation_name=clip,
        export_frame_range=False, export_anim_slide_to_zero=True, export_optimize_animation_size=False,
        export_rest_position_armature=True, export_reset_pose_bones=True, export_anim_single_armature=True,
        export_skins=True, export_all_influences=False, export_influence_nb=4, export_def_bones=False,
        export_yup=True, export_apply=False, export_extras=True, export_materials='EXPORT',
        export_image_format='AUTO', export_texcoords=True, export_normals=True, export_morph=True)


def glb_patch(path, fn):
    """edit the JSON chunk of a GLB in place"""
    data = Path(path).read_bytes()
    magic, ver, _ = struct.unpack_from('<III', data, 0)
    clen, ctype = struct.unpack_from('<II', data, 12)
    js = json.loads(data[20:20 + clen]); rest = data[20 + clen:]
    out = fn(js)
    jb = json.dumps(js, separators=(",", ":")).encode(); jb += b" " * ((4 - len(jb) % 4) % 4)
    Path(path).write_bytes(struct.pack('<III', magic, ver, 20 + len(jb) + len(rest)) + struct.pack('<II', len(jb), ctype) + jb + rest)
    return out


def glb_root_mover(js, name="m3d_mover"):
    """Root motion for RealityKit (2026-10-02, the viewer animation notes, 2026-10-02, not published "Root motion"; the same function as
    lib/motion.py's). Apple's usdextract keeps only the joint ROTATIONS of a glTF animation in its SkelAnimation
    (translations stay at rest, scales at 1) and RealityKit's skinning ignores the xformOp samples it bakes on the
    joint prims, so a hips translation (a jump, a walk's bob) never shows and the clip may not play at all.
    RealityKit does play transform animation on ordinary entities, so this moves EVERY channel of the skin's root
    joint (the hips: translation and rotation samplers, reused as they are) onto a new non-joint node that becomes
    the parent of the root joint and of the skinned mesh node(s). The mover takes the root's rest TRS and the root
    becomes identity: world placement and the inverse bind matrices are unchanged and glTF consumers see the same
    motion (the joints inherit the mover). Returns the number of channels moved, 0 when there was nothing to move."""
    nodes = js.get("nodes", []); skins = js.get("skins", [])
    if not skins or not nodes: return 0
    joints = set(skins[0]["joints"])
    parent = {c: i for i, n in enumerate(nodes) for c in n.get("children", [])}
    roots = [j for j in skins[0]["joints"] if parent.get(j) not in joints]
    if len(roots) != 1: return 0                                      # several skeleton roots: leave the plain layout
    root = roots[0]; par = parent.get(root)
    moved = [ch for an in js.get("animations", []) for ch in an["channels"] if ch["target"]["node"] == root]
    if not moved: return 0
    meshes = [i for i, n in enumerate(nodes) if n.get("skin") == 0 and parent.get(i) == par]   # siblings of the root
    mv = {"name": name, "children": [root] + meshes}
    for k in ("translation", "rotation", "scale"):
        if k in nodes[root]: mv[k] = nodes[root].pop(k)
    nodes.append(mv); mi = len(nodes) - 1
    drop = set([root] + meshes)
    if par is None:
        for scn in js.get("scenes", []): scn["nodes"] = [n for n in scn["nodes"] if n not in drop] + [mi]
    else:
        nodes[par]["children"] = [n for n in nodes[par]["children"] if n not in drop] + [mi]
    for ch in moved: ch["target"]["node"] = mi
    return len(moved)


def glb_summary(path):
    data = Path(path).read_bytes()
    clen, _ = struct.unpack_from('<II', data, 12)
    js = json.loads(data[20:20 + clen])
    anims = []
    for a in js.get("animations", []):
        acc = js["accessors"]
        t0 = min(acc[s["input"]]["min"][0] for s in a["samplers"])
        t1 = max(acc[s["input"]]["max"][0] for s in a["samplers"])
        anims.append((a.get("name"), len(a["channels"]), t0, t1))
    return anims, len(js.get("skins", [])), len(js.get("materials", [])), len(js.get("images", []))


# ---------------------------------------------------------------- main

def main():
    a = parse_args()
    t_start = time.time()
    glb, bvh, outdir = Path(a.glb).expanduser(), Path(a.bvh).expanduser(), Path(a.outdir).expanduser()
    for p in (glb, bvh, Path(a.map).expanduser()):
        if not p.exists():
            fail(f"missing input {p}")
    outdir.mkdir(parents=True, exist_ok=True)
    clip = a.clip_name or re.sub(r"_ik$", "-ik", bvh.stem)
    stem = glb.stem
    fps_out = a.fps

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene

    tgt, meshes = import_target(glb)
    src = import_source(bvh)
    n_src, frametime = bvh_header(bvh)
    fps_src = 1.0 / frametime
    log(f"target {tgt.name}: {len(tgt.data.bones)} bones, {len(meshes)} mesh(es); "
        f"source {src.name}: {len(src.data.bones)} joints, {n_src} frames @ {fps_src:g} fps")

    if a.prune_weights > 0:
        changed, moved = prune_weights(tgt, meshes, a.prune_weights)
        log(f"pruned influences farther than {a.prune_weights} joints from the dominant bone: "
            f"{changed} vertices changed, {moved:.1f} total weight moved")

    mapping, unresolved = load_map(a.map, [b.name for b in src.data.bones], [b.name for b in tgt.data.bones])
    if unresolved:
        log("unresolved map entries: " + ", ".join(unresolved))
    if not mapping:
        fail("no bone of the rig maps onto a BVH joint")
    inv = {s: t for t, s in mapping.items()}
    src_roots = [b for b in src.data.bones if b.parent is None]
    root_src = src_roots[0].name if src_roots else None
    root_tgt = inv.get(root_src)
    if root_tgt is None:
        log(f"source root {root_src} is not mapped: no root translation will be keyed")

    # rest-pose directions and alignment
    src_bones = {b.name: b for b in src.data.bones}
    tgt_bones = {b.name: b for b in tgt.data.bones}
    s_child = {s: primary_child(src_bones[s], lambda c: norm(c.name).lower() in AXIAL) for s in mapping.values()}
    dirs_S = {s: rest_direction(src, src_bones[s], s_child[s]) for s in mapping.values()}
    dirs_T = {}
    for t, s in mapping.items():
        want = inv.get(s_child[s].name) if s_child[s] is not None else None
        child = primary_child(tgt_bones[t], lambda c, want=want: c.name == want or norm(mapping.get(c.name, "")).lower() in AXIAL)
        dirs_T[t] = rest_direction(tgt, tgt_bones[t], child)
    rest = {b.name: world_rest(tgt, b) for b in tgt.data.bones}
    rest_q = {n: rot_q(M) for n, M in rest.items()}
    q_s0_inv = {s: rot_q(world_rest(src, src_bones[s])).inverted() for s in mapping.values()}

    # target pose order: parents first
    order = []

    def walk(b):
        order.append(b)
        for c in b.children:
            walk(c)
    for b in tgt.data.bones:
        if b.parent is None:
            walk(b)

    # heights for the root scale: hips joint above the lowest joint, both rest poses
    floor_T = min(head_world(tgt, b).z for b in tgt.data.bones)
    floor_S_rest = min(head_world(src, b).z for b in src.data.bones)
    p_T0 = rest[root_tgt].to_translation() if root_tgt else None
    h_T = (p_T0.z - floor_T) if root_tgt else 0.0
    h_S = head_world(src, src_bones[root_src]).z - floor_S_rest if root_src else 0.0
    if a.root_scale == "auto":
        scale = h_T / h_S if h_S > 1e-6 else 1.0
    else:
        scale = float(a.root_scale)

    # sample the source at the output frame times (fractional source frames; F-curves interpolate)
    duration = (n_src - 1) / fps_src
    n_out = math.ceil(duration * fps_out - 1e-6) + 1
    src_names = sorted(set(mapping.values()))
    q_src = {s: [] for s in src_names}
    p_hips, floor_S = [], math.inf
    for k in range(n_out):
        s_frame = 1 + min(k / fps_out * fps_src, n_src - 1)
        fi = int(math.floor(s_frame))
        scene.frame_set(fi, subframe=s_frame - fi)
        for s in src_names:
            q_src[s].append(rot_q(src.matrix_world @ src.pose.bones[s].matrix))
        if root_src:
            p_hips.append((src.matrix_world @ src.pose.bones[root_src].matrix).to_translation())
        floor_S = min(floor_S, min((src.matrix_world @ pb.matrix).to_translation().z for pb in src.pose.bones))
    if root_tgt:
        p_ref = Vector((p_hips[0].x, p_hips[0].y, floor_S + h_S))
        log(f"root: target hips height {h_T:.3f}, source {h_S:.3f} (floor at z={floor_S:.3f} in the clip) "
            f"→ scale {scale:.3f}{' (in place)' if a.in_place else ''}")

    # Rest alignment, walked down the chain: a bone first follows its parent's alignment rigidly,
    # then gets the minimal (swing-only) rotation that points it along the source's reference
    # bone. Aligning every bone independently with its own minimal rotation would put a relative
    # twist between parent and child wherever their rest directions differ in a different plane
    # (the knight's forearm hangs a little forward of its upper arm): a candy-wrapper elbow in
    # every frame. Unmapped bones follow their parent rigidly.
    # Reference per bone: the source rest (T-pose), or for arms with --arm-clearance keep the
    # clip's FIRST FRAME with no direction alignment at all, so a source arm that hangs straight
    # down leaves the target arm at its own rest spread (the knight holds its arms out over a
    # flared skirt; copying the hanging arm absolutely pressed it into the skirt) and only the
    # motion relative to the first frame is transferred.
    arm_bones = {t for t, s in mapping.items() if a.arm_clearance == "keep"
                 and any(x in norm(s).lower() for x in ("shoulder", "arm", "hand"))}
    for t in arm_bones:
        q_s0_inv[mapping[t]] = q_src[mapping[t]][0].inverted()
    A, align = {}, {}
    for b in order:
        pre = rest_q[b.name] if b.parent is None else A[b.parent.name] @ rest_q[b.parent.name].inverted() @ rest_q[b.name]
        if b.name in mapping and b.name not in arm_bones:
            d_now = pre @ (rest_q[b.name].inverted() @ dirs_T[b.name])
            rot = d_now.rotation_difference(dirs_S[mapping[b.name]])
            A[b.name], align[b.name] = rot @ pre, min(rot.angle, 2 * math.pi - rot.angle)
        else:
            A[b.name] = pre
    log("rest alignment (per joint, relative to the aligned parent): "
        + ", ".join(f"{t}→{s} {math.degrees(align[t]):.0f}°" if t in align else f"{t}→{s} kept (first frame)"
                    for t, s in mapping.items()))

    # target pose per frame, parents first
    keys = {t: [] for t in mapping}
    loc_keys, root_targets = [], []
    for k in range(n_out):
        pose = {}
        for b in order:
            pre = rest[b.name] if b.parent is None else pose[b.parent.name] @ rest[b.parent.name].inverted() @ rest[b.name]
            if b.name in mapping:
                s = mapping[b.name]
                qw = q_src[s][k] @ q_s0_inv[s] @ A[b.name]
                p = pre.to_translation()
                if b.name == root_tgt:
                    delta = (p_hips[k] - p_ref) * scale
                    if a.in_place:
                        delta.x = delta.y = 0.0
                    p = p_T0 + delta
                    root_targets.append(p)
                desired = Matrix.Translation(p) @ qw.to_matrix().to_4x4()
                basis = pre.inverted() @ desired
                keys[b.name].append(basis.to_quaternion())
                if b.name == root_tgt:
                    loc_keys.append(basis.to_translation())
                pose[b.name] = desired
            else:
                pose[b.name] = pre

    # keyframes: one slotted action, quaternion rotations, LINEAR, frames 0..n_out-1
    for pb in tgt.pose.bones:
        pb.rotation_mode = 'QUATERNION'
        pb.location = (0, 0, 0)
        pb.rotation_quaternion = (1, 0, 0, 0)
        pb.scale = (1, 1, 1)
    act, slot, cb = make_action(clip, tgt)
    for t, qs in keys.items():
        for i in range(1, len(qs)):
            if qs[i].dot(qs[i - 1]) < 0:
                qs[i].negate()
        grp = None
        try:
            grp = cb.groups.new(t)
        except Exception:
            pass
        for i in range(4):
            add_curve(cb, f'pose.bones["{t}"].rotation_quaternion', i, [q[i] for q in qs], grp)
        if t == root_tgt and loc_keys:
            for i in range(3):
                add_curve(cb, f'pose.bones["{t}"].location', i, [p[i] for p in loc_keys], grp)
    ad = tgt.animation_data_create()
    ad.action = act
    ad.action_slot = slot
    scene.frame_start, scene.frame_end = 0, n_out - 1
    scene.render.fps, scene.render.fps_base = fps_out, 1.0

    # numeric self-check: the target bones must point where the source bones point
    check_frames = sorted({int(round(f * (n_out - 1))) for f in SHEET_FRACTIONS} | {n_out - 1})
    err_dir, err_bone, err_root = 0.0, "", 0.0
    for k in check_frames:
        scene.frame_set(k)
        for t, s in mapping.items():
            Mw = tgt.matrix_world @ tgt.pose.bones[t].matrix
            dT = (Mw.to_3x3().normalized() @ (rest[t].to_3x3().inverted() @ dirs_T[t])).normalized()
            dS = (q_src[s][k] @ q_s0_inv[s] @ A[t] @ (rest_q[t].inverted() @ dirs_T[t])).normalized()
            ang = math.degrees(math.acos(max(-1.0, min(1.0, dT.dot(dS)))))
            if ang > err_dir:
                err_dir, err_bone = ang, t
        if root_tgt:
            err_root = max(err_root, ((tgt.matrix_world @ tgt.pose.bones[root_tgt].matrix).to_translation() - root_targets[k]).length)
    log(f"self-check over frames {check_frames}: max bone direction error {err_dir:.2f}° ({err_bone}), root position error {err_root * 1000:.1f} mm")
    if err_dir > 1.0 or err_root > 0.005:
        fail("retarget self-check failed: the keyed pose does not reproduce the source directions")

    # loop report: first vs last sampled pose (world deltas), never fixed here
    loop_ang, loop_bone = 0.0, ""
    for t, s in mapping.items():
        d = (q_src[s][0] @ q_s0_inv[s]).inverted() @ (q_src[s][-1] @ q_s0_inv[s])
        ang = math.degrees(2 * math.acos(min(1.0, abs(d.normalized().w))))
        if ang > loop_ang:
            loop_ang, loop_bone = ang, t
    root_delta = (p_hips[-1] - p_hips[0]) * scale if root_tgt else Vector((0, 0, 0))
    log(f"loop: first/last pose differ by {loop_ang:.1f}° max ({loop_bone}); root travelled "
        f"({root_delta.x:+.3f}, {root_delta.y:+.3f}, {root_delta.z:+.3f}) in target units"
        f"{' → does not loop' if loop_ang > 5 or root_delta.length > 0.02 else ' → loops'}")

    # inspection scene: stick mesh of the source, cameras; the .blend keeps everything
    stick = stick_mesh(src, 0.018, "source-stick", (1.0, 0.45, 0.10, 1.0))
    tstick = stick_mesh(tgt, 0.012, "target-stick", (0.85, 0.10, 0.10, 1.0))
    bounds_T = [tgt.matrix_world @ Vector(c) for m in meshes for c in m.bound_box] or [head_world(tgt, b) for b in tgt.data.bones]
    zmin, zmax = min(v.z for v in bounds_T), max(v.z for v in bounds_T)
    center_T = Vector(((p_T0.x if root_tgt else 0.0), (p_T0.y if root_tgt else 0.0), (zmin + zmax) / 2))
    height_T = (zmax - zmin) * 1.25
    src_heads = [head_world(src, b) for b in src.data.bones]
    height_S = (max(h.z for h in src_heads) - floor_S_rest) * 1.35
    setup_render(scene)
    cam_front = make_camera(scene, "cam-front", center_T, height_T, 0, f"{clip}: target, front")
    cam_quarter = make_camera(scene, "cam-quarter", center_T, height_T, -35, f"{clip}: target, 3/4 from its right")
    cam_src = make_camera(scene, "cam-source", Vector((p_hips[0].x, p_hips[0].y, floor_S + height_S / 1.35 / 2)), height_S, 0, f"{clip}: source BVH, front")
    blend = outdir / f"{stem}.{clip}.blend"
    bpy.context.preferences.filepaths.save_version = 0  # no .blend1 backups beside the clips
    bpy.ops.wm.save_as_mainfile(filepath=str(blend), compress=True)

    sheet = None
    if a.check:
        tiles_dir = outdir / f".tiles-{clip}"
        tiles_dir.mkdir(exist_ok=True)
        frames = [int(round(f * (n_out - 1))) for f in SHEET_FRACTIONS]
        rows = ((cam_front, meshes, "target front"), (cam_quarter, meshes, "target 3/4"),
                (cam_front, [tstick], "target skeleton"), (cam_src, [stick], "source BVH"))
        tiles = [[] for _ in rows]
        for k in frames:
            scene.frame_set(k)
            root_now = (tgt.matrix_world @ tgt.pose.bones[root_tgt].matrix).to_translation() if root_tgt else center_T
            for cam in (cam_front, cam_quarter):
                aim_camera(cam, Vector((root_now.x, root_now.y, center_T.z)))
            hips_now = (src.matrix_world @ src.pose.bones[root_src].matrix).to_translation() if root_src else Vector()
            aim_camera(cam_src, Vector((hips_now.x, hips_now.y, cam_src["center"][2])))
            for row, (cam, vis, what) in enumerate(rows):
                bpy.data.objects[cam["label"]].data.body = f"{clip}  f{k}/{n_out - 1}  {k / fps_out:.2f}s  {what}"
                p = tiles_dir / f"r{row}-f{k}.png"
                render_tile(scene, cam, set(vis), p)
                tiles[row].append(p)
        sheet = outdir / f"{stem}.{clip}-sheet.png"
        compose_sheet([p for row in tiles for p in row], len(rows), len(frames), sheet)
        for p in tiles_dir.iterdir():
            p.unlink()
        tiles_dir.rmdir()
        log(f"sheet → {sheet}")

    # export: only the rig and its meshes, one action
    for o in list(bpy.data.objects):
        if o is not tgt and o not in meshes:
            bpy.data.objects.remove(o)
    for act_other in list(bpy.data.actions):
        if act_other is not act:
            bpy.data.actions.remove(act_other)
    out_glb = outdir / f"{stem}.{clip}.glb"
    export_glb(out_glb, tgt, meshes, clip)
    moved = glb_patch(out_glb, glb_root_mover) if a.usdz_root_motion else 0
    anims, n_skins, n_mats, n_imgs = glb_summary(out_glb)
    log(f"exported {out_glb.name}: animations {anims}, skins {n_skins}, materials {n_mats}, images {n_imgs}, "
        f"{out_glb.stat().st_size / 1e6:.1f} MB, root channels on m3d_mover {moved}, {time.time() - t_start:.0f}s total")
    if len(anims) != 1 or anims[0][0] != clip:
        fail(f"export holds {len(anims)} animations {[x[0] for x in anims]}, expected one named {clip}")
    print(f"M3D_RETARGET clip={clip} frames={n_out} fps={fps_out} mapped={len(mapping)}/{len(tgt.data.bones)} out={out_glb}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        fail("unhandled exception (above)")

# graft_skin.py — headless Blender: skeleton + skin weights of a rigged GLB -> the ORIGINAL textured GLB.
# Written 2026-10-02 for `m3d rig`: skin-tokens.cpp's `rig` output keeps positions, normals and
# vertex colours only (UVs, textures and materials are dropped), so the rig has to be carried
# back onto the textured input. lib/skintokens.py first tries an exact per-vertex graft (same
# vertex order); this is the robust fallback: nearest-vertex weight transfer, so it also works
# when the rigger re-indexed, welded or dropped vertices.
# Usage:
#   blender -b --factory-startup --python graft_skin.py -- ORIGINAL.glb RIGGED.glb OUTDIR [--name STEM]
# Writes OUTDIR/<STEM>-rigged.glb (STEM = ORIGINAL's stem) with the rigged GLB's armature, the
# original meshes parented to it with transferred vertex groups, the original materials and
# textures, and no animations (the pipeline makes one GLB per clip later). Verifies the file it
# wrote and prints "M3D_GRAFT joints=… textures=… out=…"; Blender exits 0 after a Python
# traceback, so callers must look for that line (same lesson as ai2game.py, 2026-09-26).
import bpy, mathutils, sys, os, json, struct, argparse, math

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("original"); ap.add_argument("rigged"); ap.add_argument("outdir")
ap.add_argument("--name", default="")                   # output stem; default: the original's
ap.add_argument("--far", type=float, default=0.01)       # nearest-vertex distance counted as "far", as a fraction of the bbox diagonal
a = ap.parse_args(argv)
os.makedirs(a.outdir, exist_ok=True)
name = a.name or os.path.splitext(os.path.basename(a.original))[0]
out = os.path.join(a.outdir, f"{name}-rigged.glb")


def glb_json(path):
    with open(path, "rb") as f:
        magic, _, _ = struct.unpack("<III", f.read(12))
        if magic != 0x46546C67: raise SystemExit(f"graft_skin: not a GLB: {path}")
        length, kind = struct.unpack("<II", f.read(8))
        if kind != 0x4E4F534A: raise SystemExit(f"graft_skin: first chunk is not JSON: {path}")
        return json.loads(f.read(length))


def import_glb(path):
    """Imports a GLB and returns the objects it created. merge_vertices stays off so the vertex
    set matches the file (the exact graft in skintokens.py relies on the same order);
    disable_bone_shape stops the importer adding a 42-vertex custom bone-shape mesh that would
    otherwise be taken for geometry (it was, 2026-10-02)."""
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=path, merge_vertices=False, disable_bone_shape=True)
    return [o for o in bpy.data.objects if o not in before]


def flatten(objects):
    """Clear parents and bake object transforms into the data so every mesh and the armature
    share world coordinates (the glTF importer may leave transforms on the root nodes)."""
    bpy.ops.object.select_all(action="DESELECT")
    for o in objects:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]
    bpy.ops.object.parent_clear(type="CLEAR_KEEP_TRANSFORM")
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    bpy.ops.object.select_all(action="DESELECT")


bpy.ops.wm.read_factory_settings(use_empty=True)

# ---------- 1. the rigged GLB: armature + weighted mesh(es) ----------
rig_objs = import_glb(a.rigged)
arms = [o for o in rig_objs if o.type == "ARMATURE"]
rig_meshes = [o for o in rig_objs if o.type == "MESH" and o.vertex_groups]
if len(arms) != 1 or not rig_meshes:
    raise SystemExit(f"graft_skin: {a.rigged} must hold one armature and a weighted mesh "
                     f"(found {len(arms)} armatures, {len(rig_meshes)} weighted meshes)")
arm = arms[0]
flatten([arm] + rig_meshes)
bone_names = [b.name for b in arm.data.bones]

# ---------- 2. the original GLB: meshes, materials, textures ----------
orig_objs = import_glb(a.original)
orig_meshes = [o for o in orig_objs if o.type == "MESH"]
if not orig_meshes:
    raise SystemExit(f"graft_skin: {a.original} has no mesh")
for o in orig_objs:  # an already-rigged original: drop its armature, keep its meshes
    if o.type == "ARMATURE":
        print(f"[graft_skin] original carries an armature '{o.name}'; replacing it")
for om in orig_meshes:
    for mod in [mod for mod in om.modifiers if mod.type == "ARMATURE"]:
        om.modifiers.remove(mod)
    for vg in list(om.vertex_groups):
        om.vertex_groups.remove(vg)
flatten(orig_meshes)
for o in orig_objs:
    if o.type == "ARMATURE":
        bpy.data.objects.remove(o, do_unlink=True)
orig_textures = len(glb_json(a.original).get("textures", []))

# ---------- 3. nearest-vertex weight transfer ----------
rig_verts = []  # (world co, [(bone name, weight)]) over every weighted rigged mesh
for rm in rig_meshes:
    names = [vg.name for vg in rm.vertex_groups]
    for v in rm.data.vertices:
        rig_verts.append((v.co.copy(), [(names[g.group], g.weight) for g in v.groups if g.weight > 0.0]))
kd = mathutils.kdtree.KDTree(len(rig_verts))
for i, (co, _) in enumerate(rig_verts):
    kd.insert(co, i)
kd.balance()
lo = mathutils.Vector([min(c[k] for c, _ in rig_verts) for k in range(3)])
hi = mathutils.Vector([max(c[k] for c, _ in rig_verts) for k in range(3)])
diag = (hi - lo).length or 1.0
max_dist, far, transferred = 0.0, 0, 0
for om in orig_meshes:
    groups = {b: om.vertex_groups.new(name=b) for b in bone_names}
    for v in om.data.vertices:
        _, idx, dist = kd.find(v.co)
        max_dist = max(max_dist, dist)
        if dist > a.far * diag: far += 1
        for bone, w in rig_verts[idx][1]:
            if bone in groups:
                groups[bone].add([v.index], w, "REPLACE")
        transferred += 1
    om.parent = arm
    om.matrix_parent_inverse = arm.matrix_world.inverted()
    mod = om.modifiers.new("Armature", "ARMATURE")
    mod.object = arm
    mod.use_vertex_groups = True
print(f"[graft_skin] {transferred} vertices in {len(orig_meshes)} mesh(es) weighted from {len(rig_verts)} rigged "
      f"vertices; max nearest distance {max_dist:.4g} ({100 * max_dist / diag:.2f}% of the diagonal), {far} far")

# ---------- 4. drop the untextured rigged mesh and the rigger's 1-frame rest clip ----------
for rm in rig_meshes:
    bpy.data.objects.remove(rm, do_unlink=True)
for action in list(bpy.data.actions):
    bpy.data.actions.remove(action)
arm.animation_data_clear()

# ---------- 5. export: armature + original meshes, skin, materials, no animation ----------
bpy.ops.object.select_all(action="DESELECT")
arm.select_set(True)
for om in orig_meshes:
    om.select_set(True)
bpy.context.view_layer.objects.active = arm
bpy.ops.export_scene.gltf(filepath=out, use_selection=True, export_format="GLB", export_yup=True,
                          export_apply=False, export_skins=True, export_all_influences=False,
                          export_influence_nb=4, export_rest_position_armature=True,
                          export_def_bones=False, export_animations=False, export_morph=False,
                          export_materials="EXPORT", export_texcoords=True, export_normals=True)

# ---------- 6. verify what was written ----------
doc = glb_json(out)
skins = doc.get("skins", [])
if not skins:
    raise SystemExit(f"graft_skin: {out} has no skin")
joints = len(skins[0].get("joints", []))
skinned = [n for n in doc.get("nodes", []) if "skin" in n and "mesh" in n]
if not skinned:
    raise SystemExit(f"graft_skin: {out}: no mesh node references the skin")
for n in skinned:
    for p in doc["meshes"][n["mesh"]]["primitives"]:
        if "JOINTS_0" not in p["attributes"] or "WEIGHTS_0" not in p["attributes"]:
            raise SystemExit(f"graft_skin: {out}: a skinned primitive lacks JOINTS_0/WEIGHTS_0")
textures = len(doc.get("textures", []))
if textures < orig_textures:
    raise SystemExit(f"graft_skin: {out} has {textures} textures, the original had {orig_textures}")
print(f"M3D_GRAFT joints={joints} skinned_meshes={len(skinned)} textures={textures} orig_textures={orig_textures} "
      f"animations={len(doc.get('animations', []))} max_nn_dist={max_dist:.4g} far_verts={far} out={out}")

# puppet.py — headless Blender: textured AI GLB (one shell, no skeleton) -> deterministic "puppet" rig:
# a small armature placed from a JSON spec (or a heuristic), numpy skin weights (no bone-heat, which
# fails on fused AI shells), squash/stretch shape keys, exported as a skinned GLB plus a .blend for
# motion.py. Written 2026-10-02 for the m3d rig stage (task T3 of the rig/animate plan); Blender 5.2.2 LTS.
# Usage:
#   blender -b --factory-startup --python puppet.py -- IN.glb OUTDIR --spec spec.json|auto [--check] [--size 600]
# Outputs: OUTDIR/<stem>-rigged.glb, OUTDIR/<stem>-rigged.blend, OUTDIR/<stem>-rigged.roles.json (bone -> role
# sidecar; the same roles are glTF node extras "m3d_role"), OUTDIR/<stem>-spec.json (the spec used, so an auto
# rig can be edited and re-run), and with --check OUTDIR/check-rest.png + check-posed.png.
# m3d treats the stage as successful only when the last line is  M3D_RIG bones=<n> verts=<n> out=<glb>
#
# ---------------------------------------------------------------------------------------------------
# SPEC FORMAT (JSON)
# ---------------------------------------------------------------------------------------------------
# {
#   "name": "07-lamp-robot",                      # optional label
#   "units": "normalized",                         # positions: "normalized" (default) or "metres"
#   "notes": "free text",
#   "bones": [ {bone}, {bone}, ... ]
# }
# Frame: the asset's glTF frame, Y up, +X right, +Z front (towards the viewer). The character faces +Z, so
# its LEFT side is +X: ".L" bones have x > 0.5, ".R" bones x < 0.5 (Blender/Mixamo convention).
# "normalized" positions are bbox fractions 0..1 per axis ((p - bbox_min) / bbox_size), so a spec survives
# rescaling and recentring of the mesh. "metres" positions are raw asset coordinates.
# Radii (sphere, capsule) are fractions of the LARGEST bbox dimension (so a sphere stays a sphere), or
# metres when "units" is "metres". Ellipsoid radii are per-axis normalized (or metres).
#
# bone:
#   "name"     unique; chains get ".01", ".02", ... appended
#   "role"     root | body | head | arm.L | arm.R | leg.L | leg.R | lid | tail | antenna | wheel | spout | other
#              (motion.py keys its presets on roles, never on names; stored on the bone as "m3d_role")
#   "parent"   name of the parent bone (null for the root)
#   "head", "tail"   one bone, or
#   "points"   [p0, p1, ..., pN]: a chain of N connected bones (tails, arms); a wheel bone must point along
#              its axle (motion.py spins it about its own axis)
#   "mask"     which vertices this bone claims:
#              null                       claims nothing (the root)
#              "rest"                     everything no other bone claimed (also the fallback for leftovers)
#              {"box": [[x,y,z],[x,y,z]]} min/max corner
#              {"sphere": {"c": [x,y,z], "r": 0.1}}
#              {"ellipsoid": {"c": [x,y,z], "r": [rx,ry,rz]}}
#              {"capsule": {"r": 0.05, "extend": 0.0}}   around the bone's own segment (every chain bone gets
#                                         its own); "extend" lengthens the segment by that fraction at both ends
#              [shape, shape, ...]        union
#              {"any": [shapes], "not": [shapes]}   union minus a union (e.g. a spout box minus the body ellipsoid)
#              Any shape may carry "feather": 0..1 (default 0.5): for smooth bones the fraction of the
#              shape's radius (or smallest half-size) over which influence fades to zero at the edge.
#   "priority" int, default 0. When masks overlap, only the claimants with the highest priority keep the
#              vertex, so a priority-1 body box carves the body column out of a priority-0 head box.
#   "weights"  "hard" (default): every claimed vertex gets 1.0 on the nearest claiming bone (segment distance).
#              "smooth": the bone's influence is its feathered mask falloff g (1 deep inside, 0 at the edge);
#              overlapping smooth bones (a chain) share g by softened inverse distance 1/(d^2 + soft^2);
#              what is left (1 - g) goes to the nearest hard claimant, else to the smooth bone's anchor (its
#              first non-smooth ancestor: a steam chain fades into the spout, a tail into the body). This
#              keeps the weights continuous at mask edges. At most 4 influences per vertex, normalized.
#   "soft"     softening distance for smooth sharing (same units as radii; default = the capsule radius,
#              else 0.03 of the largest dimension)
#   "cap"      true | "ball". true: close the hard cut around this bone with fan discs (one per connected cut loop of >= 8
#              edges, on BOTH sides of the cut, each weighted to its own side, same material slot, UVs averaged
#              from the loop) so the never-modelled interior does not show when a lid lifts or a head turns.
#              On a chain (arm.R) it caps the cut around the whole chain (the shoulder), not the joints inside.
#              Put a limb's pivot 1-2 cm inside the body surface so the rotation does not open a gap. "cap_reach"
#              (default 0.1 of the largest dimension) limits a chain cap to the cut within that distance of the pivot.
#              "ball": keep the body-side disc but replace the limb-side disc with a UV sphere (24x12) on the
#              pivot, radius 1.15 x the cut loop's, weighted to the chain root, painted with a nearby body texel:
#              a socket that stays closed when a player treats joints as rigid parts (RealityKit via usdextract).
#   "rip"      (top level, optional) true: split the mesh along every hard cut so no triangle straddles two parts
#              (what rigid-part players do anyway); use with caps/balls, which cover the openings.
#   "presets"  (top level, optional) {"wave": {"abduct": 70, ...}}: per-character motion.py tuning, copied into
#              the roles sidecar.
#
# The "auto" spec (--spec auto), for an unknown character, is described at make_auto_spec() below.
# ---------------------------------------------------------------------------------------------------
import bpy, bmesh, sys, os, json, math, argparse
import numpy as np
from mathutils import Vector, Quaternion, Matrix

ROLES = ("root", "body", "head", "arm.L", "arm.R", "leg.L", "leg.R", "lid", "tail", "antenna", "wheel", "spout", "other")

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("inp"); ap.add_argument("outdir")
ap.add_argument("--spec", default="auto")               # spec.json path or "auto"
ap.add_argument("--check", action="store_true")         # Workbench renders of the rest and a test pose
ap.add_argument("--size", type=int, default=600)        # pixel size of each check panel
a = ap.parse_args(argv)
os.makedirs(a.outdir, exist_ok=True)
stem = os.path.splitext(os.path.basename(a.inp))[0]
log = lambda *s: print("[puppet]", *s, flush=True)

# ---------- frame helpers: glTF (x right, y up, z front) <-> Blender (x right, y back, z up) ----------
def gl2bl(p): return Vector((p[0], -p[2], p[1]))
def bl2gl_np(V): return np.stack([V[:, 0], V[:, 2], -V[:, 1]], axis=1)

# ---------- 0. clean scene, import, one mesh in world space ----------
bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
sc.render.engine = "BLENDER_WORKBENCH"
bpy.ops.import_scene.gltf(filepath=a.inp, merge_vertices=True)
meshes = [o for o in sc.objects if o.type == "MESH"]
if not meshes: raise SystemExit("[puppet] no mesh in input")
bpy.ops.object.select_all(action="DESELECT")
for o in meshes: o.select_set(True)
bpy.context.view_layer.objects.active = meshes[0]
if len(meshes) > 1: bpy.ops.object.join()
ob = bpy.context.view_layer.objects.active
bpy.ops.object.parent_clear(type="CLEAR_KEEP_TRANSFORM")
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
for o in list(sc.objects):                              # drop empties / cameras / lights the importer made
    if o.type != "MESH": bpy.data.objects.remove(o, do_unlink=True)
if ob.data.shape_keys:                                  # start from a clean basis
    ob.shape_key_clear()
me = ob.data
nv = len(me.vertices)
Vb = np.empty(nv * 3, np.float32); me.vertices.foreach_get("co", Vb); Vb = Vb.reshape(-1, 3).astype(np.float64)
Vg = bl2gl_np(Vb)                                       # vertices in the glTF frame, metres
lo, hi = Vg.min(0), Vg.max(0); size = hi - lo; maxdim = float(size.max())
Nrm = (Vg - lo) / np.where(size > 1e-9, size, 1.0)      # bbox-normalized 0..1
log(f"mesh verts={nv} tris={sum(len(p.vertices) - 2 for p in me.polygons)} bbox_min={np.round(lo, 3).tolist()} size={np.round(size, 3).tolist()}")

# ---------- 1. the spec ----------
def make_auto_spec():
    """Heuristic default rig for an unknown character (documented here because it IS the documentation):
    - root at the bbox bottom centre, pointing up.
    - body: a hard bone from 2% to 60% of the height on the vertical axis through the bbox centre; it is
      the "rest" bone, so anything not claimed below goes to it.
    - head: 60% -> 98% of the height, hard, claiming the box y >= 0.6 (the body carves nothing out: whatever
      is above 60% is head, so a tilted lampshade reaching below 60% will tear at that plane).
    - arms: look at the band 40-60% of the height for vertices beyond 66% of the half-width on each side
      (|x - 0.5| > 0.33 normalized). A side with >= 1% of all vertices there gets a single hard arm bone
      from a shoulder on the body's flank (x = 0.5 -/+ 0.28, y = 0.57) down to the lateral cluster's lowest
      point, claiming the box beyond x = 0.5 -/+ 0.27 between 10% and 60% of the height, within +-0.25 of
      the cluster's depth. -X is the character's right (arm.R), +X its left (arm.L).
    - everything else is hard-weighted to the body. No legs, lid, tail: those need a hand spec."""
    bones = [
        {"name": "root", "role": "root", "parent": None, "head": [0.5, 0.0, 0.5], "tail": [0.5, 0.1, 0.5], "mask": None},
        {"name": "body", "role": "body", "parent": "root", "head": [0.5, 0.02, 0.5], "tail": [0.5, 0.6, 0.5], "mask": "rest"},
        {"name": "head", "role": "head", "parent": "body", "head": [0.5, 0.6, 0.5], "tail": [0.5, 0.98, 0.5],
         "mask": {"box": [[0, 0.6, 0], [1, 1, 1]]}},
    ]
    band = (Nrm[:, 1] >= 0.4) & (Nrm[:, 1] <= 0.6)
    for side, sign in (("R", -1), ("L", 1)):
        lateral = band & (sign * (Nrm[:, 0] - 0.5) > 0.33)
        if lateral.sum() < 0.01 * nv:
            log(f"auto: no {side} arm (lateral verts in the 40-60% band: {int(lateral.sum())})"); continue
        zc = float(Nrm[lateral, 2].mean())
        arm = (sign * (Nrm[:, 0] - 0.5) > 0.27) & (Nrm[:, 1] >= 0.1) & (Nrm[:, 1] <= 0.6) & (abs(Nrm[:, 2] - zc) < 0.25)
        hand = [float(Nrm[arm, 0].mean()), float(Nrm[arm, 1].min()) + 0.02, float(Nrm[arm, 2].mean())]
        xin = 0.5 + sign * 0.27
        bones.append({"name": f"arm.{side}", "role": f"arm.{side}", "parent": "body",
                      "head": [0.5 + sign * 0.28, 0.57, zc], "tail": hand,
                      "mask": {"box": [[min(xin, 0.5 + sign), 0.1, zc - 0.25], [max(xin, 0.5 + sign), 0.6, zc + 0.25]]},
                      "priority": 1})
        log(f"auto: arm.{side} shoulder=({0.5 + sign * 0.28:.2f},0.57,{zc:.2f}) hand={np.round(hand, 2).tolist()}")
    return {"name": stem, "units": "normalized", "notes": "auto heuristic rig (puppet.py make_auto_spec)", "bones": bones}

if a.spec == "auto":
    spec = make_auto_spec()
else:
    with open(a.spec) as f: spec = json.load(f)
units = spec.get("units", "normalized")
def P(p):
    """spec position -> glTF metres"""
    p = np.asarray(p, np.float64)
    return lo + p * size if units == "normalized" else p
def R(r):
    """spec radius -> metres (fraction of the largest dimension, or metres)"""
    return float(r) * maxdim if units == "normalized" else float(r)
def Rvec(r):
    r = np.asarray(r, np.float64)
    return r * size if units == "normalized" else r

# expand chains into bones; each entry: name, role, parent, head, tail (glTF metres), spec ref, chain index
bones = []
for b in spec["bones"]:
    role = b.get("role", "other")
    if role not in ROLES: raise SystemExit(f"[puppet] bone {b['name']}: unknown role {role!r}; roles: {ROLES}")
    if "points" in b:
        pts = [P(p) for p in b["points"]]
        if len(pts) < 2: raise SystemExit(f"[puppet] chain {b['name']} needs >= 2 points")
        prev = b.get("parent")
        for i in range(len(pts) - 1):
            nm = f"{b['name']}.{i + 1:02d}"
            bones.append(dict(name=nm, role=role, parent=prev, head=pts[i], tail=pts[i + 1], spec=b, chain=i, connect=i > 0))
            prev = nm
    else:
        bones.append(dict(name=b["name"], role=role, parent=b.get("parent"), head=P(b["head"]), tail=P(b["tail"]), spec=b, chain=0, connect=False))
names = [b["name"] for b in bones]
if len(set(names)) != len(names): raise SystemExit(f"[puppet] duplicate bone names: {names}")
for b in bones:
    if b["parent"] is not None and b["parent"] not in names: raise SystemExit(f"[puppet] bone {b['name']}: parent {b['parent']!r} not found")
    if np.linalg.norm(b["tail"] - b["head"]) < 1e-4: b["tail"] = b["head"] + np.array([0, 0.02 * maxdim, 0])
with open(os.path.join(a.outdir, f"{stem}-spec.json"), "w") as f: json.dump(spec, f, indent=1)

# ---------- 2. weights in numpy ----------
def seg_dist(Vp, h, t):
    """distance from every vertex to the segment h-t, and the parameter along it"""
    d = t - h; L2 = float(d @ d)
    u = np.clip(((Vp - h) @ d) / L2, 0, 1) if L2 > 0 else np.zeros(len(Vp))
    return np.linalg.norm(Vp - (h + u[:, None] * d), axis=1), u

def smoothstep(x):
    x = np.clip(x, 0, 1); return x * x * (3 - 2 * x)

def shape_inside(shape, b):
    """-> (inside bool array, g falloff 0..1 for smooth use)"""
    feather = float(shape.get("feather", 0.5))
    if "box" in shape:
        mn, mx = P(shape["box"][0]), P(shape["box"][1])
        mn, mx = np.minimum(mn, mx), np.maximum(mn, mx)
        inside_d = np.minimum(Vg - mn, mx - Vg).min(axis=1)              # metres to the nearest face, >0 inside
        ins = inside_d >= 0
        half = max(float((mx - mn).min()) * 0.5, 1e-6)
        g = smoothstep(inside_d / max(feather * half, 1e-6)) if feather > 0 else ins.astype(float)
    elif "sphere" in shape:
        c, r = P(shape["sphere"]["c"]), R(shape["sphere"]["r"])
        inside_d = r - np.linalg.norm(Vg - c, axis=1)
        ins = inside_d >= 0
        g = smoothstep(inside_d / max(feather * r, 1e-6)) if feather > 0 else ins.astype(float)
    elif "ellipsoid" in shape:
        c, r = P(shape["ellipsoid"]["c"]), Rvec(shape["ellipsoid"]["r"])
        rho = np.linalg.norm((Vg - c) / np.maximum(r, 1e-9), axis=1)
        inside_d = (1 - rho) * float(r.min())
        ins = rho <= 1
        g = smoothstep(inside_d / max(feather * float(r.min()), 1e-6)) if feather > 0 else ins.astype(float)
    elif "capsule" in shape:
        r = R(shape["capsule"].get("r", 0.05)); ext = float(shape["capsule"].get("extend", 0.0))
        h, t = b["head"], b["tail"]; d = t - h
        d_, _ = seg_dist(Vg, h - d * ext, t + d * ext)
        inside_d = r - d_
        ins = inside_d >= 0
        g = smoothstep(inside_d / max(feather * r, 1e-6)) if feather > 0 else ins.astype(float)
    else:
        raise SystemExit(f"[puppet] bone {b['name']}: unknown mask shape {list(shape)}")
    return ins, np.where(ins, g, 0.0)

def mask_eval(mask, b):
    """-> (claims bool array, g falloff) for a bone's mask (None/'rest' handled by the caller)"""
    if isinstance(mask, list): mask = {"any": mask}
    elif "any" not in mask and "not" not in mask: mask = {"any": [mask]}
    anys = mask.get("any", []); nots = mask.get("not", [])
    if isinstance(anys, dict): anys = [anys]
    if isinstance(nots, dict): nots = [nots]
    ins = np.zeros(nv, bool); g = np.zeros(nv)
    for s in anys:
        i, gi = shape_inside(s, b); ins |= i; g = np.maximum(g, gi)
    for s in nots:
        i, gi = shape_inside(s, b); ins &= ~i; g = np.minimum(g, 1 - gi)
    return ins, np.where(ins, g, 0.0)

nb = len(bones)
claims = np.zeros((nb, nv), bool); G = np.zeros((nb, nv)); D = np.zeros((nb, nv))
prio = np.full(nb, -10 ** 6, dtype=np.int64); smooth = np.zeros(nb, bool); soft = np.zeros(nb)
rest_idx = None
for k, b in enumerate(bones):
    D[k], _ = seg_dist(Vg, b["head"], b["tail"])
    s = b["spec"]; mask = s.get("mask")
    smooth[k] = s.get("weights", "hard") == "smooth"
    cap = None
    if isinstance(mask, dict) and "capsule" in mask: cap = mask["capsule"].get("r")
    elif isinstance(mask, list): cap = next((m["capsule"].get("r") for m in mask if "capsule" in m), None)
    elif isinstance(mask, dict) and "any" in mask: cap = next((m["capsule"].get("r") for m in mask["any"] if "capsule" in m), None)
    soft[k] = R(s["soft"]) if "soft" in s else (R(cap) if cap is not None else 0.03 * maxdim)
    if mask is None: continue
    if mask == "rest":
        if rest_idx is not None: raise SystemExit("[puppet] only one bone may have mask 'rest'")
        rest_idx = k; continue
    claims[k], G[k] = mask_eval(mask, b)
    prio[k] = int(s.get("priority", 0))
# fallback bone for leftovers: the rest bone, else the body, else the root
fallback = rest_idx if rest_idx is not None else next((k for k, b in enumerate(bones) if b["role"] == "body"),
                                                       next((k for k, b in enumerate(bones) if b["role"] == "root"), 0))
# priority filter: keep only the highest-priority claimants of each vertex
pr = np.where(claims, prio[:, None], -10 ** 7)
best = pr.max(axis=0)
claims &= (pr == best[None, :])
W = np.zeros((nb, nv))
hard_cl = claims & ~smooth[:, None]; smooth_cl = claims & smooth[:, None]
has_hard = hard_cl.any(axis=0); has_smooth = smooth_cl.any(axis=0)
# anchor of a smooth bone: its first non-smooth ancestor (else the fallback bone)
anchor = np.arange(nb)
for k, b in enumerate(bones):
    if not smooth[k]: continue
    p = b["parent"]; anchor[k] = fallback
    while p is not None:
        j = names.index(p)
        if not smooth[j]: anchor[k] = j; break
        p = bones[j]["parent"]
# the hard share goes to the nearest hard claimant, else to the anchor of the strongest smooth claimant,
# else (unclaimed vertex) to the fallback bone
Dh = np.where(hard_cl, D, np.inf)
hard_target = np.where(has_hard, Dh.argmin(axis=0), fallback)
if has_smooth.any():
    Gs = np.where(smooth_cl, G, 0.0)
    cover = 1 - np.prod(1 - Gs, axis=0)                                       # probabilistic union of falloffs
    share = np.where(smooth_cl, Gs / (D ** 2 + soft[:, None] ** 2), 0.0)
    ssum = share.sum(axis=0)
    share = np.where(ssum > 0, share / np.where(ssum > 0, ssum, 1), 0.0)
    W += cover[None, :] * share
    hard_share = 1 - cover
    hard_target = np.where(has_hard, hard_target, np.where(has_smooth, anchor[share.argmax(axis=0)], fallback))
else:
    hard_share = np.ones(nv)
W[hard_target, np.arange(nv)] += hard_share
# at most 4 influences, renormalize
if nb > 4:
    order = np.argsort(-W, axis=0)
    keep = np.zeros_like(W, dtype=bool)
    keep[order[:4], np.arange(nv)] = True
    W = np.where(keep, W, 0.0)
tot = W.sum(axis=0)
W = W / np.where(tot > 0, tot, 1)
unweighted = int((tot <= 1e-9).sum())
dominant = W.argmax(axis=0)

# ---------- 2b. caps: close hard cuts with fan discs so interiors stay hidden when a part moves ----------
def add_caps():
    global W, dominant, Vb, Vg, nv
    cap_bones = [k for k, b in enumerate(bones) if b["spec"].get("cap") and b["chain"] == 0]
    if not cap_bones: return
    E = np.empty(len(me.edges) * 2, np.int32); me.edges.foreach_get("vertices", E); E = E.reshape(-1, 2)
    bm = bmesh.new(); bm.from_mesh(me); bm.verts.ensure_lookup_table(); bm.faces.ensure_lookup_table()
    uv_layer = bm.loops.layers.uv.active
    vert_uv = {}                                            # first UV seen per vertex, for the new loops
    for f in bm.faces:
        for l in f.loops: vert_uv.setdefault(l.vert.index, tuple(l[uv_layer].uv))
    new_owner = []
    # every new vertex carries its owning bone in a layer: the mesh's vertex order after to_mesh() is not the
    # creation order (found 2026-10-03: ball rings ended up weighted to the wrong bone)
    ownl = bm.verts.layers.int.new("m3d_newowner")
    for v in bm.verts: v[ownl] = -1
    for k in cap_bones:
        # a chain root (arm.R.01) caps the cut around its whole chain, not the elbow inside it
        chain_ids = [j for j, bj in enumerate(bones) if bj["spec"] is bones[k]["spec"]]
        on_k = np.isin(dominant, chain_ids)
        cut = on_k[E[:, 0]] != on_k[E[:, 1]]
        if len(chain_ids) > 1:
            # a limb touches the body elsewhere too (the lamp's forearm rests on its flank): keep only the cut
            # near the pivot, or the shoulder loop merges with that contact seam and the disc slices the chest
            reach = R(bones[k]["spec"].get("cap_reach", 0.1))
            pivot_b = np.array(gl2bl(bones[k]["head"]))
            near_piv = np.linalg.norm(Vb - pivot_b, axis=1) <= reach
            cut &= near_piv[E[:, 0]] & near_piv[E[:, 1]]
        cut_verts = np.unique(E[cut])
        A = cut_verts[on_k[cut_verts]]; B = cut_verts[~on_k[cut_verts]]
        for side in (A, B):
            if len(side) < 3: continue
            inside = np.isin(E[:, 0], side) & np.isin(E[:, 1], side)
            ring = E[inside]
            # connected loops among the ring edges (union-find), one disc each
            parent = {int(v): int(v) for v in side}
            def find(v):
                while parent[v] != v: parent[v] = parent[parent[v]]; v = parent[v]
                return v
            for i, j in ring: parent[find(int(i))] = find(int(j))
            loops = {}
            for i, j in ring: loops.setdefault(find(int(i)), []).append((int(i), int(j)))
            if bones[k]["spec"].get("cap") == "ball" and side is A:
                # socket ball instead of the limb-side disc: a UV sphere on the pivot, 1.15 x the biggest
                # loop's radius, rotating with the chain root so the joint never shows a gap when the app
                # plays the rig as rigid parts (seen on the headset 2026-10-03: the arm visibly separated from the body at the joint)
                edges = max(loops.values(), key=len, default=[])
                if len(edges) < 8: continue
                vids = np.unique(np.array(edges)); lc = Vb[vids].mean(0)
                r = 1.15 * float(np.linalg.norm(Vb[vids] - lc, axis=1).mean())
                pivot_b = np.array(gl2bl(bones[k]["head"]))
                ret = bmesh.ops.create_uvsphere(bm, u_segments=24, v_segments=12, radius=r,
                                                matrix=Matrix.Translation(Vector(pivot_b.tolist())))
                bm.verts.ensure_lookup_table()
                bverts = np.array(sorted(B)); near = int(bverts[np.argmin(np.linalg.norm(Vb[bverts] - lc, axis=1))])
                suv = vert_uv.get(near, (0.5, 0.5))         # one body texel: reads as the shoulder ball's paint
                mat_idx = next((f.material_index for f in bm.verts[near].link_faces), 0)
                for f in {f for v in ret["verts"] for f in v.link_faces}:
                    f.material_index = mat_idx; f.smooth = True
                    for l in f.loops: l[uv_layer].uv = suv
                for v in ret["verts"]: v[ownl] = k
                new_owner.extend([k] * len(ret["verts"]))
                log(f"cap: {bones[k]['name']} socket ball r={r:.3f} m ({len(ret['verts'])} verts) at {np.round(pivot_b, 3).tolist()}, loop {len(edges)} edges")
                continue
            for edges in loops.values():
                if len(edges) < 8: continue
                vids = np.unique(np.array(edges))
                owner = int(np.bincount(dominant[vids], minlength=nb).argmax())
                c = Vb[vids].mean(0)
                region_c = Vb[dominant == owner].mean(0)
                n = c - region_c; n = n / (np.linalg.norm(n) + 1e-12)   # disc faces away from its own region
                vc = bm.verts.new(c.tolist()); bm.verts.ensure_lookup_table(); vc[ownl] = owner
                uvs = [vert_uv.get(int(v), (0.5, 0.5)) for v in vids]
                cuv = (sum(u[0] for u in uvs) / len(uvs), sum(u[1] for u in uvs) / len(uvs))
                mat_idx = next((f.material_index for f in bm.verts[int(vids[0])].link_faces), 0)
                made = 0
                for i, j in edges:
                    vi, vj = bm.verts[i], bm.verts[j]
                    if np.dot(np.cross(Vb[i] - c, Vb[j] - c), n) < 0: vi, vj = vj, vi
                    try: f = bm.faces.new((vc, vi, vj))
                    except ValueError: continue
                    f.material_index = mat_idx; f.smooth = True
                    for l in f.loops: l[uv_layer].uv = cuv if l.vert is vc else vert_uv.get(l.vert.index, cuv)
                    made += 1
                new_owner.append(owner)
                log(f"cap: {bones[k]['name']} cut, {bones[owner]['name']} side: {len(edges)} edges -> {made} triangles at {np.round(c, 3).tolist()}")
    if not new_owner:
        bm.free(); return
    bm.to_mesh(me); bm.free(); me.update()
    nv = len(me.vertices)
    Vb = np.empty(nv * 3, np.float32); me.vertices.foreach_get("co", Vb); Vb = Vb.reshape(-1, 3).astype(np.float64)
    Vg = bl2gl_np(Vb)
    own_attr = np.empty(nv, np.int32); me.attributes["m3d_newowner"].data.foreach_get("value", own_attr)
    me.attributes.remove(me.attributes["m3d_newowner"])
    old_n = W.shape[1]
    if (own_attr[:old_n] >= 0).any() or (own_attr[old_n:] < 0).any():
        log("cap: note: new vertices are not appended in creation order; owners taken from the vertex layer")
    W2 = np.zeros((nb, nv)); W2[:, own_attr < 0] = W[:, :int((own_attr < 0).sum())]
    for vi in np.nonzero(own_attr >= 0)[0]: W2[own_attr[vi], vi] = 1.0
    W = W2; dominant = W.argmax(axis=0)
add_caps()

# ---------- 2c. rip: split the mesh along hard cuts so no triangle straddles two rigid parts ----------
# Players that treat joints as rigid parts (RealityKit via usdextract splits by dominant joint) never show
# the straddling triangles; skinned players stretch them into long dark wedges once a limb swings far
# (lamp wave, 2026-10-03). With "rip": true the hard cuts become open edges, which caps/balls cover.
def rip_hard_cuts():
    global W, dominant, Vb, Vg, nv
    if not spec.get("rip"): return
    bm = bmesh.new(); bm.from_mesh(me); bm.verts.ensure_lookup_table(); bm.faces.ensure_lookup_table()
    orig = bm.verts.layers.int.new("m3d_orig"); own = bm.faces.layers.int.new("m3d_owner")
    for v in bm.verts: v[orig] = v.index
    for f in bm.faces:
        ds = [int(dominant[v.index]) for v in f.verts]
        f[own] = max(set(ds), key=lambda d: (ds.count(d), -ds.index(d)))    # majority, ties -> first vertex
    split = [e for e in bm.edges if len({f[own] for f in e.link_faces}) > 1
             and all(not smooth[f[own]] for f in e.link_faces)]
    if not split:
        bm.free(); return
    pairs = {}
    for e in split:
        key = tuple(sorted({bones[f[own]]["name"] for f in e.link_faces}))
        c = (e.verts[0].co + e.verts[1].co) / 2
        pairs.setdefault(key, []).append((c.x, c.y, c.z))
    for key, cs in sorted(pairs.items(), key=lambda kv: -len(kv[1])):
        log(f"rip: {len(cs):4d} edges between {' / '.join(key)} around {np.round(np.mean(cs, axis=0), 3).tolist()}")
    bmesh.ops.split_edges(bm, edges=split)
    bm.to_mesh(me); bm.free(); me.update()
    nv = len(me.vertices)
    o = np.empty(nv, np.int32); me.attributes["m3d_orig"].data.foreach_get("value", o)
    fo = np.empty(len(me.polygons), np.int32); me.attributes["m3d_owner"].data.foreach_get("value", fo)
    W = W[:, o]
    # a ripped vertex now touches faces of one owner: a hard vertex whose faces belong to another hard bone
    # moves over to that bone (weight 1.0)
    vert_owner = np.full(nv, -1, np.int64); conflict = np.zeros(nv, bool)
    for p in me.polygons:
        for vi in p.vertices:
            if vert_owner[vi] == -1: vert_owner[vi] = fo[p.index]
            elif vert_owner[vi] != fo[p.index]: conflict[vi] = True
    dom_o = dominant[o]
    move = (vert_owner >= 0) & ~conflict & (vert_owner != dom_o) & ~smooth[np.maximum(vert_owner, 0)] & ~smooth[dom_o]
    for vi in np.nonzero(move)[0]:
        W[:, vi] = 0.0; W[vert_owner[vi], vi] = 1.0
    me.attributes.remove(me.attributes["m3d_orig"]); me.attributes.remove(me.attributes["m3d_owner"])
    Vb = np.empty(nv * 3, np.float32); me.vertices.foreach_get("co", Vb); Vb = Vb.reshape(-1, 3).astype(np.float64)
    Vg = bl2gl_np(Vb); dominant = W.argmax(axis=0)
    log(f"rip: split {len(split)} edges along hard cuts -> {nv} verts; {int(move.sum())} boundary verts moved to their face's part")
rip_hard_cuts()

# ---------- 3. armature, vertex groups, modifier ----------
arm_data = bpy.data.armatures.new(f"{stem}-rig")
arm = bpy.data.objects.new(f"{stem}-rig", arm_data)
sc.collection.objects.link(arm)
bpy.ops.object.select_all(action="DESELECT"); arm.select_set(True)
bpy.context.view_layer.objects.active = arm
bpy.ops.object.mode_set(mode="EDIT")
eb = arm_data.edit_bones
for b in bones:
    e = eb.new(b["name"]); e.head = gl2bl(b["head"]); e.tail = gl2bl(b["tail"]); e.roll = 0.0
for b in bones:
    if b["parent"] is not None:
        eb[b["name"]].parent = eb[b["parent"]]
        eb[b["name"]].use_connect = bool(b["connect"])
bpy.ops.object.mode_set(mode="OBJECT")
for b in bones:
    arm_data.bones[b["name"]]["m3d_role"] = b["role"]
    arm.pose.bones[b["name"]]["m3d_role"] = b["role"]
    arm.pose.bones[b["name"]].rotation_mode = "QUATERNION"
for k, b in enumerate(bones):
    vg = ob.vertex_groups.new(name=b["name"])
    idx = np.nonzero(W[k] > 1e-6)[0]
    for i in idx: vg.add([int(i)], float(W[k, i]), "REPLACE")
ob.parent = arm
mod = ob.modifiers.new("Armature", "ARMATURE"); mod.object = arm

# ---------- 4. shape keys: squash / stretch about the bottom centre ----------
# glTF Y is Blender Z. Scale about the bbox bottom centre so feet stay planted.
basis = ob.shape_key_add(name="Basis", from_mix=False)
pivot = np.array([(lo[0] + hi[0]) / 2, -(lo[2] + hi[2]) / 2, lo[1]])     # Blender coords
def add_key(name, sxy, sz):
    k = ob.shape_key_add(name=name, from_mix=False)
    co = (Vb - pivot) * np.array([sxy, sxy, sz]) + pivot
    k.data.foreach_set("co", co.astype(np.float32).ravel())
    k.slider_min, k.slider_max = 0.0, 1.5
    k.value = 0.0          # Blender 5.2 creates new keys with value 1.0 (found 2026-10-02: both keys were on)
    return k
add_key("squash", 1.08, 0.85)
add_key("stretch", 0.95, 1.12)
me.shape_keys.use_relative = True

# ---------- 5. summary ----------
print(f"[puppet] {'bone':<14}{'role':<9}{'dominant':>9}{'influenced':>11}")
for k, b in enumerate(bones):
    print(f"[puppet] {b['name']:<14}{b['role']:<9}{int((dominant == k).sum()):>9}{int((W[k] > 1e-6).sum()):>11}")
print(f"[puppet] unweighted verts: {unweighted} (must be 0); max influences/vertex: {int((W > 1e-6).sum(axis=0).max())}")

# ---------- 6. export GLB + blend (rest pose) ----------
bpy.ops.object.select_all(action="DESELECT"); ob.select_set(True); arm.select_set(True)
bpy.context.view_layer.objects.active = arm
glb = os.path.join(a.outdir, f"{stem}-rigged.glb")
bpy.ops.export_scene.gltf(filepath=glb, export_format="GLB", use_selection=True, export_yup=True,
                          export_apply=False, export_skins=True, export_morph=True, export_morph_normal=True,
                          export_extras=True, export_animations=False, export_image_format="AUTO")
blend = os.path.join(a.outdir, f"{stem}-rigged.blend")
bpy.ops.wm.save_as_mainfile(filepath=blend, copy=True)
# Role contract (team, 2026-10-02): roles travel as glTF node extras {"m3d_role": ...} (export_extras above)
# AND as a sidecar <stem>.roles.json = {bone name: role}, so a rig from any engine can be tagged the same way.
roles_path = os.path.join(a.outdir, f"{stem}-rigged.roles.json")
with open(roles_path, "w") as f:
    json.dump({"roles": {b["name"]: b["role"] for b in bones}, "summary": "puppet",
               "presets": spec.get("presets", {})}, f, indent=1)      # "presets": per-character motion.py tuning
log(f"wrote {glb}, {blend} and {roles_path}")

# ---------- 7. --check: Workbench renders, rest and a test pose ----------
def palette(i):
    h = (i * 0.618034) % 1.0; s, v = 0.65, 0.95
    j = int(h * 6); f = h * 6 - j; p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    return [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)][j % 6]

def rot_world(pb, axis_bl, angle):
    """pose rotation about an armature-space axis, expressed in the bone's rest frame"""
    M = pb.bone.matrix_local.to_3x3()
    q = M.inverted() @ Quaternion(axis_bl, angle).to_matrix() @ M
    pb.rotation_quaternion = q.to_quaternion() @ pb.rotation_quaternion

def bones_with_role(role):
    return sorted([pb for pb in arm.pose.bones if pb.get("m3d_role") == role], key=lambda pb: pb.name)

if a.check:
    import numpy as _np
    size_px = a.size
    # bone colours as a vertex colour attribute, blended by weight
    cols = _np.zeros((nv, 4), _np.float32); cols[:, 3] = 1
    for k in range(nb): cols[:, :3] += (W[k])[:, None] * _np.array(palette(k), _np.float32)
    ca = me.color_attributes.new("bone", "FLOAT_COLOR", "POINT")
    ca.data.foreach_set("color", cols.ravel())
    me.color_attributes.active_color = ca
    me.color_attributes.render_color_index = me.color_attributes.find("bone")
    for m in me.materials:                              # Workbench 'TEXTURE' shows the active image node
        if m and m.node_tree:
            tex = next((n for n in m.node_tree.nodes if n.type == "TEX_IMAGE" and n.outputs["Color"].is_linked
                        and any(l.to_socket.name == "Base Color" for l in n.outputs["Color"].links)), None)
            tex = tex or next((n for n in m.node_tree.nodes if n.type == "TEX_IMAGE"), None)
            if tex: m.node_tree.nodes.active = tex
    world = bpy.data.worlds.new("w"); sc.world = world; world.color = (0.32, 0.33, 0.36)
    sh = sc.display.shading
    sh.light = "STUDIO"; sh.show_shadows = False; sh.show_cavity = True; sh.show_object_outline = True
    sc.display.render_aa = "FXAA"
    sc.render.resolution_x = sc.render.resolution_y = size_px; sc.render.resolution_percentage = 100
    cam = bpy.data.objects.new("check-cam", bpy.data.cameras.new("check-cam")); sc.collection.objects.link(cam)
    cam.data.type = "ORTHO"; sc.camera = cam
    centre = gl2bl((lo + hi) / 2)
    diag = float(np.linalg.norm(size))
    cam.data.ortho_scale = diag * 1.12
    def aim(yaw_deg, elev_deg=14):
        yaw, el = math.radians(yaw_deg), math.radians(elev_deg)
        d = Vector((math.sin(yaw) * math.cos(el), -math.cos(yaw) * math.cos(el), math.sin(el)))
        cam.location = centre + d * diag * 3
        cam.rotation_euler = (-d).to_track_quat("-Z", "Y").to_euler()
        cam.data.clip_end = diag * 10
    def render(path, color_type, yaw):
        aim(yaw); sh.color_type = color_type
        sc.render.filepath = path; bpy.ops.render.render(write_still=True)
        im = bpy.data.images.load(path); px = _np.empty(len(im.pixels), _np.float32); im.pixels.foreach_get(px)
        bpy.data.images.remove(im)
        return px.reshape(size_px, size_px, 4)
    def strip(name):
        tmp = os.path.join(a.outdir, "_panel.png")
        panels = [render(tmp, "TEXTURE", 35), render(tmp, "VERTEX", 35), render(tmp, "VERTEX", 215)]
        out = _np.concatenate(panels, axis=1)
        im = bpy.data.images.new(name, out.shape[1], out.shape[0], alpha=True)
        im.pixels.foreach_set(out.ravel()); im.filepath_raw = os.path.join(a.outdir, name); im.file_format = "PNG"; im.save()
        bpy.data.images.remove(im)
        try: os.remove(tmp)
        except OSError: pass
        log(f"wrote {os.path.join(a.outdir, name)} (panels: textured yaw 35 | bone colours yaw 35 | bone colours yaw 215)")
    strip("check-rest.png")
    # test pose: head yaw 30 deg, arm.R raised 45 deg away from the body, squash 1.0
    for pb in bones_with_role("head"): rot_world(pb, Vector((0, 0, 1)), math.radians(30))
    armR = bones_with_role("arm.R")
    if armR:
        pb = armR[0]
        d = (pb.bone.tail_local - pb.bone.head_local).normalized()
        out_x = -1.0 if pb.bone.head_local.x <= centre.x else 1.0        # away from the body centre
        target = (Vector((out_x, 0, 0)) * 0.6 + Vector((0, 0, 1)) * 0.8).normalized()   # up-and-out, as motion.py
        axis = d.cross(target)
        if axis.length < 1e-3: axis = Vector((0, 1, 0))
        rot_world(pb, axis.normalized(), math.radians(45))
    me.shape_keys.key_blocks["squash"].value = 1.0
    strip("check-posed.png")
    for pb in arm.pose.bones: pb.rotation_quaternion = (1, 0, 0, 0)
    me.shape_keys.key_blocks["squash"].value = 0.0

print(f"M3D_RIG bones={nb} verts={nv} out={glb}")

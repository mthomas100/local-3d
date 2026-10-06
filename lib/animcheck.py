"""animcheck.py clip.glb [--anim 0] [--fps 24] [--frames DIR] [--render] [--rig RIG.glb] [--roles R.json] [--out PREFIX] [--json]
The numeric and visual QA of a skinned animation (worker venv python: numpy, scipy, pygltflib, PIL, matplotlib;
Blender only for --render). Written 2026-10-02 after the lamp-wave clip, whose left arm swung through its own torso
(the lamp animation QA notes, 2026-10-02, not published). Input: an ANIMATED glTF/GLB (skin + animation channels), or a rigged GLB
without animation plus --frames (then only the rest-pose checks and the sheet).

Numeric part (no Blender): samples the glTF animation at --fps, evaluates linear blend skinning in numpy
(node hierarchy → joint world matrices × inverse bind matrices) and reports per frame
  penetration  per limb chain, the fraction of the chain's vertices (those owned by its bones outside the trunk)
               that sit inside the posed trunk's convex hull, inset by 2% of the trunk size, and were not inside
               at rest. A limb swung into the body scores high.
  stretch      posed / rest length over every mesh edge: p99 and max (tearing, candy-wrapper, a clavicle
               dragging torso skin); squash p1 too; hotspots name the bone pairs owning edges stretched > 2x.
  web          skin pulled into existence (2026-10-03, the teddy wave: an arm fused to the torso along its side
               dragged a sheet of belly skin along with the arm, yet the clip only warned):
               per frame, over triangles with an edge > 1.5x rest and a larger posed area, the summed area growth
               as a share of the rest surface (web_area), and the largest connected sheet of them (web_sheet) with
               the bones whose weight it carries. A thin hard seam scores near 0, a taffy web scores a percent.
  bleed        (rest pose) the share of all limb weight that lib/weightfix.py's bleed pass would move: weight of
               a limb's internal bones (clavicles) on trunk skin the limb dominates (limb weight ≥ 0.5), off the
               limb's own surface and away from its attachment joint. Weaker tails are blend falloff. A rig
               defect, not a motion one. Internal bones come from weightfix's record in the skin's extras, else
               from --rig (Blender's glTF export drops skin extras), else they are measured here.
  root         root joint height range over the clip, and how far the lowest vertex sinks below the rest floor,
               both as fractions of the model height.
Limbs and trunk are found from the skeleton alone (lib/weightfix.py auto_limbs): no bone names or roles needed.

Visual part (--frames DIR of frame_####.png, or --render to make them with Blender Workbench, three-quarter view):
<out>-sheet.png, 8 time-stamped frames, a row of difference heatmaps against the first frame (red = changed),
and a strip plotting penetration and stretch per frame with the worst frame marked. LOOK at the worst frame.

Thresholds (fail beats warn beats ok; calibrated on the lamp: bad clip vs fixed clip, see the bench note):
  penetration_max   ok < 0.03  warn < 0.10  fail ≥ 0.10
  stretch_p99       ok < 1.30  warn < 1.60  fail ≥ 1.60    (stretch_max > 4 is a warn by itself)
  web_area          ok < 0.003 warn < 0.010 fail ≥ 0.010   (share of the rest surface area; 2026-10-03 calibration:
  web_sheet         ok < 0.002 warn < 0.005 fail ≥ 0.005    the fixed lamp's four clips 0.0000, the teddy wave
                                                            0.0195/0.0119 (belly skin dragged by the arm),
                                                            the knight MoMask wave 0.103)
  bleed             ok < 0.01  warn < 0.03  fail ≥ 0.03
  floor_sink        ok < 0.02  warn ≥ 0.02 (of model height)
Outputs <out>.json; the last line is
  M3D_ANIMCHECK frames=<n> penetration_max=<f> stretch_p99=<f> web_area=<f> bleed=<f> verdict=<ok|warn|fail>
A metric that cannot be measured prints nan (null in the JSON): a rig with no trunk-owned skin (puppet rigs whose
root/body carry no vertices) has no body to penetrate or bleed into; stretch and floor sink still decide."""
import sys, os, json, glob, argparse, shutil, subprocess, tempfile, numpy as np
from scipy.spatial import ConvexHull
from pygltflib import GLTF2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from weightfix import accessor_view, topology, limbs_and_trunk, load_roles, fix, hull_sd

TH = dict(penetration=(0.03, 0.10), stretch_p99=(1.30, 1.60), bleed=(0.01, 0.03), floor_sink=(0.02, None),
          stretch_max_warn=4.0, web_area=(0.003, 0.010), web_sheet=(0.002, 0.005))


def tri_area(X, F):
    return 0.5 * np.linalg.norm(np.cross(X[F[:, 1]] - X[F[:, 0]], X[F[:, 2]] - X[F[:, 0]]), axis=1)


def web(R, P, F, a0, Wd, names):
    """(web_area, web_sheet, sheet bones): area growth of triangles stretched > 1.5x on some edge, as a share of
    the rest area a0.sum(); the largest vertex-connected sheet of them, and the bones carrying its weight."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    a1 = tri_area(P, F); tot = max(a0.sum(), 1e-12)
    er = np.zeros(len(F))
    for i, j in ((0, 1), (1, 2), (2, 0)):
        er = np.maximum(er, np.linalg.norm(P[F[:, i]] - P[F[:, j]], axis=1) / np.maximum(np.linalg.norm(R[F[:, i]] - R[F[:, j]], axis=1), 1e-12))
    hot = np.where((er > 1.5) & (a1 > a0))[0]
    if not len(hot): return 0.0, 0.0, {}
    grow = a1[hot] - a0[hot]
    A = coo_matrix((np.ones(3 * len(hot)), (np.repeat(np.arange(len(hot)), 3), F[hot].ravel() + len(hot))),
                   shape=(len(hot) + len(R),) * 2)
    _, lab = connected_components(A, directed=False); L = lab[:len(hot)]
    sums = np.bincount(L, weights=grow); k = int(sums.argmax())
    vs = np.unique(F[hot[L == k]]); mw = Wd[vs].mean(0); top = np.argsort(-mw)[:4]
    return float(grow.sum() / tot), float(sums[k] / tot), {names[b]: round(float(mw[b]), 2) for b in top if mw[b] >= 0.05}


def quat_mat(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def slerp(a, b, t):
    d = np.dot(a, b)
    if d < 0: b, d = -b, -d
    if d > 0.9995: r = a + t * (b - a); return r / np.linalg.norm(r)
    th = np.arccos(d); return (np.sin((1 - t) * th) * a + np.sin(t * th) * b) / np.sin(th)


class Clip:
    def __init__(self, path):
        self.g = g = GLTF2().load(path); self.blob = bytearray(g.binary_blob() or b"")
        self.parent = {c: i for i, n in enumerate(g.nodes) for c in (n.children or [])}
        self.rest = []
        for n in g.nodes:
            if n.matrix: self.rest.append(("M", np.array(n.matrix, float).reshape(4, 4).T)); continue
            self.rest.append(("TRS", np.array(n.translation or [0, 0, 0], float), np.array(n.rotation or [0, 0, 0, 1], float),
                              np.array(n.scale or [1, 1, 1], float)))

    def acc(self, i): return accessor_view(self.g, self.blob, i)[0]

    def channels(self, anim):
        a = self.g.animations[anim]; out = []
        for ch in a.channels:
            s = a.samplers[ch.sampler]
            out.append((ch.target.node, ch.target.path, self.acc(s.input).ravel().astype(float),
                        self.acc(s.output).astype(float), s.interpolation or "LINEAR"))
        return out

    @staticmethod
    def sample(times, vals, interp, t, path):
        if interp == "CUBICSPLINE": vals = vals.reshape(len(times), 3, -1)[:, 1]   # keyframe values; tangents ignored
        if t <= times[0]: return vals[0]
        if t >= times[-1]: return vals[-1]
        k = np.searchsorted(times, t) - 1; u = (t - times[k]) / max(times[k + 1] - times[k], 1e-9)
        if interp == "STEP": return vals[k]
        if path == "rotation": return slerp(vals[k], vals[k + 1], u)
        return vals[k] * (1 - u) + vals[k + 1] * u

    def world(self, over=None):
        """world matrices of all nodes, with optional {node: {path: value}} overrides of the rest TRS."""
        L = []
        for i, r in enumerate(self.rest):
            if r[0] == "M": L.append(r[1]); continue
            t, q, s = r[1], r[2], r[3]
            o = (over or {}).get(i, {})
            t = o.get("translation", t); q = o.get("rotation", q); s = o.get("scale", s)
            M = np.eye(4); M[:3, :3] = quat_mat(q / np.linalg.norm(q)) * s; M[:3, 3] = t; L.append(M)
        W = [None] * len(L)
        def w(i):
            if W[i] is None: W[i] = L[i] if i not in self.parent else w(self.parent[i]) @ L[i]
            return W[i]
        return np.array([w(i) for i in range(len(L))])


def skinned_mesh(c):
    g = c.g; skin = g.skins[0]
    prims = [(n, p) for n in g.nodes if n.mesh is not None and n.skin == 0 for p in g.meshes[n.mesh].primitives
             if p.attributes.JOINTS_0 is not None]
    V, J, W, F, o = [], [], [], [], 0
    for n, p in prims:
        v = c.acc(p.attributes.POSITION).astype(float); V.append(v)
        J.append(c.acc(p.attributes.JOINTS_0).astype(int)); W.append(c.acc(p.attributes.WEIGHTS_0).astype(float))
        f = c.acc(p.indices).reshape(-1, 3).astype(int) if p.indices is not None else np.arange(len(v)).reshape(-1, 3)
        F.append(f + o); o += len(v)
    V, J, W, F = np.concatenate(V), np.concatenate(J), np.concatenate(W), np.concatenate(F)
    W = W / np.maximum(W.sum(1, keepdims=True), 1e-12)
    IB = c.acc(skin.inverseBindMatrices).reshape(-1, 4, 4).transpose(0, 2, 1) if skin.inverseBindMatrices is not None \
        else np.repeat(np.eye(4)[None], len(skin.joints), 0)
    return V, J, W, F, np.array(skin.joints), IB


def lbs(V, J, W, mats):
    M = np.einsum("nk,nkij->nij", W, mats[J])
    return np.einsum("nij,nj->ni", M[:, :3, :3], V) + M[:, :3, 3]


def hull_eq(P):
    H = ConvexHull(P); return H.equations


def inside(P, eq, inset):
    return hull_sd(P, eq) <= -inset


def level(x, lo, hi):
    if x is None: return "ok"
    if hi is not None and x >= hi: return "fail"
    return "warn" if x >= lo else "ok"


def _beside(path, ext):
    """<clip stem>{ext} and <clip stem minus .clip>{ext} beside the clip or one folder up: m3d animate writes
    <rig stem>.<clip>.glb into anim/ next to <rig stem>.glb and <rig stem>.roles.json."""
    stem = os.path.basename(path)[:-len(".glb")] if path.endswith(".glb") else os.path.basename(path)
    here = os.path.dirname(os.path.abspath(path))
    return [os.path.join(d, st + ext) for st in (stem, stem.rsplit(".", 1)[0]) for d in (here, os.path.dirname(here))]


def find_rig(path, rig=None):
    """--rig, else the rig the clip was made from by m3d's naming (Blender's export drops the skin extras where
    lib/weightfix.py records its decisions, so the rig GLB is where they survive)."""
    if rig: return rig
    me = os.path.abspath(path)
    return next((c for c in _beside(path, ".glb") if os.path.abspath(c) != me and os.path.exists(c)), None)


def node_roles(g):
    """roles stored on the bones themselves (puppet rigs: node extras {"m3d_role": "arm.L"})."""
    r = {n.name: n.extras["m3d_role"] for n in g.nodes
         if n.name and isinstance(n.extras, dict) and isinstance(n.extras.get("m3d_role"), str)}
    return r or None


def find_roles(path, rig=None, roles=None, clip_gltf=None):
    """--roles, else bone extras on the clip or the rig, else <rig stem>.roles.json, else a sidecar found by name."""
    if roles: return load_roles(roles)
    for g in (clip_gltf, GLTF2().load(rig) if rig else None):
        r = node_roles(g) if g is not None else None
        if r: return r
    cands = ([os.path.splitext(rig)[0] + ".roles.json"] if rig else []) + _beside(path, ".roles.json")
    return next((r for r in map(load_roles, cands) if r), None)


def check(path, anim=0, fps=24.0, rig=None, roles=None):
    c = Clip(path); g = c.g
    V, J, W, F, joints, IB = skinned_mesh(c)
    nb = len(joints); Wd = np.zeros((len(V), nb)); np.add.at(Wd, (np.arange(len(V))[:, None].repeat(J.shape[1], 1), J), W)
    dom = Wd.argmax(1)
    rest_mats = c.world()[joints] @ IB
    R = lbs(V, J, W, rest_mats)                                                  # rest (bind) pose in world space
    parent, children = topology(g, g.skins[0])
    rig = find_rig(path, rig)
    role_map = find_roles(path, rig, roles, g)
    limbs, trunk = limbs_and_trunk(parent, children, np.bincount(dom, minlength=nb),
                                   [g.nodes[j].name or str(j) for j in joints], role_map)
    height = float(R[:, 1].max() - R[:, 1].min())
    # rest-pose bleed: limb weight weightfix would hand back to the trunk (bleed pass only)
    jpos = (c.world()[joints])[:, :3, 3]
    # internal (clavicle) bones: weightfix's record in the clip's skin, else in --rig (Blender's export drops skin
    # extras), else measured here; ripped contacts leave hip-weighted copies at the hands that fool the measurement
    names_ = [g.nodes[j].name or str(j) for j in joints]; internal = None
    for src in (g, GLTF2().load(rig) if rig else None):
        ex = (src.skins[0].extras or {}) if src is not None and src.skins else {}
        if isinstance(ex, dict) and "internal" in ex.get("m3d_weightfix", {}):
            internal = np.isin(names_, ex["m3d_weightfix"]["internal"]); break
    # a body to measure against: trunk-owned skin with a non-degenerate hull (puppet rigs may have neither)
    # the body: trunk-owned skin; a trunk with no skin of its own (puppet root/body bones) falls back to everything
    # no limb owns except the head's subtree (a lamp shade in the hull would swallow a raised arm), then to all of it
    teq = None; limb_b = [j for L in limbs for j in L]
    body = np.isin(dom, trunk); body_src = "trunk"
    if limbs and body.sum() < 4:
        nonlimb = ~np.isin(dom, limb_b); head = int(np.argmax(np.bincount(dom[nonlimb], minlength=nb))) if nonlimb.any() else -1
        from weightfix import subtree
        nohead = nonlimb & ~np.isin(dom, subtree(children, head) if head >= 0 else [])
        body, body_src = (nohead, "non-limb minus head") if nohead.sum() >= 4 else (nonlimb, "non-limb")
    if limbs and body.sum() >= 4:
        try: teq = hull_eq(R[body]); tsize = float(np.linalg.norm(np.ptp(R[body], 0)))
        except Exception: teq = None
    bleed, rep = None, {"note": "no trunk-owned skin: penetration and bleed not measured"}
    if teq is not None and body_src == "trunk":
        Wn, rep, _ = fix(R, Wd, parent, children, jpos, limbs, trunk, far=1e9, F=F, internal=internal)
        allL = [j for L in limbs for j in L]
        wl, wl_new = Wd[:, allL].sum(1), Wn[:, allL].sum(1)
        strong = (wl - wl_new > 1e-3) & (wl >= 0.5)      # torso skin that mostly follows a limb (not blend falloff)
        bleed = round(float((wl - wl_new)[strong].sum() / max(wl.sum(), 1e-9)), 4)
    # per-chain vertex sets: vertices owned by the chain's bones whose joint is outside the trunk at rest
    jin = inside(jpos, teq, 0.0) if teq is not None else np.zeros(nb, bool)
    chains = []
    for L in limbs:
        ext = [j for j in L if not jin[j]]
        chains.append(dict(bones=[g.nodes[joints[j]].name or str(j) for j in L], verts=np.isin(dom, ext)))
    inset = 0.02 * tsize if teq is not None else 0.0
    rest_in = inside(R, teq, inset) if teq is not None else None
    e = np.unique(np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), 1), axis=0)
    rl = np.linalg.norm(R[e[:, 0]] - R[e[:, 1]], axis=1); ok_e = rl > 1e-6 * height; e, rl = e[ok_e], rl[ok_e]
    root_j = int(np.where(parent < 0)[0][0])
    res = dict(input=path, verts=len(V), joints=nb, trunk=[g.nodes[joints[j]].name for j in trunk],
               limbs=[ch["bones"] for ch in chains], roles=bool(role_map), rig=rig,
               body=body_src if teq is not None else None, bleed=bleed, bleed_detail=rep, frames=[])
    if not g.animations:
        res.update(n_frames=0, note="no animation: rest-pose checks only"); return res, R, None
    chs = c.channels(anim)
    t0 = min(ch[2][0] for ch in chs); t1 = max(ch[2][-1] for ch in chs)
    n = max(1, int(round((t1 - t0) * fps)) + 1); ts = np.linspace(t0, t1, n) if n > 1 else np.array([t0])
    res["animation"] = g.animations[anim].name; res["duration"] = round(t1 - t0, 3)
    floor = R[:, 1].min(); root_h = []; hotspots = {}; names = [g.nodes[j].name or str(j) for j in joints]
    a0 = tri_area(R, F); web_best = (0.0, 0.0, {}, None)
    for fi, t in enumerate(ts):
        over = {}
        for node, path_, times, vals, interp in chs:
            if path_ in ("translation", "rotation", "scale"):
                over.setdefault(node, {})[path_] = Clip.sample(times, vals, interp, t, path_)
        Wm = c.world(over); P = lbs(V, J, W, Wm[joints] @ IB)
        root_h.append(Wm[joints[root_j]][1, 3])
        pen = {}
        if teq is not None:
            try:
                peq = hull_eq(P[body]); pin = inside(P, peq, inset) & ~rest_in
                pen = {"+".join(ch["bones"][:2]) + "…": float(pin[ch["verts"]].mean()) if ch["verts"].any() else 0.0 for ch in chains}
            except Exception: pen = {}
        ratio = np.linalg.norm(P[e[:, 0]] - P[e[:, 1]], axis=1) / rl
        hot = ratio > 2.0
        if hot.sum():
            pairs = np.sort(dom[e[hot]], 1); u, cnt = np.unique(pairs, axis=0, return_counts=True)
            for (b0, b1), k in zip(u, cnt):
                key_ = f"{names[b0]}~{names[b1]}"; hotspots[key_] = max(hotspots.get(key_, 0), int(k))
        wa, ws, wb = web(R, P, F, a0, Wd, names)
        if ws > web_best[1] or (ws == web_best[1] and wa > web_best[0]): web_best = (max(wa, web_best[0]), ws, wb, round(float(t), 3))
        else: web_best = (max(wa, web_best[0]),) + web_best[1:]
        res["frames"].append(dict(i=fi, t=round(float(t), 3), web_area=round(wa, 4), penetration=round(max(pen.values()), 4) if pen else None,
                                  penetration_by_limb={k: round(v, 4) for k, v in pen.items()},
                                  stretch_p99=round(float(np.percentile(ratio, 99)), 3), stretch_max=round(float(ratio.max()), 3),
                                  squash_p1=round(float(np.percentile(ratio, 1)), 3),
                                  floor_sink=round(float(max(0.0, floor - P[:, 1].min()) / height), 4)))
    fr = res["frames"]
    pf = [f for f in fr if f["penetration"] is not None]
    res.update(n_frames=n, penetration_max=max(f["penetration"] for f in pf) if pf else None,
               penetration_worst_t=max(pf, key=lambda f: f["penetration"])["t"] if pf else None,
               stretch_p99=max(f["stretch_p99"] for f in fr), stretch_max=max(f["stretch_max"] for f in fr),
               stretch_worst_t=max(fr, key=lambda f: f["stretch_p99"])["t"],
               squash_p1=min(f["squash_p1"] for f in fr), floor_sink=max(f["floor_sink"] for f in fr),
               root_height_range=round(float((max(root_h) - min(root_h)) / height), 4),
               stretch_hotspots=dict(sorted(hotspots.items(), key=lambda kv: -kv[1])[:6]),
               web_area=round(web_best[0], 4), web_sheet=round(web_best[1], 4), web_sheet_t=web_best[3],
               web_sheet_bones=web_best[2])
    return res, R, ts


def verdict(res):
    why = []
    lv = {"ok": 0, "warn": 1, "fail": 2}; worst = "ok"
    def bump(name, val, lo, hi):
        nonlocal worst
        l = level(val, lo, hi)
        if l != "ok": why.append(f"{name}={val} ({l}: ok<{lo}" + (f", fail>={hi})" if hi else ")"))
        if lv[l] > lv[worst]: worst = l
    bump("bleed", res["bleed"], *TH["bleed"])
    if res.get("n_frames"):
        bump("penetration_max", res["penetration_max"], *TH["penetration"])
        bump("stretch_p99", res["stretch_p99"], *TH["stretch_p99"])
        bump("floor_sink", res["floor_sink"], *TH["floor_sink"])
        bump("web_area", res["web_area"], *TH["web_area"])
        bump("web_sheet", res["web_sheet"], *TH["web_sheet"])
        if res["web_sheet"] >= TH["web_sheet"][0] and res.get("web_sheet_bones"):
            why.append(f"web sheet at t={res['web_sheet_t']} s carries weight of {res['web_sheet_bones']}: skin shared by "
                       "parts that move apart (a limb fused to the body in the mesh; re-make the concept in an A-pose, "
                       "or rig with --engine puppet and a spec with rip + ball caps)")
        if res["stretch_max"] > TH["stretch_max_warn"]:
            why.append(f"stretch_max={res['stretch_max']} (warn > {TH['stretch_max_warn']})")
            if worst == "ok": worst = "warn"
    return worst, why


RENDER_PY = r'''
import sys, math, os, bpy, mathutils
a = sys.argv[sys.argv.index("--") + 1:]; glb, out, n = a[0], a[1], int(a[2])
bpy.ops.wm.read_factory_settings(use_empty=True); bpy.ops.import_scene.gltf(filepath=glb)
for o in list(bpy.data.objects):
    if o.type == "MESH" and o.parent is None and o.name.startswith("Icosphere"): bpy.data.objects.remove(o)
meshes = [o for o in bpy.data.objects if o.type == "MESH"]
lo = mathutils.Vector((1e9,) * 3); hi = mathutils.Vector((-1e9,) * 3)
for o in meshes:
    for c in o.bound_box:
        w = o.matrix_world @ mathutils.Vector(c); lo = mathutils.Vector(map(min, lo, w)); hi = mathutils.Vector(map(max, hi, w))
ctr = (lo + hi) / 2; size = max(hi - lo)
cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam")); bpy.context.scene.collection.objects.link(cam)
cam.data.type = "ORTHO"; cam.data.ortho_scale = size * 1.35; ang = math.radians(30)
cam.location = (ctr.x + math.sin(ang) * size * 3, ctr.y - math.cos(ang) * size * 3, ctr.z + size * 0.35)
cam.rotation_euler = (math.radians(83), 0, ang)
sc = bpy.context.scene; sc.camera = cam; sc.render.engine = "BLENDER_WORKBENCH"
sc.display.shading.light = "STUDIO"; sc.display.shading.show_cavity = True; sc.display.shading.show_shadows = True
sc.render.resolution_x = 480; sc.render.resolution_y = 600
arm = next((o for o in bpy.data.objects if o.type == "ARMATURE"), None)
act = arm.animation_data.action if arm and arm.animation_data else None
f0, f1 = (int(act.frame_range[0]), int(act.frame_range[1])) if act else (1, 1)
for k in range(n):
    f = round(f0 + (f1 - f0) * k / max(n - 1, 1)); sc.frame_set(f)
    sc.render.filepath = os.path.join(out, f"frame_{f:04d}.png"); bpy.ops.render.render(write_still=True)
print("RENDER_OK")
'''


def render_frames(glb, n=12):
    blender = shutil.which("blender") or "/opt/homebrew/bin/blender"
    d = tempfile.mkdtemp(prefix="animcheck-"); s = os.path.join(d, "r.py"); open(s, "w").write(RENDER_PY)
    r = subprocess.run([blender, "-b", "--factory-startup", "--python", s, "--", os.path.abspath(glb), d, str(n)],
                       capture_output=True, text=True, timeout=600)
    if "RENDER_OK" not in r.stdout: raise RuntimeError("blender render failed:\n" + r.stdout[-2000:] + r.stderr[-2000:])
    return d


def sheet(frames_dir, out_png, res, fps=24.0, n=8):
    from PIL import Image, ImageDraw
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    files = sorted(glob.glob(os.path.join(frames_dir, "frame_*.png")))
    if not files: raise SystemExit(f"no frame_*.png in {frames_dir}")
    num = [int(''.join(ch for ch in os.path.basename(f) if ch.isdigit()) or 0) for f in files]
    t0 = res["frames"][0]["t"] if res.get("frames") else 0.0
    tf = [t0 + (x - num[0]) / fps for x in num]                  # first frame file = first sample (Blender: t=1/fps)
    pick = sorted(set(np.linspace(0, len(files) - 1, n).round().astype(int)))
    worst_i = -1
    if res.get("n_frames") and (res.get("penetration_max") or 0) > 0:   # show the worst-penetration frame
        worst_i = int(np.argmin([abs(x - res["penetration_worst_t"]) for x in tf]))
        if worst_i not in pick: pick = sorted(pick[:-1] + [worst_i])
    def metrics(t):
        fr = res.get("frames") or []
        return min(fr, key=lambda f: abs(f["t"] - t)) if fr else None
    ims = [Image.open(files[i]).convert("RGB") for i in pick]
    w, h = ims[0].size; s = min(1.0, 320 / w); tw, th = int(w * s), int(h * s)
    base = np.asarray(Image.open(files[0]).convert("RGB")).astype(float)
    strip_h = 220 if res.get("n_frames") else 0
    S = Image.new("RGB", (tw * len(pick), th * 2 + strip_h), "white")
    for k, (i, im) in enumerate(zip(pick, ims)):
        a = im.resize((tw, th)); d = ImageDraw.Draw(a)
        label = f"#{num[i]}  t={tf[i]:.2f}s"
        fr = metrics(tf[i])
        if fr: label += (f"\npen {fr['penetration']:.3f}" if fr["penetration"] is not None else "\npen n/a") + f" str {fr['stretch_p99']:.2f}"
        d.text((4, 4), label, fill=(255, 60, 60))
        if i == worst_i: d.rectangle([0, 0, tw - 1, th - 1], outline=(255, 0, 0), width=4)
        S.paste(a, (k * tw, 0))
        diff = np.abs(np.asarray(im).astype(float) - base).sum(2)
        hm = np.clip(diff / max(diff.max(), 1e-6) * 255 * 1.5, 0, 255).astype(np.uint8)
        grey = (np.asarray(im).astype(float).mean(2) * 0.4).astype(np.uint8)
        rgb = np.where((diff > 20)[..., None], np.stack([hm, np.zeros_like(hm), 255 - hm], 2), np.stack([grey] * 3, 2))
        b = Image.fromarray(rgb.astype(np.uint8)).resize((tw, th)); ImageDraw.Draw(b).text((4, 4), "diff vs first", fill=(255, 255, 255))
        S.paste(b, (k * tw, th))
    if strip_h:
        t = [f["t"] for f in res["frames"]]
        fig, ax = plt.subplots(figsize=(S.width / 100, strip_h / 100), dpi=100)
        ax.plot(t, [f["penetration"] if f["penetration"] is not None else np.nan for f in res["frames"]], label="penetration (limb share inside trunk)", color="#d62728")
        ax.axhline(TH["penetration"][1], color="#d62728", ls=":", lw=0.8)
        ax2 = ax.twinx(); ax2.plot(t, [f["stretch_p99"] for f in res["frames"]], label="edge stretch p99", color="#1f77b4")
        ax2.axhline(TH["stretch_p99"][1], color="#1f77b4", ls=":", lw=0.8)
        ax.set_xlabel("t (s)"); ax.set_ylabel("penetration"); ax2.set_ylabel("stretch p99")
        ax.legend(loc="upper left", fontsize=7); ax2.legend(loc="upper right", fontsize=7); fig.tight_layout()
        tmp = out_png + ".strip.png"; fig.savefig(tmp); plt.close(fig)
        S.paste(Image.open(tmp).convert("RGB").resize((S.width, strip_h)), (0, th * 2)); os.remove(tmp)
    S.save(out_png)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("glb"); ap.add_argument("--anim", type=int, default=0); ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--frames"); ap.add_argument("--render", action="store_true"); ap.add_argument("--out")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--rig", help="the rigged GLB the clip was made from (reads lib/weightfix.py's record and its "
                                  "roles); default: found from the clip's name (find_rig)")
    ap.add_argument("--roles", help="roles sidecar; default: found from --rig or the clip's name (see find_roles)")
    a = ap.parse_args()
    out = a.out or os.path.splitext(a.glb)[0] + "-animcheck"
    res, _, _ = check(a.glb, a.anim, a.fps, a.rig, a.roles)
    f4 = lambda x, d=4: "n/a" if x is None else f"{x:.{d}f}"
    m4 = lambda x, d=4: "nan" if x is None else f"{x:.{d}f}"   # the marker stays float()-parseable
    v, why = verdict(res); res.update(verdict=v, reasons=why, thresholds=TH)
    frames = a.frames or (render_frames(a.glb) if a.render else None)
    if frames:
        sheet(frames, out + "-sheet.png", res, a.fps); res["sheet"] = out + "-sheet.png"
    json.dump(res, open(out + ".json", "w"), indent=1)
    if a.json:
        print(json.dumps({k: v_ for k, v_ in res.items() if k not in ("frames", "bleed_detail")}))
    else:
        print(f"{a.glb}: {res['verts']} verts, {res['joints']} joints, trunk {res['trunk']}, {len(res['limbs'])} limbs")
        print(f"  bleed (rest): {f4(res['bleed'])} of limb weight on trunk skin" + ("" if res['bleed'] is not None else
              " (no trunk-owned skin: puppet rig with a skinless root/body, or no limbs found)"))
        if res.get("n_frames"):
            print(f"  {res['n_frames']} frames over {res['duration']} s ('{res['animation']}')")
            print(f"  penetration max {f4(res['penetration_max'])} at t={res['penetration_worst_t']} s")
            print(f"  edge stretch p99 {res['stretch_p99']:.3f} (worst t={res['stretch_worst_t']} s), max {res['stretch_max']:.2f}, squash p1 {res['squash_p1']:.3f}")
            if res["stretch_hotspots"]: print(f"  edges stretched > 2x, by owning bones (max count in a frame): {res['stretch_hotspots']}")
            print(f"  web (skin pulled into sheets): area {res['web_area']:.4f}, largest sheet {res['web_sheet']:.4f} at t={res['web_sheet_t']} s, bones {res['web_sheet_bones']}")
            print(f"  root height range {res['root_height_range']:.4f}, floor sink {res['floor_sink']:.4f} (of height {1:.0f})")
        print(f"  verdict {v}" + ("".join(f"\n    - {r}" for r in why)))
        print(f"  report {out}.json" + (f", sheet {res['sheet']}" if frames else ""))
    print(f"M3D_ANIMCHECK frames={res.get('n_frames', 0)} penetration_max={m4(res.get('penetration_max', 0.0))} "
          f"stretch_p99={m4(res.get('stretch_p99', 1.0), 3)} web_area={m4(res.get('web_area', 0.0))} bleed={m4(res['bleed'])} verdict={v}")


if __name__ == "__main__":
    main()

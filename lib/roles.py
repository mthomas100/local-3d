# roles.py — role tagger for auto-rigged skeletons: a skinned GLB whose joints are just `bone_0…bone_N`
# (skin-tokens.cpp output, or any glTF skin) -> a role per bone, so the procedural presets in motion.py
# (keyed on roles, never on names) can drive it. Written 2026-10-02 for the m3d rig stage (T6 of
# the rig/animate plan). CPU only; runs with the worker venv python (numpy, pygltflib, matplotlib).
# Usage (under the memory guard, like every numpy/trimesh job since the 2026-10-02 jetsam incident):
#   bin/memguard 16 ~/.hy3d/worker-venv/bin/python lib/roles.py RIGGED.glb [--out-json path]
#       [--write-glb [path] | --in-place] [--humanoid-names] [--plot path] [--facing=-z]
#       [--override bone_7=arm.L,bone_8=arm.L] [--json]
# Outputs (the contract shared with motion.py and puppet.py; consumers resolve sidecar -> node extras -> name):
#   <stem>.roles.json  beside the GLB (or --out-json):
#       {"roles": {"bone_5": "head", …}, "confidence": {"bone_5": 0.97, …}, "chains": [["bone_1","bone_2"], …],
#        "summary": "biped+fingers", "mixamo": {"bone_0": "mixamorig:Hips", …} (bipeds, or --humanoid-names),
#        "reasons": {…one line per chain…}, "notes": […], "facing": "+z", "overrides": {…}}
#   --write-glb [path]  a copy of the GLB (default <stem>-roles.glb) with node.extras = {"m3d_role": role}
#                       on every joint; --in-place rewrites the input. The copy is reloaded and checked.
#   --plot path         rigplot-style front/side picture, bones coloured and labelled by role.
# The last stdout line is  M3D_ROLES bones=<n> tagged=<n> summary=<type> out=<json>  (tagged = roles other
# than "other"); with --json the line before it is one JSON object (the sidecar plus "glb"/"plot" paths)
# and the table goes to stderr.
#
# ---------------------------------------------------------------------------------------------------
# ROLES: root, body, head, arm.L, arm.R, leg.L, leg.R, lid, tail, antenna, wheel, spout, other.
# A chain (maximal path of single-child joints, listed root->tip) shares one role, except where the
# geometry says a chain changes nature along its length (body->lid, body->head, spout->antenna/tail).
#
# FRAME: glTF, +Y up, the character faces +Z (toward the viewer), so its LEFT is +X (Blender/Mixamo
# convention, same as specs/puppet/*.json). --facing -z flips both the left/right sign and back/forward.
# Skinned vertices are read raw from the skin's primitives (the mesh node transform does not apply to a
# skinned mesh), so joints and vertices share one frame. Joint positions come from the node hierarchy,
# cross-checked against the inverse bind matrices (the bind pose wins if they disagree).
#
# HEURISTICS (what each rule looks at; confidence = product of clamped ramps on the same quantities):
#   root     the skin's parentless joint; if several, the one heading the largest tree (a detached leg must
#            not win because it sits lower), ties broken by distance to the bbox bottom centre; the others
#            start "floating" chains classified by geometry alone.
#   chains   start at every child of the root or of a branching joint, follow single children. Per chain:
#            direction u (tip - first joint; for a single joint, attach -> joint), path length, tortuosity
#            (path / straight distance), start/tip heights as bbox fractions yn, dominant-vertex blobs.
#   body     the upward (u.y >= 0.8) chain from the root with the most descendants; it continues through a
#            branch point when the upward child there ends in another branch with >= 2 non-upward children
#            (Root -> Hips -> Spine -> Chest with shoulders).
#   head     at the body's last branch point (the chest), the upward child (u.y >= 0.6, |horizontal| <= 0.6)
#            with the biggest vertex blob (ties: the highest tip); extra upward chains there -> antenna.
#            If the body chain ends in a leaf (a closed column, no shoulders): the tip joint is the head
#            when its blob holds >= 10% of the vertices and sits in the upper mesh (yn >= 0.55)…
#   lid      …otherwise the top of the column is a lid: scanning from the tip, joints whose blobs are small
#            (< 10% of verts) and high (centroid yn >= 0.5), within 0.35 H of the column top. A thin
#            horizontal slab (blob height << width) raises the confidence. The whole column may be lid
#            (the teapot: the dome disc + knob); the root then carries the body and the notes say so.
#   arm.L/R  non-upward chains attached at the chest (any hang angle: T-pose or A-pose) with a lateral
#            offset, tortuosity < 1.7; also chains from the root / lower body that start in the middle
#            band (0.25 <= yn <= 0.85), reach sideways and do not reach the floor (stub arms on objects).
#            L/R: sign of the chain's mean x relative to the root's x (left = +x when facing +z).
#            Clavicles: a chain-start bone whose joint lies inside the torso silhouette (lateral offset
#            < 0.85 of the root+body vertices' half-width at that height) AND whose bone runs sideways
#            (|dx| >= |dy|) is tagged `body`; the arm starts at the first joint outside (the lamp's
#            bone_6/bone_13 own torso skin and would drag a bulge if raised; motion test 2026-10-02).
#            A thigh also starts inside the pelvis but runs down, so legs keep their hip bone. At least
#            two joints always stay in the limb. Fingers: trailing joints after the 4th whose bones are
#            < half the first three bones' mean; sub-chains hanging off an arm inherit the arm role.
#   leg.L/R  chains from the root or lower body going down (u.y <= -0.6, little sideways travel) whose tip
#            reaches the bottom 12% of the mesh, starting below 55% height. The clavicle rule applies
#            (a sideways pelvis stub inside the hip mass -> body).
#   tail     chains from the root / lower body going backwards (-Z component >= 0.6), or long diagonal
#            chains (>= 3 joints, path >= 0.3 H) trailing down with a big horizontal component.
#   spout    chains of >= 2 joints from the root / lower body, path >= 0.25 H, leaving sideways or forward
#            (|horizontal| >= 0.45) and rising (u.y >= 0.25), or horizontal from above mid-height with
#            the tip outside the root/body core. The first joints are spout; from the first joint whose
#            blob is a separate puff (radius >= 1.8x and count >= 3x the earlier joints' median, higher
#            than the chain start) the rest is antenna (rising) or tail (going back/down).
#   antenna  upward chains that are not the body/head/lid and own < 10% of the verts (also chains off the
#            head going up, and the spout puff above).
#   wheel    a leaf joint whose blob is a thin round disc with a horizontal axle (sorted bbox dims a<=b<=c:
#            a <= 0.4 c, b >= 0.75 c) near the bottom (centroid yn <= 0.3). None in the 2026-10-02 rigs.
#   other    arcs that return toward the body (tortuosity >= 1.7: the teapot handle), chains with no
#            dominant vertices, and anything unmatched.
#   summary  biped (legs L+R, arms L+R, body; +fingers), quadruped (>= 2 legs per side, no arms),
#            object (lid, spout or wheel present, or no limbs at all), else unknown.
#   mixamo   for bipeds (or --humanoid-names): Hips = root; the body joints -> Spine, Spine1, Spine2 by
#            count (1-3; more than 3: first, middle, last); the head chain -> Head (1 joint), Neck+Head (2),
#            Neck+Head+HeadTop_End (3+: first, second, last); each arm's first 4 joints -> Shoulder, Arm,
#            ForeArm, Hand (3 joints: Arm, ForeArm, Hand; 2: Arm, ForeArm), fingers skipped, a clavicle
#            tagged `body` still maps to Shoulder (that is what Mixamo's Shoulder is); each leg ->
#            UpLeg, Leg, Foot, ToeBase (, Toe_End). With several arms/legs per side only the first maps.
# KNOWN WEAKNESSES (2026-10-02, three SkinTokens rigs: lamp robot, knight, teapot robot):
#   - Everything is rest-pose geometry. Arms raised above the shoulders (u.y >= 0.6, little sideways travel)
#     would be taken for a head/antenna; T-pose and A-pose are fine.
#   - An object whose shell is weighted to the root (the teapot: 36% of the verts on bone_0) gets no `body`
#     bone; the note says so and presets must fall back to the root. `--override bone_1=body` would make
#     the lid disc the body, which is worse.
#   - Legs on objects: a tripod gets leg.R, leg.R, leg.L by x sign (there is no tripod role); L/R
#     confidence drops for feet near the centre line.
#   - The lid rule needs a closed body column (no shoulders): a jar robot with arms and a flat lid gets
#     `head` on the lid. The head rule assumes the chest's biggest upward blob is the head.
#   - Fingers are only the short trailing bones of an arm chain or sub-chains at its tip; a thumb branching
#     mid-hand still inherits the arm role (correct) but is not counted as a finger.
#   - The handle is `other` by tortuosity (>= 1.7); a straight bar handle would read as an arm or a spout.
#   - `wheel` has no fixture yet; `tail` was only exercised synthetically.
#   - L/R assume the character faces +Z; a mesh facing -Z silently swaps sides unless --facing=-z is given.
# Fix any of these with --override bone=role (confidence 1.0, recorded under "overrides" in the sidecar).
# ---------------------------------------------------------------------------------------------------
import sys, os, json, math, argparse, tempfile
import numpy as np

ROLES = ("root", "body", "head", "arm.L", "arm.R", "leg.L", "leg.R", "lid", "tail", "antenna", "wheel", "spout", "other")
ROLE_COLOURS = {"root": "#222222", "body": "#ff8c00", "head": "#d62728", "arm.L": "#1f77b4", "arm.R": "#17becf",
                "leg.L": "#2ca02c", "leg.R": "#8bc34a", "lid": "#9467bd", "spout": "#8c564b", "antenna": "#e377c2",
                "tail": "#f06292", "wheel": "#555555", "other": "#aaaaaa"}
COMPONENT = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16}


def ramp(x, a, b):
    """0 at x <= a, 1 at x >= b, linear between (a may exceed b for a falling ramp)."""
    return float(min(1.0, max(0.0, (x - a) / (b - a))))


def unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else np.zeros(3)


# ---------------------------------------------------------------------------------------------------
# glTF reading
# ---------------------------------------------------------------------------------------------------
def read_accessor(g, idx):
    """Accessor -> (count, ncomp) array; handles byteStride and normalized integer attributes."""
    a = g.accessors[idx]
    bv = g.bufferViews[a.bufferView]
    blob = g.binary_blob()
    dt = np.dtype(COMPONENT[a.componentType])
    nc = NCOMP[a.type]
    off = (bv.byteOffset or 0) + (a.byteOffset or 0)
    stride = bv.byteStride or 0
    if stride and stride != dt.itemsize * nc:
        raw = np.frombuffer(blob, dtype=np.uint8, count=stride * (a.count - 1) + dt.itemsize * nc, offset=off)
        arr = np.lib.stride_tricks.as_strided(raw, shape=(a.count, dt.itemsize * nc), strides=(stride, 1))
        arr = np.ascontiguousarray(arr).view(dt).reshape(a.count, nc)
    else:
        arr = np.frombuffer(blob, dtype=dt, count=a.count * nc, offset=off).reshape(a.count, nc)
    if a.normalized and dt.kind in "ui":
        arr = arr.astype(np.float64) / np.iinfo(dt).max
    return arr


def node_local(g, n):
    node = g.nodes[n]
    if node.matrix:
        return np.array(node.matrix, dtype=float).reshape(4, 4).T
    T = np.eye(4)
    t = node.translation or [0, 0, 0]; q = node.rotation or [0, 0, 0, 1]; s = node.scale or [1, 1, 1]
    x, y, z, w = q
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    T[:3, :3] = R * np.array(s, dtype=float); T[:3, 3] = t
    return T


class Rig:
    """Skeleton + skinned mesh of skins[0]: joint positions, parents within the skin, dominant-vertex blobs."""

    def __init__(self, path):
        from pygltflib import GLTF2
        self.path = path
        g = self.g = GLTF2().load(path)
        if not g.skins:
            raise SystemExit(f"roles: {path} has no skin")
        skin = g.skins[0]
        self.joints = list(skin.joints)
        self.n = len(self.joints)
        self.names = [g.nodes[j].name or f"node_{j}" for j in self.joints]
        parent = {}
        for i, node in enumerate(g.nodes):
            for c in (node.children or []):
                parent[c] = i

        def world(n):
            M = node_local(g, n); p = parent.get(n)
            while p is not None:
                M = node_local(g, p) @ M; p = parent.get(p)
            return M
        pos = np.array([world(j)[:3, 3] for j in self.joints])
        self.notes = []
        if skin.inverseBindMatrices is not None:
            ibm = read_accessor(g, skin.inverseBindMatrices).reshape(-1, 4, 4)
            bind = np.array([np.linalg.inv(m.T) for m in ibm])[:, :3, 3]
            if np.abs(bind - pos).max() > 1e-3 * max(1e-9, np.linalg.norm(pos.max(0) - pos.min(0))):
                self.notes.append("joint positions from the inverse bind matrices (the node hierarchy disagrees)")
                pos = bind
        self.pos = pos
        # parent within the skin: walk up through non-joint nodes (an Armature node, say)
        jidx = {j: k for k, j in enumerate(self.joints)}
        self.jparent = []
        for j in self.joints:
            p = parent.get(j)
            while p is not None and p not in jidx:
                p = parent.get(p)
            self.jparent.append(jidx[p] if p is not None else None)
        self.children = [[] for _ in range(self.n)]
        for k, p in enumerate(self.jparent):
            if p is not None:
                self.children[p].append(k)
        # skinned vertices of every primitive of every mesh node that uses skins[0]
        P, J, W = [], [], []
        used = set()
        for i, node in enumerate(g.nodes):
            if node.mesh is None or node.skin != 0 or node.mesh in used:
                continue
            used.add(node.mesh)
            for prim in g.meshes[node.mesh].primitives:
                at = prim.attributes
                if getattr(at, "JOINTS_0", None) is None or getattr(at, "WEIGHTS_0", None) is None:
                    self.notes.append(f"mesh {node.mesh}: a primitive without JOINTS_0/WEIGHTS_0 was skipped")
                    continue
                P.append(read_accessor(g, at.POSITION).astype(np.float64))
                J.append(read_accessor(g, at.JOINTS_0).astype(np.int64))
                W.append(read_accessor(g, at.WEIGHTS_0).astype(np.float64))
        if not P:
            raise SystemExit(f"roles: {path} has no skinned primitive using skins[0]")
        self.V = np.concatenate(P); J = np.concatenate(J); W = np.concatenate(W)
        self.nv = len(self.V)
        self.dom = J[np.arange(self.nv), W.argmax(1)]
        self.dom[W.max(1) <= 0] = -1
        self.cnt = np.bincount(self.dom[self.dom >= 0], minlength=self.n)[: self.n]
        self.share = self.cnt / max(1, self.nv)
        self.lo, self.hi = self.V.min(0), self.V.max(0)
        self.H = float(self.hi[1] - self.lo[1]) or 1.0
        self.W = float(max(self.hi[0] - self.lo[0], self.hi[2] - self.lo[2])) or 1.0
        # per-joint blob of dominant vertices: bbox, centroid, RMS radius
        self.blob_lo = np.zeros((self.n, 3)); self.blob_hi = np.zeros((self.n, 3))
        self.blob_ctr = self.pos.copy(); self.blob_r = np.zeros(self.n)
        for k in range(self.n):
            sel = self.dom == k
            if sel.any():
                Q = self.V[sel]
                self.blob_lo[k], self.blob_hi[k] = Q.min(0), Q.max(0)
                self.blob_ctr[k] = Q.mean(0)
                self.blob_r[k] = float(np.sqrt(((Q - self.blob_ctr[k]) ** 2).sum(1).mean()))
        self.blob_size = self.blob_hi - self.blob_lo

    def yn(self, y):
        return float((y - self.lo[1]) / self.H)

    def ctr_yn(self, k):
        return self.yn(self.blob_ctr[k][1])

    def subtree(self, k):
        out = [k]
        for c in self.children[k]:
            out += self.subtree(c)
        return out


# ---------------------------------------------------------------------------------------------------
# chains
# ---------------------------------------------------------------------------------------------------
class Chain:
    def __init__(self, rig, joints, attach, fwd):
        P = rig.pos
        self.joints = joints; self.attach = attach
        self.first, self.last = joints[0], joints[-1]
        self.start, self.tip = P[self.first], P[self.last]
        if len(joints) > 1:
            self.vec = self.tip - self.start
            self.path = float(sum(np.linalg.norm(P[b] - P[a]) for a, b in zip(joints, joints[1:])))
        elif attach is not None:
            self.vec = self.start - P[attach]; self.path = float(np.linalg.norm(self.vec))
        else:
            self.vec = np.zeros(3); self.path = 0.0
        self.length = float(np.linalg.norm(self.vec))
        self.u = unit(self.vec)
        self.tort = self.path / self.length if self.length > 1e-9 else 1.0
        self.horiz = float(math.hypot(self.u[0], self.u[2]))
        self.back = float(-self.u[2] * fwd)             # > 0 when the chain goes backwards
        self.desc = len(rig.subtree(self.first))
        self.cnt = int(rig.cnt[joints].sum()); self.share = float(rig.share[joints].sum())
        self.mean_x = float(P[joints, 0].mean())
        self.start_yn, self.tip_yn = rig.yn(self.start[1]), rig.yn(self.tip[1])
        self.parent_chain = None; self.child_chains = []; self.depth = 0
        a = abs(self.u)
        if self.length <= 1e-9: self.dir = "-"
        elif a[1] >= 0.6: self.dir = "up" if self.u[1] > 0 else "down"
        elif a[0] >= a[2]: self.dir = "lateral"
        else: self.dir = "back" if self.back > 0 else "forward"

    def names(self, rig):
        return [rig.names[j] for j in self.joints]


def build_chains(rig, root, fwd):
    starts = [k for k in range(rig.n) if k != root and
              (rig.jparent[k] is None or rig.jparent[k] == root or len(rig.children[rig.jparent[k]]) >= 2)]
    chains = [Chain(rig, [root], None, fwd)]
    for s in starts:
        js = [s]
        while len(rig.children[js[-1]]) == 1:
            js.append(rig.children[js[-1]][0])
        chains.append(Chain(rig, js, rig.jparent[s], fwd))
    chain_of = {}
    for i, ch in enumerate(chains):
        for j in ch.joints:
            chain_of[j] = i
    for ch in chains:
        if ch.attach is not None:
            ch.parent_chain = chains[chain_of[ch.attach]]
            ch.parent_chain.child_chains.append(ch)
    for ch in chains:                                      # depth for parents-first processing
        d, p = 0, ch.parent_chain
        while p is not None:
            d += 1; p = p.parent_chain
        ch.depth = d
    return chains, chain_of


# ---------------------------------------------------------------------------------------------------
# tagging
# ---------------------------------------------------------------------------------------------------
def tag_rig(rig, facing="+z"):
    fwd = 1.0 if facing == "+z" else -1.0
    left = fwd                                             # the character's left is +x when it faces +z
    H, W = rig.H, rig.W
    roles = ["other"] * rig.n; conf = [0.3] * rig.n
    reasons = {}; notes = list(rig.notes); fingers = set()

    def side(x):
        return "L" if (x - axis_x) * left >= 0 else "R"

    def side_conf(x):
        return ramp(abs(x - axis_x) / W, 0.03, 0.12)

    def put(js, role, c, why):
        for j in js:
            roles[j] = role; conf[j] = round(float(min(1.0, max(0.05, c))), 2)
        reasons[rig.names[js[0]]] = f"{role}: {why}"

    # root: the parentless joint; if several, the one heading the largest tree, then nearest the bbox bottom centre
    parentless = [k for k in range(rig.n) if rig.jparent[k] is None]
    bottom_centre = np.array([(rig.lo[0] + rig.hi[0]) / 2, rig.lo[1], (rig.lo[2] + rig.hi[2]) / 2])
    root = min(parentless, key=lambda k: (-len(rig.subtree(k)), np.linalg.norm(rig.pos[k] - bottom_centre)))
    rig.root = root
    axis_x = float(rig.pos[root][0])
    chains, chain_of = build_chains(rig, root, fwd)
    put([root], "root", 1.0 if len(parentless) == 1 else 0.7,
        "parentless joint" + ("" if len(parentless) == 1 else f" heading the largest of {len(parentless)} trees"))
    if len(parentless) > 1:
        notes.append(f"{len(parentless)} parentless joints: {', '.join(rig.names[k] for k in parentless)}; "
                     f"{rig.names[root]} is the root, the others start floating chains")
    is_up = lambda ch: ch.u[1] >= 0.6 and ch.horiz <= 0.6

    # body: the upward chain from the root with the most descendants, continued through pelvis-like branches
    cands = [ch for ch in chains if ch.attach == root and ch.u[1] >= 0.8]
    no_upward = not cands
    if no_upward:
        cands = [ch for ch in chains if ch.attach == root]
    body, head, chest, lid = [], [], None, []
    body_conf = 1.0
    cur = max(cands, key=lambda ch: (ch.desc, ch.u[1])) if cands else None
    if cur is not None and no_upward:
        notes.append("no upward chain from the root: the chain with the most descendants is the body")
        body_conf = 0.5
    while cur is not None:
        body += cur.joints
        body_conf *= ramp(cur.u[1], 0.5, 0.8) if not no_upward else 1.0
        kids = cur.child_chains
        ups = [k for k in kids if is_up(k)]
        if not kids or not ups:
            break
        best = max(ups, key=lambda k: (round(k.share, 2), k.tip[1]))
        if len([c for c in best.child_chains if not is_up(c)]) >= 2:   # best is a trunk with shoulders
            cur = best; continue
        head = best.joints; chest = cur.last
        head_c = ramp(best.share, 0.03, 0.12) * ramp(rig.ctr_yn(best.last), 0.45, 0.65)
        put(head, "head", 0.5 + 0.5 * head_c,
            f"upward chain from the chest {rig.names[chest]} with the biggest blob there ({best.share:.0%} of verts)")
        for k in ups:
            if k is not best:
                put(k.joints, "antenna" if k.share < 0.1 else "other", 0.6,
                    f"second upward chain from the chest beside the head ({k.share:.0%} of verts)")
        break
    if body and chest is None:                            # closed column: head or lid at the top
        tipj = body[-1]
        if rig.share[tipj] >= 0.10 and rig.ctr_yn(tipj) >= 0.55:
            head = [tipj]; body = body[:-1]
            put(head, "head", 0.5 + 0.5 * ramp(rig.share[tipj], 0.1, 0.25),
                f"top joint of the body column owns {rig.share[tipj]:.0%} of the verts high in the mesh")
        else:
            col_top = max(rig.blob_hi[j][1] if rig.cnt[j] else rig.pos[j][1] for j in body)
            for j in reversed(body):
                low = rig.blob_lo[j][1] if rig.cnt[j] else rig.pos[j][1]
                if rig.share[j] < 0.10 and rig.ctr_yn(j) >= 0.5 and (col_top - low) <= 0.35 * H:
                    lid.insert(0, j)
                else:
                    break
            if lid:
                body = body[: len(body) - len(lid)]
                blo = rig.blob_lo[[j for j in lid if rig.cnt[j]] or lid].min(0)
                bhi = rig.blob_hi[[j for j in lid if rig.cnt[j]] or lid].max(0)
                hw = (bhi[1] - blo[1]) / max(1e-9, bhi[0] - blo[0], bhi[2] - blo[2])
                share = float(rig.share[lid].sum())
                c = ramp(share, 0.10, 0.04) * ramp(float(np.mean([rig.ctr_yn(j) for j in lid])), 0.5, 0.65) \
                    * (0.6 + 0.4 * ramp(hw, 1.0, 0.5))
                put(lid, "lid", c, f"top of a closed body column: {share:.1%} of the verts, blob height/width {hw:.2f}"
                                   + (" (a thin slab)" if hw < 0.5 else ""))
    if body:
        put(body, "body", body_conf, "upward chain from the root with the most descendants" +
            (f"; ends at the chest {rig.names[chest]}" if chest is not None else "; a closed column"))
    else:
        notes.append(f"body: none — {rig.names[root]} (root) owns {rig.share[root]:.0%} of the vertices"
                     + (", the column above it is the lid" if lid else "") + "; presets should fall back to the root")

    def is_disc(k):
        if rig.cnt[k] < 0.003 * rig.nv or rig.ctr_yn(k) > 0.3:
            return False
        s = rig.blob_size[k]; order = np.argsort(s); a, b, c = s[order]
        return a <= 0.4 * c and b >= 0.75 * c and order[0] in (0, 2)

    def spout_split(js):
        """First joint whose blob is a separate puff above the chain start -> (spout part, continuation)."""
        for i in range(1, len(js)):
            prev = [j for j in js[:i] if rig.cnt[j]]
            if not prev:
                continue
            pr, pc = float(np.median(rig.blob_r[prev])), float(np.median(rig.cnt[prev]))
            if rig.cnt[js[i]] and rig.blob_r[js[i]] >= 1.8 * pr and rig.cnt[js[i]] >= 3 * pc \
                    and rig.pos[js[i]][1] > rig.pos[js[0]][1] + 0.1 * H:
                return js[:i], js[i:], rig.blob_r[js[i]] / pr, rig.cnt[js[i]] / pc
        return js, [], 0.0, 0.0

    def tag_spout(ch, why):
        main, rest, r_ratio, c_ratio = spout_split(ch.joints)
        c = ramp(ch.path / H, 0.25, 0.4) * ramp(max(ch.u[1], 0.25 if ch.start_yn >= 0.5 else 0), 0.2, 0.5)
        put(main, "spout", c, why)
        if rest:
            v = unit(rig.pos[rest[-1]] - rig.pos[main[-1]])
            role = "tail" if (v[1] < 0 and -v[2] * fwd >= 0.3) else "antenna"
            put(rest, role, c * max(ramp(r_ratio, 1.6, 3.0), ramp(c_ratio, 3.0, 8.0)),
                f"separate blob at the end of the spout chain (blob radius x{r_ratio:.1f}, verts x{c_ratio:.1f}), "
                f"going {'back/down' if role == 'tail' else 'up'}")

    def torso_halfwidth(y):
        """Lateral half-width of the root+body vertices in a slab at height y (0 when none there)."""
        s = core_v[np.abs(core_v[:, 1] - y) < 0.05 * H]
        return float(np.abs(s[:, 0] - axis_x).max()) if len(s) else 0.0

    def split_clavicle(ch, role):
        """Chain-start bones inside the torso silhouette that run sideways -> body (clavicle / pelvis stub)."""
        js = ch.joints; clav = []
        for i in range(len(js) - 2):                       # the limb keeps at least two joints
            j, nxt = js[i], js[i + 1]
            hw = torso_halfwidth(rig.pos[j][1]); off = abs(rig.pos[j][0] - axis_x)
            b = rig.pos[nxt] - rig.pos[j]
            if hw > 0 and off < 0.85 * hw and abs(b[0]) >= abs(b[1]):
                clav.append(j)
            else:
                break
        if clav:
            ratio = abs(rig.pos[clav[0]][0] - axis_x) / torso_halfwidth(rig.pos[clav[0]][1])
            put(clav, "body", 0.6 + 0.4 * ramp(ratio, 0.7, 0.4),
                f"clavicle of the {role} chain: joint inside the torso (offset {ratio:.2f} of its half-width), bone runs sideways")
        return js[len(clav):]

    def tag_arm(ch, why, from_chest=False):
        s = side(ch.mean_x)
        c = side_conf(ch.mean_x) * ramp(ch.tort, 1.9, 1.5)
        limb = split_clavicle(ch, f"arm.{s}") if from_chest else ch.joints
        put(limb, f"arm.{s}", c, f"{why}; mean x {ch.mean_x - axis_x:+.2f} -> {s}, tortuosity {ch.tort:.2f}")
        limbs[f"arm.{s}"].append(ch)
        if len(ch.joints) > 4:                              # fingers: short trailing bones after the hand
            segs = [float(np.linalg.norm(rig.pos[b] - rig.pos[a])) for a, b in zip(ch.joints, ch.joints[1:])]
            ref = float(np.mean(segs[:3]))
            tail = [j for i, j in enumerate(ch.joints[4:]) if segs[3 + i] < 0.5 * ref]
            if tail and tail[0] == ch.joints[4]:
                for j in tail:
                    fingers.add(j); conf[j] = round(conf[j] * 0.9, 2)
                reasons[rig.names[limb[0]]] += f"; {len(tail)} trailing short bone(s) = fingers"

    def tag_leg(ch, why):
        s = side(ch.mean_x)
        c = side_conf(ch.mean_x) * ramp(-ch.u[1], 0.6, 0.9) * ramp(ch.tip_yn, 0.15, 0.08)
        limb = split_clavicle(ch, f"leg.{s}")
        put(limb, f"leg.{s}", c, f"{why}; mean x {ch.mean_x - axis_x:+.2f} -> {s}, tip at {ch.tip_yn:.0%} height")
        limbs[f"leg.{s}"].append(ch)

    limbs = {"arm.L": [], "arm.R": [], "leg.L": [], "leg.R": []}   # limb chains (clavicles included)
    core_js = [j for j in [root] + body if rig.cnt[j]]  # the root/body blobs: what "outside the body" means
    core_lo = rig.blob_lo[core_js].min(0) if core_js else rig.lo
    core_hi = rig.blob_hi[core_js].max(0) if core_js else rig.hi
    core_v = rig.V[np.isin(rig.dom, core_js)] if core_js else rig.V[:0]   # torso silhouette for clavicles

    def outside_core(p):
        return bool((p < core_lo - 0.05 * W).any() or (p > core_hi + 0.05 * W).any())

    def root_side(ch):
        """Chains from the root, the lower body or a floating root: classified by geometry alone."""
        n = len(ch.joints)
        if ch.cnt == 0:
            return put(ch.joints, "other", 0.2, f"{ch.dir} chain with no dominant vertices")
        if is_disc(ch.last):
            if n > 1:
                put(ch.joints[:-1], "other", 0.5, "axle of a wheel")
            return put([ch.last], "wheel", 0.7, "leaf with a thin round disc blob near the bottom")
        if ch.tort >= 1.7 and n >= 3:
            return put(ch.joints, "other", 0.5 + 0.3 * ramp(ch.tort, 1.7, 2.2),
                       f"arc returning toward the body (tortuosity {ch.tort:.2f}): a handle/loop, not a limb")
        if ch.back >= 0.6:
            return put(ch.joints, "tail", 0.5 + 0.5 * ramp(ch.back, 0.6, 0.9), f"leaves the root going backwards")
        if ch.u[1] <= -0.6 and ch.horiz < 0.5 and ch.tip_yn <= 0.12 and ch.start_yn <= 0.55:
            return tag_leg(ch, "downward chain reaching the bottom")
        if ch.u[1] <= -0.4 and ch.horiz >= 0.5 and n >= 3 and ch.path >= 0.3 * H:
            return put(ch.joints, "tail", 0.6, "long diagonal chain trailing down and away")
        if n >= 2 and ch.path >= 0.25 * H and ch.horiz >= 0.45 and \
                (ch.u[1] >= 0.25 or (ch.start_yn >= 0.5 and outside_core(ch.tip))):
            return tag_spout(ch, f"long chain leaving the body sideways and {'rising' if ch.u[1] >= 0.25 else 'level, ending outside the body'}"
                                 f" (path {ch.path / H:.2f} H, start at {ch.start_yn:.0%} height)")
        if ch.u[1] >= 0.8:
            if ch.share < 0.1 and ch.start_yn >= 0.5:
                return put(ch.joints, "lid", 0.5, "second upward chain high on the body with a small blob")
            return put(ch.joints, "antenna" if ch.share < 0.1 else "other", 0.5, "upward chain beside the body")
        if ch.horiz >= 0.4 and 0.25 <= ch.start_yn <= 0.85 and ch.tip_yn > 0.12:
            return tag_arm(ch, f"sideways chain from {rig.names[ch.attach] if ch.attach is not None else 'a floating root'} starting at {ch.start_yn:.0%} height")
        return put(ch.joints, "other", 0.3, f"{ch.dir} chain matched no rule")

    def chest_side(ch):
        if ch.cnt == 0:
            return put(ch.joints, "other", 0.2, f"{ch.dir} chain from the chest with no dominant vertices")
        if ch.tort >= 1.7 and len(ch.joints) >= 3:
            return put(ch.joints, "other", 0.6, f"arc from the chest (tortuosity {ch.tort:.2f})")
        if ch.horiz >= 0.3 or abs(ch.mean_x - axis_x) >= 0.05 * W:
            return tag_arm(ch, f"chain from the chest {rig.names[ch.attach]} ({ch.dir})", from_chest=True)
        return put(ch.joints, "other", 0.4, f"{ch.dir} chain from the chest centre")

    for ch in sorted(chains, key=lambda c: c.depth):
        if roles[ch.first] != "other" or rig.names[ch.first] in reasons:   # body/head/lid/antenna already placed
            continue
        att = ch.attach
        ar = roles[att] if att is not None else None
        if att is None or ar in ("root", "other") or (ar == "body" and att != chest):
            root_side(ch)                                  # "other" says nothing about its children: geometry decides
        elif att == chest:
            chest_side(ch)
        elif ar in ("arm.L", "arm.R"):
            for j in ch.joints:
                fingers.add(j)
            put(ch.joints, ar, conf[att] * 0.9, f"sub-chain of {rig.names[att]} ({ar}): fingers")
        elif ar in ("leg.L", "leg.R"):
            put(ch.joints, ar, conf[att] * 0.9, f"sub-chain of {rig.names[att]} ({ar}): toes")
        elif ar == "head":
            put(ch.joints, "antenna" if ch.u[1] >= 0.3 else "other", 0.5,
                f"chain off the head going {ch.dir}")
        elif ar == "spout":
            v = ch.u
            role = "tail" if (v[1] < 0 and ch.back >= 0.3) else "antenna"
            put(ch.joints, role, 0.6, f"chain off the spout tip going {ch.dir}")
        else:
            put(ch.joints, ar, conf[att] * 0.8, f"sub-chain of {rig.names[att]} ({ar})")

    # summary
    nl, nr, al, ar_ = (len(limbs[r]) for r in ("leg.L", "leg.R", "arm.L", "arm.R"))
    has = lambda r: r in roles
    if has("lid") or has("spout") or has("wheel") or (nl + nr + al + ar_ == 0):
        summary = "object"
    elif nl >= 1 and nr >= 1 and al >= 1 and ar_ >= 1 and body:
        summary = "biped+fingers" if fingers else "biped"
    elif nl >= 2 and nr >= 2 and al + ar_ == 0:
        summary = "quadruped"
    else:
        summary = "unknown"
    shares = {r: float(rig.share[[k for k in range(rig.n) if roles[k] == r]].sum()) for r in ROLES if r in roles}
    return dict(roles=roles, confidence=conf, chains=chains, chain_of=chain_of, reasons=reasons, notes=notes,
                summary=summary, fingers=fingers, body=body, head=head, chest=chest, lid=lid, root=root, shares=shares,
                limbs=limbs)


# ---------------------------------------------------------------------------------------------------
# Mixamo names for bipeds
# ---------------------------------------------------------------------------------------------------
SPINE_NAMES = {1: ["Spine"], 2: ["Spine", "Spine1"], 3: ["Spine", "Spine1", "Spine2"]}


def mixamo_map(rig, T):
    names, roles, chains = rig.names, T["roles"], T["chains"]
    m, notes = {}, []
    pre = "mixamorig:"
    m[names[T["root"]]] = pre + "Hips"
    body = T["body"]
    if body:
        if len(body) <= 3:
            picks = list(zip(body, SPINE_NAMES[len(body)]))
        else:
            picks = [(body[0], "Spine"), (body[len(body) // 2], "Spine1"), (body[-1], "Spine2")]
            notes.append(f"{len(body)} body joints: Spine/Spine1/Spine2 = first, middle, last")
        for j, nm in picks:
            m[names[j]] = pre + nm
    head = T["head"]
    if head:
        if len(head) == 1:
            picks = [(head[0], "Head")]
        elif len(head) == 2:
            picks = [(head[0], "Neck"), (head[1], "Head")]
        else:
            picks = [(head[0], "Neck"), (head[1], "Head"), (head[-1], "HeadTop_End")]
        for j, nm in picks:
            m[names[j]] = pre + nm
    for role, side in (("arm.L", "Left"), ("arm.R", "Right")):
        arms = T["limbs"][role]                            # whole chains: a clavicle tagged body is the Shoulder
        if not arms:
            continue
        if len(arms) > 1:
            notes.append(f"{len(arms)} {role} chains: only the first ({names[arms[0].first]}) maps")
        js = [j for j in arms[0].joints if j not in T["fingers"]]
        order = {4: ["Shoulder", "Arm", "ForeArm", "Hand"], 3: ["Arm", "ForeArm", "Hand"], 2: ["Arm", "ForeArm"], 1: ["Arm"]}
        for j, nm in zip(js, order[min(4, len(js))]):
            m[names[j]] = pre + side + nm
    for role, side in (("leg.L", "Left"), ("leg.R", "Right")):
        legs = T["limbs"][role]
        if not legs:
            continue
        if len(legs) > 1:
            notes.append(f"{len(legs)} {role} chains: only the first ({names[legs[0].first]}) maps")
        js = legs[0].joints
        for j, nm in zip(js, ["UpLeg", "Leg", "Foot", "ToeBase", "Toe_End"]):
            m[names[j]] = pre + side + nm
    if T["fingers"]:
        notes.append(f"{len(T['fingers'])} finger joint(s) left unmapped: " + ", ".join(names[j] for j in sorted(T["fingers"])))
    return m, notes


# ---------------------------------------------------------------------------------------------------
# outputs
# ---------------------------------------------------------------------------------------------------
def write_glb(rig, roles_by_name, out_path):
    """Copy of the GLB with node.extras["m3d_role"] on every joint, written to a temp file, reloaded and checked."""
    from pygltflib import GLTF2
    g = rig.g
    for j, name in zip(rig.joints, rig.names):
        node = g.nodes[j]
        extras = dict(node.extras or {})
        extras["m3d_role"] = roles_by_name[name]
        node.extras = extras
    blob = g.binary_blob()
    d = os.path.dirname(os.path.abspath(out_path)) or "."
    fd, tmp = tempfile.mkstemp(prefix=".roles-", suffix=".glb", dir=d)
    os.close(fd); os.chmod(tmp, 0o644)                   # mkstemp makes it 0600
    try:
        g.set_binary_blob(blob)
        g.save(tmp)
        back = GLTF2().load(tmp)
        got = {back.nodes[j].name: (back.nodes[j].extras or {}).get("m3d_role") for j in back.skins[0].joints}
        if got != roles_by_name:
            raise SystemExit(f"roles: the written GLB does not carry the roles ({tmp})")
        if len(back.binary_blob()) != len(blob) or len(back.accessors) != len(g.accessors):
            raise SystemExit(f"roles: the written GLB lost buffer data ({tmp})")
        os.replace(tmp, out_path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return out_path


def plot(rig, T, out, title):
    """rigplot-style front/side picture: vertices grey, bones coloured by the child joint's role, labels."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    roles, pos = T["roles"], rig.pos
    idx = np.random.default_rng(0).choice(rig.nv, min(rig.nv, 20000), replace=False)
    bones = [(rig.jparent[k], k) for k in range(rig.n) if rig.jparent[k] is not None]
    fig, axes = plt.subplots(1, 2, figsize=(13, 7.5))
    for ax, (a, b, name) in zip(axes, [(0, 1, "front (x,y)"), (2, 1, "side (z,y)")]):
        ax.scatter(rig.V[idx, a], rig.V[idx, b], s=0.2, c="#cccccc")
        for p, c in bones:
            ax.plot([pos[p, a], pos[c, a]], [pos[p, b], pos[c, b]], "-", c=ROLE_COLOURS[roles[c]], lw=2.5)
        ax.scatter(pos[:, a], pos[:, b], s=22, c=[ROLE_COLOURS[r] for r in roles], zorder=3, edgecolors="black", linewidths=0.4)
        for k in range(rig.n):
            ax.annotate(f"{rig.names[k]} {roles[k]}", (pos[k, a], pos[k, b]), fontsize=6, color=ROLE_COLOURS[roles[k]],
                        xytext=(3, 2), textcoords="offset points")
        ax.set_aspect("equal"); ax.set_title(name)
    present = [r for r in ROLES if r in roles]
    axes[0].legend([Line2D([0], [0], color=ROLE_COLOURS[r], lw=3) for r in present], present, fontsize=7, loc="upper left")
    fig.suptitle(title, fontsize=10)
    plt.tight_layout(); plt.savefig(out, dpi=110); plt.close(fig)
    return out


def table(rig, T, overrides):
    roles, conf, chain_of = T["roles"], T["confidence"], T["chain_of"]
    lines = [f"{'bone':<10}{'parent':<10}{'role':<9}{'conf':>5}{'verts':>7}{'share':>7}  {'chain':<6}note"]
    for k in range(rig.n):
        p = rig.jparent[k]
        extra = "override" if rig.names[k] in overrides else ("fingers" if k in T["fingers"] else "")
        lines.append(f"{rig.names[k]:<10}{(rig.names[p] if p is not None else '-'):<10}{roles[k]:<9}{conf[k]:>5.2f}"
                     f"{rig.cnt[k]:>7}{rig.share[k]:>7.1%}  {chain_of[k]:<6}{extra}")
    lines.append("chains:")
    for i, ch in enumerate(T["chains"]):
        why = " | ".join(T["reasons"][n] for n in ch.names(rig) if n in T["reasons"])   # a split chain has several
        lines.append(f"  [{i}] {' > '.join(ch.names(rig))}  ({ch.dir}, {len(ch.joints)} joints, {ch.share:.1%} verts)  {why}")
    lines.append("vertex share by role: " + ", ".join(f"{r}={s:.0%}" for r, s in T["shares"].items()))
    for n in T["notes"]:
        lines.append(f"note: {n}")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description="tag the bones of a rigged GLB with roles for motion presets")
    ap.add_argument("glb")
    ap.add_argument("--out-json", default=None, help="sidecar path (default <stem>.roles.json beside the GLB)")
    ap.add_argument("--write-glb", nargs="?", const="auto", default=None, help="GLB copy with node extras (default <stem>-roles.glb)")
    ap.add_argument("--in-place", action="store_true", help="write the node extras into the input GLB")
    ap.add_argument("--humanoid-names", action="store_true", help="emit the Mixamo name map even when the rig is not a biped")
    ap.add_argument("--plot", default=None, help="write a front/side picture coloured by role")
    ap.add_argument("--facing", choices=["+z", "-z"], default="+z", help="which way the character faces (glTF: +z); write --facing=-z")
    ap.add_argument("--override", default="", help="bone_7=arm.L,bone_8=arm.L: force roles after tagging")
    ap.add_argument("--json", action="store_true", help="one JSON object on stdout (table on stderr), then the marker line")
    argv = list(sys.argv[1:] if argv is None else argv)
    for i, s in enumerate(argv[:-1]):                      # argparse reads a bare "-z" as an option: join it
        if s == "--facing" and argv[i + 1] in ("+z", "-z"):
            argv[i:i + 2] = [f"--facing={argv[i + 1]}"]
    a = ap.parse_args(argv)

    rig = Rig(a.glb)
    T = tag_rig(rig, a.facing)
    overrides = {}
    for item in filter(None, a.override.split(",")):
        if "=" not in item:
            raise SystemExit(f"roles: bad --override item {item!r} (want bone=role)")
        bone, role = (s.strip() for s in item.split("=", 1))
        if bone not in rig.names:
            raise SystemExit(f"roles: --override names unknown bone {bone!r}; bones: {', '.join(rig.names)}")
        if role not in ROLES:
            raise SystemExit(f"roles: --override gives unknown role {role!r}; roles: {', '.join(ROLES)}")
        k = rig.names.index(bone)
        T["roles"][k] = role; T["confidence"][k] = 1.0; overrides[bone] = role
    if overrides:                                          # the summary and shares follow the overrides
        roles = T["roles"]
        T["shares"] = {r: float(rig.share[[k for k in range(rig.n) if roles[k] == r]].sum()) for r in ROLES if r in roles}
        T["notes"].append("overrides applied: " + ", ".join(f"{b}={r}" for b, r in overrides.items()))

    roles_by_name = {n: r for n, r in zip(rig.names, T["roles"])}
    stem = os.path.splitext(os.path.basename(a.glb))[0]
    folder = os.path.dirname(os.path.abspath(a.glb))
    out_json = a.out_json or os.path.join(folder, f"{stem}.roles.json")
    side = {"roles": roles_by_name,
            "confidence": {n: c for n, c in zip(rig.names, T["confidence"])},
            "chains": [ch.names(rig) for ch in T["chains"]],
            "summary": T["summary"],
            "reasons": T["reasons"], "notes": T["notes"], "facing": a.facing, "overrides": overrides,
            "vertex_share": {r: round(s, 4) for r, s in T["shares"].items()},
            "source": os.path.abspath(a.glb)}
    if a.humanoid_names or T["summary"].startswith("biped"):
        mix, mnotes = mixamo_map(rig, T)
        if len(mix) <= 1 and not T["summary"].startswith("biped"):
            T["notes"].append("no Mixamo map: the rig has no body/arms/legs to name")
        else:
            side["mixamo"] = mix
            if mnotes:
                side["mixamo_notes"] = mnotes
    with open(out_json, "w") as f:
        json.dump(side, f, indent=1)
    glb_out = None
    if a.in_place:
        glb_out = write_glb(rig, roles_by_name, a.glb)
    elif a.write_glb:
        glb_out = write_glb(rig, roles_by_name, os.path.join(folder, f"{stem}-roles.glb") if a.write_glb == "auto" else a.write_glb)
    plot_out = plot(rig, T, a.plot, f"{stem} — {T['summary']}") if a.plot else None

    text = table(rig, T, overrides)
    if a.json:
        print(text, file=sys.stderr)
        print(json.dumps({**side, "glb": glb_out, "plot": plot_out, "json": out_json}))
    else:
        print(text)
        if "mixamo" in side:
            print("mixamo: " + ", ".join(f"{b}={n.split(':')[1]}" for b, n in side["mixamo"].items()))
        if glb_out:
            print(f"glb with extras: {glb_out}")
        if plot_out:
            print(f"plot: {plot_out}")
    tagged = sum(1 for r in T["roles"] if r != "other")
    print(f"M3D_ROLES bones={rig.n} tagged={tagged} summary={T['summary']} out={out_json}")


if __name__ == "__main__":
    main()

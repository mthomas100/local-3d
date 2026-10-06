"""weightfix.py in.glb out.glb [--roles roles.json] [--rip] [--limbs auto|6-12,13-19] [--trunk auto|0-3]
                               [--margin 0.01] [--ball 2.0] [--far 2.5] [--no-misbound] [--harden none|all|5,0] [--json]
Cleans auto-rig skin weights (SkinTokens and the like) so limbs stop dragging the body and the body stops holding
limbs, then writes a new GLB (worker venv python: numpy, scipy, pygltflib; run under bin/memguard). Without --rip
everything but JOINTS_0/WEIGHTS_0 stays byte-identical. Written 2026-10-02 for two SkinTokens faults
(the lamp animation QA notes, 2026-10-02, not published): the lamp's clavicle owned a patch of its torso wall (arm swings sheared the
torso), and the knight's inner arms were bound to its hips and thighs (a raised arm tore into a fan).

--roles is the role tagger's sidecar ({"roles": {bone: role}} or a flat map); default <input stem>.roles.json.
Without one, groups come from the skeleton (each limb chain, each trunk bone, each other subtree). The decisions a
later run must not re-derive are written to skins[0].extras.m3d_weightfix (internal bones, rip count).

Passes, in order:
  misbound  skin bound to a group not adjacent to the surface it sits on. Groups are adjacent when a bone of one is
          the parent of a bone of the other. Each limb's surface is the source side of a minimum cut (capacity =
          edge length, so the shortest seam) between its confident vertices and everyone else's (labels()); off
          that surface a vertex the weights give to a limb keeps it. A limb-surface vertex hands non-adjacent
          weight to the limb's nearest bone; any other vertex hands non-adjacent limb weight to its own remaining
          weights. Shoulder, hip and neck blends are adjacent, spine blends trunk-to-trunk: untouched. Foreign
          weight under 1% is left. (A first version labelled by geodesic distance from weight-picked seeds and
          iterated; on the knight it crept across the hand/hip contact and bound a slab of skirt to the forearm.)
  bleed   weight of a limb's internal bones (clavicles: ≥ 40% of the segment to the next joint inside the trunk
          hull, internal_bones()) on vertices on or inside the trunk hull (outset --margin × bbox diagonal), off the
          limb's own surface and farther than --ball × limb radius from its first outer joint, moves to the bone
          the limb hangs from; the region grows over welded mesh edges through vertices those internal bones own,
          since the hull cuts box corners. Limbs without an internal bone (the knight's legs) are skipped: a skirt
          hem with some thigh weight is normal skinning. One pass (a second, re-measured pass eats the shoulder).
  far     limb-bone weight farther than --far × the chain's own vertex spread from every segment of the chain moves
          to the vertex's other influences (or the attach bone). Stray blobs.
  harden  optional argmax: vertices dominated by the listed bones get weight 1.0 on that bone (rigid parts).
  rip     --rip, needs --roles: weights cannot stop a fan where the mesh itself joins a limb to a non-adjacent part
          (the knight's hands are fused into its hips: ~200 bridging triangles). Bridging triangles go to their
          majority side and their other-side corners are duplicated (nudged 1e-5 × size so a later weld by position
          keeps them apart); the limb's attachment ring is never ripped. Adds vertices (new accessors for that
          primitive) and leaves small openings where the surfaces were fused. In auto mode the per-bone trunk
          groups make an arm's contact with the lower torso look foreign (lamp: 224 copies vs 0 with roles).

Limbs (auto): every child subtree of a branching joint with ≥ 3 joints that does not hold the most-weighted bone
(the head); the trunk is the path from the root to the limbs' attach joints. A rig with no trunk-owned vertices
(puppet rigs) passes through unchanged. Last line: M3D_WEIGHTFIX changed=… misbound_verts=… bleed_verts=… ."""
import sys, json, argparse, numpy as np
from scipy.spatial import ConvexHull
from pygltflib import GLTF2

COMP = {5121: np.uint8, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def accessor_view(g, blob, ai):
    """(array copy, writer) for accessor ai; the writer puts an array of the same shape back into blob in place."""
    a = g.accessors[ai]; bv = g.bufferViews[a.bufferView]; dt = np.dtype(COMP[a.componentType]); n = NCOMP[a.type]
    off = (bv.byteOffset or 0) + (a.byteOffset or 0); stride = bv.byteStride or dt.itemsize * n
    raw = np.frombuffer(blob, np.uint8, count=stride * (a.count - 1) + dt.itemsize * n, offset=off)
    arr = np.lib.stride_tricks.as_strided(raw, (a.count, n * dt.itemsize), (stride, 1)).copy().view(dt).reshape(a.count, n)
    def write(new):
        new = np.asarray(new)
        if a.normalized and dt.kind == "u":
            new = np.round(np.clip(new, 0, 1) * np.iinfo(dt).max)
        b = np.ascontiguousarray(new.astype(dt)).view(np.uint8).reshape(a.count, -1)
        for i in range(a.count): blob[off + i * stride: off + i * stride + b.shape[1]] = b[i].tobytes()
    if a.normalized and dt.kind == "u": arr = arr.astype(np.float64) / np.iinfo(dt).max
    return arr, write


def parse_sets(s):
    out = []
    for grp in s.split(","):
        lo, _, hi = grp.partition("-"); out.append(list(range(int(lo), int(hi or lo) + 1)))
    return out


def topology(g, skin):
    """parent index (in joint space, -1 for root) and children lists for the skin's joints."""
    jidx = {n: i for i, n in enumerate(skin.joints)}
    parent = np.full(len(skin.joints), -1)
    for i, n in enumerate(skin.joints):
        for c in g.nodes[n].children or []:
            if c in jidx: parent[jidx[c]] = i
    children = [[] for _ in skin.joints]
    for i, p in enumerate(parent):
        if p >= 0: children[p].append(i)
    return parent, children


def subtree(children, r):
    out, st = [], [r]
    while st:
        x = st.pop(); out.append(x); st.extend(children[x])
    return out


def load_roles(path):
    """role-tagger sidecar ({"roles": {bone: role}, …}) or a flat {bone: role}; None if absent or empty."""
    import os
    if not path or not os.path.exists(path): return None
    d = json.load(open(path)); d = d.get("roles", d) if isinstance(d, dict) else {}
    return {k: v for k, v in d.items() if isinstance(v, str)} or None


def limbs_and_trunk(parent, children, dom_counts, names, roles=None):
    """auto_limbs (≥ 3-joint chains, validated on SkinTokens rigs); when that finds none (puppet rigs: 2-bone arms,
    1-bone legs), the roles sidecar's arm.*/leg.* chains with root/body as trunk, else ≥ 1-joint chains."""
    limbs, trunk = auto_limbs(parent, children, dom_counts)
    if limbs: return limbs, trunk
    if roles:
        by = {}
        for b, n in enumerate(names):
            if roles.get(n, "").startswith(("arm.", "leg.")): by.setdefault(roles[n], []).append(b)
        trunk = [b for b, n in enumerate(names) if roles.get(n) in ("root", "body")]
        if by and trunk: return [sorted(v, key=lambda x: depth(parent, x)) for v in by.values()], trunk
    return auto_limbs(parent, children, dom_counts, min_len=1)


def auto_limbs(parent, children, dom_counts, min_len=3):
    head = int(np.argmax(dom_counts))
    limbs = []
    for b, ch in enumerate(children):
        if len(ch) < 2: continue
        for c in ch:
            st = subtree(children, c)
            if len(st) >= min_len and head not in st: limbs.append(sorted(st, key=lambda x: depth(parent, x)))
    trunk = set()
    for L in limbs:
        p = parent[L[0]]
        while p >= 0: trunk.add(int(p)); p = parent[p]
    return limbs, sorted(trunk)


def depth(parent, x):
    d = 0
    while parent[x] >= 0: d += 1; x = parent[x]
    return d


def hull_sd(P, eq, chunk=2048):
    """max over hull planes of the signed distance (≤ 0 inside). Chunked: a curved trunk's hull has thousands of
    facets, and points × facets in one go took 1.3 GB per call (5.9 GB peak in animcheck, 2026-10-02)."""
    out = np.empty(len(P))
    for i in range(0, len(P), chunk):
        out[i:i + chunk] = (P[i:i + chunk] @ eq[:, :3].T + eq[:, 3]).max(1)
    return out


def seg_dist(P, a, b):
    ab = b - a; t = np.clip(((P - a) @ ab) / max(ab @ ab, 1e-12), 0, 1)
    return np.linalg.norm(P - (a + t[:, None] * ab), axis=1)


def chain_segments(L, parent, jp):
    """segments of a limb: each joint to its children; a leaf joint gets a zero-length segment at itself."""
    segs = []
    for j in L:
        kids = [k for k in L if parent[k] == j]
        segs += [(jp[j], jp[k]) for k in kids] or [(jp[j], jp[j])]
    return segs


def grow(F, cand, seed, n, V=None):
    """vertices of cand connected (over mesh edges inside cand) to a seed vertex, plus the seeds."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    cand0 = cand
    if V is not None:                                     # weld split-normal seams: same position, same vertex
        _, wid = np.unique(np.round(V, 6), axis=0, return_inverse=True); wid = wid.ravel()
        rep = np.zeros(wid.max() + 1, int); rep[wid] = np.arange(n); F = rep[wid][F]
        cand = cand | (np.bincount(wid, weights=cand, minlength=wid.max() + 1)[wid] > 0)
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]); e = e[cand[e[:, 0]] & cand[e[:, 1]]]
    _, lab = connected_components(coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)), directed=False)
    if V is not None: lab = lab[rep[wid]]                 # every copy of a welded vertex takes its label
    hit = np.unique(lab[seed & cand])
    return seed | (cand0 & np.isin(lab, hit))


def bone_groups(parent, children, limbs, trunk, names, roles=None):
    """group label per bone and the group adjacency matrix (parent/child across groups)."""
    nb = len(parent); lab = [None] * nb
    if roles:
        for b, n in enumerate(names): lab[b] = roles.get(n)
    if any(x is None for x in lab):
        base = [None] * nb
        for i, L in enumerate(limbs):
            for j in L: base[j] = f"limb{i}"
        for j in trunk: base[j] = f"trunk.{names[j]}"     # one group per trunk bone: hips ≠ chest, like root ≠ body
        auto = list(base)
        for b in range(nb):
            if auto[b] is None:                           # other subtrees (head, tail …): named after their top bone
                x = b
                while parent[x] >= 0 and base[parent[x]] is None: x = parent[x]
                auto[b] = f"sub.{names[x]}"
        lab = [l if l is not None else auto[b] for b, l in enumerate(lab)]
    gnames = sorted(set(lab)); gi = {g: i for i, g in enumerate(gnames)}
    grp = np.array([gi[l] for l in lab]); adj = np.eye(len(gnames), dtype=bool)
    for b in range(nb):
        if parent[b] >= 0: adj[grp[b], grp[parent[b]]] = adj[grp[parent[b]], grp[b]] = True
    return grp, gnames, adj


def bone_segments(parent, children, jp):
    """per bone: head → mean of the children's heads; a leaf continues its parent's direction at half length."""
    segs = []
    for b in range(len(parent)):
        if children[b]: e = jp[children[b]].mean(0)
        elif parent[b] >= 0: e = jp[b] + 0.5 * (jp[b] - jp[parent[b]])
        else: e = jp[b]
        segs.append((jp[b], e))
    return segs


def welded_graph(V, F):
    """welded vertex ids (split-normal/UV seams merged by position) and the undirected edge list with lengths."""
    _, wid = np.unique(np.round(V, 6), axis=0, return_inverse=True); wid = wid.ravel(); nw = wid.max() + 1
    Vw = np.zeros((nw, 3)); Vw[wid] = V
    Fw = wid[F]; e = np.concatenate([Fw[:, [0, 1]], Fw[:, [1, 2]], Fw[:, [2, 0]]]); e = e[e[:, 0] != e[:, 1]]
    e = np.unique(np.sort(e, 1), axis=0)
    return wid, nw, e, np.linalg.norm(Vw[e[:, 0]] - Vw[e[:, 1]], axis=1)


def min_cut(nw, e, length, src, snk):
    """source side of the minimum cut separating welded vertices src from snk; capacity = edge length, so the
    cut is the shortest closed seam between them (on a fused hand/hip contact: around the contact patch)."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import maximum_flow, breadth_first_order
    S, T = nw, nw + 1; big = np.int32(2 ** 30)
    cap = np.maximum(1, np.round(length / max(length.mean(), 1e-12) * 1000)).astype(np.int32)
    si, ti = np.where(src)[0], np.where(snk)[0]
    r = np.concatenate([e[:, 0], e[:, 1], np.full(len(si), S), ti])
    c = np.concatenate([e[:, 1], e[:, 0], si, np.full(len(ti), T)])
    w = np.concatenate([cap, cap, np.full(len(si) + len(ti), big, np.int32)])
    G = csr_matrix((w, (r, c)), shape=(nw + 2, nw + 2)); G.sum_duplicates()
    res = maximum_flow(G, S, T, method="dinic")
    resid = G - res.flow; resid.data[resid.data < 0] = 0; resid.eliminate_zeros()
    side = np.zeros(nw + 2, bool); side[breadth_first_order(resid, S, directed=True, return_predecessors=False)] = True
    return side[:nw], int(res.flow_value)


def labels(V, F, Wd, parent, children, jp, limbs, trunk, names, roles=None, conf=0.5, internal=None):
    """bone groups, adjacency and a per-vertex surface group. Each limb group gets the source side of a minimum
    cut between its confident vertices (dominated by the limb and nearer its segments than conf × any other
    group's) and everyone else's confident vertices (dominated elsewhere, nearer elsewhere); every other vertex
    takes its nearest non-limb group. Geometry decides the seam; the input weights only pick the seeds, so a
    second run sees the same cut. Limb bones whose joint is inside the trunk (clavicles, hip bones) do not count
    as anyone's nearest segment: their in-body segments would claim torso skin."""
    grp, gnames, adj = bone_groups(parent, children, limbs, trunk, names, roles)
    jin = skip = internal if internal is not None else internal_bones(V, Wd, trunk, jp, parent, limbs)
    segs = bone_segments(parent, children, jp)
    D = np.stack([seg_dist(V, a, e) for a, e in segs], 1); Ds = np.where(skip[None, :], np.inf, D)
    ng = len(gnames); Dg = np.stack([Ds[:, grp == k].min(1) if np.any((grp == k) & ~skip) else np.full(len(V), np.inf)
                                     for k in range(ng)], 1)
    limb_groups = np.zeros(ng, bool)
    for L in limbs: limb_groups[grp[[j for j in L if not jin[j]] or L]] = True
    domg = grp[Wd.argmax(1)]
    wid, nw, e, length = welded_graph(V, F)
    lab = np.where(limb_groups[None, :], np.inf, Dg).argmin(1)    # non-limb default: nearest non-limb group
    best = np.full(len(V), np.inf); cuts = {}
    for k in np.where(limb_groups)[0]:
        other = np.delete(Dg, k, 1).min(1)
        src = np.zeros(nw, bool); snk = np.zeros(nw, bool)
        src[wid[(domg == k) & (Dg[:, k] < conf * other)]] = True
        snk[wid[(domg != k) & (other < conf * Dg[:, k])]] = True
        snk &= ~src
        if not src.any(): continue
        side, flow = min_cut(nw, e, length, src, snk)
        on = side[wid]; score = Dg[:, k] / np.maximum(other, 1e-9)
        take = on & (score < best)                        # two limbs claim a vertex: the relatively nearer wins
        lab[take] = k; best[take] = score[take]
        cuts[gnames[k]] = dict(side_verts=int(on.sum()), seeds=int(src.sum()), sinks=int(snk.sum()), cut=flow)
    # the cut decides a limb's surface; off it, a vertex the weights give to a limb keeps that limb (where the cut
    # and the weights disagree off-limb, the weights win: lamp torso top lies nearest the neck segment, the
    # knight's inner forearm strip is fused into the skirt)
    flip = ~limb_groups[lab] & limb_groups[domg] & ~adj[lab, domg]
    lab = np.where(flip, domg, lab)
    return dict(grp=grp, gnames=gnames, adj=adj, lab=lab, D=D, limb_groups=limb_groups, cuts=cuts, internal=jin)


def misbound(Wd, grp, gnames, adj, lab, D, limb_groups):
    """move weight on bones of groups not adjacent to each vertex's surface group to that group's nearest bone.
    Only pairs involving a limb group: trunk-to-trunk and trunk-to-head blends (a spine) are never cut."""
    Wd = Wd.copy()
    foreign = ~adj[lab][:, grp]                           # verts x bones: bone's group not adjacent to the surface
    foreign &= limb_groups[lab][:, None] | limb_groups[grp][None, :]
    fw = np.where(foreign, Wd, 0).sum(1); hit = fw > 0.01  # slivers under 1% are left (they reshuffle every run)
    near_own = np.where(grp[None, :] == lab[:, None], D, np.inf).argmin(1)
    keep = np.where(foreign, 0, Wd); tgt = np.where(limb_groups[lab], near_own,     # on a limb: its nearest bone
                                                    np.where(keep.sum(1) > 1e-6, keep.argmax(1), near_own))  # else: own weights
    pairs = {}
    for i in np.where(hit)[0]:
        for b in np.where(foreign[i] & (Wd[i] > 1e-3))[0]:
            k = f"{gnames[grp[b]]}→{gnames[lab[i]]}"; pairs[k] = pairs.get(k, 0.0) + float(Wd[i, b])
    idx = np.where(hit)[0]
    Wd[idx] = np.where(foreign[idx], 0, Wd[idx]); Wd[idx, tgt[idx]] += fw[idx]
    rep = dict(verts=int(hit.sum()), weight=round(float(fw.sum()), 2), strong_verts=int((fw > 0.5).sum()),
               by_pair={k: round(v, 1) for k, v in sorted(pairs.items(), key=lambda kv: -kv[1])})
    return Wd, rep


def trunk_hull(V, Wd, trunk, jp):
    """convex-hull plane equations of the trunk-dominated vertices, and which joints lie inside."""
    H = ConvexHull(V[np.isin(Wd.argmax(1), trunk)]).equations
    return H, hull_sd(jp, H) <= 0


def internal_bones(V, Wd, trunk, jp, parent, limbs, frac=0.4):
    """limb bones that live inside the body (clavicles): ≥ frac of the segment to their child joint lies inside the
    trunk hull. Measured 2026-10-02: clavicles 42–62% (lamp, knight), lamp hips 32–38%, knight thighs 18–24%
    (inside only because the skirt flares the hull); "joint inside the hull" called the knight's thighs internal
    and the bleed pass then cut the hip joint into a hard seam."""
    out = np.zeros(len(parent), bool); t = np.linspace(0, 1, 41)[:, None]
    if np.isin(Wd.argmax(1), trunk).sum() < 4: return out    # no trunk skin (puppet rigs with an empty root)
    H, _ = trunk_hull(V, Wd, trunk, jp)
    for L in limbs:
        for j in L:
            kids = [k for k in L if parent[k] == j]
            if not kids: break
            if np.mean(hull_sd(jp[j] + t * (jp[kids[0]] - jp[j]), H) <= 0) >= frac: out[j] = True
            else: break                                   # only a run of internal bones from the limb's root
    return out


def rip(V, F, Wd, S, jp, parent, limbs, internal, ball=2.0):
    """split fused contacts (--rip). Triangles joining a limb's surface to a group not adjacent to it (hand fused
    into hip, pauldron into hair) are given to their majority side and their other-side corners are duplicated,
    so the two surfaces separate instead of stretching into a fan; the copy is weighted to its triangle's side.
    A limb's attachment ring (within ball × limb radius of its first joint outside the body) is never ripped.
    Returns (new faces, source vertex of each copy, dense weights of the copies, report)."""
    grp, adj, lab, D, lg = S["grp"], S["adj"], S["lab"], S["D"], S["limb_groups"]
    tl = lab[F]; tri_side = np.full(len(F), -1); keep_out = np.zeros(len(V), bool); rep = {}
    for L in limbs:
        ext = [j for j in L if not internal[j]] or L; k = int(np.bincount(grp[ext]).argmax())
        if not lg[k]: continue
        own = np.isin(Wd.argmax(1), ext)
        radius = float(np.median(np.min([seg_dist(V[own], jp[j], jp[c]) for j in ext for c in [x for x in L if parent[x] == j]] or
                                        [np.linalg.norm(V[own] - jp[ext[0]], axis=1)], axis=0))) if own.any() else 0.05
        near_anchor = np.linalg.norm(V - jp[ext[0]], axis=1) < ball * radius
        on = tl == k; foreign = ~adj[k][tl] & ~on
        bridge = on.any(1) & foreign.any(1) & ~near_anchor[F].any(1)
        if not bridge.any(): continue
        maj = on.sum(1) >= 2
        tri_side[bridge & maj] = k
        for t in np.where(bridge & ~maj)[0]:
            f = tl[t][foreign[t]]; tri_side[t] = int(np.bincount(f).argmax())
        rep[S["gnames"][k]] = int(bridge.sum())
    # duplicate corners whose label differs from their triangle's side
    F = F.copy(); dup_src, dup_side, key = [], [], {}
    for t in np.where(tri_side >= 0)[0]:
        for c in range(3):
            v = F[t, c]
            if lab[v] == tri_side[t]: continue
            kk = (int(v), int(tri_side[t]))
            if kk not in key: key[kk] = len(V) + len(dup_src); dup_src.append(int(v)); dup_side.append(int(tri_side[t]))
            F[t, c] = key[kk]
    dup_src = np.array(dup_src, int); dup_side = np.array(dup_side, int)
    Wc = Wd[dup_src].copy() if len(dup_src) else np.zeros((0, Wd.shape[1]))
    for i, (v, sd) in enumerate(zip(dup_src, dup_side)):
        ok = adj[sd][grp]                                 # keep only weights the copy's side may carry
        Wc[i] = np.where(ok, Wc[i], 0)
        if Wc[i].sum() < 1e-6: Wc[i, np.where(grp == sd, D[v], np.inf).argmin()] = 1.0
        Wc[i] /= Wc[i].sum()
    # nudge each copy 1e-5 × model size toward the centroid of its triangles: welding by position (UV and
    # normal seams) would otherwise re-join the rip on the next run
    Vc = V[dup_src].copy() if len(dup_src) else np.zeros((0, 3)); eps = 1e-5 * np.linalg.norm(np.ptp(V, 0))
    if len(dup_src):
        Vall = np.concatenate([V, Vc]); cen = np.zeros_like(Vc); cnt = np.zeros(len(Vc))
        for t in np.where(tri_side >= 0)[0]:
            for v in F[t]:
                if v >= len(V): cen[v - len(V)] += Vall[F[t]].mean(0); cnt[v - len(V)] += 1
        d = cen / np.maximum(cnt, 1)[:, None] - Vc; Vc += eps * d / np.maximum(np.linalg.norm(d, axis=1), 1e-12)[:, None]
    rep["copies"] = int(len(dup_src))
    return F, dup_src, Wc, rep, Vc


def append_accessor(g, blob, arr, like):
    """append arr as a new accessor shaped like accessor `like` (type, component type, normalization)."""
    from pygltflib import Accessor, BufferView
    a = g.accessors[like]; dt = np.dtype(COMP[a.componentType])
    if a.normalized and dt.kind == "u": arr = np.round(np.clip(arr, 0, 1) * np.iinfo(dt).max)
    data = np.ascontiguousarray(np.asarray(arr).astype(dt)).tobytes()
    while len(blob) % 4: blob.append(0)
    bv = g.bufferViews[a.bufferView] if a.bufferView is not None else None
    g.bufferViews.append(BufferView(buffer=0, byteOffset=len(blob), byteLength=len(data), target=bv.target if bv else None))
    blob.extend(data)
    na = Accessor(bufferView=len(g.bufferViews) - 1, byteOffset=0, componentType=a.componentType, normalized=a.normalized,
                  count=len(arr), type=a.type, min=a.min, max=a.max)
    g.accessors.append(na); return len(g.accessors) - 1


def fix(V, Wd, parent, children, jp, limbs, trunk, margin=0.01, ball=2.0, far=2.5, harden=(), F=None, iters=1,
        lab=None, grp=None, internal=None):
    """returns (new dense weights, per-limb report, on-trunk mask of the first pass). iters > 1 re-runs the bleed
    pass with the hull of the new weights; it does not converge on the lamp (each pass grows the hull into the
    shoulder ball and eats it), so the default is one pass against the input's hull, corners handled by grow()."""
    Wd = Wd.copy(); diag = np.linalg.norm(V.max(0) - V.min(0)); rep = {}; first_on = None
    if np.isin(Wd.argmax(1), trunk).sum() < 4 or not limbs:   # nothing to measure a body against: no-op
        return Wd, {"note": "no trunk-owned vertices; bleed/far skipped"}, np.zeros(len(V), bool)
    def hull_of(W):
        H = ConvexHull(V[np.isin(W.argmax(1), trunk)]).equations
        return hull_sd(V, H) <= margin * diag, hull_sd(jp, H) <= 0
    on_trunk, _ = hull_of(Wd); first_on = on_trunk
    jin = internal if internal is not None else internal_bones(V, Wd, trunk, jp, parent, limbs)
    dom0 = Wd.argmax(1); geo = {}
    for L in limbs:                                       # chain geometry, fixed from the input weights
        anchor = next((j for j in L if not jin[j]), L[-1])        # first joint of the chain outside the trunk
        ext = [j for j in L if not jin[j]]
        segs = chain_segments(ext or L, parent, jp)
        own = np.isin(dom0, ext or L) & ~on_trunk           # the limb proper: weights this pass never moves
        dseg = np.min([seg_dist(V, a, b) for a, b in segs], axis=0)
        radius = float(np.median(dseg[own])) if own.any() else 0.02 * diag
        spread = float(np.percentile(dseg[own], 95)) if own.any() else radius
        away = np.linalg.norm(V - jp[anchor], axis=1) > ball * radius
        if lab is not None:                               # never bleed the limb's own surface (knight: arms inside the hip hull)
            away = away & ~np.isin(lab, np.unique(grp[ext or L]))
        geo[tuple(L)] = dict(att=int(parent[L[0]]), anchor=int(anchor), ext=ext, dseg=dseg, radius=radius, spread=spread,
                             away=away, bleed_verts=0, bleed_weight=0.0, passes=0)
    for it in range(iters):
        if it: on_trunk, _ = hull_of(Wd)
        moved = 0.0; dom = Wd.argmax(1)
        for L in limbs:
            G = geo[tuple(L)]; inn = [j for j in L if jin[j]]
            if not inn: continue                          # only bones living inside the body (clavicles) bleed
            w = Wd[:, inn].sum(1)
            bleed = (w > 1e-3) & on_trunk & G["away"]
            if F is not None and bleed.any():
                # grow over mesh edges: the hull cuts corners that the limb's internal bones (clavicle, hip) still own
                cand = (w > 1e-3) & G["away"] & ~np.isin(dom, G["ext"])
                bleed = grow(F, cand, bleed, len(V), V)
            if not bleed.any(): continue
            Wd[np.ix_(bleed, [G["att"]])] += w[bleed][:, None]; Wd[np.ix_(bleed, inn)] = 0
            G["bleed_verts"] += int(bleed.sum()); G["bleed_weight"] += float(w[bleed].sum()); G["passes"] = it + 1
            moved += float(w[bleed].sum())
        if moved < 1e-6: break
    for L in limbs:
        G = geo[tuple(L)]; att = G["att"]
        # far pass: the chain's own spread (p95 of owned-vertex distance to the chain) sets the cut-off
        farv = (Wd[:, L].sum(1) > 1e-3) & (G["dseg"] > far * G["spread"]) & ~first_on
        wf = Wd[farv][:, L].sum(1); Wd[np.ix_(farv, L)] = 0
        rest = Wd[farv].sum(1); empty = rest < 1e-6; idx = np.where(farv)[0]
        Wd[idx[empty], att] = 1.0
        Wd[idx[~empty]] /= rest[~empty, None]
        rep[",".join(f"{j}" for j in L)] = dict(attach=att, anchor=G["anchor"], radius=round(G["radius"], 4),
                                               spread=round(G["spread"], 4), bleed_verts=G["bleed_verts"],
                                               bleed_weight=round(G["bleed_weight"], 2), bleed_passes=G["passes"],
                                               far_verts=int(farv.sum()), far_weight=round(float(wf.sum()), 2))
    if len(harden):
        d2 = Wd.argmax(1); hv = np.isin(d2, list(harden))
        Wd[hv] = 0; Wd[np.where(hv)[0], d2[hv]] = 1.0; rep["harden_verts"] = int(hv.sum())
    return Wd, rep, first_on


def top4(Wd):
    J = np.argsort(-Wd, axis=1)[:, :4]; W = np.take_along_axis(Wd, J, 1)
    W /= np.maximum(W.sum(1, keepdims=True), 1e-12)
    return J, W


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("inp"); ap.add_argument("out")
    ap.add_argument("--limbs", default="auto"); ap.add_argument("--trunk", default="auto")
    ap.add_argument("--margin", type=float, default=0.01); ap.add_argument("--ball", type=float, default=2.0)
    ap.add_argument("--far", type=float, default=2.5); ap.add_argument("--harden", default="none")
    ap.add_argument("--roles", help="role sidecar (.roles.json) of the rig; default: <input stem>.roles.json if present")
    ap.add_argument("--no-misbound", action="store_true"); ap.add_argument("--json", action="store_true")
    ap.add_argument("--rip", action="store_true", help="split fused limb contacts (adds vertices; see rip())")
    a = ap.parse_args()
    g = GLTF2().load(a.inp); blob = bytearray(g.binary_blob()); skin = g.skins[0]; nb = len(skin.joints)
    IB = accessor_view(g, blob, skin.inverseBindMatrices)[0].reshape(-1, 4, 4).transpose(0, 2, 1)
    jp = np.array([np.linalg.inv(m)[:3, 3] for m in IB])           # joint positions in bind space
    prims = [p for n in g.nodes if n.mesh is not None and n.skin == 0 for p in g.meshes[n.mesh].primitives
             if p.attributes.JOINTS_0 is not None]
    seen, parts = set(), []
    for p in prims:
        key = (p.attributes.POSITION, p.attributes.JOINTS_0, p.attributes.WEIGHTS_0)
        if key in seen: continue
        seen.add(key)
        V = accessor_view(g, blob, p.attributes.POSITION)[0].astype(float)
        J, wj = accessor_view(g, blob, p.attributes.JOINTS_0); W, ww = accessor_view(g, blob, p.attributes.WEIGHTS_0)
        Fp = accessor_view(g, blob, p.indices)[0].reshape(-1, 3).astype(int) if p.indices is not None else None
        parts.append((V, J.astype(int), W.astype(float), wj, ww, Fp, p))
    V = np.concatenate([x[0] for x in parts]); Wd = np.zeros((len(V), nb)); o = 0
    offs = np.cumsum([0] + [len(x[0]) for x in parts])
    F = np.concatenate([x[5] + o_ for x, o_ in zip(parts, offs) if x[5] is not None]) if all(x[5] is not None for x in parts) else None
    offsF = np.cumsum([0] + [len(x[5]) if x[5] is not None else 0 for x in parts])
    for Vp, J, W, _, _, _, _ in parts:
        np.add.at(Wd, (np.arange(o, o + len(Vp))[:, None].repeat(J.shape[1], 1), J), W); o += len(Vp)
    parent, children = topology(g, skin)
    import os
    rpath = a.roles or (os.path.splitext(a.inp)[0] + ".roles.json")
    roles = load_roles(rpath); names = [g.nodes[j].name or str(j) for j in skin.joints]
    limbs, trunk = limbs_and_trunk(parent, children, np.bincount(Wd.argmax(1), minlength=nb), names, roles)
    if a.limbs != "auto": limbs = parse_sets(a.limbs)
    if a.trunk != "auto": trunk = parse_sets(a.trunk)[0]
    harden = [] if a.harden == "none" else (list(range(nb)) if a.harden == "all" else [int(x) for x in a.harden.split(",")])

    mrep = None; W0 = Wd; S = dict(lab=None, grp=None)
    has_trunk = np.isin(Wd.argmax(1), trunk).sum() >= 4
    if F is not None and has_trunk and limbs:
        # internal bones from the input weights, used by every pass; a GLB this tool already wrote carries its
        # decision (rip copies at a hand/hip contact are hip-weighted and would grow the trunk hull next time)
        prev = (skin.extras or {}).get("m3d_weightfix", {}) if isinstance(skin.extras, dict) else {}
        internal = (np.isin(names, prev["internal"]) if "internal" in prev
                    else internal_bones(V, Wd, trunk, jp, parent, limbs))
        S = labels(V, F, Wd, parent, children, jp, limbs, trunk, names, roles, internal=internal)
        if not a.no_misbound:
            Wd, mrep = misbound(Wd, S["grp"], S["gnames"], S["adj"], S["lab"], S["D"], S["limb_groups"])
            mrep["cuts"] = S["cuts"]
    Wn, rep, _ = fix(V, Wd, parent, children, jp, limbs, trunk, a.margin, a.ball, a.far, harden, F,
                     lab=S["lab"], grp=S["grp"], internal=S.get("internal"))
    Wd = W0
    rrep = None
    if a.rip and F is not None and not roles:
        print("  rip: skipped, needs a roles sidecar (auto groups split the trunk per bone, so an arm's contact "
              "with the lower torso looks foreign: lamp 224 copies in auto mode, 0 with roles)")
    if a.rip and F is not None and roles and S.get("lab") is not None:
        S2 = labels(V, F, Wn, parent, children, jp, limbs, trunk, names, roles, internal=S["internal"])
        Fn, dup_src, Wc, rrep, Vc = rip(V, F, Wn, S2, jp, parent, limbs, S["internal"])
    Jn, Wn4 = top4(Wn); o = 0
    for pi, (Vp, J, W, wj, ww, Fp, prim) in enumerate(parts):
        n = len(Vp)
        mine = (dup_src >= o) & (dup_src < o + n) if rrep and len(dup_src) else np.zeros(0, bool)
        if not len(mine) or not mine.any():
            wj(Jn[o:o + n, :J.shape[1]]); ww(Wn4[o:o + n, :W.shape[1]]); o += n; continue
        # this primitive gains vertices: every attribute (and morph target) gets a new, longer accessor
        src = dup_src[mine] - o; local = np.full(len(V) + len(dup_src), -1); local[o:o + n] = np.arange(n)
        local[len(V) + np.where(mine)[0]] = n + np.arange(mine.sum())
        Jc, Wc4 = top4(Wc[mine])
        for name, ai in list(prim.attributes.__dict__.items()):
            if ai is None or not isinstance(ai, int): continue
            arr = accessor_view(g, blob, ai)[0]
            if name == "JOINTS_0": new = np.concatenate([Jn[o:o + n, :arr.shape[1]], Jc[:, :arr.shape[1]]])
            elif name == "WEIGHTS_0": new = np.concatenate([Wn4[o:o + n, :arr.shape[1]], Wc4[:, :arr.shape[1]]])
            elif name.startswith(("JOINTS_", "WEIGHTS_")): new = np.concatenate([arr * 0, arr[src] * 0])
            elif name == "POSITION": new = np.concatenate([arr, Vc[mine].astype(arr.dtype)])
            else: new = np.concatenate([arr, arr[src]])
            setattr(prim.attributes, name, append_accessor(g, blob, new, ai))
        for tg in prim.targets or []:
            for name, ai in list(tg.items()):
                arr = accessor_view(g, blob, ai)[0]; tg[name] = append_accessor(g, blob, np.concatenate([arr, arr[src]]), ai)
        tri = Fn[offsF[pi]:offsF[pi + 1]]                 # this primitive's faces (global ids, copies at the end)
        idx = local[tri].ravel()
        ia = g.accessors[prim.indices]
        if idx.max() > 65535 and ia.componentType != 5125: ia.componentType = 5125
        prim.indices = append_accessor(g, blob, idx[:, None] if ia.type == "SCALAR" else idx, prim.indices)
        g.accessors[prim.indices].min = None; g.accessors[prim.indices].max = None
        o += n
    if S.get("internal") is not None:
        ex = dict(skin.extras) if isinstance(skin.extras, dict) else {}
        old = ex.get("m3d_weightfix", {})
        ex["m3d_weightfix"] = dict(version=1, internal=[names[i] for i in np.where(S["internal"])[0]],
                                   roles=os.path.basename(rpath) if roles else None,
                                   ripped=int(old.get("ripped", 0)) + (rrep["copies"] if rrep else 0))
        skin.extras = ex
    g.buffers[0].byteLength = len(blob)
    g.set_binary_blob(bytes(blob)); g.save(a.out)
    changed = int((np.abs(Wn - Wd).sum(1) > 1e-4).sum())
    res = dict(input=a.inp, output=a.out, joints=nb, verts=len(V), roles=rpath if roles else None, trunk=trunk,
               misbound=mrep, limbs=rep, rip=rrep, changed_verts=changed)
    if a.json: print(json.dumps(res))
    else:
        print(f"roles {rpath if roles else 'auto (skeleton)'}; trunk joints {trunk}")
        if mrep: print(f"  misbound: {mrep}")
        if rrep: print(f"  rip: {rrep}")
        for k, r in rep.items(): print(f"  limb [{k}]: {r}")
        print(f"changed {changed}/{len(V)} vertices → {a.out}")
    bl = sum(r["bleed_verts"] for k, r in rep.items() if isinstance(r, dict) and "bleed_verts" in r)
    fv = sum(r["far_verts"] for k, r in rep.items() if isinstance(r, dict) and "far_verts" in r)
    print(f"M3D_WEIGHTFIX changed={changed} misbound_verts={mrep['verts'] if mrep else 0} bleed_verts={bl} far_verts={fv} rip_copies={rrep['copies'] if rrep else 0} out={a.out}")


if __name__ == "__main__":
    main()

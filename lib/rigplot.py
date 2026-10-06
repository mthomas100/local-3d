"""rigplot.py rigged.glb out.png — joints and bones over the mesh (front and side), plus a text summary of the
hierarchy and the skin weights. The "look at the rig" step for agents (2026-10-02), run with the worker venv python."""
import sys, numpy as np, trimesh
from pygltflib import GLTF2
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

glb, out = sys.argv[1], sys.argv[2]
g = GLTF2().load(glb)
# node world transforms
def tr(n):
    node = g.nodes[n]
    if node.matrix: return np.array(node.matrix, dtype=float).reshape(4, 4).T
    T = np.eye(4); t = node.translation or [0, 0, 0]; q = node.rotation or [0, 0, 0, 1]; s = node.scale or [1, 1, 1]
    x, y, z, w = q
    R = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)], [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
    T[:3, :3] = R * np.array(s); T[:3, 3] = t; return T
parent = {}
for i, n in enumerate(g.nodes):
    for c in (n.children or []): parent[c] = i
def world(n):
    M = tr(n); p = parent.get(n)
    while p is not None: M = tr(p) @ M; p = parent.get(p)
    return M
skin = g.skins[0] if g.skins else None
joints = skin.joints if skin else []
pos = np.array([world(j)[:3, 3] for j in joints]) if joints else np.zeros((0, 3))
names = [g.nodes[j].name or f"j{j}" for j in joints]
jset = set(joints)
bones = [(joints.index(parent[j]), joints.index(j)) for j in joints if parent.get(j) in jset]
# mesh verts and weights
sc = trimesh.load(glb, force="scene"); m = trimesh.util.concatenate(list(sc.geometry.values()))
v = m.vertices; idx = np.random.default_rng(0).choice(len(v), min(len(v), 20000), replace=False)
summary = [f"joints={len(joints)} bones={len(bones)} verts={len(v)} tris={len(m.faces)}"]
# weights
import struct
def acc(ai):
    a = g.accessors[ai]; bv = g.bufferViews[a.bufferView]; buf = g.binary_blob()
    comp = {5121: np.uint8, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}[a.componentType]
    n = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[a.type]
    off = (bv.byteOffset or 0) + (a.byteOffset or 0)
    arr = np.frombuffer(buf, dtype=comp, count=a.count * n, offset=off).reshape(a.count, n)
    return arr
prim = g.meshes[0].primitives[0]
if hasattr(prim.attributes, "JOINTS_0") and prim.attributes.JOINTS_0 is not None:
    J = acc(prim.attributes.JOINTS_0); Wt = acc(prim.attributes.WEIGHTS_0)
    dom = J[np.arange(len(J)), Wt.argmax(1)]
    counts = np.bincount(dom, minlength=len(joints))
    hard = (Wt.max(1) > 0.99).mean()
    summary.append(f"dominant-joint vertex counts: " + ", ".join(f"{names[i]}={c}" for i, c in enumerate(counts) if c))
    summary.append(f"verts with one weight>0.99: {hard:.0%}; mean influences>0.01: {(Wt > 0.01).sum(1).mean():.2f}")
else:
    summary.append("no JOINTS_0 on primitive 0")
print("\n".join(summary))
print("hierarchy:"); 
for j in joints:
    d = 0; p = parent.get(j)
    while p is not None: d += 1; p = parent.get(p)
    print("  " * d + (g.nodes[j].name or f"j{j}"))
fig, axes = plt.subplots(1, 2, figsize=(12, 7))
for ax, (a, b, title) in zip(axes, [(0, 1, "front (x,y)"), (2, 1, "side (z,y)")]):
    ax.scatter(v[idx, a], v[idx, b], s=0.2, c="#bbbbbb")
    for p, c in bones: ax.plot([pos[p, a], pos[c, a]], [pos[p, b], pos[c, b]], "-", c="red", lw=2)
    ax.scatter(pos[:, a], pos[:, b], s=18, c="blue", zorder=3)
    for i, nm in enumerate(names): ax.annotate(nm, (pos[i, a], pos[i, b]), fontsize=6, color="darkblue")
    ax.set_aspect("equal"); ax.set_title(title)
plt.tight_layout(); plt.savefig(out, dpi=110)

"""usdbindfix.py <layer.usd[c|a]> — make every Skeleton's bindTransforms equal its chained
restTransforms, moving the bound meshes' points by the same offset.

glb_root_mover (lib/motion.py, lib/retarget.py) parents the root joint and the skinned
mesh under a non-joint "m3d_mover" node that takes the root's rest TRS, but leaves the
GLB's inverseBindMatrices in the old scene space. glTF tolerates that (skinning is relative
to the mesh node), and usdextract carries it through: bindTransforms = chained rest + D,
restTransforms chain from identity. UsdSkel allows a bind pose that differs from rest,
RealityKit does not honour the difference: the skeleton poses are applied correctly (the
SkeletalPosesComponent matches a Blender export joint for joint) but the mesh deforms as
a straight-arm over-swing (measured 2026-10-03; the plain layout, where bind == chained
rest, renders the Blender pose). This rewrites bindTransforms := chained restTransforms
and subtracts D from the points of every Mesh bound to that Skeleton, which is the layout
Blender's own USD export produces. Only applied when the rotation parts already agree
(D is a pure translation); otherwise the layer is left alone. usdcat in, usdcat out (no pxr
module on this Mac). Run after usdjointfix.py in bin/m3d's glb_to_usdz.
"""
import pathlib
import re
import subprocess
import sys

import numpy as np

src = pathlib.Path(sys.argv[1])
txt = subprocess.run(["usdcat", str(src)], capture_output=True, text=True, check=True).stdout

MAT = re.compile(r"\( \((.*?)\), \((.*?)\), \((.*?)\), \((.*?)\) \)")


def mats(block, name):
    m = re.search(rf"uniform matrix4d\[\] {name} = \[(.*)\]\n", block)
    if m is None:
        return None, None
    return m, [np.array([[float(x) for x in r.split(",")] for r in row]) for row in MAT.findall(m.group(1))]


def fmt(M):
    return "( " + ", ".join("(" + ", ".join(repr(float(v)) for v in row) + ")" for row in M) + " )"


changed = 0
for sk_m in list(re.finditer(r'def Skeleton "([^"]*)"[^{]*\{', txt))[::-1]:   # last first: offsets stay valid
    # the Skeleton's body ends at the first child prim or its closing brace
    body_start = sk_m.end()
    end_rel = re.compile(r"^\s+def |^\s*\}\s*$", re.M).search(txt, body_start)
    body_end = end_rel.start() if end_rel else len(txt)
    sk = txt[body_start:body_end]
    jm = re.search(r"uniform token\[\] joints = \[(.*?)\]", sk)
    rest_m, rest = mats(sk, "restTransforms")
    bind_m, bind = mats(sk, "bindTransforms")
    if not (jm and rest and bind):
        continue
    joints = jm.group(1).replace('"', "").split(", ")
    idx = {j: i for i, j in enumerate(joints)}
    chained = []
    for i, j in enumerate(joints):
        p = j.rsplit("/", 1)[0] if "/" in j else None
        chained.append(rest[i] @ chained[idx[p]] if p in idx else rest[i])   # usda row-vector form
    rot_ok = all(np.allclose(c[:3, :3], b[:3, :3], atol=1e-4) for c, b in zip(chained, bind))
    D = bind[0][3, :3] - chained[0][3, :3]
    if not rot_ok:
        print(f"usdbindfix: {sk_m.group(1)}: bind and rest rotations differ; left as is")
        continue
    if np.linalg.norm(D) < 1e-6 and all(np.allclose(c, b, atol=1e-5) for c, b in zip(chained, bind)):
        print(f"usdbindfix: {sk_m.group(1)}: bind == chained rest; left as is")
        continue
    # Skeleton prim path → which meshes bind to it (rel skel:skeleton = </…/Name>)
    skel_name = sk_m.group(1)
    new_bind = "uniform matrix4d[] bindTransforms = [" + ", ".join(fmt(M) for M in chained) + "]\n"
    sk2 = sk[:bind_m.start()] + new_bind + sk[bind_m.end():]
    txt = txt[:body_start] + sk2 + txt[body_end:]
    n_mesh = 0
    for mm in list(re.finditer(r'def Mesh "[^"]*"', txt))[::-1]:
        mesh_end = txt.find("\ndef ", mm.end()) if False else None
        # mesh body: up to the next prim definition at any depth
        nxt = re.search(r"^\s+def ", txt[mm.end():], flags=re.M)
        mesh_body_end = mm.end() + (nxt.start() if nxt else len(txt) - mm.end())
        body = txt[mm.end():mesh_body_end]
        rel = re.search(r"rel skel:skeleton = <([^>]*)>", body)
        if not rel or rel.group(1).rsplit("/", 1)[-1] != skel_name:
            continue
        pm = re.search(r"(point3f\[\] points = \[)(.*?)(\])", body)
        if not pm:
            continue
        P = np.array([[float(v) for v in p.split(",")] for p in re.findall(r"\(([^)]*)\)", pm.group(2))]) - D
        body2 = body[:pm.start()] + pm.group(1) + ", ".join("(" + ", ".join(f"{v:.7g}" for v in p) + ")" for p in P) + pm.group(3) + body[pm.end():]
        txt = txt[:mm.end()] + body2 + txt[mesh_body_end:]
        n_mesh += 1
    changed += 1
    print(f"usdbindfix: {skel_name}: bind := chained rest, D={np.round(D, 4).tolist()} removed from {n_mesh} mesh(es)")

if changed:
    tmp = src.with_suffix(".bindfix.usda")
    tmp.write_text(txt)
    subprocess.run(["usdcat", "-o", str(src), str(tmp)], check=True)
    tmp.unlink()

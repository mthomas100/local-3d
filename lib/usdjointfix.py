"""usdjointfix.py <layer.usd[c|a]> — make every Skeleton's root joint a single token.

usdextract (Apple USD Tools 0.25.2) names UsdSkel joints by the full glTF node path, so a
skin whose root joint sits under a non-joint node gets joints like "n10/n8", "n10/n8/n7".
UsdSkel accepts that (the parent "n10" is not a joint, so "n10/n8" is a root) but
RealityKit's importer does not build the skeleton at all (`mesh.contents.skeletons` is
empty, no SkeletalPosesComponent, the SkelAnimation never plays; only node-xform tracks
move). Measured 2026-10-03 against Apple's toy_drummer.usdz (root joint "root") and a
Blender export ("root"): stripping the root joint's parent path from every joint token
("n10/n8" → "n8") is the one edit that makes RealityKit create the skeleton and play the
clip. Applied to `joints` on Skeleton and SkelAnimation prims and `skel:joints` on meshes,
through usdcat (no pxr module on this Mac: a regex over the usda text), written back in the
layer's original format. Run after usdretime.py in bin/m3d's glb_to_usdz.
"""
import pathlib
import re
import subprocess
import sys

src = pathlib.Path(sys.argv[1])
txt = subprocess.run(["usdcat", str(src)], capture_output=True, text=True, check=True).stdout

# The prefix comes from the Skeleton prims' own joint lists: a SkelAnimation may animate a
# subset without the root (the mover layout), and skel:joints on a mesh is a subset too.
skel_joints = re.findall(r'def Skeleton "[^"]*"[^{]*\{(?:[^{}]*\n)*?\s+uniform token\[\] joints = \[([^\]]*)\]', txt)
if not skel_joints:
    print(f"usdjointfix: {src.name}: no joints; left as is")
    sys.exit(0)
prefixes = set()
for lst in skel_joints:
    toks = re.findall(r'"([^"]*)"', lst)
    root = min(toks, key=lambda p: p.count("/"))
    if "/" in root:
        prefixes.add(root.rsplit("/", 1)[0] + "/")
if not prefixes:
    print(f"usdjointfix: {src.name}: root joints are single tokens; left as is")
    sys.exit(0)
n = 0
for prefix in sorted(prefixes, key=len, reverse=True):
    def strip(m):
        global n
        toks = m.group(2).split(prefix)
        n += len(toks) - 1
        return m.group(1) + "".join(toks) + "]"
    txt = re.sub(r'(^\s+uniform token\[\] (?:skel:)?joints = \[)([^\]]*)\]', strip, txt, flags=re.M)
tmp = src.with_suffix(".jointfix.usda")
tmp.write_text(txt)
subprocess.run(["usdcat", "-o", str(src), str(tmp)], check=True)
tmp.unlink()
print(f"usdjointfix: {src.name}: stripped {sorted(prefixes)} from {n} joint tokens")

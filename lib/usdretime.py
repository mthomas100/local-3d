"""usdretime.py <layer.usd[c|a]> [fps] — re-time a usdextract layer in place.

usdextract (macOS 27) writes `timeCodesPerSecond = 1` with sample keys in seconds, and
RealityKit evaluates animation at whole time codes: every Apple-route clip played at one
sample per second (a 2 s hop became a 3-point triangle, a 2 s breathing idle sampled to
nothing; measured 2026-10-02, see the viewer animation notes, 2026-10-02, not published "Root motion").
This multiplies every sample key, start/endTimeCode by `fps` and sets timeCodesPerSecond
and framesPerSecond to `fps`, through `usdcat` (no pxr module on this Mac: a regex over
the usda text), then writes the layer back in its original format. Run by bin/m3d's
glb_to_usdz between usdextract and usdzip.
"""
import pathlib
import re
import subprocess
import sys

src = pathlib.Path(sys.argv[1])
fps = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
txt = subprocess.run(["usdcat", str(src)], capture_output=True, text=True, check=True).stdout
if re.search(r"^\s*timeCodesPerSecond = 1(\.0)?\s*$", txt, flags=re.M) is None:
    print(f"usdretime: {src.name} is not at 1 code/s; left as is")
    sys.exit(0)
n = [0]


def key(m):
    n[0] += 1
    return f"{m.group(1)}{float(m.group(2)) * fps:g}:"


txt = re.sub(r"^(\s+)(-?\d+(?:\.\d+)?(?:e-?\d+)?):", key, txt, flags=re.M)        # "    0.0333: [...]" sample keys
txt = re.sub(r"^(\s*)endTimeCode = (\S+)", lambda m: f"{m.group(1)}endTimeCode = {float(m.group(2)) * fps:g}",
             txt, count=1, flags=re.M)
txt = re.sub(r"^(\s*)startTimeCode = (\S+)", lambda m: f"{m.group(1)}startTimeCode = {float(m.group(2)) * fps:g}",
             txt, count=1, flags=re.M)
txt = re.sub(r"^(\s*)timeCodesPerSecond = \S+",
             lambda m: f"{m.group(1)}timeCodesPerSecond = {fps:g}\n{m.group(1)}framesPerSecond = {fps:g}",
             txt, count=1, flags=re.M)
tmp = src.with_suffix(".retimed.usda")
tmp.write_text(txt)
subprocess.run(["usdcat", "-o", str(src), str(tmp)], check=True)
tmp.unlink()
print(f"usdretime: {src.name}: {n[0]} sample keys x{fps:g}")

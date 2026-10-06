#!/usr/bin/env bash
# viewer/burst.sh OUT_DIR clip.usdz [more.usdz …]   env: N=10 DT=0.25 YAW=0 SETTLE=10 FIT=0.25 (model size, m) LIFT=0
# The animation acceptance test that is not fooled: launches the viewer LOOPING the clip(s)
# in the visionOS simulator and takes N screenshots DT seconds apart, then crops every frame
# to the region that changed between the first and the middle frame and writes
# OUT_DIR/burst-sheet.png (+ burst-NN.png). render.sh --anim's seek-and-pause frames render
# the rest pose (2026-10-02/03), so a burst of the live loop is the truth: a clip passes
# when the frames differ in the mesh, and a reviewer judges the motion from the sheet.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=$1; shift
N=${N:-10}; DT=${DT:-0.25}; YAW=${YAW:-0}; SETTLE=${SETTLE:-10}; FIT=${FIT:-0.25}; LIFT=${LIFT:-0}
SIM=${M3D_SIM:-"Apple Vision Pro"}
mkdir -p "$OUT"
rm -f "$HERE"/Models/*.usdz
i=0; for u in "$@"; do [ -f "$u" ] || { echo "burst.sh: no such USDZ: $u" >&2; exit 2; }; i=$((i+1)); cp "$u" "$HERE/Models/$(printf %02d $i)-$(basename "$u")"; done
cd "$HERE" && xcodegen generate --quiet
UDID=$(xcrun simctl list devices available | grep -F "$SIM (" | head -1 | grep -oE '[0-9A-F-]{36}')
xcrun simctl boot "$UDID" 2>/dev/null || true
xcrun simctl bootstatus "$UDID" -b >/dev/null
xcodebuild -quiet -project M3DViewer.xcodeproj -scheme M3DViewer -configuration Debug \
  -destination "id=$UDID" -derivedDataPath .build CODE_SIGNING_ALLOWED=NO build
APP=$(find .build/Build/Products -name "M3DViewer.app" -maxdepth 3 | head -1)
xcrun simctl install "$UDID" "$APP"
xcrun simctl terminate "$UDID" local.m3d.viewer 2>/dev/null || true
SIMCTL_CHILD_M3D_ANIM=1 SIMCTL_CHILD_M3D_YAW=$YAW SIMCTL_CHILD_M3D_FIT=$FIT SIMCTL_CHILD_M3D_LIFT=$LIFT xcrun simctl launch "$UDID" local.m3d.viewer >/dev/null
sleep "$SETTLE"
for k in $(seq 0 $((N-1))); do
  xcrun simctl io "$UDID" screenshot "$OUT/burst-$(printf %02d $k).png" >/dev/null 2>&1
  sleep "$DT"
done
xcrun simctl terminate "$UDID" local.m3d.viewer 2>/dev/null || true
"${M3D_PY:-$HOME/.hy3d/worker-venv/bin/python}" - "$OUT" "$DT" <<'PY'
import sys, glob, numpy as np
from PIL import Image, ImageDraw
out, dt = sys.argv[1], float(sys.argv[2])
fs = sorted(glob.glob(out + "/burst-*.png")); ims = [Image.open(f).convert("RGB") for f in fs]
arrs = [np.asarray(im).astype(int) for im in ims]
# the region that moves: union of every consecutive-frame difference (two grounded hop frames once read as "no motion")
m = np.zeros(arrs[0].shape[:2], bool); changed = 0
for a, b in zip(arrs, arrs[1:]):
    d = np.abs(a - b).sum(2) > 40; m |= d; changed = max(changed, int(d.sum()))
ys, xs = np.nonzero(m); w, h = ims[0].size
if len(xs) > 50:
    x0, x1 = np.percentile(xs, [1, 99]); y0, y1 = np.percentile(ys, [1, 99]); pad = 0.6 * max(x1 - x0, y1 - y0) + 60
    box = (int(max(0, x0 - pad)), int(max(0, y0 - pad)), int(min(w, x1 + pad)), int(min(h, y1 + pad)))
else:
    box = (int(w * 0.35), 0, int(w * 0.65), int(h * 0.45))
tw = 420; th = int(tw * (box[3] - box[1]) / (box[2] - box[0]))
sheet = Image.new("RGB", (tw * len(ims), th + 24), "white"); d = ImageDraw.Draw(sheet)
for i, im in enumerate(ims):
    sheet.paste(im.crop(box).resize((tw, th)), (i * tw, 24)); d.text((i * tw + 6, 4), f"t={i * dt:.2f}s", fill="black")
sheet.save(out + "/burst-sheet.png")
print(f"{out}/burst-sheet.png  max_changed_px_between_frames={changed}  ({'MOTION' if changed > 500 else 'NO MOTION'})")
PY

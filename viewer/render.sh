#!/usr/bin/env bash
# viewer/render.sh OUT_DIR a.usdz b.usdz …   (full paths, or names inside OUT_DIR)
# Builds the viewer with these USDZs, runs it in the visionOS simulator and
# writes front (35°) and back (215°) screenshots to OUT_DIR. M3D_LIFT=0.3 raises the
# models above the room's TV so a light wall is behind them (for glass).
#
# Animation modes (2026-10-02, the acceptance test for `m3d animate`):
#   render.sh --anim OUT_DIR clip.usdz …      front view only. Plays clip M3D_ANIM_INDEX
#       (default 0) of every file and captures paused frames at t = 0, ⅓ and ⅔ of the
#       first clip's duration (or M3D_ANIM_TIMES="0 0.4 0.8"), cropped with one shared
#       box and tiled, time-stamped, into OUT_DIR/anim-sheet.png.
#   render.sh --lib OUT_DIR character.usdz clip.usdz …   the first file is the character;
#       each other file's first clip goes into an AnimationLibraryComponent on it and the
#       clip M3D_ANIM_CLIP (default: the second file's basename) is played. Same frames.
#   Both print the sheet path and the viewer's report.json (per file: clips, durations,
#   skeleton joints, materials, what was played and where it was paused).
#   M3D_SETTLE (default 6) is the wait, in seconds, between the viewer's report.json
#   appearing and the screenshot.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
MODE=default
case ${1:-} in --anim) MODE=anim; shift;; --lib) MODE=lib; shift;; esac
OUT=$1; shift
SIM=${M3D_SIM:-"Apple Vision Pro"}
PY=${M3D_PY:-$HOME/.hy3d/worker-venv/bin/python}
mkdir -p "$OUT"
rm -f "$HERE"/Models/*.usdz
# A bare file name is looked up in OUT_DIR (pi passed one and cp failed, 2026-10-02).
i=0; for u in "$@"; do
  [ -f "$u" ] || { [ -f "$OUT/$u" ] && u="$OUT/$u"; }
  [ -f "$u" ] || { echo "render.sh: no such USDZ: $u" >&2; exit 2; }
  i=$((i+1)); cp "$u" "$HERE/Models/$(printf %02d $i)-$(basename "$u")"
done
cd "$HERE" && xcodegen generate --quiet
UDID=$(xcrun simctl list devices available | grep -F "$SIM (" | head -1 | grep -oE '[0-9A-F-]{36}')
xcrun simctl boot "$UDID" 2>/dev/null || true
xcrun simctl bootstatus "$UDID" -b >/dev/null
xcodebuild -quiet -project M3DViewer.xcodeproj -scheme M3DViewer -configuration Debug \
  -destination "id=$UDID" -derivedDataPath .build CODE_SIGNING_ALLOWED=NO build
APP=$(find .build/Build/Products -name "M3DViewer.app" -maxdepth 3 | head -1)
xcrun simctl install "$UDID" "$APP"
shot() {  # shot NAME YAW HIDE
  xcrun simctl terminate "$UDID" local.m3d.viewer 2>/dev/null || true
  SIMCTL_CHILD_M3D_YAW=$2 SIMCTL_CHILD_M3D_HIDE=$3 SIMCTL_CHILD_M3D_LIFT=${M3D_LIFT:-0} \
    xcrun simctl launch "$UDID" local.m3d.viewer >/dev/null
  sleep 12
  xcrun simctl io "$UDID" screenshot "$OUT/$1.png" >/dev/null
}
closeup() {  # closeup EMPTY.png FULL.png OUT.png
  # Close-up: the models' bounding box from the difference with the empty room, plus a
  # margin, enlarged. An agent judging the full 4K room shot missed pane defects that
  # were obvious in a close-up (live test 2, 2026-10-02).
  "$PY" - "$1" "$2" "$3" <<'PY' || true
import sys, numpy as np
from PIL import Image
a = np.asarray(Image.open(sys.argv[1]).convert("RGB")).astype(int)
b = np.asarray(Image.open(sys.argv[2]).convert("RGB")).astype(int)
m = np.abs(a - b).sum(2) > 40
ys, xs = np.nonzero(m)
if len(xs) < 50:
    sys.exit(0)
x0, x1 = np.percentile(xs, [0.5, 99.5]); y0, y1 = np.percentile(ys, [0.5, 99.5])
pad = 0.08 * max(x1 - x0, y1 - y0)
box = (int(max(0, x0 - pad)), int(max(0, y0 - pad)), int(min(b.shape[1], x1 + pad)), int(min(b.shape[0], y1 + pad)))
im = Image.open(sys.argv[2]).convert("RGB").crop(box)
s = 1600 / max(im.size)
im.resize((int(im.width * s), int(im.height * s))).save(sys.argv[3])
PY
}

if [ "$MODE" = default ]; then
  shot viewer-empty 0 1
  for yaw in 35 215; do
    shot viewer-yaw$yaw $yaw 0
    closeup "$OUT/viewer-empty.png" "$OUT/viewer-yaw$yaw.png" "$OUT/viewer-yaw$yaw-closeup.png"
  done
  xcrun simctl terminate "$UDID" local.m3d.viewer 2>/dev/null || true
  echo "$OUT/viewer-yaw35-closeup.png $OUT/viewer-yaw215-closeup.png (full room shots: viewer-yaw35.png, viewer-yaw215.png)"
  exit 0
fi

# --anim / --lib. The viewer writes Documents/report.json once every file is loaded and
# playing; each launch waits for it, settles, then screenshots.
T0=$(date +%s)
REPORT="$(xcrun simctl get_app_container "$UDID" local.m3d.viewer data)/Documents/report.json"
launch() {  # launch NAME SIMCTL_CHILD_VAR=VALUE …  → OUT/NAME.png, OUT/NAME-report.json
  local name=$1; shift
  xcrun simctl terminate "$UDID" local.m3d.viewer 2>/dev/null || true
  rm -f "$REPORT"
  env SIMCTL_CHILD_M3D_YAW=${M3D_YAW:-35} SIMCTL_CHILD_M3D_LIFT=${M3D_LIFT:-0} "$@" \
    xcrun simctl launch "$UDID" local.m3d.viewer >/dev/null
  for _ in $(seq 1 80); do [ -f "$REPORT" ] && break; sleep 0.5; done
  [ -f "$REPORT" ] || echo "render.sh: $name: no report.json after 40 s" >&2
  sleep "${M3D_SETTLE:-6}"
  xcrun simctl io "$UDID" screenshot "$OUT/$name.png" >/dev/null
  cp "$REPORT" "$OUT/$name-report.json" 2>/dev/null || true
}
if [ "$MODE" = anim ]; then
  MODE_ENV=(SIMCTL_CHILD_M3D_ANIM=1 SIMCTL_CHILD_M3D_ANIM_INDEX=${M3D_ANIM_INDEX:-0})
else
  second=$(ls "$HERE"/Models/02-*.usdz 2>/dev/null | head -1)
  clip=${M3D_ANIM_CLIP:-$(basename "${second#*/02-}" .usdz)}
  MODE_ENV=(SIMCTL_CHILD_M3D_ANIM_LIB=1 SIMCTL_CHILD_M3D_ANIM_CLIP=$clip)
fi
launch viewer-empty SIMCTL_CHILD_M3D_HIDE=1
launch anim-play "${MODE_ENV[@]}"
TIMES=${M3D_ANIM_TIMES:-$("$PY" -c '
import json, sys
r = json.load(open(sys.argv[1]))
d = next((f["playback"]["clipDuration"] for f in r["files"] if f.get("playback")), 0)
print(" ".join(f"{d * k / 3:.3f}" for k in range(3)) if d > 0 else "")' "$OUT/anim-play-report.json")}
[ -n "$TIMES" ] || { echo "render.sh: nothing played; report:" >&2; cat "$OUT/anim-play-report.json" >&2; exit 4; }
FRAMES=()
for t in $TIMES; do
  launch "anim-t$t" "${MODE_ENV[@]}" SIMCTL_CHILD_M3D_ANIM_T=$t
  FRAMES+=("$OUT/anim-t$t.png")
done
xcrun simctl terminate "$UDID" local.m3d.viewer 2>/dev/null || true
# One crop box for every frame (the union of their differences with the empty room), so
# poses can be compared side by side; each tile is stamped with its time.
"$PY" - "$OUT/viewer-empty.png" "$OUT/anim-sheet.png" "${FRAMES[@]}" <<'PY'
import sys, re, numpy as np
from PIL import Image, ImageDraw, ImageFont
empty, sheet, frames = sys.argv[1], sys.argv[2], sys.argv[3:]
a = np.asarray(Image.open(empty).convert("RGB")).astype(int)
mask = np.zeros(a.shape[:2], bool)
for f in frames:
    b = np.asarray(Image.open(f).convert("RGB")).astype(int)
    mask |= np.abs(a - b).sum(2) > 40
ys, xs = np.nonzero(mask)
if len(xs) < 50:
    box = (0, 0, a.shape[1], a.shape[0])
else:
    x0, x1 = np.percentile(xs, [0.5, 99.5]); y0, y1 = np.percentile(ys, [0.5, 99.5])
    pad = 0.08 * max(x1 - x0, y1 - y0)
    box = (int(max(0, x0 - pad)), int(max(0, y0 - pad)), int(min(a.shape[1], x1 + pad)), int(min(a.shape[0], y1 + pad)))
tiles = []
for f in frames:
    im = Image.open(f).convert("RGB").crop(box)
    s = 800 / max(im.size)
    im = im.resize((int(im.width * s), int(im.height * s)))
    im.save(f.replace(".png", "-closeup.png"))
    t = re.search(r"anim-t([0-9.]+)\.png$", f).group(1)
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, 190, 44), fill=(0, 0, 0))
    d.text((8, 6), f"t={t}s", fill=(255, 255, 255), font=ImageFont.load_default(size=30))
    tiles.append(im)
gap = 12
w = sum(t.width for t in tiles) + gap * (len(tiles) - 1); h = max(t.height for t in tiles)
out = Image.new("RGB", (w, h), (40, 40, 40))
x = 0
for t in tiles:
    out.paste(t, (x, 0)); x += t.width + gap
out.save(sheet)
PY
echo "$OUT/anim-sheet.png  (frames: ${FRAMES[*]}; playing: $OUT/anim-play.png; $(( $(date +%s) - T0 )) s after the build)"
for t in $TIMES; do
  "$PY" -c '
import json, sys
r = json.load(open(sys.argv[1]))
p = next((f["playback"] for f in r["files"] if f.get("playback")), {})
print(f"  t={sys.argv[2]}: requested {p.get(\"requestedTime\")} applied {p.get(\"appliedTime\")} of clip {p.get(\"clip\")!r} ({p.get(\"clipDuration\")} s)")' \
    "$OUT/anim-t$t-report.json" "$t" 2>/dev/null || true
done
cat "$OUT/anim-play-report.json"

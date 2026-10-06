"""posecheck.py cutout-rgba.png [...] [--out report.json] [--json] — is a rig-ready concept's pose actually rig-ready?
Written 2026-10-03 after the pi teddy test: the concept drew the arms against the body, Hunyuan fused each arm to
the torso, and every arm raise dragged belly skin. The suspicion that the flaw started in the image (not enough
arm separation in the concept) was right: the four original concepts score 0.00-0.01 here.

From the cutout's alpha (worker venv python: numpy, PIL), per row of the figure: the number of opaque runs wider than
1% of the figure width. In the arm band (40-72% of the height from the top: shoulders to hips on a standing figure)
3+ runs = arm, body, arm stand apart; in the leg band (85-97%) 2+ runs = the legs stand apart.
  arm_gap  share of arm-band rows with 3+ runs     leg_gap  share of leg-band rows with 2+ runs
Measured 2026-10-03 (teddy): arms-at-sides concepts 0.00-0.01; the first --rig-ready prompt 0.25-0.40, and the 0.40
one still meshed with the upper arm fused from the armpit to near the elbow. arm_gap assumes hanging arms (a true
T-pose scores low on it), so the verdict uses notch (below), ok >= NOTCH_OK.
Last line: M3D_POSECHECK n=<k> best=<path> arm_gap=<f> leg_gap=<f>"""
import sys, json, numpy as np
from PIL import Image

ARM_OK = 0.70      # arm_gap (hanging/A-pose arms only); kept for the report
NOTCH_OK = 0.40    # the verdict. Meshed and checked 2026-10-03: notch 0.246 (A-pose) and 0.261 (wide T-pose) fused
                   # under the arm; 0.453 (airplane T-pose) clean at the armpit through a wave and a jump


def score(path):
    a = np.asarray(Image.open(path).convert("RGBA"))[..., 3] > 127
    ys, xs = np.where(a)
    if not len(ys): return dict(image=path, arm_gap=0.0, leg_gap=0.0, note="empty cutout")
    y0, y1 = ys.min(), ys.max(); h = y1 - y0 + 1; w = xs.max() - xs.min() + 1
    def runs(row):
        r = np.diff(np.concatenate([[0], row.astype(int), [0]])); s, e = np.where(r == 1)[0], np.where(r == -1)[0]
        return int(((e - s) > 0.01 * w).sum())
    n = np.array([runs(a[y]) for y in range(y0, y1 + 1)])
    arm = n[int(0.40 * h):int(0.72 * h)]; leg = n[int(0.85 * h):int(0.97 * h)]
    # notch: empty area inside the convex hull of the shoulders-to-hips band (30-75% of the height), per opaque
    # pixel. Pose-independent (hanging, A-pose, T-pose, raised arms): arms held off the body leave big notches under
    # them. 2026-10-03 teddy: arms-at-sides 0.145-0.178, "A-pose" 0.20-0.25, "wide T-pose" 0.24-0.35; arm_gap ranked
    # those T-pose variants differently and missed the two with the most air under the arms (s48, s49: 0.35).
    from PIL import ImageDraw
    from scipy.spatial import ConvexHull
    b0, b1 = y0 + int(0.30 * h), y0 + int(0.75 * h); sub = a[b0:b1]; yy, xx = np.where(sub)
    notch = 0.0
    if len(xx) > 3:
        pts = np.c_[xx, yy]; hv = ConvexHull(pts).vertices
        img = Image.new("L", (sub.shape[1], sub.shape[0]), 0); ImageDraw.Draw(img).polygon([tuple(map(int, pts[i])) for i in hv], fill=1)
        notch = float((np.asarray(img).astype(bool) & ~sub).sum() / max(sub.sum(), 1))
    return dict(image=path, arm_gap=round(float(np.mean(arm >= 3)), 3), leg_gap=round(float(np.mean(leg >= 2)), 3),
                notch=round(notch, 3))


def main():
    args = sys.argv[1:]; outp = None
    if "--out" in args: i = args.index("--out"); outp = args[i + 1]; del args[i:i + 2]
    paths = [p for p in args if not p.startswith("--")]
    res = sorted((score(p) for p in paths), key=lambda r: (-(r["notch"] + 0.1 * r["leg_gap"])))   # notch ranks
    for r in res: r["verdict"] = "ok" if r["notch"] >= NOTCH_OK else "fused"
    best = res[0] if res else None
    out = dict(ranked=res, best=best["image"] if best else None, ok_threshold=NOTCH_OK, ok_metric="notch")
    if outp: json.dump(out, open(outp, "w"), indent=1)
    if "--json" in sys.argv: print(json.dumps(out))
    else:
        for r in res: print(f"  notch {r['notch']:.3f}  arm_gap {r['arm_gap']:.2f}  leg_gap {r['leg_gap']:.2f}  {r['verdict']:5s}  {r['image']}")
    print(f"M3D_POSECHECK n={len(res)} best={out['best']} arm_gap={best['arm_gap'] if best else 0} "
          f"leg_gap={best['leg_gap'] if best else 0}")


if __name__ == "__main__":
    main()

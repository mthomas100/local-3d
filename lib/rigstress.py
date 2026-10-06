"""rigstress.py rigged.glb [--roles R.json] [--out PREFIX] [--json]
Rig-time stress test (worker venv python: numpy, scipy, pygltflib; no Blender). Written 2026-10-03 after the pi
teddy test: the teddy's concept had its arms against its sides, the mesh fused each arm to the torso from armpit to
hip, and nothing said so until a wave dragged a sheet of belly skin. This poses the rig in numpy (linear blend
skinning) the way the procedural clips will, one limb at a time:
  arm   the chain swung up and out to 130 deg from hanging, in the frontal plane (a chibi wave, a jump's arm fling)
  leg   the chain swung forward 60 deg (a jump's knee tuck, a step)
and measures, per pose, lib/animcheck.py's web metric: skin area pulled into stretched sheets (web_area, share of
the rest surface) and the largest connected sheet (web_sheet) with the bones carrying its weight. Same thresholds
as animcheck (ok < 0.003 / 0.002, fail >= 0.010 / 0.005). A fused limb fails here, before any clip is made.
Last line: M3D_RIGSTRESS poses=<n> web_area_max=<f> web_sheet_max=<f> verdict=<ok|warn|fail>"""
import sys, os, json, argparse, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from animcheck import Clip, skinned_mesh, lbs, tri_area, web, level, TH, find_roles
from weightfix import topology, limbs_and_trunk, subtree

RELAX_TO, RELAX_OVER = 30.0, 50.0      # keep in step with lib/motion.py relax_arms()


def rodrigues(axis, deg):
    a = np.asarray(axis, float); a = a / max(np.linalg.norm(a), 1e-12); t = np.radians(deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("glb"); ap.add_argument("--roles"); ap.add_argument("--out"); ap.add_argument("--json", action="store_true")
    ap.add_argument("--arm-deg", type=float, default=130.0); ap.add_argument("--leg-deg", type=float, default=60.0)
    a = ap.parse_args()
    out = a.out or os.path.splitext(a.glb)[0] + "-rigstress"
    c = Clip(a.glb); g = c.g
    V, J, W, F, joints, IB = skinned_mesh(c)
    nb = len(joints); names = [g.nodes[j].name or str(j) for j in joints]
    Wd = np.zeros((len(V), nb)); np.add.at(Wd, (np.arange(len(V))[:, None].repeat(J.shape[1], 1), J), W)
    dom = Wd.argmax(1)
    Wrest = c.world()[joints]
    R = lbs(V, J, W, Wrest @ IB); a0 = tri_area(R, F)
    parent, children = topology(g, g.skins[0])
    roles = find_roles(a.glb, a.glb, a.roles, g) or {}
    limbs, trunk = limbs_and_trunk(parent, children, np.bincount(dom, minlength=nb), names, roles)
    jp = Wrest[:, :3, 3]
    centre_x = float((R[:, 0].min() + R[:, 0].max()) / 2)
    DOWN = np.array([0.0, -1.0, 0.0]); FRONT = np.array([0.0, 0.0, 1.0])        # glTF: Y up, the character faces +Z
    poses = []
    for L in limbs:
        # pivot where the clips pivot: the first arm./leg. bone (the shoulder), not a clavicle that auto_limbs puts
        # at the head of the chain (it sits by the sternum; swinging from there tears the chest, which no clip does)
        k0 = next((i for i, j in enumerate(L) if roles.get(names[j], "").startswith(("arm.", "leg."))), 0)
        root = L[k0]; tip = L[-1]
        # the chain may start at a clavicle tagged "body"; name the limb by its first arm./leg. bone
        role = next((roles.get(names[j], "") for j in L if roles.get(names[j], "").startswith(("arm.", "leg."))), roles.get(names[root], ""))
        kind = "leg" if role.startswith("leg.") or (not role and jp[tip, 1] < jp[root, 1] and abs(jp[root, 0] - centre_x) < 0.25 * np.ptp(R[:, 0]) and jp[root, 1] < np.median(R[:, 1])) else "arm"
        d = jp[tip] - jp[root]
        if np.linalg.norm(d) < 1e-9: continue
        d = d / np.linalg.norm(d)
        if kind == "arm":
            out_dir = np.array([1.0 if jp[root, 0] > centre_x else -1.0, 0.0, 0.0])
            d_fr = np.array([d[0], d[1], 0.0]); rest = np.degrees(np.arccos(np.clip(np.dot(d_fr / max(np.linalg.norm(d_fr), 1e-9), DOWN), -1, 1)))
            axis = np.cross(DOWN, out_dir)
            swings = [("raise", a.arm_deg - rest)]                                  # swing in the frontal plane
            # an arm resting far out (a T-pose rig) is lowered to RELAX_TO in every clip (motion.py relax_arms):
            # test that too, the clips' idle pose is a 60-70 degree rotation from such a rest (2026-10-03)
            if rest > RELAX_OVER: swings.append(("lower", RELAX_TO - rest))
        else:
            axis = np.cross(DOWN, FRONT); swings = [("forward", a.leg_deg)]
        for how, deg in swings:
            if how == "raise" and deg <= 0: continue
            Rm = rodrigues(axis, deg); p = jp[root]
            M = np.eye(4); M[:3, :3] = Rm; M[:3, 3] = p - Rm @ p
            Wp = Wrest.copy()
            for j in subtree(children, root): Wp[j] = M @ Wrest[j]
            P = lbs(V, J, W, Wp @ IB)
            wa, ws, wb = web(R, P, F, a0, Wd, names)
            poses.append(dict(limb=[names[j] for j in L[k0:]], role=role or None, kind=kind, pose=how, swing_deg=round(float(deg), 1),
                              web_area=round(wa, 4), web_sheet=round(ws, 4), sheet_bones=wb,
                              verdict=max((level(wa, *TH["web_area"]), level(ws, *TH["web_sheet"])), key=["ok", "warn", "fail"].index)))
    order = ["ok", "warn", "fail"]
    verdict = max((p["verdict"] for p in poses), key=order.index, default="ok")
    res = dict(input=a.glb, limbs=len(limbs), poses=poses, verdict=verdict,
               web_area_max=max((p["web_area"] for p in poses), default=0.0),
               web_sheet_max=max((p["web_sheet"] for p in poses), default=0.0), thresholds={k: TH[k] for k in ("web_area", "web_sheet")})
    if verdict != "ok":
        bad = [p for p in poses if p["verdict"] != "ok"]
        res["advice"] = ("skin is shared by parts that must move apart (" + ", ".join(f"{p['kind']} {p['role'] or p['limb'][0]}: "
                         f"web {p['web_area']}" for p in bad) + "). The mesh has the limb fused to the body: re-make the "
                         "concept with `m3d make … --rig-ready` (A-pose, gaps under the arms), or rig with --engine puppet "
                         "and a spec with rip + ball caps. Clips that move these limbs will drag skin.")
    json.dump(res, open(out + ".json", "w"), indent=1)
    if a.json: print(json.dumps(res))
    else:
        for p in poses:
            print(f"  {p['kind']:3s} {str(p['role'] or p['limb'][0]):6s} {p['pose']:7s} {p['swing_deg']:6.1f} deg: web {p['web_area']:.4f}, "
                  f"sheet {p['web_sheet']:.4f} {p['sheet_bones'] if p['web_sheet'] else ''} -> {p['verdict']}")
        if res.get("advice"): print("  " + res["advice"])
    print(f"M3D_RIGSTRESS poses={len(poses)} web_area_max={res['web_area_max']:.4f} web_sheet_max={res['web_sheet_max']:.4f} verdict={verdict}")


if __name__ == "__main__":
    main()

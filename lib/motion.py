# motion.py — headless Blender: a rigged puppet (or any rig whose bones carry roles) -> procedural
# animation clips keyed on bone ROLES -> one GLB per clip plus one GLB with every clip. Written 2026-10-02
# for the m3d animate stage (task T3 of the rig/animate plan); Blender 5.2.2 LTS (slotted actions).
# Usage:
#   blender -b --factory-startup --python motion.py -- RIGGED.blend|RIGGED.glb OUTDIR \
#       [--clips idle,hop,look,wave,spin,wobble,sleep] [--fps 30] [--seconds 2] [--check] [--no-usdz-root-motion]
# Outputs: OUTDIR/<stem>.<clip>.glb (exactly one glTF animation, named <clip>; the root joint's animation sits on a
# non-joint parent node "m3d_mover" so the Apple USDZ route keeps root motion, see glb_root_mover), OUTDIR/<stem>.all.glb
# (plain layout: root motion on the root joint, for Unity/Godot/Blender), and with --check OUTDIR/check-<clip>.png
# (Workbench frames at t = 0, 0.2, 0.45, 0.7).
# m3d treats the stage as successful only when the last line is  M3D_ANIM clips=<n> out=<dir>
#
# Roles (team contract, 2026-10-02), resolved per bone in this order:
#   1. the sidecar JSON next to the input, <stem>.roles.json = {"<bone name>": "<role>", ...} or
#      {"roles": {...}} (or --roles PATH); lib/roles.py writes these for SkinTokens rigs (bones bone_0..N);
#   2. the bone's custom property "m3d_role" (puppet.py sets it; a glTF import restores it from the joint's
#      extras {"m3d_role": ...});
#   3. the bone's own name: "arm.R.02" -> arm.R, "head" -> head; Mixamo-style "LeftArm"/"Hips"/"Spine" too.
# Roles: root body head arm.L arm.R leg.L leg.R lid tail antenna wheel spout other. Bones sharing a role form
# a chain, ordered root-to-tip by hierarchy depth. Swing axes are computed from each bone's rest orientation
# (bone.matrix_local), so imported rigs with arbitrary bone axes (SkinTokens bone_0..N) work the same way.
# Presets (every clip starts and ends in the same pose, so it loops; a role the rig lacks is skipped with a
# note, never an error; durations are --seconds except sleep = 2x):
# Squash/stretch is baked as non-uniform scale on the root bone (about the bbox bottom centre; children inherit),
# because Apple's usdextract drops glTF blend-weight animation; --shape-keys additionally drives the puppet's
# squash/stretch morph targets for GLB-only consumers.
#   idle    breathe with squash/stretch (+-3%), a 2-degree head bob, 3-degree arm sway, a soft
#           sway on antenna/tail chains, the lid rises 1% with the in-breath
#   hop     anticipation squash, root rises on a parabola (35% of the height) with stretch in the air, landing
#           squash, settle; head, arms, lid, tail and antenna follow through on the vertical velocity
#   look    head yaw +-35 degrees eased at the extremes, the body leans and turns a little with it
#   wave    friendly wave: arm.R out to the side and a little forward, forearm vertical with the hand at head
#           height, rocking at the elbow 2.5 Hz, hand lagging; distance-clamped away from the head; head tilts
#           towards it; arm.L counter-sways
#   spin    wheel bones turn once about their own axis; with no wheels the root yaws a full turn instead
#   wobble  the root rolls and pitches like a settling top (3 cycles, damped, zero at both ends); the head lags
#   sleep   slow forward lean of body and head and drooping arms over the clip, breathing at half speed and
#           half amplitude
import bpy, sys, os, re, math, json, struct, argparse
import numpy as np
from mathutils import Vector, Quaternion, Matrix

CLIPS = ("idle", "hop", "look", "wave", "spin", "wobble", "sleep")
ROLES = ("root", "body", "head", "arm.L", "arm.R", "leg.L", "leg.R", "lid", "tail", "antenna", "wheel", "spout", "other")

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("inp"); ap.add_argument("outdir")
ap.add_argument("--clips", default=",".join(CLIPS))
ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--seconds", type=float, default=2.0)
ap.add_argument("--check", action="store_true")
ap.add_argument("--size", type=int, default=400)         # pixel size of each check frame
ap.add_argument("--yaw", type=float, default=35.0)       # check camera yaw in degrees (35 = front-left 3/4, 215 = from behind)
ap.add_argument("--roles", default=None)                 # sidecar JSON {bone: role}; default <input stem>.roles.json if present
ap.add_argument("--shape-keys", action="store_true")     # ALSO drive the squash/stretch shape keys (GLB-only consumers)
ap.add_argument("--presets", default=None)               # JSON {"presets": {"wave": {...}}} overriding the preset tables
# per-clip GLBs carry the root joint's animation on a non-joint parent node so RealityKit plays the root motion
# (see glb_root_mover); --no-usdz-root-motion keeps the plain layout. The .all.glb is always plain.
ap.add_argument("--usdz-root-motion", action=argparse.BooleanOptionalAction, default=True)
a = ap.parse_args(argv)
clips = [c.strip() for c in a.clips.split(",") if c.strip()]
bad = [c for c in clips if c not in CLIPS]
if bad: raise SystemExit(f"[motion] unknown clips {bad}; known: {CLIPS}")
os.makedirs(a.outdir, exist_ok=True)
stem = os.path.splitext(os.path.basename(a.inp))[0]
log = lambda *s: print("[motion]", *s, flush=True)

# ---------- 0. load the rig ----------
if a.inp.lower().endswith(".blend"):
    bpy.ops.wm.open_mainfile(filepath=a.inp)
else:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=a.inp)
sc = bpy.context.scene
sc.render.fps = a.fps; sc.render.fps_base = 1.0
arm = next((o for o in sc.objects if o.type == "ARMATURE"), None)
if arm is None: raise SystemExit("[motion] no armature in input")
meshes = [o for o in sc.objects if o.type == "MESH" and (any(m.type == "ARMATURE" and m.object == arm for m in o.modifiers) or o.parent == arm)]
mesh = meshes[0] if meshes else next((o for o in sc.objects if o.type == "MESH"), None)
key = mesh.data.shape_keys if mesh else None
keys_available = set(k.name for k in key.key_blocks) if key else set()
for o in sc.objects:                                      # a clean slate: no leftover actions or NLA
    if o.animation_data: o.animation_data_clear()
if key and key.animation_data: key.animation_data_clear()
for pb in arm.pose.bones:
    pb.rotation_mode = "QUATERNION"

# ---------- 1. roles ----------
MIXAMO = {"hips": "root", "pelvis": "root", "spine": "body", "spine1": "body", "spine2": "body", "chest": "body",
          "neck": "head", "head": "head", "leftarm": "arm.L", "leftforearm": "arm.L", "lefthand": "arm.L",
          "rightarm": "arm.R", "rightforearm": "arm.R", "righthand": "arm.R", "leftupleg": "leg.L", "leftleg": "leg.L",
          "leftfoot": "leg.L", "rightupleg": "leg.R", "rightleg": "leg.R", "rightfoot": "leg.R"}
sidecar_path = a.roles or (os.path.splitext(a.inp)[0] + ".roles.json")
sidecar = {}; preset_overrides = {}
if os.path.exists(sidecar_path):
    with open(sidecar_path) as f: sidecar = json.load(f)
    if isinstance(sidecar.get("roles"), dict):                                   # {"roles": {bone: role}} wrapper form
        preset_overrides = sidecar.get("presets") or {}                          # puppet.py copies the spec's "presets"
        sidecar = sidecar["roles"]
    bad_roles = {k: v for k, v in sidecar.items() if v not in ROLES}
    if bad_roles: raise SystemExit(f"[motion] {sidecar_path}: unknown roles {bad_roles}; roles: {ROLES}")
    missing = [k for k in sidecar if k not in arm.pose.bones]
    log(f"roles sidecar {sidecar_path}: {len(sidecar)} bones" + (f", {len(missing)} not in the rig: {missing[:5]}" if missing else ""))
elif a.roles:
    raise SystemExit(f"[motion] --roles {a.roles}: not found")
if a.presets:
    with open(a.presets) as f: d = json.load(f)
    preset_overrides = d.get("presets", d)
# per-character tuning of the presets: the spec's "presets": {"wave": {...}} travels in the roles sidecar
WAVE = {"abduct": 78.0,      # upper arm out to the side from hanging, degrees (elbow about shoulder height)
        "forward": 20.0,     # upper arm forward, towards the viewer
        "elbow": 90.0,       # forearm bent up to vertical, hand at head / bulb height
        "twist": 0.0,        # forearm twist about its own axis, degrees (palm forward; depends on the mesh)
        "rock": 20.0,        # forearm rocking side to side at the elbow, frontal plane, degrees
        "hz": 2.5,           # rocking rate (rounded to whole cycles per clip so it loops)
        "hand_lag": 10.0,    # a third bone (hand) lags the rock by this much, degrees
        "ramp": 0.3,         # seconds to raise the arm at the start and settle at the end
        "head_tilt": 6.0,    # head tilts towards the waving arm
        "clearance": 0.02,   # metres: posed forearm/hand must stay this far from every head-role vertex
        "forward_max": 60.0} # the clamp first adds forward swing up to this, then reduces abduction
WAVE["style"] = "auto"     # auto: "chibi" when the head is >= 0.3 of the height (big head, short arms), else "human"
# the chibi wave (2026-10-03, the teddy: head 0.5 of the height, arms shorter than the head is tall). The human wave
# put its hand in front of the bow tie: elbow at shoulder height leaves a short-armed character's hand at its chin.
# A chibi says hi with the whole arm up beside the head, waving from the shoulder with a floppy wrist, the body
# swaying and the knees bouncing.
# 2026-10-03 (T-pose teddy, rendered): abduct 140 with the elbow bent 20 deg inward put the hand ON the side of the
# head (a salute, not a wave). Now the arm is up and out (128), the elbow opens slightly outward so the hand stands
# clear of the head, the shoulder rocks wider and the hand flops more; the clamp keeps 6 cm from the head.
WAVE_CHIBI = {"abduct": 128.0, "forward": 10.0, "elbow": -12.0, "rock": 0.0, "shoulder_rock": 15.0, "hand_lag": 34.0,
              "hz": 2.0, "ramp": 0.28, "head_tilt": 10.0, "body_sway": 4.0, "bounce": 0.012, "other_arm": 14.0,
              "clearance": 0.06, "forward_max": 20.0}
WAVE.update(preset_overrides.get("wave", {}))
if preset_overrides: log(f"preset overrides: {preset_overrides}")
def role_of(pb):
    r = sidecar.get(pb.name)
    if r in ROLES: return r
    r = pb.get("m3d_role") or pb.bone.get("m3d_role")
    if r in ROLES: return r
    n = re.sub(r"\.\d+$", "", pb.name)                    # chain suffix
    if n in ROLES: return n
    m = re.match(r"^(.*)[._-]([LR])$", n)                  # name.L / name_R
    base, side = (m.group(1), m.group(2)) if m else (n, None)
    b = re.sub(r"^mixamorig\d*:", "", base).lower().replace("_", "").replace(".", "")
    if b in MIXAMO: return MIXAMO[b]
    if side and b in ("arm", "upperarm", "forearm", "hand", "shoulder"): return f"arm.{side}"
    if side and b in ("leg", "upleg", "thigh", "shin", "foot"): return f"leg.{side}"
    return "other"
def depth(pb):
    d = 0
    while pb.parent: pb = pb.parent; d += 1
    return d
by_role = {}
for pb in arm.pose.bones:
    by_role.setdefault(role_of(pb), []).append(pb)
for r in by_role: by_role[r].sort(key=lambda pb: (depth(pb), pb.name))
log("roles:", {r: [pb.name for pb in v] for r, v in by_role.items()})
noted = set()
def chain(role):
    """bones of a role root-to-tip, [] with a one-time note when missing"""
    if role not in by_role:
        if role not in noted: noted.add(role); log(f"note: rig has no '{role}' bone; its motion is skipped")
        return []
    return by_role[role]
def first(role):
    c = chain(role); return c[0] if c else None

# ---------- 2. pose helpers (armature space: +Z up, -Y front (glTF +Z), +X the character's left) ----------
UP, FRONT, RIGHT = Vector((0, 0, 1)), Vector((0, -1, 0)), Vector((1, 0, 0))
H = float(max(mesh.dimensions.z, 1e-3)) if mesh else 1.0   # character height, metres
centre_x = float((Vector(mesh.bound_box[0]) + Vector(mesh.bound_box[6])).x / 2) if mesh else 0.0
pose = {}                                                # bone name -> [world quaternion, world offset]
twist = {}                                               # bone name -> degrees about the bone's own axis
shape = {}                                               # shape key name -> value
def reset_pose():
    pose.clear(); shape.clear(); twist.clear()
    for pb in arm.pose.bones:
        pb.rotation_quaternion = (1, 0, 0, 0); pb.location = (0, 0, 0); pb.scale = (1, 1, 1)
    if key:
        for k in key.key_blocks: k.value = 0.0
def rot(pb, axis, deg):
    """rotate a bone about an armature-space axis (composed with earlier rotations this frame)"""
    if pb is None or abs(deg) < 1e-9: return
    q, off = pose.setdefault(pb.name, [Quaternion((1, 0, 0, 0)), Vector((0, 0, 0))])
    pose[pb.name][0] = Quaternion(axis, math.radians(deg)) @ q
def move(pb, vec):
    if pb is None: return
    pose.setdefault(pb.name, [Quaternion((1, 0, 0, 0)), Vector((0, 0, 0))])[1] += Vector(vec)
def key_value(name, v):
    """squash / stretch amount 0..1 for this frame (1.0 = the puppet shape key's full deformation)"""
    shape[name] = shape.get(name, 0.0) + v
# Squash/stretch is baked as non-uniform SCALE on the root bone (else the body): Apple's usdextract drops glTF
# blend-weight animation (viewer probe, the viewer animation notes, 2026-10-02, not published), so shape keys never reach RealityKit.
# A pose-bone scale is about the bone head, which puppet.py puts at the bbox bottom centre, so the feet stay
# planted and every child bone inherits the squash. Same factors as the shape keys: squash Y 0.85 / XZ 1.08,
# stretch Y 1.12 / XZ 0.95, interpolated by the amount.
scale_bone = first("root") or first("body")
def world_scale_to_local(pb, s_world):
    """non-uniform world scale (x, y, z in armature space) expressed on the bone's local axes: each local axis
    takes the scale of the world axis it mostly aligns with (exact for axis-aligned puppet bones)"""
    M = pb.bone.matrix_local.to_3x3()
    out = []
    for i in range(3):
        col = Vector((M[0][i], M[1][i], M[2][i]))
        j = max(range(3), key=lambda k: abs(col[k]))
        out.append(s_world[j])
    return Vector(out)
# A SkinTokens / Mixamo root sits at the HIPS, not at the floor (puppet.py's root is at the bbox bottom). Its scale
# then squashes about the hips: the feet float up on a squash and sink on a stretch (2026-10-03 teddy hop: the body
# stretched and the jump looked stiff). apply_pose moves the root by hip_h * (up - 1) so the feet stay planted.
floor_z = float(min((mesh.matrix_world @ Vector(c)).z for c in mesh.bound_box)) if mesh else 0.0
hip_h = float(scale_bone.bone.head_local.z - floor_z) if scale_bone is not None else 0.0
def apply_pose():
    sq, st = max(0.0, shape.get("squash", 0.0)), max(0.0, shape.get("stretch", 0.0))
    up_s = (1 - 0.15 * sq) * (1 + 0.12 * st)
    if scale_bone is not None and hip_h > 0.05 * H and abs(up_s - 1) > 1e-6:
        move(scale_bone, UP * hip_h * (up_s - 1))
    for name, (q, off) in pose.items():
        pb = arm.pose.bones[name]
        M = pb.bone.matrix_local.to_3x3()
        pb.rotation_quaternion = (M.inverted() @ q.to_matrix() @ M).to_quaternion()
        if name in twist: pb.rotation_quaternion = pb.rotation_quaternion @ Quaternion((0, 1, 0), math.radians(twist[name]))
        pb.location = M.inverted() @ off
    sq, st = max(0.0, shape.get("squash", 0.0)), max(0.0, shape.get("stretch", 0.0))
    if scale_bone is not None:
        up = (1 - 0.15 * sq) * (1 + 0.12 * st); side = (1 + 0.08 * sq) * (1 - 0.05 * st)
        scale_bone.scale = world_scale_to_local(scale_bone, (side, side, up))   # armature space: Z up
    if key and a.shape_keys:
        for kn in ("squash", "stretch"):
            if kn in keys_available: key.key_blocks[kn].value = shape.get(kn, 0.0)
# Imported glTF bones get a made-up tail: Blender's importer points every SkinTokens bone straight up (seen
# 2026-10-02). TAILS_BOGUS is true when every bone of the rig points exactly +Z; then no tail is trusted, not even
# a leaf's (2026-10-03: limb_dir took the teddy's hand tail, read its hanging arm as 25 degrees below horizontal,
# and every wave swung the arm up-and-INWARD: across the bow tie, or into the body).
TAILS_BOGUS = all(b.length < 1e-6 or (b.tail_local - b.head_local).normalized().z > 0.9999 for b in arm.data.bones)
def limb_dir(pb):
    """rest direction of the limb that starts at this bone, from the hierarchy: the chain's tip joint for a role
    chain, else the children's heads, and the bone's own tail only for a leaf of a rig whose tails are real."""
    role = role_of(pb); ch = by_role.get(role, [])
    if len(ch) > 1 and pb is ch[0]:
        tip = ch[-1].bone
        real_tail = tip.length > 1e-6 and not tip.children and not TAILS_BOGUS
        v = (tip.tail_local if real_tail else tip.head_local) - pb.bone.head_local
        if v.length > 1e-6: return v.normalized()
    kids = [c for c in pb.bone.children]
    if kids:
        v = sum((c.head_local for c in kids), Vector()) / len(kids) - pb.bone.head_local
        if v.length > 1e-6: return v.normalized()
    return (pb.bone.tail_local - pb.bone.head_local).normalized()
def raise_axis(pb):
    """axis that swings a limb towards up-and-away-from-the-body (a hanging arm lifts sideways, a stub that
    already points out lifts up)"""
    d = limb_dir(pb)
    out = Vector((-1, 0, 0)) if pb.bone.head_local.x <= centre_x else Vector((1, 0, 0))
    target = (out * 0.6 + UP * 0.8).normalized()
    ax = d.cross(target)
    if ax.length < 1e-3: ax = d.cross(UP) if d.cross(UP).length > 1e-3 else FRONT
    return ax.normalized(), math.degrees(d.angle(target))
# Rigs made from a --rig-ready (T-pose) concept rest with their arms far out. Every clip starts by lowering such an
# arm to RELAX_TO degrees from hanging in the frontal plane, so the character idles, hops and looks with its arms down
# (2026-10-03). The wave measures its raise from the relaxed arm. Arms resting at <= RELAX_OVER are left alone.
RELAX_TO, RELAX_OVER = 30.0, 50.0
def frontal_axis(pb):
    out = Vector((-1, 0, 0)) if pb.bone.head_local.x <= centre_x else Vector((1, 0, 0))
    return Vector((0, 0, -1)).cross(out).normalized()      # +deg about it swings a hanging arm out and up
def rest_out_deg(pb):
    d = limb_dir(pb); d_fr = Vector((d.x, 0, d.z))
    return math.degrees(d_fr.angle(Vector((0, 0, -1)))) if d_fr.length > 1e-6 else 0.0
_relax = {}
def relax_arms():
    if not _relax:
        for side in ("arm.L", "arm.R"):
            ac = by_role.get(side, [])
            if ac:
                ro = rest_out_deg(ac[0])
                _relax[side] = (frontal_axis(ac[0]), -(ro - RELAX_TO) if ro > RELAX_OVER else 0.0, ro)
        _relax["_logged"] = True
        for k, v in _relax.items():
            if k != "_logged" and v[1]: log(f"relax: {k} rests {v[2]:.0f} deg out; clips lower it to {RELAX_TO:.0f}")
    for side, v in _relax.items():
        if side != "_logged" and v[1]: rot(by_role[side][0], v[0], v[1])
def relaxed_out(pb):
    """the arm's angle from hanging once relax_arms() has run"""
    ro = rest_out_deg(pb); return RELAX_TO if ro > RELAX_OVER else ro
def smoothstep(e0, e1, x):
    t = min(1.0, max(0.0, (x - e0) / (e1 - e0) if e1 > e0 else 1.0)); return t * t * (3 - 2 * t)
def window(t, a0, b0): return smoothstep(0, a0, t) * (1 - smoothstep(b0, 1, t))
def bump(t): return (1 - math.cos(2 * math.pi * t)) / 2
def sway_chains(t, cycles, amp, phase_lag=0.9):
    """soft secondary sway on antenna and tail chains, lagging along the chain"""
    for role in ("antenna", "tail"):
        for k, pb in enumerate(chain(role)):
            ph = 2 * math.pi * cycles * t - phase_lag * k
            rot(pb, RIGHT, amp * math.sin(ph)); rot(pb, FRONT, 0.7 * amp * math.cos(ph))

# ---------- 3. the presets: pose(t) with t in [0, 1], pose(0) == pose(1) ----------
def clip_idle(t, secs):
    c = max(1, round(secs / 2))                            # one breath per 2 s
    s = math.sin(2 * math.pi * c * t)
    key_value("stretch", 0.25 * max(0.0, s)); key_value("squash", 0.20 * max(0.0, -s))   # +-3%
    rot(first("head"), RIGHT, 2.0 * s)
    for role in ("arm.L", "arm.R"):
        rot(first(role), RIGHT, 3.0 * math.sin(2 * math.pi * c * t - 0.5))
    sway_chains(t, c, 5.0)
    move(first("lid"), UP * 0.01 * H * max(0.0, s))

def _jpos(pb): return pb.bone.head_local
def leg_bend(legc, drop, tuck=0.0, point=0.0):
    """bend a leg chain [thigh, shin, foot, (toe)] so the ankle stays where it was while the hip drops by `drop`
    metres (two-bone IK by the law of cosines, knee forward); `tuck` (0..1) folds the leg up in the air, `point`
    (-1..1) pitches the foot (toes down > 0). Rotations about RIGHT: +deg swings a hanging bone backwards."""
    if len(legc) < 2: return
    hip, knee = _jpos(legc[0]), _jpos(legc[1])
    ank = _jpos(legc[2]) if len(legc) > 2 else knee + (knee - hip)
    a, b = (knee - hip).length, (ank - knee).length; D0 = (ank - hip).length
    def angles(D):
        D = max(abs(a - b) + 1e-4, min(a + b - 1e-4, D))
        th = math.acos(max(-1, min(1, (a * a + D * D - b * b) / (2 * a * D))))
        ta = math.acos(max(-1, min(1, (b * b + D * D - a * a) / (2 * b * D))))
        return th, ta
    h0, a0 = angles(D0); h1, a1 = angles(D0 - max(0.0, drop))
    dh, da = math.degrees(h1 - h0), math.degrees(a1 - a0)
    rot(legc[0], RIGHT, -dh - 38.0 * tuck)
    rot(legc[1], RIGHT, dh + da + 70.0 * tuck)
    if len(legc) > 2: rot(legc[2], RIGHT, -da - 28.0 * tuck + 22.0 * point)
def legs_bend(drop, tuck=0.0, point=0.0):
    for side in ("leg.L", "leg.R"): leg_bend(chain(side), drop, tuck, point)
def is_biped_hips():
    """legs with a knee (>= 2 bones each) under a root that sits at the hips: knees can do the crouching"""
    return len(by_role.get("leg.L", [])) >= 2 and len(by_role.get("leg.R", [])) >= 2 and hip_h > 0.2 * H
def head_share():
    """share of the height from the neck (first head bone) up: ~0.15 for a human, ~0.5 for a chibi teddy"""
    hb = first("head")
    if hb is None or not mesh: return 0.0
    top = max((mesh.matrix_world @ Vector(c)).z for c in mesh.bound_box)
    return float((top - hb.bone.head_local.z) / H)
def ring(x, k=7.0, w=3.0):
    """damped overshoot after an impact at x = 0 (x in seconds): 0 at x <= 0, peaks near 1 then rings down"""
    return 0.0 if x <= 0 else math.exp(-k * x) * math.sin(2 * math.pi * w * x) / 0.62
def seg(t, t0, t1):
    """smoothstep from t0 to t1"""
    return smoothstep(t0, t1, t)

HOP_T0, HOP_T1, HOP_H = 0.24, 0.66, 0.35
def hop_height(t):
    if t <= HOP_T0 or t >= HOP_T1: return 0.0
    u = (t - HOP_T0) / (HOP_T1 - HOP_T0); return HOP_H * H * 4 * u * (1 - u)
def hop_vnorm(t):
    """vertical velocity normalised to [-1, 1] (0 outside the flight)"""
    if t < 0: return 0.0
    dt = 1e-3; v = (hop_height(t + dt) - hop_height(t - dt)) / (2 * dt)
    vmax = HOP_H * H * 4 / (HOP_T1 - HOP_T0)
    return max(-1.0, min(1.0, v / vmax))
HOP = {"style": "auto",   # auto: "biped" when the rig has knees and its root sits at the hips, else "object"
       "crouch": 0.10,      # anticipation: hips drop by this share of the height, knees bend, feet stay planted
       "crouch_leg": 0.16,  # ... but never more than this share of the leg (hip to ankle): a short-legged teddy's
                            # 10%-of-height crouch was half its leg, and the deep knee bend stretched the boots (2026-10-03)
       "absorb_leg": 0.13,  # the same cap for the landing absorb
       "height": 0.30,      # apex of the flight, share of the height
       "absorb": 0.08,      # landing: hips drop by this share as the knees take the impact
       "lean": 14.0,        # body leans forward in the crouch and the landing, degrees
       "arms_back": 45.0,   # arms swing back in the crouch
       "arms_up": 120.0,    # arms fling forward-and-up at take-off ("yay!"), degrees from hanging
       "arms_out": 30.0,    # and out to the sides
       "tuck": 0.6,         # knees fold up at the apex (0..1)
       "squash": 0.5, "stretch": 0.9,   # root squash/stretch amounts (feet kept on the floor, see apply_pose)
       "t_crouch": (0.08, 0.30), "t_takeoff": 0.36, "t_land": 0.66}
HOP.update(preset_overrides.get("hop", {}))
def clip_jump_biped(t, secs):
    """a cartoon jump with weight: anticipation (knees bend, body leans in, arms swing back, head looks up), take-off
    (legs snap straight, stretch, arms fling up and out), flight (knees tuck at the apex, head and arms lag), landing
    (squash, knees absorb, arms and head follow through) and a damped overshoot back to rest. Every channel is a joint
    rotation or root motion/scale, which is all the Apple USDZ route keeps (glb_root_mover)."""
    c0, c1 = HOP["t_crouch"]; T0, T1 = HOP["t_takeoff"], HOP["t_land"]
    legs = [chain(sd) for sd in ("leg.L", "leg.R") if len(chain(sd)) > 2]
    leg_len = min(((c[2].bone.head_local - c[0].bone.head_local).length for c in legs), default=H)
    crouch = min(HOP["crouch"] * H, HOP["crouch_leg"] * leg_len); Hj = HOP["height"] * H
    absorb = min(HOP["absorb"] * H, HOP["absorb_leg"] * leg_len)
    if "hop-depth" not in noted:
        noted.add("hop-depth"); log(f"hop: leg {leg_len:.3f} m, crouch {crouch:.3f} m, absorb {absorb:.3f} m, apex {Hj:.3f} m")
    # hips: down into the crouch, up through take-off, a parabola in the air, down on landing, overshoot, settle
    pre = seg(t, c0, c1) * (1 - seg(t, c1 + 0.01, T0))                     # 0 -> 1 -> 0 (crouch, then extend)
    u = min(1.0, max(0.0, (t - T0) / (T1 - T0))); air = 4 * u * (1 - u) if T0 < t < T1 else 0.0
    land_x = (t - T1) * secs                                               # seconds since touch-down
    endf = 1 - seg(t, 0.86, 0.99)          # every ring-out is faded to zero before the end, so the clip loops
    ring_ = lambda x, *k: ring(x, *k) * endf
    land = (seg(t, T1, T1 + 0.05) * (1 - seg(t, T1 + 0.07, T1 + 0.24))) if t > T1 else 0.0
    after = 0.25 * ring_(land_x - 0.30, 6.0, 1.6) if t > T1 else 0.0       # a small rebound after the absorb
    y = -crouch * pre + Hj * air - absorb * land + 0.04 * H * after
    move(first("root"), UP * y)
    legs_bend(max(0.0, -y), tuck=HOP["tuck"] * math.sin(math.pi * u) ** 1.5 if T0 < t < T1 else 0.0,
              # toes point only once the feet are off the ground (pointing them at T0 dug them into the floor)
              point=(seg(t, T0, T0 + 0.03) * (1 - seg(t, T0 + 0.06, T0 + 0.14))) - 0.6 * (seg(t, T1 - 0.06, T1 - 0.01) * (1 - seg(t, T1 - 0.01, T1 + 0.04))))
    # squash in the crouch and on landing, stretch through take-off and just before touch-down
    key_value("squash", HOP["squash"] * (pre * (1 - seg(t, c1, T0)) + land))
    key_value("stretch", HOP["stretch"] * (seg(t, c1, T0) * (1 - seg(t, T0 + 0.02, T0 + 0.14)) +
                                           0.4 * seg(t, T1 - 0.08, T1) * (1 - seg(t, T1, T1 + 0.02))))
    # body: lean in for the crouch, straighten (a touch back) on take-off, lean in again on landing, ring out
    lean = HOP["lean"] * (pre - 0.35 * seg(t, c1 + 0.02, T0) * (1 - seg(t, T0, T0 + 0.10)) + 0.8 * land) \
           - 3.0 * ring_(land_x - 0.12, 5.0, 1.8)
    rot(first("body"), RIGHT, lean)
    # head: looks up at the jump in the crouch, lags behind the body in the air (down going up, up coming down),
    # nods on landing and rings out
    vlag = hop_vnorm_biped(t - 0.03, T0, T1)
    rot(first("head"), RIGHT, -0.6 * lean - 8.0 * pre + 10.0 * vlag + 9.0 * ring_(land_x - 0.03, 6.0, 2.2))
    # arms: back in the crouch, fling forward-up and out at take-off, float down through the flight, swing
    # forward past rest on landing (follow-through), settle
    fling = seg(t, c1, T0 + 0.02) * (1 - seg(t, T0 + 0.12, T1 - 0.02))
    swing = HOP["arms_back"] * pre * (1 - seg(t, c1, T0)) - HOP["arms_up"] * fling \
            - 25.0 * ring_(land_x - 0.06, 5.0, 1.6)
    for side in ("arm.L", "arm.R"):
        ac = chain(side)
        if not ac: continue
        ax, _ = raise_axis(ac[0])
        rot(ac[0], RIGHT, swing); rot(ac[0], ax, HOP["arms_out"] * fling + 8.0 * pre)
        if len(ac) > 1: rot(ac[1], RIGHT, -20.0 * fling - 10.0 * pre + 12.0 * ring_(land_x - 0.10, 5.0, 2.0))
        if len(ac) > 2: rot(ac[2], RIGHT, 15.0 * hop_vnorm_biped(t - 0.06, T0, T1) - 15.0 * ring_(land_x - 0.14, 5.0, 2.4))
    for role in ("antenna", "tail"):
        for k, pb in enumerate(chain(role)):
            rot(pb, RIGHT, -18.0 * hop_vnorm_biped(t - 0.04 * (k + 1), T0, T1) + 10.0 * ring_(land_x - 0.05 * (k + 1)))
def hop_vnorm_biped(t, T0, T1):
    if not (T0 < t < T1): return 0.0
    u = (t - T0) / (T1 - T0); return 1 - 2 * u                               # +1 rising ... -1 falling
def clip_hop(t, secs):
    style = HOP["style"] if HOP["style"] != "auto" else ("biped" if is_biped_hips() else "object")
    if "hop-style" not in noted:
        noted.add("hop-style"); log(f"hop style: {style} (legs with knees: {len(chain('leg.L'))}/{len(chain('leg.R'))} bones, "
                                     f"root {hip_h / H:.2f} of the height above the floor)")
    if style == "biped": return clip_jump_biped(t, secs)
    antic = 0.8 * smoothstep(0.03, 0.20, t) * (1 - smoothstep(0.20, 0.26, t))
    land = 0.7 * smoothstep(0.66, 0.70, t) * (1 - smoothstep(0.72, 0.92, t))
    key_value("squash", antic + land)
    u = min(1.0, max(0.0, (t - HOP_T0) / (HOP_T1 - HOP_T0)))
    key_value("stretch", 0.8 * abs(1 - 2 * u) * smoothstep(HOP_T0, HOP_T0 + 0.05, t) * (1 - smoothstep(HOP_T1 - 0.05, HOP_T1, t)))
    move(first("root"), UP * hop_height(t))
    rot(first("head"), RIGHT, -12.0 * hop_vnorm(t - 0.04) + 5.0 * antic)
    for role in ("arm.L", "arm.R"):
        rot(first(role), RIGHT, 20.0 * hop_vnorm(t - 0.05) + 8.0 * antic)
    for role in ("antenna", "tail"):
        for k, pb in enumerate(chain(role)):
            rot(pb, RIGHT, -15.0 * hop_vnorm(t - 0.04 * (k + 1)))
    lid = first("lid")
    if lid: move(lid, UP * H * max(-0.02, min(0.06, -0.05 * hop_vnorm(t - 0.05))))

def clip_look(t, secs):
    s = math.sin(2 * math.pi * t); e = math.copysign(abs(s) ** 0.7, s)   # eases into the extremes
    rot(first("head"), UP, 35.0 * e)
    body = first("body")
    rot(body, UP, 5.0 * e); rot(body, FRONT, -3.0 * e)
    sway_chains(t, 1, 3.0)

def head_kdtree():
    """KD-tree of the vertices whose strongest weight is a head-role bone (armature space)"""
    import mathutils
    bones_ = [pb.name for pb in by_role.get("head", [])]
    if not bones_ or not mesh: return None
    gi = {g.index: g.name for g in mesh.vertex_groups}
    pts = []
    for v in mesh.data.vertices:
        if not v.groups: continue
        g = max(v.groups, key=lambda e: e.weight)
        if gi.get(g.group) in bones_: pts.append(mesh.matrix_world @ v.co)
    if not pts: return None
    kd = mathutils.kdtree.KDTree(len(pts))
    for i, q in enumerate(pts): kd.insert(q, i)
    kd.balance(); return kd

def wave_geometry(armR):
    """swing axes for the wave, and the forward/abduction angles after the distance clamp: every sample along
    the posed forearm and hand (at both rock extremes) must stay WAVE['clearance'] from the head vertices;
    on a violation the upper arm first swings 5 degrees more forward (towards the viewer), then abducts less"""
    up = armR[0]
    d = limb_dir(up)
    # the raise is in the frontal plane about a fixed axis per side; d x out (the old axis) degenerates for an arm
    # resting near horizontal (a T-pose rig, 2026-10-03: the wave drove the arm into the body, penetration 0.83)
    abd_axis = frontal_axis(up)
    fwd_axis = RIGHT if (Quaternion(RIGHT, math.radians(10)) @ d).dot(FRONT) > d.dot(FRONT) else -RIGHT
    S = up.bone.head_local
    E = armR[1].bone.head_local if len(armR) > 1 else S + d * up.bone.length
    fore = armR[1] if len(armR) > 1 else up
    H = armR[2].bone.head_local if len(armR) > 2 else E + limb_dir(fore) * max(fore.bone.length, 0.05)
    tip = armR[-1]; T = tip.bone.head_local + limb_dir(tip) * max(tip.bone.length, 0.05)
    kd = head_kdtree()
    # WAVE["abduct"] is where the upper arm should END, in degrees from hanging straight down in the frontal plane;
    # subtract how far out the arm already is at rest (an A-pose arm needs less raising than a hanging one)
    rest_out = relaxed_out(up)                           # relax_arms() has already lowered a T-pose arm
    forward, abduct = WAVE["forward"], max(0.0, WAVE["abduct"] - rest_out)
    log(f"wave: upper arm rests {rest_out:.0f} deg out from hanging; raising it {abduct:.0f} deg to {WAVE['abduct']:.0f}")
    def clear(fw, ab):
        rx = _relax.get(role_of(up))                     # the clip applies relax_arms() first: so does the clamp
        q_rel = Quaternion(rx[0], math.radians(rx[1])) if rx and rx[1] else Quaternion((1, 0, 0, 0))
        q_up = Quaternion(fwd_axis, math.radians(fw)) @ Quaternion(abd_axis, math.radians(ab)) @ q_rel
        E2 = S + q_up @ (E - S)
        for rock in (-WAVE["rock"], 0.0, WAVE["rock"]):
            q_el = q_up @ Quaternion(abd_axis, math.radians(WAVE["elbow"] + rock))
            H2 = E2 + q_el @ (H - E); T2 = H2 + q_el @ (T - H)
            for seg in ((E2, H2), (H2, T2)):
                for s in (0.0, 0.25, 0.5, 0.75, 1.0):
                    p = seg[0].lerp(seg[1], s)
                    if kd.find(p)[2] < WAVE["clearance"]: return False
        return True
    steps = []
    if kd:
        while not clear(forward, abduct):
            # a chibi's hand belongs beside the head, not in front of the face: lower the arm before swinging it forward
            if WAVE["style"] == "chibi" and abduct + rest_out - 5 >= 90: abduct -= 5; steps.append(f"abduct -> {abduct:g}")
            elif forward + 5 <= WAVE["forward_max"]: forward += 5; steps.append(f"forward -> {forward:g}")
            elif abduct - 5 >= 0: abduct -= 5; steps.append(f"abduct -> {abduct:g}")
            else: break
    log("wave clamp: " + (", ".join(steps) if steps else f"clear at forward {forward:g}, abduct {abduct:g} (no change)"))
    return abd_axis, fwd_axis, forward, abduct
_wave_cache = {}
def wave_style():
    if "style" not in _wave_cache:
        st = WAVE.get("style", "auto")
        # a spec that tunes the wave chose its look: auto never overrides it (the lamp's shade is 0.54 of its height)
        if st == "auto": st = "chibi" if head_share() >= 0.3 and not preset_overrides.get("wave") else "human"
        if st == "chibi":
            for k, v in WAVE_CHIBI.items():
                if k not in preset_overrides.get("wave", {}): WAVE[k] = v
        WAVE["style"] = st; _wave_cache["style"] = st
        log(f"wave style: {st} (head {head_share():.2f} of the height)")
    return _wave_cache["style"]
def clip_wave(t, secs):
    if wave_style() == "chibi": return clip_wave_chibi(t, secs)
    return clip_wave_human(t, secs)
def clip_wave_chibi(t, secs):
    """'hi!' from a big-headed character: the whole arm up and out beside the head (elbow soft), waving side to side
    at the shoulder with the hand flopping a beat behind; the body sways against it, the head tilts towards the
    hand, the other arm lifts a little, and the knees give a small bounce on every wave (feet planted)."""
    r = WAVE["ramp"] / secs
    env = smoothstep(0, r, t) * (1 - smoothstep(1 - r, 1, t))
    cycles = max(1, round(WAVE["hz"] * secs)); ph = 2 * math.pi * cycles * t
    armR = chain("arm.R")
    if armR:
        if "geo" not in _wave_cache: _wave_cache["geo"] = wave_geometry(armR)
        abd_axis, fwd_axis, forward, abduct = _wave_cache["geo"]
        # the raise overshoots a little and settles (follow-through), then the shoulder rocks
        raise_ = env + 0.08 * ring(t * secs - WAVE["ramp"] * 0.8, 6.0, 1.8) * (1 - smoothstep(1 - r, 1, t))
        rot(armR[0], abd_axis, abduct * raise_ + WAVE["shoulder_rock"] * env * math.sin(ph))
        rot(armR[0], fwd_axis, forward * env)
        if len(armR) > 1: rot(armR[1], abd_axis, WAVE["elbow"] * env + 0.4 * WAVE["shoulder_rock"] * env * math.sin(ph - 0.7))
        if len(armR) > 2: rot(armR[2], abd_axis, WAVE["hand_lag"] * env * math.sin(ph - 1.3))
    body = first("body")
    rot(body, FRONT, -WAVE["body_sway"] * env * math.sin(ph - 0.4))
    rot(first("head"), FRONT, WAVE["head_tilt"] * env + 3.0 * env * math.sin(ph - 0.9))
    armL = chain("arm.L")
    if armL:
        ax, _ = raise_axis(armL[0]); rot(armL[0], ax, WAVE["other_arm"] * env * (1 + 0.25 * math.sin(ph + math.pi)))
    bounce = WAVE["bounce"] * H * env * (0.5 - 0.5 * math.cos(2 * ph))      # two dips per wave cycle
    if is_biped_hips():
        move(first("root"), -UP * bounce); legs_bend(bounce)
    sway_chains(t, cycles, 5.0 * env)
def clip_wave_human(t, secs):
    """the human-standard friendly wave: upper arm out to the side and a little forward (elbow at shoulder
    height), forearm vertical with the hand at head height, palm forward, the forearm rocking side to side at the
    elbow with the hand lagging; raised over WAVE['ramp'] s and settled over the same, so it loops from hanging"""
    r = WAVE["ramp"] / secs
    env = smoothstep(0, r, t) * (1 - smoothstep(1 - r, 1, t))
    armR = chain("arm.R")
    if armR:
        if "geo" not in _wave_cache: _wave_cache["geo"] = wave_geometry(armR)
        abd_axis, fwd_axis, forward, abduct = _wave_cache["geo"]
        cycles = max(1, round(WAVE["hz"] * secs))
        rock = WAVE["rock"] * env * math.sin(2 * math.pi * cycles * t)
        rot(armR[0], abd_axis, abduct * env); rot(armR[0], fwd_axis, forward * env)
        if len(armR) > 1:
            rot(armR[1], abd_axis, WAVE["elbow"] * env + rock)
            twist[armR[1].name] = WAVE["twist"] * env
            if len(armR) > 2:
                rot(armR[2], abd_axis, WAVE["hand_lag"] * env * math.sin(2 * math.pi * cycles * t - math.pi / 2))
        else:
            rot(armR[0], abd_axis, 0.5 * rock)
    rot(first("head"), FRONT, WAVE["head_tilt"] * env)
    rot(first("arm.L"), RIGHT, -3.0 * env * math.sin(2 * math.pi * max(1, round(WAVE["hz"] * secs)) * t))   # counter-sway
    sway_chains(t, 1, 3.0)

def clip_spin(t, secs):
    wheels = chain("wheel")
    if wheels:
        for pb in wheels:                                  # about the bone's own axis: pose-local Y
            q, off = pose.setdefault(pb.name, [Quaternion((1, 0, 0, 0)), Vector((0, 0, 0))])
            axis_world = (pb.bone.matrix_local.to_3x3() @ Vector((0, 1, 0))).normalized()
            pose[pb.name][0] = Quaternion(axis_world, 2 * math.pi * t) @ q
    else:
        if "spin-fallback" not in noted:
            noted.add("spin-fallback"); log("note: no 'wheel' bones; spin turns the root a full turn instead")
        q, off = pose.setdefault(first("root").name, [Quaternion((1, 0, 0, 0)), Vector((0, 0, 0))])
        pose[first("root").name][0] = Quaternion(UP, 2 * math.pi * t) @ q

def clip_wobble(t, secs):
    env = window(t, 0.08, 0.78) * math.exp(-2.2 * t)
    ph = 2 * math.pi * 3 * t
    root = first("root")
    rot(root, FRONT, 14.0 * env * math.sin(ph)); rot(root, RIGHT, 8.0 * env * math.cos(ph))
    head = first("head")
    rot(head, FRONT, -4.0 * env * math.sin(ph - 0.6)); rot(head, RIGHT, -2.5 * env * math.cos(ph - 0.6))
    for role in ("arm.L", "arm.R"):
        pb = first(role)
        if pb:
            ax, _ = raise_axis(pb); rot(pb, ax, 12.0 * env * max(0.0, math.sin(ph + (0 if role == "arm.R" else math.pi))))
    sway_chains(t, 3, 6.0 * env)

def clip_sleep(t, secs):
    e = bump(t); s = math.sin(2 * math.pi * t)
    rot(first("body"), RIGHT, 8.0 * e); rot(first("head"), RIGHT, 14.0 * e)
    for role in ("arm.L", "arm.R"): rot(first(role), RIGHT, 6.0 * e)
    for role in ("antenna", "tail"):
        for pb in chain(role): rot(pb, RIGHT, 8.0 * e)
    key_value("stretch", 0.125 * max(0.0, s)); key_value("squash", 0.10 * max(0.0, -s))   # +-1.5%, half speed
    move(first("lid"), UP * 0.005 * H * max(0.0, s))

PRESETS = {"idle": clip_idle, "hop": clip_hop, "look": clip_look, "wave": clip_wave, "spin": clip_spin, "wobble": clip_wobble, "sleep": clip_sleep}
def clip_seconds(name): return a.seconds * (2 if name == "sleep" else 1)

# ---------- 4. bake each clip into a slotted action (armature slot + shape-key slot) ----------
def glb_patch(path, fn):
    """edit the JSON chunk of a GLB in place (used to name the single animation after its clip)"""
    with open(path, "rb") as f: d = f.read()
    magic, ver, _ = struct.unpack_from("<III", d, 0)
    jlen, jtype = struct.unpack_from("<II", d, 12)
    js = json.loads(d[20:20 + jlen]); rest = d[20 + jlen:]
    fn(js)
    jb = json.dumps(js, separators=(",", ":")).encode(); jb += b" " * ((4 - len(jb) % 4) % 4)
    with open(path, "wb") as f:
        f.write(struct.pack("<III", magic, ver, 20 + len(jb) + len(rest)) + struct.pack("<II", len(jb), jtype) + jb + rest)
def glb_animations(path):
    with open(path, "rb") as f: d = f.read()
    jlen = struct.unpack_from("<I", d, 12)[0]; js = json.loads(d[20:20 + jlen])
    return [(an.get("name"), len(an["channels"])) for an in js.get("animations", [])]
def glb_root_mover(js, name="m3d_mover"):
    """Root motion for RealityKit (2026-10-02, the viewer animation notes, 2026-10-02, not published "Root motion"). Apple's usdextract keeps
    only the joint ROTATIONS of a glTF animation in its SkelAnimation (translations stay at rest, scales at 1) and
    RealityKit's skinning ignores the xformOp samples it bakes on the joint prims, so the hop never left the ground
    and idle never breathed; a clip whose root joint has translation/scale channels did not play at all. RealityKit
    does play transform animation on ordinary entities, so this moves EVERY channel of the skin's root joint (its
    translation, rotation and scale samplers, reused as they are) onto a new non-joint node that becomes the parent
    of the root joint and of the skinned mesh node(s). The mover takes the root's rest TRS and the root becomes
    identity: world placement and the inverse bind matrices are unchanged, glTF consumers see the same motion (the
    joints inherit the mover), and usdextract puts the SkelRoot under the mover, which RealityKit then animates.
    Edits the GLB JSON in place (glb_patch); returns the number of channels moved, 0 when there was nothing to move."""
    nodes = js.get("nodes", []); skins = js.get("skins", [])
    if not skins or not nodes: return 0
    joints = set(skins[0]["joints"])
    parent = {c: i for i, n in enumerate(nodes) for c in n.get("children", [])}
    roots = [j for j in skins[0]["joints"] if parent.get(j) not in joints]
    if len(roots) != 1: return 0                                      # several skeleton roots: leave the plain layout
    root = roots[0]; par = parent.get(root)
    moved = [ch for an in js.get("animations", []) for ch in an["channels"] if ch["target"]["node"] == root]
    if not moved: return 0
    meshes = [i for i, n in enumerate(nodes) if n.get("skin") == 0 and parent.get(i) == par]   # siblings of the root
    mv = {"name": name, "children": [root] + meshes}
    for k in ("translation", "rotation", "scale"):
        if k in nodes[root]: mv[k] = nodes[root].pop(k)
    nodes.append(mv); mi = len(nodes) - 1
    drop = set([root] + meshes)
    if par is None:
        for scn in js.get("scenes", []): scn["nodes"] = [n for n in scn["nodes"] if n not in drop] + [mi]
    else:
        nodes[par]["children"] = [n for n in nodes[par]["children"] if n not in drop] + [mi]
    for ch in moved: ch["target"]["node"] = mi
    return len(moved)

arm.animation_data_create()
if key: key.animation_data_create()
actions = {}
for name in clips:
    secs = clip_seconds(name); N = max(2, round(a.fps * secs))
    act = bpy.data.actions.new(name)
    slot_ob = act.slots.new(id_type="OBJECT", name=arm.name)
    arm.animation_data.action = act; arm.animation_data.action_slot = slot_ob
    slot_key = None
    if key and a.shape_keys:
        slot_key = act.slots.new(id_type="KEY", name=key.name)
        key.animation_data.action = act; key.animation_data.action_slot = slot_key
    snaps = []
    for f in range(1, N + 2):                                  # frame N+1 repeats frame 1: a clean loop
        t = (f - 1) / N
        reset_pose(); relax_arms(); PRESETS[name](t, secs); apply_pose()
        for pb in arm.pose.bones:
            pb.keyframe_insert("rotation_quaternion", frame=f); pb.keyframe_insert("location", frame=f)
            pb.keyframe_insert("scale", frame=f)
        if key and a.shape_keys:
            for kn in ("squash", "stretch"):
                if kn in keys_available: key.key_blocks[kn].keyframe_insert("value", frame=f)
        if f in (1, N + 1):
            snaps.append(([tuple(pb.rotation_quaternion) + tuple(pb.location) + tuple(pb.scale) for pb in arm.pose.bones], dict(shape)))
    d_pose = max(abs(x - y) for p0, p1 in zip(snaps[0][0], snaps[1][0]) for x, y in zip(p0, p1))
    d_shape = max([abs(snaps[0][1].get(k, 0) - snaps[1][1].get(k, 0)) for k in set(snaps[0][1]) | set(snaps[1][1])] or [0.0])
    act.frame_range = (1, N + 1); act.use_frame_range = True
    actions[name] = (act, slot_ob, slot_key, N)
    log(f"baked {name}: {N + 1} frames at {a.fps} fps ({secs:g} s), loop delta pose {d_pose:.2e} shape {d_shape:.2e}")
    arm.animation_data.action = None
    if key: key.animation_data.action = None
reset_pose()

# ---------- 5. export: one GLB per clip (ACTIVE_ACTIONS), then every clip (ACTIONS from NLA stashes) ----------
bpy.ops.object.select_all(action="DESELECT")
arm.select_set(True)
for o in meshes or [mesh]: o.select_set(True)
bpy.context.view_layer.objects.active = arm
common = dict(export_format="GLB", use_selection=True, export_yup=True, export_apply=False, export_skins=True,
              export_morph=True, export_extras=True, export_animations=True, export_force_sampling=True,
              export_anim_slide_to_zero=True, export_frame_range=False, export_image_format="AUTO")
written = []
for name in clips:
    act, slot_ob, slot_key, N = actions[name]
    arm.animation_data.action = act; arm.animation_data.action_slot = slot_ob
    if slot_key: key.animation_data.action = act; key.animation_data.action_slot = slot_key
    sc.frame_start, sc.frame_end = 1, N + 1
    path = os.path.join(a.outdir, f"{stem}.{name}.glb")
    bpy.ops.export_scene.gltf(filepath=path, export_animation_mode="ACTIVE_ACTIONS", **common)
    def rename(js, name=name):
        if len(js.get("animations", [])) == 1: js["animations"][0]["name"] = name
    glb_patch(path, rename)
    moved = []
    if a.usdz_root_motion: glb_patch(path, lambda js: moved.append(glb_root_mover(js)))
    anims = glb_animations(path)
    if len(anims) != 1: raise SystemExit(f"[motion] {path}: expected 1 animation, got {anims}")
    written.append(path); log(f"wrote {path} animations={anims}" + (f" root channels on m3d_mover={moved[0]}" if moved else ""))
    arm.animation_data.action = None
    if key: key.animation_data.action = None
# stash every clip on its own NLA track (the exporter's ACTIONS mode collects actions from NLA strips)
for name in clips:
    act, slot_ob, slot_key, N = actions[name]
    for ad, slot in ((arm.animation_data, slot_ob), (key.animation_data if slot_key else None, slot_key)):
        if ad is None: continue
        tr = ad.nla_tracks.new(); tr.name = name
        st = tr.strips.new(name, 1, act); st.action_slot = slot
sc.frame_start, sc.frame_end = 1, max(actions[n][3] for n in clips) + 1
all_path = os.path.join(a.outdir, f"{stem}.all.glb")
bpy.ops.export_scene.gltf(filepath=all_path, export_animation_mode="ACTIONS", **common)
anims = glb_animations(all_path)
if sorted(n for n, _ in anims) != sorted(clips): raise SystemExit(f"[motion] {all_path}: expected {clips}, got {anims}")
log(f"wrote {all_path} animations={anims}")

# ---------- 6. --check: Workbench contact strips, one per clip ----------
if a.check and mesh:
    sc.render.engine = "BLENDER_WORKBENCH"
    world = bpy.data.worlds.new("w"); sc.world = world; world.color = (0.32, 0.33, 0.36)
    sh = sc.display.shading; sh.light = "STUDIO"; sh.color_type = "TEXTURE"; sh.show_shadows = False
    sh.show_cavity = True; sh.show_object_outline = True; sc.display.render_aa = "FXAA"
    for m in mesh.data.materials:
        if m and m.node_tree:
            tex = next((n for n in m.node_tree.nodes if n.type == "TEX_IMAGE"), None)
            if tex: m.node_tree.nodes.active = tex
    size_px = a.size
    sc.render.resolution_x = sc.render.resolution_y = size_px; sc.render.resolution_percentage = 100
    cam = bpy.data.objects.new("check-cam", bpy.data.cameras.new("check-cam")); sc.collection.objects.link(cam)
    cam.data.type = "ORTHO"; sc.camera = cam
    bb = [mesh.matrix_world @ Vector(c) for c in mesh.bound_box]
    lo_ = Vector((min(v.x for v in bb), min(v.y for v in bb), min(v.z for v in bb)))
    hi_ = Vector((max(v.x for v in bb), max(v.y for v in bb), max(v.z for v in bb)))
    centre = (lo_ + hi_) / 2 + UP * 0.12 * H; diag = (hi_ - lo_).length
    cam.data.ortho_scale = diag * 1.35; cam.data.clip_end = diag * 10
    yaw, el = math.radians(a.yaw), math.radians(12)
    dvec = Vector((math.sin(yaw) * math.cos(el), -math.cos(yaw) * math.cos(el), math.sin(el)))
    cam.location = centre + dvec * diag * 3; cam.rotation_euler = (-dvec).to_track_quat("-Z", "Y").to_euler()
    for tr in arm.animation_data.nla_tracks: tr.mute = True
    if key:
        for tr in key.animation_data.nla_tracks: tr.mute = True
    tmp = os.path.join(a.outdir, "_frame.png")
    for name in clips:
        act, slot_ob, slot_key, N = actions[name]
        arm.animation_data.action = act; arm.animation_data.action_slot = slot_ob
        if slot_key: key.animation_data.action = act; key.animation_data.action_slot = slot_key
        panels = []
        for frac in (0.0, 0.2, 0.45, 0.7):
            sc.frame_set(1 + round(frac * N))
            sc.render.filepath = tmp; bpy.ops.render.render(write_still=True)
            im = bpy.data.images.load(tmp); px = np.empty(len(im.pixels), np.float32); im.pixels.foreach_get(px)
            bpy.data.images.remove(im); panels.append(px.reshape(size_px, size_px, 4))
        out = np.concatenate(panels, axis=1)
        png = os.path.join(a.outdir, f"check-{name}.png")
        im = bpy.data.images.new(name, out.shape[1], out.shape[0], alpha=True); im.pixels.foreach_set(out.ravel())
        im.filepath_raw = png; im.file_format = "PNG"; im.save(); bpy.data.images.remove(im)
        arm.animation_data.action = None
        if key: key.animation_data.action = None
        log(f"wrote {png} (frames t=0, 0.2, 0.45, 0.7)")
    try: os.remove(tmp)
    except OSError: pass

print(f"M3D_ANIM clips={len(written)} out={a.outdir}")

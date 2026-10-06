#!/usr/bin/env python3
"""momask — humanoid text-to-motion on CPU (MoMask, CVPR 2024, MIT) → BVH clips.

usage: momask.py "<prompt>" OUTDIR [--seconds 4] [--seed N] [--repeats 1] [--dry-run]
                 [--max-seconds 900]

Why (2026-10-02): `m3d animate --engine momask` needs humanoid text-to-motion the way
Meshy sells it. MoMask (EricGuo5513/momask-codes, MIT; HumanML3D 22-joint skeleton; the
authors: "No GPU is required to use MoMask") is the open route. Measured here: a 4 s clip
takes ~5 s wall and 3.4 GB peak on CPU, so it never touches the shared GPU.

CPU pinning: gen_t2m.py chooses its device in one line, torch.device("cpu") when --gpu_id
is -1, which this script always passes (the code has no MPS path; CLIP is loaded with
device='cpu'); the launcher refuses any other gpu_id and strips PYTORCH_ENABLE_MPS_FALLBACK.
Verified 2026-10-02 with a TorchFunctionMode that raised on any mps/cuda tensor: none.

Layout, all outside the pristine clone (refresh it with git pull):
  ~/repos/momask-codes        upstream clone, never edited
  ~/.momask/venv              Python 3.11, torch 2.14 (CPU), numpy 2.4, openai CLIP
  ~/.momask/checkpoints/t2m   HumanML3D models (194 MB, the authors' Google Drive via gdown)
  ~/.momask/visualization     symlink into the clone: template.bvh is opened cwd-relative
  ~/.momask/gen_t2m_cpu.py    LAUNCHER (below), rewritten by this script when stale
  ~/.cache/clip/ViT-B-32.pt   text encoder (352 MB, fetched by clip.load on first run)
Override the two roots with MOMASK_HOME and MOMASK_REPO.

Output in OUTDIR: <slug>.bvh (the model's motion), <slug>_ik.bvh (upstream's naive
foot-skate removal on top; "sometimes fails"), <slug>.npy joint positions (frames, 22, 3)
in HumanML3D/SMPL order; further repeats get -2, -3 … suffixes. The raw upstream output
stays in OUTDIR/momask-raw/. 20 fps, Y up, metres, lengths rounded to a multiple of 4
frames, at most 196 (9.8 s). BVH joints in file order: Hips, LeftUpLeg, LeftLeg,
LeftFoot, LeftToe, RightUpLeg, RightLeg, RightFoot, RightToe, Spine, Spine1, Spine2,
Neck, Head, LeftShoulder, LeftArm, LeftForeArm, LeftHand, RightShoulder, RightArm,
RightForeArm, RightHand (Mixamo-style names; rotation channels Z X Y).

Prints `M3D_MOTION clips=<n> frames=<n> fps=<n> joints=22 out=<dir>` on success (bin/m3d
greps for the M3D_ line); stage lines go to stderr. --dry-run prints the command and
exits 0. A missing venv, clone or model set fails with the one-line fix.
"""
import argparse
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

HOME = Path(os.environ.get("MOMASK_HOME", Path.home() / ".momask"))
REPO = Path(os.environ.get("MOMASK_REPO", Path.home() / "repos" / "momask-codes"))
FPS = 20            # HumanML3D frame rate; the model works in blocks of 4 frames (one token)
MAX_FRAMES = 196    # max_motion_length in the checkpoints' opt.txt
CKPT = "checkpoints/t2m"
MODEL_FILES = [     # what gen_t2m.py loads, relative to HOME
    f"{CKPT}/rvq_nq6_dc512_nc512_noshare_qdp0.2/model/net_best_fid.tar",
    f"{CKPT}/rvq_nq6_dc512_nc512_noshare_qdp0.2/meta/mean.npy",
    f"{CKPT}/t2m_nlayer8_nhead6_ld384_ff1024_cdp0.1_rvq6ns/model/latest.tar",
    f"{CKPT}/tres_nlayer8_ld384_ff1024_rvq6ns_cdp0.2_sw/model/net_best_fid.tar",
    f"{CKPT}/length_estimator/model/finest.tar",
]
MODELS_GDRIVE = "1vXS7SHJBgWPt59wupQ5UUzhFObrnGkQ0"   # humanml3d_models.zip, prepare/download_models.sh
PIP = ("torch numpy scipy matplotlib einops ftfy regex tqdm pillow gdown "
       "git+https://github.com/openai/CLIP.git")

LAUNCHER = '''#!/usr/bin/env python
"""Written by local-3d/lib/momask.py (LAUNCHER) — edit it there, not here.

Runs the pristine ~/repos/momask-codes/gen_t2m.py on CPU under the ~/.momask venv
(Python 3.11, torch 2.14, numpy 2.4), pinned to the CPU (see the comment below), with
three shims the 2024 code needs:
  1. numpy >= 1.24 removed the np.float / np.int aliases; common/quaternion.py:13 reads
     np.float at import time, remove_fs.py and AnimationStructure.py use both.
  2. numpy 2 removed numpy.core.umath_tests; Animation.py and Quaternions.py use only its
     matrix_multiply gufunc, whose documented replacement is np.matmul.
  3. The stick-figure mp4 render (two per clip, matplotlib + ffmpeg) is stubbed out: the
     BVH and npy outputs do not depend on it, and it crashes on matplotlib >= 3.7 anyway
     (plot_script.py:378 assigns ax.lines, a read-only property now). MOMASK_VIDEO=1
     re-enables it.
The working directory must hold visualization/ (symlink into the clone: the BVH template
is opened as ./visualization/data/template.bvh) and checkpoints/t2m/ (the opt.txt files
say checkpoints_dir: ./checkpoints). Output goes to ./generation/<ext>/, or to <ext>
itself when it is an absolute path.
"""
import os
import runpy
import sys
import types

import numpy as np

REPO = os.environ.get("MOMASK_REPO", os.path.join(os.environ.get("M3D_REPOS", os.path.expanduser("~/repos")), "momask-codes"))
sys.path.insert(0, REPO)

# CPU pinning (2026-10-02, asked by the Mac's memory watchdog): gen_t2m.py picks its device
# in one line, torch.device("cpu" if gpu_id == -1 else "cuda:N"), and the code has no MPS
# path at all; lib/momask.py always passes --gpu_id -1. Refuse anything else and drop the
# MPS fallback switch before torch is imported. (Overriding torch's is_available probes was
# tried and dropped: torch._dynamo reads their __wrapped__ at import.) MPS runs are a
# separate, later decision.
_gpu = [a.split("=", 1)[1] if "=" in a else sys.argv[i + 1]
        for i, a in enumerate(sys.argv) if a.startswith("--gpu_id")]
if any(g != "-1" for g in _gpu):
    sys.exit("gen_t2m_cpu.py: CPU only; pass --gpu_id -1")
os.environ.pop("PYTORCH_ENABLE_MPS_FALLBACK", None)

for name, builtin in (("float", float), ("int", int)):
    if name not in np.__dict__:
        setattr(np, name, builtin)

try:
    import numpy.core.umath_tests  # noqa: F401  (still there in numpy 1.x)
except ImportError:
    ut = types.ModuleType("numpy.core.umath_tests")
    ut.matrix_multiply = np.matmul
    sys.modules["numpy.core.umath_tests"] = ut

if os.environ.get("MOMASK_VIDEO") != "1":
    import utils.plot_script as plot_script
    plot_script.plot_3d_motion = lambda *a, **k: None

sys.argv[0] = os.path.join(REPO, "gen_t2m.py")
runpy.run_path(sys.argv[0], run_name="__main__")
'''


def log(msg):
    print(f"[momask] {msg}", file=sys.stderr, flush=True)


def slugify(text):
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s[:48].rstrip("-") or "motion"


def bvh_info(path):
    """(joint names in file order, frame count, frame time) from a BVH header."""
    names, frames, frametime = [], 0, 0.0
    with open(path) as f:
        for line in f:
            t = line.split()
            if len(t) >= 2 and t[0] in ("ROOT", "JOINT"):
                names.append(t[1])
            elif t[:1] == ["Frames:"]:
                frames = int(t[1])
            elif t[:2] == ["Frame", "Time:"]:
                frametime = float(t[2])
                break  # last header line; the motion data follows
    return names, frames, frametime


def preflight(strict):
    """Each missing piece prints as `momask: <what>: <one-line fix>`; strict exits 1."""
    py = HOME / "venv" / "bin" / "python"
    problems = []
    if not py.exists():
        problems.append(f"venv missing: uv venv {HOME}/venv --python 3.11 && "
                        f"uv pip install --python {py} {PIP}")
    if not (REPO / "gen_t2m.py").exists():
        problems.append(f"clone missing: git clone https://github.com/EricGuo5513/momask-codes {REPO}")
    if not all((HOME / f).exists() for f in MODEL_FILES):
        problems.append(f"models missing: mkdir -p {HOME}/{CKPT} && cd {HOME}/{CKPT} && "
                        f"{py} -m gdown {MODELS_GDRIVE} -O m.zip && unzip m.zip && rm m.zip")
    if not (Path.home() / ".cache" / "clip" / "ViT-B-32.pt").exists():
        log("CLIP ViT-B/32 not cached yet: the first run downloads 352 MB to ~/.cache/clip")
    for p in problems:
        print(f"momask: {p}", file=sys.stderr)
    if problems and strict:
        sys.exit(1)


def prepare():
    """The two things this script owns inside HOME: the launcher and the template symlink."""
    launcher = HOME / "gen_t2m_cpu.py"
    if not launcher.exists() or launcher.read_text() != LAUNCHER:
        launcher.write_text(LAUNCHER)
        log(f"wrote {launcher}")
    link = HOME / "visualization"
    if not link.exists():           # follows the link, so a dangling one is replaced too
        if link.is_symlink():
            link.unlink()
        link.symlink_to(REPO / "visualization")


def main():
    ap = argparse.ArgumentParser(description="MoMask humanoid text-to-motion on CPU → BVH")
    ap.add_argument("prompt")
    ap.add_argument("outdir")
    ap.add_argument("--seconds", type=float, default=4.0,
                    help="clip length; rounded to 4-frame blocks at 20 fps, at most 9.8")
    ap.add_argument("--seed", type=int, default=10107, help="upstream default")
    ap.add_argument("--repeats", type=int, default=1, help="distinct clips for the same prompt")
    ap.add_argument("--dry-run", action="store_true", help="print the command, run nothing")
    ap.add_argument("--max-seconds", type=int, default=900, help="kill the run after this long")
    a = ap.parse_args()

    frames = max(4, min(MAX_FRAMES, int(round(a.seconds * FPS / 4)) * 4))
    if frames != round(a.seconds * FPS):
        log(f"length rounded to {frames} frames ({frames / FPS:.2f} s)")
    outdir = Path(a.outdir).expanduser().resolve()
    raw = outdir / "momask-raw"
    slug = slugify(a.prompt)
    py = HOME / "venv" / "bin" / "python"
    # --ext is joined onto ./generation; an absolute path replaces it (os.path.join), so
    # the raw output lands next to the clips instead of piling up in ~/.momask/generation.
    cmd = [str(py), str(HOME / "gen_t2m_cpu.py"), "--gpu_id", "-1", "--ext", str(raw),
           "--text_prompt", a.prompt, "--motion_length", str(frames),
           "--repeat_times", str(a.repeats), "--seed", str(a.seed)]
    env = {k: v for k, v in os.environ.items()
           if k not in ("MOMASK_VIDEO", "PYTORCH_ENABLE_MPS_FALLBACK")}  # mp4 off; no MPS switch
    env.update(MOMASK_REPO=str(REPO), PYTHONUNBUFFERED="1",
               PYTHONDONTWRITEBYTECODE="1")  # no __pycache__ in the pristine clone
    if a.dry_run:
        preflight(strict=False)
        print(f"cd {HOME} && MOMASK_REPO={REPO} {shlex.join(cmd)}")
        return
    preflight(strict=True)
    prepare()
    outdir.mkdir(parents=True, exist_ok=True)
    if raw.exists():
        shutil.rmtree(raw)  # an older run's files would otherwise pass for this one's
    log(f"generating {a.repeats} × {frames} frames ({frames / FPS:.1f} s) on CPU: {a.prompt!r}")
    t0 = time.time()
    try:
        p = subprocess.run(cmd, cwd=HOME, env=env, stdout=sys.stderr, timeout=a.max_seconds)
    except subprocess.TimeoutExpired:
        sys.exit(f"momask: gen_t2m.py still running after {a.max_seconds}s; killed")
    if p.returncode != 0:
        sys.exit(f"momask: gen_t2m.py exited {p.returncode} after {time.time() - t0:.0f}s (output above)")

    copied = []
    for r in range(a.repeats):
        base = f"sample0_repeat{r}_len{frames}"
        name = slug if r == 0 else f"{slug}-{r + 1}"
        for src, dst in ((raw / "animations" / "0" / f"{base}.bvh", outdir / f"{name}.bvh"),
                         (raw / "animations" / "0" / f"{base}_ik.bvh", outdir / f"{name}_ik.bvh"),
                         (raw / "joints" / "0" / f"{base}.npy", outdir / f"{name}.npy")):
            if not src.exists():
                sys.exit(f"momask: expected output missing: {src}")
            shutil.copyfile(src, dst)
            copied.append(dst.name)
    names, nframes, frametime = bvh_info(outdir / f"{slug}.bvh")
    fps = round(1 / frametime) if frametime else 0
    if len(names) != 22 or nframes != frames:
        sys.exit(f"momask: {slug}.bvh has {len(names)} joints and {nframes} frames; "
                 f"expected 22 and {frames}")
    log(f"done in {time.time() - t0:.0f}s → {', '.join(copied)}")
    print(f"M3D_MOTION clips={a.repeats} frames={nframes} fps={fps} joints={len(names)} out={outdir}")


if __name__ == "__main__":
    main()

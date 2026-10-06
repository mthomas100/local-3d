#!/usr/bin/env python3
"""skintokens — auto-rig a GLB (skeleton + skin weights) with SkinTokens through skin-tokens.cpp.

usage: skintokens.py IN.glb OUTDIR [--device auto|cpu|metal] [--seed N] [--dry-run]
                     [--postprocess] [--beams N] [--max-tokens N] [--temperature F]
                     [--graft auto|exact|blender] [--from-raw RAW.glb]

Why (2026-10-02): SkinTokens / TokenRig (VAST, MIT weights) is the only open auto-rigger
that runs on Apple silicon, through the GGML port skin-tokens.cpp (Apache-2.0; CPU, and
Metal through `--device auto` since its PR #3, 2026-09-17). Its `rig` command generates
the skeleton AND the learned skin weights in one pass, so no separate `skin` step runs.
The CLI has no seed flag (its NumPy-compatible PCG64 sampler accepts only the fixed seeds
0 and 1 and the CLI always uses 0), so runs are deterministic; --seed is recorded only.
`--device metal` is spelled `--device auto` for the CLI: that takes the first GGML GPU
device, which is Metal on this Mac (the CLI knows no "metal" word).

What the port drops, and what this wrapper does about it: its GLB writer keeps positions,
recomputed normals and vertex colours only. UVs, textures and materials are gone ("atlas
textures are not yet round-tripped", upstream README). So `rig` writes a raw GLB and the rig
is grafted back onto the ORIGINAL textured input, two ways:
  exact   — the port's loader keeps vertex order (mesh instance order, primitive order,
            accessor order), so the joint nodes, the skin (inverse bind matrices) and the
            per-vertex JOINTS_0 / WEIGHTS_0 are copied in 1:1 with numpy; weights stay
            bit-exact and everything else in the input GLB stays untouched. Guarded: vertex
            counts and positions must line up, no instancing, no skipped primitives.
  blender — lib/graft_skin.py in headless Blender: nearest-vertex weight transfer onto the
            original meshes, parented to the imported armature, original materials and
            textures re-exported. Works when the rigger re-indexed, welded or dropped vertices.
--graft auto (default) tries exact, then blender. The port's one-frame rest-pose animation
is never copied: the pipeline makes one GLB per clip and usdextract keeps only the first.

Upstream figures (skin-tokens.cpp PR #3 and #2, F16, 10 beams): M4 Pro, 104k verts, `rig`
2m41s on Metal / 6m57s on CPU; M1 Max, 41k verts, `rig` 210 s on Metal. No memory figure is
published; the TokenRig KV cache is F32, (mesh tokens + max-tokens) × beams × 28 layers ×
2 × 8 × 128 × 4 B ≈ 5.9 GB at the defaults, plus 1.25 GB of weights. The run prints a
heartbeat with the CLI's RSS every minute so the first run measures it.

Run this with ~/.hy3d/worker-venv/bin/python (pygltflib + numpy).
"""
import argparse
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pygltflib

CLI = Path(os.environ.get("SKINTOKENS_CLI",
                          Path(os.environ.get("M3D_REPOS", Path.home() / "repos")) / "skin-tokens.cpp/build/release/bin/skintokens-cli"))
MODEL_DIR = Path(os.environ.get("SKINTOKENS_MODEL_DIR", Path.home() / ".skintokens/gguf/F16"))
BLENDER = Path(os.environ.get("M3D_BLENDER", "/opt/homebrew/bin/blender"))  # m3d.toml [paths] blender
GRAFT_SKIN = Path(__file__).resolve().with_name("graft_skin.py")
GGUFS = ("mesh-encoder.gguf", "tokenrig.gguf", "skin-vae.gguf")
BUILD_FIX = ("cd ~/repos/skin-tokens.cpp && cmake -S . -B build/release -G Ninja "
             "-DCMAKE_BUILD_TYPE=Release -DSKINTOKENS_ENABLE_VULKAN=OFF && cmake --build build/release -j")
WEIGHTS_FIX = "~/.local/bin/hf download LocalAI-io/SkinTokens-GGUF --include 'F16/*' --local-dir ~/.skintokens/gguf"
DEVICE_ARG = {"auto": "auto", "metal": "auto", "cpu": "cpu"}

DTYPES = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16}


def log(msg):
    print(f"[skintokens] {msg}", file=sys.stderr, flush=True)


def die(msg):
    sys.exit(f"skintokens: {msg}")


class GraftError(Exception):
    """The exact graft cannot map the rig back 1:1; the Blender graft may still."""


# ---------- glTF accessors ----------
def read_accessor(g, blob, index):
    """Accessor → (count, components) numpy array; honours byteStride and normalized ints."""
    a = g.accessors[index]
    if a.sparse is not None:
        die("sparse glTF accessors are unsupported")
    dt = np.dtype(DTYPES[a.componentType])
    n = NCOMP[a.type]
    elem = dt.itemsize * n
    bv = g.bufferViews[a.bufferView]
    start = (bv.byteOffset or 0) + (a.byteOffset or 0)
    stride = bv.byteStride or elem
    raw = np.frombuffer(blob, dtype=np.uint8, count=stride * (a.count - 1) + elem, offset=start)
    rows = np.lib.stride_tricks.as_strided(raw, shape=(a.count, elem), strides=(stride, 1))
    out = np.ascontiguousarray(rows).view(dt).reshape(a.count, n)
    if a.normalized and dt.kind == "u":
        out = out.astype(np.float32) / np.iinfo(dt).max
    return out


class BlobWriter:
    """Appends arrays to a GLB's single buffer as new bufferView + accessor pairs."""

    def __init__(self, g):
        self.g = g
        self.blob = bytearray(g.binary_blob())

    def add(self, array, component_type, type_, target=None, minmax=False):
        while len(self.blob) % 4:
            self.blob += b"\0"
        data = np.ascontiguousarray(array).tobytes()
        self.g.bufferViews.append(pygltflib.BufferView(buffer=0, byteOffset=len(self.blob),
                                                       byteLength=len(data), target=target))
        self.blob += data
        acc = pygltflib.Accessor(bufferView=len(self.g.bufferViews) - 1, byteOffset=0,
                                 componentType=component_type, count=len(array), type=type_)
        if minmax:
            acc.min = array.min(0).tolist()
            acc.max = array.max(0).tolist()
        self.g.accessors.append(acc)
        return len(self.g.accessors) - 1

    def finish(self):
        self.g.buffers[0].byteLength = len(self.blob)
        self.g.set_binary_blob(bytes(self.blob))


# ---------- node transforms (the port bakes the world transform into the vertices) ----------
def quat_matrix(q):
    x, y, z, w = (np.asarray(q, dtype=np.float64) / np.linalg.norm(q)).tolist()
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def local_matrix(n):
    if n.matrix:
        return np.array(n.matrix, dtype=np.float64).reshape(4, 4).T  # glTF is column-major
    m = np.eye(4)
    if n.scale:
        m[:3, :3] = np.diag(n.scale)
    if n.rotation:
        m[:3, :3] = quat_matrix(n.rotation) @ m[:3, :3]
    if n.translation:
        m[:3, 3] = n.translation
    return m


def mesh_instances(g):
    """[(node index, parent node index, world matrix)] in skin-tokens.cpp's loader order:
    scene roots in order, depth first, children in order."""
    if not g.scenes:
        raise GraftError("input GLB has no scene (skin-tokens.cpp read its meshes directly, but a skin needs nodes)")
    out = []

    def visit(i, parent, parent_world):
        n = g.nodes[i]
        world = parent_world @ local_matrix(n)
        if n.mesh is not None:
            out.append((i, parent, world))
        for c in n.children or []:
            visit(c, i, world)

    for r in g.scenes[g.scene or 0].nodes:
        visit(r, None, np.eye(4))
    return out


def loader_primitives(g, mesh_index):
    """The primitives skin-tokens.cpp keeps: indexed triangle lists with POSITION."""
    kept, skipped = [], []
    for k, p in enumerate(g.meshes[mesh_index].primitives):
        if (p.mode in (None, 4)) and p.indices is not None and p.attributes.POSITION is not None:
            kept.append(p)
        else:
            skipped.append(k)
    return kept, skipped


# ---------- exact graft (numpy, same vertex order) ----------
def graft_exact(inp, raw_path, out):
    """Copy the skeleton, skin and JOINTS_0/WEIGHTS_0 of a skin-tokens.cpp output onto the
    textured input GLB, vertex for vertex. Raises GraftError when the two do not line up."""
    src = pygltflib.GLTF2().load(str(inp))
    raw = pygltflib.GLTF2().load(str(raw_path))
    src_blob, raw_blob = src.binary_blob(), raw.binary_blob()
    if not raw.skins or not raw.skins[0].joints:
        die(f"{raw_path} has no skin: skin-tokens.cpp wrote no rig")
    if src.skins:
        raise GraftError(f"{inp} is already rigged (has a skin)")
    rskin = raw.skins[0]
    rprims = [p for m in raw.meshes for p in m.primitives if p.attributes.JOINTS_0 is not None]
    if len(rprims) != 1:
        raise GraftError(f"expected one skinned primitive in {raw_path}, found {len(rprims)}")
    raw_pos = read_accessor(raw, raw_blob, rprims[0].attributes.POSITION)
    raw_joints = read_accessor(raw, raw_blob, rprims[0].attributes.JOINTS_0).astype(np.uint16)
    raw_weights = read_accessor(raw, raw_blob, rprims[0].attributes.WEIGHTS_0).astype(np.float32)

    instances = mesh_instances(src)
    if not instances:
        raise GraftError("input GLB has no mesh node in its scene")
    if len({src.nodes[i].mesh for i, _, _ in instances}) != len(instances):
        raise GraftError("a mesh is instanced by several nodes, so skin-tokens.cpp duplicated its vertices")
    total = sum(src.accessors[p.attributes.POSITION].count
                for i, _, _ in instances for p in loader_primitives(src, src.nodes[i].mesh)[0])
    if total != len(raw_pos):
        raise GraftError(f"input has {total} vertices but {raw_path} has {len(raw_pos)}")
    writer = BlobWriter(src)
    scene = src.scenes[src.scene or 0]
    diag = float(np.linalg.norm(raw_pos.max(0) - raw_pos.min(0))) or 1.0
    base = 0
    for node_index, parent, world in instances:
        node = src.nodes[node_index]
        kept, skipped = loader_primitives(src, node.mesh)
        if skipped:
            raise GraftError(f"mesh {node.mesh} primitives {skipped} are not indexed triangle lists "
                             "(skin-tokens.cpp ignored them)")
        baked = not np.allclose(world, np.eye(4), atol=1e-7)
        if baked and node.children:
            raise GraftError(f"node '{node.name}' has a transform and children")
        for p in kept:
            pos = read_accessor(src, src_blob, p.attributes.POSITION).astype(np.float64)
            count = len(pos)
            if base + count > len(raw_pos):
                raise GraftError(f"input has more vertices than {raw_path} ({base + count} > {len(raw_pos)})")
            world_pos = pos @ world[:3, :3].T + world[:3, 3]
            err = float(np.abs(world_pos - raw_pos[base:base + count]).max())
            if err > 1e-4 * diag:
                raise GraftError(f"vertex order differs from {raw_path} (max position error {err:.3g})")
            if baked:
                # The port baked the node's world transform into its vertices and placed the
                # joints in that space; do the same here and clear the node's transform.
                normal_m = np.linalg.inv(world[:3, :3]).T
                p.attributes.POSITION = writer.add(world_pos.astype(np.float32), pygltflib.FLOAT,
                                                   pygltflib.VEC3, pygltflib.ARRAY_BUFFER, minmax=True)
                if p.attributes.NORMAL is not None:
                    nrm = read_accessor(src, src_blob, p.attributes.NORMAL).astype(np.float64) @ normal_m.T
                    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
                    p.attributes.NORMAL = writer.add(nrm.astype(np.float32), pygltflib.FLOAT,
                                                     pygltflib.VEC3, pygltflib.ARRAY_BUFFER)
                if p.attributes.TANGENT is not None:
                    tan = read_accessor(src, src_blob, p.attributes.TANGENT).astype(np.float64)
                    xyz = tan[:, :3] @ world[:3, :3].T
                    xyz /= np.maximum(np.linalg.norm(xyz, axis=1, keepdims=True), 1e-12)
                    tan = np.hstack([xyz, tan[:, 3:4] * np.sign(np.linalg.det(world[:3, :3]))])
                    p.attributes.TANGENT = writer.add(tan.astype(np.float32), pygltflib.FLOAT,
                                                      pygltflib.VEC4, pygltflib.ARRAY_BUFFER)
            p.attributes.JOINTS_0 = writer.add(raw_joints[base:base + count], pygltflib.UNSIGNED_SHORT,
                                               pygltflib.VEC4, pygltflib.ARRAY_BUFFER)
            p.attributes.WEIGHTS_0 = writer.add(raw_weights[base:base + count], pygltflib.FLOAT,
                                                pygltflib.VEC4, pygltflib.ARRAY_BUFFER)
            base += count
        if baked:
            node.translation = node.rotation = node.scale = node.matrix = None
            if parent is not None:  # the parent chain carried the transform: make it a scene root
                src.nodes[parent].children.remove(node_index)
                scene.nodes.append(node_index)
        node.skin = 0
    if base != len(raw_pos):
        raise GraftError(f"input has {base} vertices but {raw_path} has {len(raw_pos)}")

    # Joint nodes: appended in skin.joints order, so JOINTS_0 values stay valid.
    raw_parent = {}
    for i, n in enumerate(raw.nodes):
        for c in n.children or []:
            raw_parent[c] = i
    offset = len(src.nodes)
    nmap = {rn: offset + k for k, rn in enumerate(rskin.joints)}
    for rn in rskin.joints:
        n = raw.nodes[rn]
        src.nodes.append(pygltflib.Node(name=n.name, translation=n.translation, rotation=n.rotation,
                                        scale=n.scale, matrix=n.matrix,
                                        children=[nmap[c] for c in (n.children or []) if c in nmap]))
    roots = [nmap[rn] for rn in rskin.joints if raw_parent.get(rn) not in nmap]
    ibm = read_accessor(raw, raw_blob, rskin.inverseBindMatrices).astype(np.float32)
    skeleton = nmap.get(rskin.skeleton, roots[0]) if rskin.skeleton is not None else roots[0]
    src.skins = [pygltflib.Skin(name=rskin.name or "SkinTokens rig", skeleton=skeleton,
                                joints=[nmap[rn] for rn in rskin.joints],
                                inverseBindMatrices=writer.add(ibm, pygltflib.FLOAT, pygltflib.MAT4))]
    scene.nodes.extend(roots)
    writer.finish()
    src.asset.generator = f"{src.asset.generator or 'glTF'}; skintokens.py (skin-tokens.cpp rig)"
    src.save(str(out), asset=src.asset)  # save() would otherwise stamp pygltflib's own asset


# ---------- Blender graft (nearest vertex, lib/graft_skin.py) ----------
def graft_blender(inp, raw_path, outdir, stem):
    """Runs lib/graft_skin.py headless; it writes OUTDIR/<stem>-rigged.glb and prints M3D_GRAFT.
    Blender exits 0 after a Python traceback, so the line is the success signal."""
    if not BLENDER.is_file():
        die(f"Blender not found at {BLENDER} (set M3D_BLENDER); needed for --graft blender")
    cmd = [str(BLENDER), "-b", "--factory-startup", "--python", str(GRAFT_SKIN), "--",
           str(inp), str(raw_path), str(outdir), "--name", stem]
    log("blender graft: " + " ".join(cmd))
    p = subprocess.run(cmd, capture_output=True, text=True)
    text = p.stdout + p.stderr
    for line in text.splitlines():
        if line.startswith(("[graft_skin]", "M3D_GRAFT", "graft_skin:", "Error", "Traceback")):
            print("  | " + line, file=sys.stderr, flush=True)
    m = re.search(r"^M3D_GRAFT .*$", text, re.M)
    if p.returncode != 0 or not m:
        die(f"Blender graft failed (exit {p.returncode}); last output:\n{text[-1500:]}")


# ---------- verify ----------
def texture_count(path):
    return len(pygltflib.GLTF2().load(str(path)).textures)


def verify(path, expect_textures):
    """Checks the rigged GLB: one skin, every skinned primitive has JOINTS_0/WEIGHTS_0 of the
    right length, joint indices in range, weights normalized, textures not lost.
    Returns (joints, skinned verts, total verts, textures, joint names)."""
    g = pygltflib.GLTF2().load(str(path))
    blob = g.binary_blob()
    if not g.skins or not g.skins[0].joints:
        die(f"{path} has no skin")
    joints = len(g.skins[0].joints)
    nodes = [n for n in g.nodes if n.skin is not None and n.mesh is not None]
    if not nodes:
        die(f"{path}: no mesh node references the skin")
    total = skinned = bad = 0
    for n in nodes:
        for p in g.meshes[n.mesh].primitives:
            if p.attributes.JOINTS_0 is None or p.attributes.WEIGHTS_0 is None:
                die(f"{path}: a skinned primitive lacks JOINTS_0/WEIGHTS_0")
            jn = read_accessor(g, blob, p.attributes.JOINTS_0)
            wt = read_accessor(g, blob, p.attributes.WEIGHTS_0).astype(np.float32)
            count = g.accessors[p.attributes.POSITION].count
            if len(jn) != count or len(wt) != count:
                die(f"{path}: JOINTS_0/WEIGHTS_0 length differs from POSITION ({len(jn)}, {len(wt)} vs {count})")
            if int(jn.max()) >= joints:
                die(f"{path}: JOINTS_0 index {int(jn.max())} outside {joints} joints")
            bad += int((np.abs(wt.sum(1) - 1) > 1e-3).sum())
            skinned += int((wt > 0).any(1).sum())
            total += count
    if bad:
        die(f"{path}: {bad} vertices have weights that do not sum to 1")
    if skinned == 0:
        die(f"{path}: no vertex carries a weight")
    textures = len(g.textures)
    if textures < expect_textures:
        die(f"{path} has {textures} textures, the input had {expect_textures}")
    return joints, skinned, total, textures, [g.nodes[j].name for j in g.skins[0].joints]


# ---------- run the CLI ----------
def run_cli(cmd, env):
    """Runs skintokens-cli, relays its output to stderr, logs a heartbeat with its RSS every
    minute (the CLI is silent for minutes; m3d stops a stage after 15 silent minutes), and
    reports whether ggml's Metal backend announced itself. Returns (exit code, device, peak GB)."""
    state = {"metal": False}
    p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    def relay():
        for line in p.stdout:
            if "ggml_metal" in line:
                state["metal"] = True
            print("  | " + line.rstrip(), file=sys.stderr, flush=True)

    threading.Thread(target=relay, daemon=True).start()
    t0, last, peak = time.time(), time.time(), 0.0
    try:
        while p.poll() is None:
            time.sleep(1)
            if time.time() - last >= 60:
                last = time.time()
                rss = subprocess.run(["ps", "-o", "rss=", "-p", str(p.pid)], capture_output=True, text=True).stdout
                gb = int(rss or 0) / 1e6
                peak = max(peak, gb)
                log(f"rig running {time.time() - t0:.0f}s, CLI rss {gb:.1f} GB (peak {peak:.1f})")
    finally:
        if p.poll() is None:  # SIGTERM from m3d's stall watchdog, or Ctrl-C
            p.terminate()
            try:
                p.wait(30)
            except subprocess.TimeoutExpired:
                p.kill()
    return p.returncode, ("metal" if state["metal"] else "cpu"), peak


def _raise_on_term(signum, frame):
    raise KeyboardInterrupt(f"signal {signum}")  # so `finally` stops the CLI (see mlxserve3d.py)


def main():
    signal.signal(signal.SIGTERM, _raise_on_term)
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inp"); ap.add_argument("outdir")
    ap.add_argument("--device", default="auto", choices=sorted(DEVICE_ARG),
                    help="metal is passed to the CLI as auto (its first GGML GPU device)")
    ap.add_argument("--seed", type=int, default=0,
                    help="recorded only: skintokens-cli 0.1 has no seed flag (fixed seed 0, deterministic)")
    ap.add_argument("--dry-run", action="store_true", help="print the commands, run nothing")
    ap.add_argument("--postprocess", action="store_true",
                    help="upstream voxel_skin surface-locality heuristic (default: raw learned weights)")
    ap.add_argument("--beams", type=int, help="TokenRig beams (CLI default 10; KV cache grows with it)")
    ap.add_argument("--max-tokens", type=int, help="TokenRig token budget (CLI default 2048)")
    ap.add_argument("--temperature", type=float, help="TokenRig sampling temperature (CLI default 1.0)")
    ap.add_argument("--graft", default="auto", choices=["auto", "exact", "blender"],
                    help="exact: same vertex order, bit-exact weights, input otherwise untouched; "
                         "blender: lib/graft_skin.py nearest-vertex transfer; auto: exact, then blender")
    ap.add_argument("--from-raw", metavar="RAW.glb",
                    help="skip the CLI: graft this existing skin-tokens.cpp output onto IN.glb")
    a = ap.parse_args()

    inp = Path(a.inp)
    if inp.suffix.lower() != ".glb" or not inp.is_file():
        die(f"input must be an existing .glb: {inp}")
    outdir = Path(a.outdir)
    out = outdir / f"{inp.stem}-rigged.glb"
    raw = Path(a.from_raw) if a.from_raw else outdir / f"{inp.stem}-skintokens-raw.glb"
    if a.seed:
        log(f"--seed {a.seed} recorded only: skintokens-cli has no seed flag (it always samples with seed 0)")

    cmd = [str(CLI), "rig", str(MODEL_DIR), str(inp), str(raw), "--device", DEVICE_ARG[a.device]]
    if a.postprocess:
        cmd.append("--postprocess")
    for flag, value in (("--beams", a.beams), ("--max-tokens", a.max_tokens), ("--temperature", a.temperature)):
        if value is not None:
            cmd += [flag, str(value)]
    env = {**os.environ, "SKINTOKENS_PROFILE": "1"}  # per-phase timings in the stage log

    problems = []
    if not CLI.is_file():
        problems.append(f"skintokens-cli not found at {CLI}; build it: {BUILD_FIX}")
    missing = [f for f in GGUFS if not (MODEL_DIR / f).is_file()]
    if missing:
        problems.append(f"weights missing in {MODEL_DIR} ({', '.join(missing)}); fetch them: {WEIGHTS_FIX}")
    if a.dry_run:
        print(f"would run: SKINTOKENS_PROFILE=1 {' '.join(cmd)}")
        print(f"then graft ({a.graft}) the skeleton, skin and JOINTS_0/WEIGHTS_0 of {raw} onto {inp} -> {out}, "
              f"and verify; the blender graft runs: {BLENDER} -b --factory-startup --python {GRAFT_SKIN} -- "
              f"{inp} {raw} {outdir} --name {inp.stem}")
        for p in problems:
            print(f"warning: {p}")
        return
    outdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    device, peak = a.device, 0.0
    if a.from_raw:
        if not raw.is_file():
            die(f"--from-raw file not found: {raw}")
        log(f"grafting existing {raw} (CLI not run)")
    else:
        if problems:
            die("\n".join(problems))
        log(f"rig {inp.name} on {a.device} (CLI device {DEVICE_ARG[a.device]}) …")
        log(" ".join(cmd))
        code, device, peak = run_cli(cmd, env)
        if code != 0 or not raw.is_file():
            die(f"skintokens-cli exited {code} after {time.time() - t0:.0f}s"
                + ("" if raw.is_file() else f"; no output at {raw}"))
        if a.device == "metal" and device != "metal":
            log("warning: asked for Metal but ggml's Metal backend never reported in; the run may have used the CPU")
        log(f"skin-tokens.cpp done in {time.time() - t0:.0f}s (peak CLI rss {peak:.1f} GB) → {raw}")

    how = a.graft
    if how in ("auto", "exact"):
        try:
            graft_exact(inp, raw, out)
            how = "exact"
        except GraftError as e:
            if a.graft == "exact":
                die(f"exact graft not possible: {e}")
            log(f"exact graft not possible ({e}); falling back to the Blender nearest-vertex graft")
            how = "blender"
    if how == "blender":
        graft_blender(inp, raw, outdir, inp.stem)
    joints, skinned, total, textures, names = verify(out, texture_count(inp))
    log(f"{joints} joints: {', '.join(names)}")
    log(f"{skinned}/{total} vertices weighted, {textures} textures kept; rig grafted ({how}) → {out}")
    print(f"M3D_RIG joints={joints} skinned_verts={skinned} verts={total} textures={textures} graft={how} "
          f"device={device} seconds={time.time() - t0:.0f} peak_rss_gb={peak:.1f} out={out}")


if __name__ == "__main__":
    main()

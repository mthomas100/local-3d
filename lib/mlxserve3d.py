#!/usr/bin/env python3
"""mlxserve3d — Hunyuan3D-2.1 (real 2.1 shape) + PBR paint + UniRig auto-rig through mlx-serve.

usage: mlxserve3d.py IMAGE OUT.glb [--rig] [--no-texture] [--seed N] [--steps N] [--octree N]
                     [--port 11234] [--keep-server]

Why (2026-10-02): mlx-serve (ddalcu, Homebrew tap) is the
only local route found to the real Hunyuan3D 2.1 shape model and to UniRig auto-rigging
(skeleton + skin) on Apple silicon, through one HTTP call: POST /v1/3d/generations with
{"texture": true, "rig": true} returns a textured, rigged GLB (model card,
ddalcu/Hunyuan3D-2.1-MLX-Serve-8bit). Like comfy3d.py, this starts a private server bound
to 127.0.0.1 (the default is 0.0.0.0, open to the network), makes the call, writes the
GLB and stops the server so nothing stays resident on the shared GPU.

The weights are read from ~/.mlx-serve/models/ddalcu/Hunyuan3D-2.1-MLX-Serve-8bit, a
symlink to the Hugging Face cache copy (no second download).
The app sends a cutout composited on white; so does this (from an RGBA input).
"""
import argparse
import base64
import io
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

MODEL = "ddalcu/Hunyuan3D-2.1-MLX-Serve-8bit"
MODEL_DIR = Path.home() / ".mlx-serve" / "models"


def http(port, path, data=None, timeout=30):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
                                 data=json.dumps(data).encode() if data is not None else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def up(port):
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3)
        return True
    except Exception:
        return False


def on_white_png_b64(path):
    """RGBA → composited on white (what the MLX Core app sends); RGB passes through."""
    from PIL import Image  # worker venv has Pillow; run this script with it
    im = Image.open(path)
    if im.mode in ("RGBA", "LA"):
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im.convert("RGBA"), mask=im.convert("RGBA").split()[3])
        im = bg
    buf = io.BytesIO()
    im.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _raise_on_term(signum, frame):
    raise KeyboardInterrupt(f"signal {signum}")  # so `finally` stops the server (see comfy3d.py)


def main():
    signal.signal(signal.SIGTERM, _raise_on_term)
    ap = argparse.ArgumentParser()
    ap.add_argument("image"); ap.add_argument("out")
    ap.add_argument("--rig", action="store_true", help="UniRig skeleton + skin weights in the GLB")
    ap.add_argument("--no-texture", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, help="shape steps (server default if omitted)")
    ap.add_argument("--octree", type=int, help="octree_resolution (server default if omitted)")
    ap.add_argument("--port", type=int, default=11234)
    ap.add_argument("--keep-server", action="store_true")
    a = ap.parse_args()

    if not (MODEL_DIR / MODEL / "config.json").exists():
        sys.exit(f"mlxserve3d: weights not found at {MODEL_DIR / MODEL}")
    body = {"model": MODEL, "image": on_white_png_b64(a.image), "seed": a.seed,
            "texture": not a.no_texture, "rig": a.rig}
    if a.steps:
        body["steps"] = a.steps
    if a.octree:
        body["octree_resolution"] = a.octree

    server = None
    if not up(a.port):
        log = open(str(a.out) + ".mlxserve.log", "w")
        server = subprocess.Popen(["mlx-serve", "--serve", "--model-dir", str(MODEL_DIR),
                                   "--host", "127.0.0.1", "--port", str(a.port)],
                                  stdout=log, stderr=subprocess.STDOUT)
        for _ in range(120):
            if up(a.port):
                break
            if server.poll() is not None:
                sys.exit(f"mlxserve3d: mlx-serve exited during start; see {a.out}.mlxserve.log")
            time.sleep(1)
    try:
        t0 = time.time()
        print(f"[mlxserve3d] POST /v1/3d/generations texture={body['texture']} rig={a.rig}",
              file=sys.stderr, flush=True)
        try:
            r = http(a.port, "/v1/3d/generations", body, timeout=3600)
        except OSError as e:
            sys.exit(f"mlxserve3d: request failed ({e}); server log: {a.out}.mlxserve.log")
        if r.get("format") != "glb" or not r.get("data"):
            sys.exit(f"mlxserve3d: unexpected response: {json.dumps(r)[:800]}")
        Path(a.out).write_bytes(base64.b64decode(r["data"]))
        if a.rig:
            # 2026-10-02: mlx-serve 26.10.1 accepted "rig": true but returned a GLB with no
            # skin (its server code had no rig stage yet; the model card was ahead of it).
            # Fail loudly rather than hand back an unrigged model as rigged.
            import pygltflib
            g = pygltflib.GLTF2().load(a.out)
            if not g.skins:
                sys.exit(f"mlxserve3d: asked for a rig but the GLB has no skin/joints "
                         f"(mlx-serve ignored \"rig\": true). Unrigged model kept at {a.out}")
        print(f"[mlxserve3d] done in {time.time() - t0:.0f}s → {a.out}", file=sys.stderr)
    finally:
        if server and not a.keep_server:
            for sig, wait in ((signal.SIGINT, 15), (signal.SIGTERM, 30)):  # never SIGKILL
                if server.poll() is not None:
                    break
                server.send_signal(sig)
                try:
                    server.wait(wait)
                except subprocess.TimeoutExpired:
                    pass


if __name__ == "__main__":
    main()

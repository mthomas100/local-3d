#!/usr/bin/env python3
"""comfy3d — run ComfyUI's native Pixal3D / TRELLIS.2 image-to-3D template headless.

usage: comfy3d.py IMAGE OUT.glb [--model pixal3d|trellis2] [--seed N] [--faces N]
                  [--comfy DIR] [--port 8189] [--keep-server]

Why (2026-09-26): ComfyUI v0.34+ ships TRELLIS.2 and Pixal3D as core nodes
with MIT weights, no gated downloads and no CUDA extensions, and one user has
run them on an M5 Max 128 GB (ComfyUI issue #16017). It is the licence-clean
route for game assets: the Hunyuan licence excludes the EU, UK and South Korea.
ComfyUI has no CLI, so this converts the shipped UI template to an API
prompt, starts a private server, submits, waits, copies the GLB and stops
the server so no model stays resident (this Mac runs one GPU job at a time).

Known Mac issue (#16017): RemeshMesh / DecimateMesh can crash on MPS with
multi-million-face meshes; the fix there was forcing those nodes to CPU. If
it bites, the error is in OUT.glb.log and the next step is that patch.
"""
import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

TEMPLATE = "3d_pixal3d_trellis2_image_to_model.json"
FACE_KEY = "target_face_count"  # DecimateMesh input name (ComfyUI v0.37, 2026-09-26)
SKIP_TYPES = {"Note", "MarkdownNote"}
NON_INPUT_WIDGETS = {"control_after_generate"}


def find_template(comfy):
    hits = list(Path(comfy, ".venv").glob(f"lib/python3*/site-packages/comfyui_workflow_templates_json/templates/{TEMPLATE}"))
    if not hits:
        sys.exit(f"comfy3d: template {TEMPLATE} not found under {comfy}/.venv")
    return json.loads(hits[0].read_text())


def to_api(wf):
    """UI workflow → API prompt. Widget values come from widgets_values_named
    (present in current templates); a linked input overrides its widget."""
    links = {l[0]: (str(l[1]), l[2]) for l in wf["links"]}
    api = {}
    for n in wf["nodes"]:
        if n["type"] in SKIP_TYPES or n.get("mode", 0) in (2, 4):  # 2 muted, 4 bypassed
            continue
        inputs = {k: v for k, v in (n.get("widgets_values_named") or {}).items() if k not in NON_INPUT_WIDGETS}
        for i in n.get("inputs", []):
            if i.get("link") is not None:
                inputs[i["name"]] = list(links[i["link"]])
        api[str(n["id"])] = {"class_type": n["type"], "inputs": inputs}
    return api


def normalize(api, oi):
    """Reconcile the template with the running ComfyUI's node definitions:
    drop UI-only widgets ('upload', a mis-named 'fixed'), keep dotted
    DynamicCombo sub-inputs ('sign_mode.qef' is the API form), and fill
    required inputs newer than the template (MoGe 'refine_steps', 2026-09)
    with their defaults."""
    for v in api.values():
        spec = oi[v["class_type"]]["input"]
        known = {**spec.get("required", {}), **spec.get("optional", {})}
        v["inputs"] = {k: x for k, x in v["inputs"].items() if k in known or "." in k}
        for k, s_ in spec.get("required", {}).items():
            if k not in v["inputs"] and len(s_) > 1 and isinstance(s_[1], dict) and "default" in s_[1]:
                v["inputs"][k] = s_[1]["default"]
    return api


def by_type(api, t):
    return [k for k, v in api.items() if v["class_type"] == t]


def http(port, path, data=None):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
                                 data=json.dumps(data).encode() if data is not None else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def up(port):
    try:
        http(port, "/system_stats")
        return True
    except Exception:
        return False


def _raise_on_term(signum, frame):
    # SIGTERM skips `finally` by default. On 2026-09-26 a watchdog SIGTERM
    # left the server orphaned, holding ~70 GB and stalling the film rig for
    # 12 minutes. Turning it into an exception makes the shutdown below run.
    raise KeyboardInterrupt(f"signal {signum}")


def stop_server(server):
    """Graceful stop that also works mid-node: SIGINT only reaches ComfyUI's
    main thread while a worker thread keeps baking, so escalate to SIGTERM
    (which exited in 2 s when tested), never to SIGKILL (can leak Metal memory)."""
    for sig, wait in ((signal.SIGINT, 15), (signal.SIGTERM, 30)):
        if server.poll() is not None:
            return
        server.send_signal(sig)
        try:
            server.wait(wait)
        except subprocess.TimeoutExpired:
            pass
    if server.poll() is None:
        print(f"comfy3d: server pid {server.pid} still running; stop it by hand", file=sys.stderr)


def main():
    signal.signal(signal.SIGTERM, _raise_on_term)
    ap = argparse.ArgumentParser()
    ap.add_argument("image"); ap.add_argument("out")
    ap.add_argument("--model", default="pixal3d", choices=["pixal3d", "trellis2"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--faces", type=int, help="DecimateMesh target (template default 700000)")
    ap.add_argument("--comfy", default=os.path.join(os.environ.get("M3D_REPOS", os.path.expanduser("~/repos")), "ComfyUI"))
    ap.add_argument("--port", type=int, default=8189)
    ap.add_argument("--keep-server", action="store_true")
    # Lighter defaults than the template (1536 / 768 / 64) after the 12M-face hang
    # of 2026-09-26; pass the template values to reproduce it exactly.
    ap.add_argument("--upsample", type=int, default=1024, help="Trellis2UpsampleStage target (1024-2048)")
    ap.add_argument("--remesh", type=int, default=512, help="RemeshMesh resolution (template 768)")
    ap.add_argument("--ao-samples", type=int, default=16, help="BakeAmbientOcclusion rays per texel (template 64)")
    a = ap.parse_args()

    api = to_api(find_template(a.comfy))
    # input image: copy into ComfyUI/input under a unique name
    name = f"m3d-{int(time.time())}-{Path(a.image).name}"
    shutil.copy(a.image, Path(a.comfy, "input", name))
    for k in by_type(api, "LoadImage"):
        api[k]["inputs"]["image"] = name
    # the template's one PrimitiveBoolean drives every Pixal3D/TRELLIS.2 switch
    for k in by_type(api, "PrimitiveBoolean"):
        api[k]["inputs"]["value"] = a.model == "trellis2"
    # The template loads both diffusion models; the lazy switch only executes
    # the selected one, but validation still wants every filename to exist.
    # If the unused model is not downloaded, point its loader at the used one.
    use = "pixal3d_int8_convrot.safetensors" if a.model == "pixal3d" else "trellis_2_int8_convrot.safetensors"
    for k in by_type(api, "UNETLoader"):
        f = api[k]["inputs"].get("unet_name")
        if f != use and not Path(a.comfy, "models", "diffusion_models", f).exists():
            api[k]["inputs"]["unet_name"] = use
    for k in by_type(api, "KSampler"):
        # shift every stage by the same amount so the template's per-stage seeds stay distinct
        api[k]["inputs"]["seed"] = api[k]["inputs"]["seed"] - 42 + a.seed
    if a.faces:
        for k in by_type(api, "DecimateMesh"):
            api[k]["inputs"][FACE_KEY] = a.faces
    for k in by_type(api, "Trellis2UpsampleStage"):
        api[k]["inputs"]["target_resolution"] = a.upsample
    for k in by_type(api, "RemeshMesh"):
        api[k]["inputs"]["resolution"] = a.remesh
    for k in by_type(api, "BakeAmbientOcclusion"):
        api[k]["inputs"]["samples"] = a.ao_samples
    prefix = f"m3d/{Path(a.out).stem}"
    for k in by_type(api, "Save3DAdvanced"):
        api[k]["inputs"]["filename_prefix"] = prefix

    server = None
    if not up(a.port):
        log = open(str(a.out) + ".comfy.log", "w")
        server = subprocess.Popen([f"{a.comfy}/.venv/bin/python", "main.py", "--port", str(a.port),
                                   "--listen", "127.0.0.1", "--disable-auto-launch",
                                   "--extra-model-paths-config", str(Path(__file__).resolve().parent.parent / "comfy_nodes" / "extra_paths.yaml")],
                                  cwd=a.comfy, stdout=log, stderr=subprocess.STDOUT,
                                  env={**os.environ, "PYTORCH_ENABLE_MPS_FALLBACK": "1"})
        for _ in range(180):
            if up(a.port):
                break
            if server.poll() is not None:
                sys.exit(f"comfy3d: ComfyUI exited during start; see {a.out}.comfy.log")
            time.sleep(1)
    try:
        t0 = time.time()
        api = normalize(api, http(a.port, "/object_info"))
        r = http(a.port, "/prompt", {"prompt": api})
        pid = r["prompt_id"]
        print(f"[comfy3d] {a.model} queued {pid}", file=sys.stderr, flush=True)
        while True:
            try:
                h = http(a.port, f"/history/{pid}")
            except OSError as e:  # URLError is an OSError
                # 2026-10-01: when another session stopped the server mid-job this
                # surfaced as a raw urllib traceback; say what happened instead.
                sys.exit(f"comfy3d: ComfyUI stopped answering mid-job ({e}); the server died or was "
                         f"stopped. Its log: {a.out}.comfy.log")
            if pid in h:
                st = h[pid].get("status", {})
                if st.get("status_str") == "error" or (st.get("completed") is False and st.get("messages")
                                                        and any(m[0] == "execution_error" for m in st["messages"])):
                    err = next((m[1] for m in st.get("messages", []) if m[0] == "execution_error"), st)
                    sys.exit(f"comfy3d: execution error: {json.dumps(err)[:2000]}")
                if st.get("completed"):
                    break
            time.sleep(3)
        outs = sorted(Path(a.comfy, "output", "m3d").glob(f"{Path(a.out).stem}*.glb"), key=os.path.getmtime)
        if not outs:
            sys.exit(f"comfy3d: finished but no GLB under output/m3d/{Path(a.out).stem}*")
        shutil.copy(outs[-1], a.out)
        print(f"[comfy3d] done in {time.time() - t0:.0f}s → {a.out}", file=sys.stderr)
    finally:
        if server and not a.keep_server:
            try:
                http(a.port, "/interrupt", {})
            except Exception:
                pass
            stop_server(server)


if __name__ == "__main__":
    main()

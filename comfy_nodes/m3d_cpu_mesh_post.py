"""Run ComfyUI's heavy mesh post-processing nodes on the CPU instead of MPS.

Loaded by comfy3d.py through ComfyUI's extra-paths config (custom_nodes:), so
the ComfyUI clone stays unedited. Why (2026-10-02):
- ComfyUI issue #16017 (an M5 Max 128 GB user): RemeshMesh and DecimateMesh
  fail on MPS with multi-million-face meshes (scatter/index_put_ indices out of
  range); forcing their device to CPU fixed both and was not slower.
- Here, on 2026-09-26 a Pixal3D run on a 12M-face mesh sat silent ~12 h after
  the colour bake, in the normal-map / ambient-occlusion bakes against the
  high-poly remesh.
The nodes ask comfy.model_management.get_torch_device() inside their helpers,
so swapping that function for the duration of these nodes' execute() moves
just these nodes to the CPU; the diffusion stages stay on MPS.
"""
import logging

import comfy.model_management as mm
import torch

CPU_NODES = {"RemeshMesh", "DecimateMesh", "BakeNormalMapFromMesh", "BakeAmbientOcclusion"}
_orig_device = mm.get_torch_device
patched = []

try:
    from comfy_extras import nodes_mesh_postprocess as pp

    for obj in list(vars(pp).values()):
        if not (isinstance(obj, type) and hasattr(obj, "define_schema") and hasattr(obj, "execute")):
            continue
        try:
            nid = obj.define_schema().node_id
        except Exception:
            continue
        if nid not in CPU_NODES:
            continue
        func = obj.execute.__func__

        def make(f):
            def wrapped(cls, *args, **kwargs):
                mm.get_torch_device = lambda: torch.device("cpu")
                try:
                    return f(cls, *args, **kwargs)
                finally:
                    mm.get_torch_device = _orig_device
            return classmethod(wrapped)

        obj.execute = make(func)
        patched.append(nid)
    # logging, not print: print to a redirected stdout stays buffered until exit
    logging.info(f"[m3d_cpu_mesh_post] CPU for: {', '.join(sorted(patched))}")
except Exception as e:  # never break ComfyUI start-up over this patch
    logging.warning(f"[m3d_cpu_mesh_post] not applied: {e}")

NODE_CLASS_MAPPINGS = {}

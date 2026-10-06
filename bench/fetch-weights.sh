#!/usr/bin/env bash
# Weight downloads for the m3d spike, in the order the GPU tests need them.
# 2026-09-26: the link gave ~10 MB/s, so the order matters. Queue rewritten after research found ComfyUI's native
# Pixal3D/TRELLIS.2 (MIT, ungated) and mlx-serve's Hunyuan 2.1 + UniRig; the
# 16 GB original TRELLIS.2 weights for trellis2mlx wait on DINOv3 access.
# hf resumes .incomplete files, so re-running this is safe.
set -u
R=${M3D_REPOS:-$HOME/repos}
W=$R/Hunyuan3D-Swift/weights
C=$R/ComfyUI/models
get() { local t0=$(date +%s); if "$@" >/dev/null 2>&1; then echo "$(date +%H:%M) done  $* ($(( $(date +%s)-t0 ))s)"; else echo "$(date +%H:%M) FAIL  $*"; fi; }
# an orphaned shape-small download may still be running from the first queue
while pgrep -f "hunyuan3d-mlx-shape-small --local-dir" >/dev/null; do sleep 10; done
get hf download zimengxiong/hunyuan3d-mlx-shape-small --local-dir $W/shape-small
get hf download zimengxiong/hunyuan3d-mlx-paint-large --local-dir $W/paint-large
get hf download filipstrand/Z-Image-Turbo-mflux-4bit
get hf download Comfy-Org/Pixal3D diffusion_models/pixal3d_int8_convrot.safetensors \
    vae/trellis_2_shape_vae_bf16.safetensors vae/trellis_2_texture_vae_bf16.safetensors \
    clip_vision/dino_v3_L_naf_fp32.safetensors --local-dir $C
get hf download Comfy-Org/BiRefNet background_removal/birefnet.safetensors --local-dir $C
get hf download Comfy-Org/MoGe geometry_estimation/moge_2_vitl_normal_fp16.safetensors --local-dir $C
get hf download Comfy-Org/TRELLIS.2 diffusion_models/trellis_2_int8_convrot.safetensors --local-dir $C
get hf download ddalcu/Hunyuan3D-2.1-MLX-Serve-8bit
get hf download zimengxiong/hunyuan3d-mlx-shape-large --local-dir $W/shape-large
echo "$(date +%H:%M) all attempted"

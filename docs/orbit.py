"""Orbit render of an animated m3d clip GLB: the camera circles once while the clip loops.
Cycles on the CPU only (no GPU), for the README demo GIF (docs/media/teddy-orbit.gif, 2026-10-05).

blender -b --factory-startup --python orbit.py -- clip.glb OUTDIR [--frames 72] [--step 2] [--size 420] [--samples 16]
    [--only N] (render just frame N, for timing)
"""
import math
import sys

import bpy
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:]
src, outdir = argv[0], argv[1]
opt = lambda k, d: type(d)(argv[argv.index(k) + 1]) if k in argv else d
n, step, size, samples, only = opt("--frames", 72), opt("--step", 2), opt("--size", 420), opt("--samples", 16), opt("--only", -1)

bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
sc.render.engine = "CYCLES"
sc.cycles.device = "CPU"
sc.cycles.samples = samples
sc.cycles.use_denoising = True
sc.render.resolution_x = sc.render.resolution_y = size
sc.render.fps = 30
bpy.ops.import_scene.gltf(filepath=src)

acts = [a for a in bpy.data.actions]
lo_f = min(int(a.frame_range[0]) for a in acts) if acts else 1
hi_f = max(int(a.frame_range[1]) for a in acts) if acts else 1
clip = max(1, hi_f - lo_f)
print(f"ORBIT clip frames {lo_f}..{hi_f}")

sc.frame_set(lo_f)
objs = [o for o in sc.objects if o.type == "MESH"]
pts = [o.matrix_world @ Vector(c) for o in objs for c in o.bound_box]
lo = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
hi = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
center, radius = (lo + hi) / 2, (hi - lo).length / 2
center.z += radius * 0.02

world = bpy.data.worlds.new("w"); sc.world = world; world.use_nodes = True
world.node_tree.nodes["Background"].inputs[0].default_value = (0.86, 0.86, 0.88, 1)
world.node_tree.nodes["Background"].inputs[1].default_value = 0.55
key = bpy.data.objects.new("key", bpy.data.lights.new("key", "SUN")); sc.collection.objects.link(key)
key.data.energy = 3.5; key.rotation_euler = (math.radians(45), 0, math.radians(35))
# a floor that only catches the shadow
bpy.ops.mesh.primitive_plane_add(size=radius * 12, location=(center.x, center.y, lo.z))
floor = bpy.context.object; floor.is_shadow_catcher = True
sc.render.film_transparent = False
sc.view_settings.view_transform = "Standard"

cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam")); sc.collection.objects.link(cam)
sc.camera = cam; cam.data.lens = 50
dist = radius * 2.3
for i in range(n):
    if only >= 0 and i != only:
        continue
    sc.frame_set(lo_f + (i * step) % clip)
    a = 2 * math.pi * i / n  # glTF -Z forward lands on Blender -Y: start in front
    cam.location = center + Vector((dist * math.sin(a), -dist * math.cos(a), radius * 0.55))
    cam.rotation_euler = (center - cam.location).to_track_quat("-Z", "Y").to_euler()
    sc.render.filepath = f"{outdir}/f{i:03d}.png"
    bpy.ops.render.render(write_still=True)
print("ORBIT done")

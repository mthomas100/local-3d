"""Four-view preview of a GLB, rendered headless in Blender (Cycles, CPU by default).

Run as: blender -b --factory-startup --python turntable.py -- in.glb out.png [--size 512] [--device CPU|METAL]
Writes one 2x2 sheet: front, right, back, three-quarter. Used to judge
generated meshes by eye (2026-09-26): the concept image only shows one side,
so the back and sides are where image-to-3D models fail.
"""
import math
import os
import sys

import bpy
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:]
src, out = argv[0], argv[1]
size = int(argv[argv.index("--size") + 1]) if "--size" in argv else 512
device = argv[argv.index("--device") + 1] if "--device" in argv else "CPU"

bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
sc.render.engine = "CYCLES"
sc.cycles.samples = 24
sc.cycles.use_denoising = True
if device == "METAL":
    p = bpy.context.preferences.addons["cycles"].preferences
    p.compute_device_type = "METAL"; p.get_devices()
    for d in p.devices: d.use = d.type == "METAL"
    sc.cycles.device = "GPU"
sc.render.resolution_x = sc.render.resolution_y = size
sc.render.film_transparent = False
bpy.ops.import_scene.gltf(filepath=src)
objs = [o for o in sc.objects if o.type == "MESH"]
pts = [o.matrix_world @ Vector(c) for o in objs for c in o.bound_box]
lo = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
hi = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
center, radius = (lo + hi) / 2, (hi - lo).length / 2

world = bpy.data.worlds.new("w"); sc.world = world; world.use_nodes = True
world.node_tree.nodes["Background"].inputs[0].default_value = (0.8, 0.8, 0.82, 1)
world.node_tree.nodes["Background"].inputs[1].default_value = 1.0
key = bpy.data.objects.new("key", bpy.data.lights.new("key", "SUN")); sc.collection.objects.link(key)
key.data.energy = 3; key.rotation_euler = (math.radians(50), 0, math.radians(30))

cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam")); sc.collection.objects.link(cam)
sc.camera = cam; cam.data.lens = 50
dist = radius * 3.2
frames = []
# glTF imports Y-up as Blender Z-up; the model faces -Y in Blender.
for i, az in enumerate((0, 90, 180, 45)):
    a = math.radians(az)
    cam.location = center + Vector((dist * math.sin(a), -dist * math.cos(a), radius * 0.6))
    cam.rotation_euler = (center - cam.location).to_track_quat("-Z", "Y").to_euler()
    f = f"{out}.{i}.png"; sc.render.filepath = f
    bpy.ops.render.render(write_still=True); frames.append(f)

sheet = bpy.data.images.new("sheet", size * 2, size * 2)
import numpy as np
buf = np.zeros((size * 2, size * 2, 4), np.float32)
for i, f in enumerate(frames):
    im = bpy.data.images.load(f); px = np.array(im.pixels[:], np.float32).reshape(size, size, 4)
    r, c = 1 - i // 2, i % 2  # Blender image origin is bottom-left
    buf[r * size:(r + 1) * size, c * size:(c + 1) * size] = px
    os.remove(f)
sheet.pixels = buf.ravel(); sheet.filepath_raw = out; sheet.file_format = "PNG"; sheet.save()
print(f"M3D_TURNTABLE {out}")

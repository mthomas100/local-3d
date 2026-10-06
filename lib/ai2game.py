# ai2game.py — headless Blender: raw AI GLB -> low-poly, re-UV'd, re-baked PBR asset -> GLB + USDZ.
# Written by a research agent for the local-3d spike and tested 2026-09-26 on this Mac
# (CPU bake; 95k-tri test mesh -> 3-5 s; every USDZ passed `usdchecker --arkit`).
# Usage:
#   blender -b --factory-startup --python ai2game.py -- IN.glb OUTDIR \
#       --faces 30000 --tex 2048 --mode decimate|voxel|quadriflow --device METAL|CPU --samples 16
# Tested 2026-09-26 with Blender 5.2.2 LTS on macOS 27 (CPU device only in the test run).
import bpy, bmesh, sys, os, math, argparse

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("inp"); ap.add_argument("outdir")
ap.add_argument("--faces", type=int, default=30000)      # target triangle count for LOD0
ap.add_argument("--tex", type=int, default=2048)         # baked texture size
ap.add_argument("--mode", default="decimate", choices=["decimate", "voxel", "quadriflow"])
ap.add_argument("--device", default="METAL", choices=["METAL", "CPU"])
ap.add_argument("--samples", type=int, default=16)
ap.add_argument("--cage", type=float, default=0.02)      # cage extrusion as fraction of bbox diagonal
ap.add_argument("--lods", default="0.5,0.25")           # extra LOD ratios relative to LOD0
ap.add_argument("--glass", default="")                    # tint of the glass to find: green|blue|cyan|red|yellow|amber|magenta or a hue in degrees
ap.add_argument("--glass-alpha", type=float, default=0.45)  # opacity of the glass material (0.35 read as clear, 2026-10-02)
a = ap.parse_args(argv)
os.makedirs(a.outdir, exist_ok=True)
name = os.path.splitext(os.path.basename(a.inp))[0]

# ---------- 0. clean scene, Cycles on Metal ----------
bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
sc.render.engine = "CYCLES"
if a.device == "METAL":
    cp = bpy.context.preferences.addons["cycles"].preferences
    cp.compute_device_type = "METAL"
    cp.get_devices()
    for dev in cp.devices:
        dev.use = dev.type == "METAL"
    sc.cycles.device = "GPU"
else:
    sc.cycles.device = "CPU"
sc.cycles.samples = a.samples
sc.render.bake.margin = 8

# ---------- 1. import + join + weld ----------
bpy.ops.import_scene.gltf(filepath=a.inp, merge_vertices=True)
meshes = [o for o in sc.objects if o.type == "MESH"]
bpy.ops.object.select_all(action="DESELECT")
for o in meshes: o.select_set(True)
bpy.context.view_layer.objects.active = meshes[0]
if len(meshes) > 1: bpy.ops.object.join()
high = bpy.context.view_layer.objects.active
high.name = "high"
bpy.ops.object.parent_clear(type="CLEAR_KEEP_TRANSFORM")
bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
bm = bmesh.new(); bm.from_mesh(high.data)
bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-5)   # AI meshes are often unwelded soup
bm.to_mesh(high.data); bm.free()
nf = sum(len(p.vertices) - 2 for p in high.data.polygons)
print(f"[ai2game] high tris={nf}")
diag = max(high.dimensions) * math.sqrt(3)

# ---------- 2. build low-poly ----------
low = high.copy(); low.data = high.data.copy(); low.name = name
sc.collection.objects.link(low)
bpy.ops.object.select_all(action="DESELECT"); low.select_set(True)
bpy.context.view_layer.objects.active = low
if a.mode == "decimate":
    m = low.modifiers.new("dec", "DECIMATE"); m.decimate_type = "COLLAPSE"
    m.ratio = min(1.0, a.faces / max(nf, 1)); m.use_collapse_triangulate = True
    bpy.ops.object.modifier_apply(modifier=m.name)
elif a.mode == "voxel":
    low.data.remesh_voxel_size = diag / 400          # watertight, uniform; then decimate to budget
    bpy.ops.object.voxel_remesh()
    t = sum(len(p.vertices) - 2 for p in low.data.polygons)
    m = low.modifiers.new("dec", "DECIMATE"); m.ratio = min(1.0, a.faces / max(t, 1))
    bpy.ops.object.modifier_apply(modifier=m.name)
else:  # quadriflow: needs a manifold, consistently-oriented input of modest size.
    # Tested 2026-09-26 (Blender 5.2.2): raw AI soup always fails; after voxel remesh, QuadriFlow
    # passes or CANCELs ("mesh needs to be manifold...") depending on tiny voxel-size changes
    # (62662 faces -> FINISHED, 62654 faces -> CANCELLED). So: retry across voxel sizes.
    src = low.data.copy(); ok = False
    for div in (200, 190, 180, 170, 160, 150, 140, 120, 100):
        low.data = src.copy()
        low.data.remesh_voxel_size = diag / div; bpy.ops.object.voxel_remesh()
        r = bpy.ops.object.quadriflow_remesh(mode="FACES", target_faces=max(a.faces // 2, 100),
                                             use_mesh_symmetry=False, use_preserve_sharp=False, seed=0)
        print(f"[ai2game] quadriflow voxel div={div}: {r}")
        if r == {"FINISHED"}: ok = True; break
    if not ok: raise SystemExit("[ai2game] QuadriFlow failed at every voxel size; use --mode voxel")
    bpy.ops.object.mode_set(mode="EDIT"); bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.quads_convert_to_tris(); bpy.ops.object.mode_set(mode="OBJECT")
bpy.ops.object.shade_smooth()
print(f"[ai2game] low tris={sum(len(p.vertices)-2 for p in low.data.polygons)}")

# ---------- 3. fresh UVs on low ----------
low.data.materials.clear()
while low.data.uv_layers: low.data.uv_layers.remove(low.data.uv_layers[0])
low.data.uv_layers.new(name="UVMap")
bpy.ops.object.mode_set(mode="EDIT"); bpy.ops.mesh.select_all(action="SELECT")
bpy.ops.uv.smart_project(angle_limit=math.radians(66), island_margin=0.002)
bpy.ops.uv.pack_islands(margin=0.002)
bpy.ops.object.mode_set(mode="OBJECT")

# ---------- 4. bake targets on low ----------
def new_img(n, data):
    im = bpy.data.images.new(f"{name}_{n}", a.tex, a.tex, alpha=False, float_buffer=False)
    im.colorspace_settings.name = "Non-Color" if data else "sRGB"
    return im
imgs = {k: new_img(k, k != "basecolor") for k in ("basecolor", "roughness", "metallic", "normal")}
mat = bpy.data.materials.new(f"{name}_mat")
mat.use_backface_culling = True   # -> glTF doubleSided=false, USD doubleSided=0 (RealityKit ignores USD doubleSided anyway)
if not mat.node_tree: mat.use_nodes = True
low.data.materials.append(mat)
nt = mat.node_tree
bake_node = nt.nodes.new("ShaderNodeTexImage")
nt.nodes.active = bake_node                                  # bake writes into the ACTIVE image node

# emission-rewire helper: bake any Principled input by routing its source to an Emission shader
def rewire(input_name):
    saved = []
    for ms in high.data.materials:
        if not ms or not ms.node_tree: continue
        t = ms.node_tree
        out = next((n for n in t.nodes if n.type == "OUTPUT_MATERIAL" and n.is_active_output), None)
        bsdf = next((n for n in t.nodes if n.type == "BSDF_PRINCIPLED"), None)
        if not out or not bsdf: continue
        em = t.nodes.new("ShaderNodeEmission")
        inp = bsdf.inputs[input_name]
        if inp.is_linked: t.links.new(inp.links[0].from_socket, em.inputs["Color"])
        else:
            v = inp.default_value
            em.inputs["Color"].default_value = (v[0], v[1], v[2], 1) if hasattr(v, "__len__") else (v, v, v, 1)
        prev = out.inputs["Surface"].links[0].from_socket if out.inputs["Surface"].is_linked else None
        t.links.new(em.outputs["Emission"], out.inputs["Surface"])
        saved.append((t, out, em, prev))
    return saved
def restore(saved):
    for t, out, em, prev in saved:
        if prev: t.links.new(prev, out.inputs["Surface"])
        t.nodes.remove(em)

bpy.ops.object.select_all(action="DESELECT")
high.select_set(True); low.select_set(True); bpy.context.view_layer.objects.active = low
common = dict(use_selected_to_active=True, cage_extrusion=a.cage * diag,
              max_ray_distance=a.cage * diag * 2, margin=8)
for key, inp in (("basecolor", "Base Color"), ("roughness", "Roughness"), ("metallic", "Metallic")):
    bake_node.image = imgs[key]
    s = rewire(inp)
    bpy.ops.object.bake(type="EMIT", **common)
    restore(s)
bake_node.image = imgs["normal"]
bpy.ops.object.bake(type="NORMAL", normal_space="TANGENT", **common)   # OpenGL (+Y) = what glTF & RealityKit expect

# ---------- 5. pack ORM (R=AO(1.0) G=rough B=metal) and save ----------
import numpy as np
def px(im): a_ = np.empty(len(im.pixels), np.float32); im.pixels.foreach_get(a_); return a_.reshape(-1, 4)
r, mtl = px(imgs["roughness"]), px(imgs["metallic"])
orm = bpy.data.images.new(f"{name}_orm", a.tex, a.tex, alpha=False); orm.colorspace_settings.name = "Non-Color"
o = np.ones_like(r); o[:, 1] = r[:, 0]; o[:, 2] = mtl[:, 0]
orm.pixels.foreach_set(o.ravel())
paths = {}
for key, im in (("basecolor", imgs["basecolor"]), ("normal", imgs["normal"]), ("orm", orm)):
    p = os.path.join(a.outdir, f"{name}_{key}.png")
    im.filepath_raw = p; im.file_format = "PNG"; im.save(); paths[key] = p

# ---------- 6. final material on low (glTF-exporter-friendly layout) ----------
nt.nodes.remove(bake_node)
bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
def tex(p, cs):
    n = nt.nodes.new("ShaderNodeTexImage"); n.image = bpy.data.images.load(p); n.image.colorspace_settings.name = cs; return n
tb, to, tn = tex(paths["basecolor"], "sRGB"), tex(paths["orm"], "Non-Color"), tex(paths["normal"], "Non-Color")
sep = nt.nodes.new("ShaderNodeSeparateColor"); nm = nt.nodes.new("ShaderNodeNormalMap")
nt.links.new(tb.outputs["Color"], bsdf.inputs["Base Color"])
nt.links.new(to.outputs["Color"], sep.inputs["Color"])
nt.links.new(sep.outputs["Green"], bsdf.inputs["Roughness"])
nt.links.new(sep.outputs["Blue"], bsdf.inputs["Metallic"])
nt.links.new(tn.outputs["Color"], nm.inputs["Color"]); nt.links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])

# ---------- 6b. optional glass: faces whose baked colour matches a tint get a see-through material ----------
# Why (2026-10-02): image-to-3D engines paint opaque PBR only (no opacity channel), so a
# lantern's green panes came out solid. The panes are recognisable in the baked colour
# texture, so faces that sample as that tint move to a second, semi-transparent, glossy
# material. The mesh is a shell, so through the glass you see the far side or the room;
# anything inside (a burner) was never modelled.
glass_info = None
if a.glass:
    import colorsys
    HUES = {"red": 0, "amber": 35, "yellow": 55, "green": 120, "cyan": 180, "blue": 225, "magenta": 300}
    target = HUES.get(a.glass.lower())
    if target is None:
        target = float(a.glass)
    img = bpy.data.images.load(paths["basecolor"])
    W, H = img.size
    px = np.empty(W * H * 4, np.float32); img.pixels.foreach_get(px); px = px.reshape(H, W, 4)
    me = low.data
    uv = me.uv_layers.active.data
    n = len(me.polygons)
    cols = np.zeros((n, 3), np.float32)
    for p in me.polygons:  # mean of the face's corner UVs and its UV centroid
        uvs = [uv[li].uv for li in p.loop_indices]
        pts = uvs + [sum((u for u in uvs), uvs[0] * 0) / len(uvs)]
        acc = np.zeros(3, np.float32)
        for u in pts:
            x = min(W - 1, max(0, int(u[0] * W))); y = min(H - 1, max(0, int(u[1] * H)))
            acc += px[y, x, :3]
        cols[p.index] = acc / len(pts)
    hsv = np.array([colorsys.rgb_to_hsv(*c) for c in cols])
    def hue_dist(h):
        return np.abs(((hsv[:, 0] * 360 - h) + 180) % 360 - 180)
    # Adaptive centre: the named tint is only a starting point. On the lantern the pane
    # centres were dark teal (~165 deg), outside a fixed window around green (120), and
    # only the pane rims were caught (debug render, 2026-10-02). So find where the
    # saturated faces near the tint actually cluster (circular mean), then select around it.
    cand = (hue_dist(target) < 60) & (hsv[:, 1] > 0.25) & (hsv[:, 2] > 0.05)
    if cand.any():
        ang = hsv[cand, 0] * 2 * np.pi
        target = (np.degrees(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())) + 360) % 360
    mask = (hue_dist(target) < 32) & (hsv[:, 1] > 0.22) & (hsv[:, 2] > 0.05)
    # neighbour majority, twice: removes speckle and fills pinholes
    bm = bmesh.new(); bm.from_mesh(me); bm.faces.ensure_lookup_table()
    nbrs = [[f2.index for e in f.edges for f2 in e.link_faces if f2.index != f.index] for f in bm.faces]
    bm.free()
    for _ in range(2):
        mask = np.array([(mask[i] + sum(mask[j] for j in nb)) * 2 > (1 + len(nb)) for i, nb in enumerate(nbrs)])
    # Fill holes: highlights painted into the concept (white glints on the panes) are not
    # tinted, so they were left opaque as speckles (2026-10-02). A face becomes glass when
    # most of its neighbours are; repeated, this closes glints up to a few faces wide.
    for _ in range(6):
        mask = mask | np.array([len(nb) > 0 and sum(mask[j] for j in nb) * 3 >= 2 * len(nb) for nb in nbrs])
    # Islands: reflections and shading baked into a pane leave patches of non-glass faces
    # inside it (up to a quarter of a pane on pi's lantern, live test 2, 2026-10-02).
    # A patch is a small connected group of non-glass faces whose border is mostly glass;
    # frame ribs are not, because they connect to the body. Patches become glass.
    seen = np.zeros(n, bool)
    for start in range(n):
        if mask[start] or seen[start]:
            continue
        comp, stack, seen[start] = [], [start], True
        while stack:
            f = stack.pop(); comp.append(f)
            for j in nbrs[f]:
                if not mask[j] and not seen[j]:
                    seen[j] = True; stack.append(j)
        if len(comp) > 0.02 * n:
            continue
        # A component is closed under non-glass adjacency, so all its outside neighbours
        # are glass: a small one with any glass border sits inside a glass region.
        if any(mask[j] for f in comp for j in nbrs[f]):
            mask[comp] = True
    # Dark bands: thick glass drawn near-black in the concept (a bottle's lip and base,
    # live test 3, 2026-10-02) fails the hue test and, joined to the cork or base, is not
    # an island. Group only the dark non-glass faces; a small dark group touching glass is
    # shaded glass. Coloured parts (cork, label, brass) are never dark, so they stay.
    dark = (~mask) & (hsv[:, 2] < 0.18)
    seen = np.zeros(n, bool)
    for start in np.nonzero(dark)[0]:
        if seen[start]:
            continue
        comp, stack, seen[start] = [], [int(start)], True
        while stack:
            f = stack.pop(); comp.append(f)
            for j in nbrs[f]:
                if dark[j] and not seen[j]:
                    seen[j] = True; stack.append(j)
        if len(comp) <= 0.05 * n and any(mask[j] for f in comp for j in nbrs[f]):
            mask[comp] = True
    if mask.any():
        tint = np.median(cols[mask], axis=0)
        # Baked shading darkens the panes and blending washes them out: at the first
        # settings the glass read as nearly clear (live test 2). Keep the hue, make it
        # clearly coloured and mid-bright.
        hh, ss, vv = colorsys.rgb_to_hsv(*tint)
        tint = np.array(colorsys.hsv_to_rgb(hh, max(0.55, min(ss, 0.8)), min(max(vv, 0.45), 0.7)))
        gm = bpy.data.materials.new(f"{name}_glass"); gm.use_nodes = True
        gb = next(nd for nd in gm.node_tree.nodes if nd.type == "BSDF_PRINCIPLED")
        gb.inputs["Base Color"].default_value = (*tint, 1.0)
        gb.inputs["Alpha"].default_value = a.glass_alpha
        gb.inputs["Roughness"].default_value = 0.05
        gb.inputs["Metallic"].default_value = 0.0
        for attr, val in (("surface_render_method", "BLENDED"), ("blend_method", "BLEND")):
            try: setattr(gm, attr, val)
            except Exception: pass
        me.materials.append(gm)
        gi = len(me.materials) - 1
        for p in me.polygons:
            if mask[p.index]: p.material_index = gi
        glass_info = (int(mask.sum()), n, tuple(round(float(c), 3) for c in tint))
    print(f"[ai2game] glass '{a.glass}' (hue centre {target:.0f} deg): {int(mask.sum())}/{n} faces ({100 * mask.mean():.1f}%)"
          + (f", tint {glass_info[2]}" if glass_info else " — nothing matched, no glass material"))

# ---------- 7. LODs (share UVs + textures; plain collapse decimation of LOD0) ----------
bpy.data.objects.remove(high, do_unlink=True)
lods = [low]
for i, ratio in enumerate(float(x) for x in a.lods.split(",") if x):
    l = low.copy(); l.data = low.data.copy(); l.name = f"{name}_LOD{i+1}"; sc.collection.objects.link(l)
    bpy.context.view_layer.objects.active = l
    m = l.modifiers.new("dec", "DECIMATE"); m.ratio = ratio; m.use_collapse_triangulate = True
    bpy.ops.object.select_all(action="DESELECT"); l.select_set(True)
    bpy.ops.object.modifier_apply(modifier=m.name)
    lods.append(l)
low.name = f"{name}_LOD0"

# ---------- 8. export ----------
for ob in lods:
    bpy.ops.object.select_all(action="DESELECT"); ob.select_set(True); bpy.context.view_layer.objects.active = ob
    base = os.path.join(a.outdir, ob.name)
    bpy.ops.export_scene.gltf(filepath=base + ".glb", use_selection=True, export_format="GLB",
                              export_yup=True, export_apply=True)
    # No USDZ from Blender: its USD export passed `usdchecker --arkit` but rendered
    # solid black in RealityKit on visionOS 27 (2026-10-02, simulator A/B test).
    # m3d converts each GLB with Apple's usdextract + usdzip instead.
    print(f"[ai2game] wrote {base}.glb tris={sum(len(p.vertices)-2 for p in ob.data.polygons)}")
# m3d treats a post stage as successful only when this line is printed:
# Blender exits 0 even after a Python traceback (2026-09-26).
print(f"M3D_POST lods={len(lods)} outdir={a.outdir} glass={glass_info}")

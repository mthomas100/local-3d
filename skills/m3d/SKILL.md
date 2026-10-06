---
name: m3d
description: Make a textured 3D model (GLB and USDZ) from a text prompt or a picture, and rig and animate it, entirely on this Mac and free, through the `m3d` command — concept image, background cutout, image-to-3D (Hunyuan3D, Pixal3D or TRELLIS.2), a re-baked game- or Vision Pro-ready export, then `m3d rig` (SkinTokens auto-rig or a deterministic puppet rig, bone roles) and `m3d animate` (procedural clips for any rig, text-to-motion for humanoids, one GLB + USDZ per clip, an animation check). The DEFAULT for any request for a 3D model, mesh, asset, prop, USDZ or GLB, rigged or animated (games, Vision Pro, AR, Reality Composer, 3D printing, "text to 3D", "a local Meshy", "make it wave"). Not when the user names Meshy as the tool ("with Meshy", "meshy make") — that goes to the `meshy` skill.
---

# m3d — local prompt → 3D model on this Mac

One command, `m3d` (on PATH; source and README in the m3d checkout, `$M3D_ROOT`
below, e.g. `~/repos/local-3d`). Every model it uses is a row in
`$M3D_ROOT/m3d.toml`. Nothing leaves the machine. `m3d --help` carries
the recipes below; `m3d models` lists engines, licences and presets as JSON.

## Which tool (the same table is in the optional meshy skill)

| The request | Use |
|---|---|
| Just asks for a 3D model, USDZ, GLB, asset, prop | **m3d** (this skill, default) |
| Says local, offline, free, private, or m3d | **m3d** |
| Names Meshy ("with Meshy", "on Meshy", "meshy make …") | **meshy** skill if installed (cloud, costs credits) |
| Needs a rig, skeleton, skin weights, animation, a clip ("make it wave", "idle loop", "a person walks") | **m3d** (`m3d rig`, then `m3d animate`; see below) |
| Needs quad retopology, or a Meshy-specific animation preset by name | **meshy** skill (credits, ask first) |

If it is unclear, ask once: "Locally with m3d (free), or with Meshy (cloud,
costs credits)?"

## Do this

0. **Start a clock:** run `date +%s` first and keep the number. At the end, report
   the elapsed time from it (`echo $(( $(date +%s) - START ))` seconds). Never add
   up stage times: they overlap and miss your own thinking time (live test 2 said
   "~8 min" for a 9 min 41 s run).
1. **Check the GPU.** `m3d gpu` — exit 3 means another job holds the shared GPU
   (llama-swap model, a video render, ComfyUI). Do not pass `--force`; ask the
   user or the other session first. If you are a local model and the only holder
   is you, see "From a local model" below (`--yield-llm`).
2. **Pick the recipe from the user's intent:**

        # Vision Pro, best quality (one showcase object)
        m3d concept "<object>" --variants 4            # ~20 s each; look at the PNGs
        m3d mesh <best.png> --post hero                # ~5 min; 150k tris, 4096 textures
        # Vision Pro, normal (several objects in a scene)
        m3d make "<object>" --post visionos            # ~7 min end to end; 50k tris
        # Game asset
        m3d make "<object>" --post game                # 20k tris + LODs
        # Game sold in the EU, UK or South Korea: add --engine pixal3d (MIT + Meta DINOv3 licence; Hunyuan excludes them)
        # Shape check first: m3d make "<object>" --draft   (~15 s, untextured)
        # From a picture: m3d make photo.png --post visionos
        # Plan only: --dry-run (and --json for a machine-readable result)
        # Object with glass in it: see "Glass" below (--glass <tint>)

3. Run it with the bash tool and a long timeout: **pi `timeout: 1800` (seconds);
   Claude Code `timeout: 1800000` and Codex `timeout_ms: 1800000` (milliseconds)**,
   or in the background. Pass
   `--json` and read the result: it has top-level `usdz`, `glb`, `dir`,
   `wall_seconds` (full paths), plus `yield` and `sheet` where they apply.
4. **Check the concepts before meshing** when you used `m3d concept`: read the
   one contact sheet it returns (`sheet`), not each image. Want one whole object,
   three-quarter view, plain background, no shadow on the floor, and the right
   construction. A shadow or busy background becomes geometry. Regenerating a
   concept costs 20 s; a bad mesh costs 5 minutes.
5. **Report** to the user:
   - the **full absolute paths**, copied from the JSON (`usdz` for Vision Pro, iPhone
     and Reality Composer Pro; `glb` for game engines). Never shorten them with `...`;
   - the **wall-clock time from your step-0 clock**, not a sum of stage times;
   - whether the GPU is free, **only after running `m3d gpu` and `lsof -i :8189`**.
     A local model will see itself loaded: say "free apart from my own model".
   Double-clicking a USDZ opens it in Preview; AirDrop opens it on Vision Pro.
6. **Optional RealityKit check:** `$M3D_ROOT/viewer/render.sh <dir> <full path to .usdz>`
   writes `viewer-yaw35-closeup.png` (front) and `viewer-yaw215-closeup.png` (back)
   into `<dir>`, cropped to the model. **Judge from the close-ups**, not the full
   room shots: the model is small in those, and defects get missed. Look for
   black or missing textures, a wall or floor disc, and, with `--glass`, panes
   that are see-through rather than solid or patchy. Report what you see, flaws
   included.

## Rig and animate (`m3d rig`, `m3d animate`)

Both are local and run on the CPU, except the SkinTokens rigger (the default
`m3d rig` engine, Metal): it is a GPU job, so `m3d gpu` and, from a local model,
`--yield-llm` apply to `m3d rig` unless `--engine puppet`. `m3d animate` and
`m3d rig --engine puppet` never need `--yield-llm` (on `animate` it is accepted and
only logs "nothing to unload": Blender runs beside your model). Start from a finished
model: the `_LOD0.glb` of an `m3d … --post` run.

**A character you will animate must be made with `--rig-ready`** (2026-10-03):

        m3d make "<character>" --rig-ready --variants 4 --post visionos --json

It draws the character front-on in a strict T-pose (arms straight out like airplane
wings, open space under each arm, legs apart), ranks the variants by how much air is under
the arms (`pose` in the JSON, `notch`; ok >= 0.40) and meshes the best one; a
`[posecheck] WARNING` means no variant is clear: try more variants or another seed.
Without it the concept shows arms against the body, the mesh fuses each arm to the torso,
and every arm raise drags a sheet of belly skin with it (seen in the pi teddy test). The clips lower T-pose arms to a relaxed 30 degrees by themselves.

        # 1. rig: a humanoid or anything near one (knight, character)
        m3d rig <name>_LOD0.glb --json                      # SkinTokens, ~1 min; writes rig/<name>_LOD0-rigged.glb
        # 1. rig: an object or robot (lamp, teapot, kite, vehicle)
        m3d rig <name>_LOD0.glb --engine puppet --spec auto --json       # ~10 s; or --spec $M3D_ROOT/specs/puppet/<x>.json
                                                            # (a spec path is taken as given: relative paths fail from another cwd)
        # 2. animate: procedural clips, any rig (idle,hop,look,wave,spin,wobble,sleep; all loop)
        m3d animate rig/<name>_LOD0-rigged.glb --clips idle,wave,hop --usdz --check --json
        # 2. animate: text-to-motion, biped rigs only (MoMask on CPU; research-data licence, not for sale)
        m3d animate rig/<name>_LOD0-rigged.glb --text "a person waves hello" --usdz --check --json
        # your own Mixamo-named BVH onto a biped rig:  --bvh clip.bvh   (add --in-place for a treadmill walk)

Rules for these two:

- **Where things land, what `--json` returns.** `m3d rig` writes `rig/` beside the
  input GLB (`rig/<name>_LOD0-rigged.glb`, `.roles.json`, `…-rigplot.png`, `rig.json`);
  `m3d animate` writes `anim/` beside the rigged GLB (`<name>_LOD0-rigged.<clip>.glb`
  and `.usdz` per clip, `animate.json`). With `--json` each command prints **one JSON
  object on stdout** (progress and `[yield]` lines go to stderr): `rig` has `glb`,
  `roles`, `plot`, `roles_summary`, `wall_seconds`; `animate` has `clips`, `usdz`,
  `all_clips_glb`, `animcheck`, `verdict`, `wall_seconds`, all full paths.
- **The rig's stress test.** `m3d rig` swings each limb in numpy (arms up and out,
  legs forward) and returns `stress` (`verdict`, `web_area_max`, `advice`). `fail`
  means skin is shared by parts that must move apart: a limb fused to the body. Every
  clip that moves that limb will drag skin, so do not animate it and call it fine:
  re-make the model with `--rig-ready` (or a puppet spec with rip + ball caps), and
  tell the user why.
- **Look at the rig before animating.** `m3d rig` prints and returns `plot`
  (joints and bones over the mesh, front and side) and `roles_summary` (how many
  bones got which role, and `biped` / `object`). A skeleton outside the mesh,
  an arm tagged as a leg, or a lamp with "fingers" means: use `--engine puppet`
  (objects) or re-run SkinTokens with a cleaner `_LOD0.glb`. Read the PNG with
  your image tool if you can.
- **`--check` and look at the sheets.** `animate --check` returns `verdict`
  (ok / warn / fail / error) and, per clip, `animcheck.<clip>.sheet`: eight
  frames with the worst one boxed. Exit 4 on `fail` (use `--allow-fail` only to
  keep files for debugging). Report the verdict and the reasons; a `fail` is a
  torn or self-penetrating mesh, do not ship it as fine. `web_area` is skin pulled
  into stretched sheets (share of the surface): 0 on a clean rig; >= 0.010 fails.
  A `warn` is not "looks good": say what it warns about. Describe what the worst
  frame shows (where is the hand, is skin stretched between arm and body) rather
  than "clean wave" — the 2026-10-03 run called a floppy wave "clean, no tearing".
- **Non-bipeds cannot take `--text`**: m3d refuses (exit 2) and says so; use
  `--clips`. The `.roles.json` beside the rig says what the rig is.
- **One USDZ per clip.** Apple's converter keeps only the first animation of a
  file; RealityKit assembles the clips at runtime. `clips` and `usdz` in the JSON
  map clip name → full path; `all_clips_glb` holds every clip of that run for
  engines that take one file.
- Procedural clips are 2 s loops at 30 fps by default (`--seconds`, `--fps`);
  `--text` clips default to 4 s (max 9.8).
- **Which clip for what.** `wave` picks its style from the proportions: a big-headed
  character (head >= 0.3 of the height: teddies, chibis, mascots) gets the arm up beside
  the head, waving from the shoulder with a floppy wrist and a knee bounce; others the
  human elbow-up wave. `hop` on a rig with knees under the hips (SkinTokens bipeds) is a
  full cartoon jump: crouch with the knees, arms back, take-off stretch and arm fling,
  knee tuck, landing squash, follow-through; on objects and puppets it is the squash
  hop. For "get ready, jump, land" use `hop`. MoMask `--text` motion is human motion
  capture: on short, big-headed characters it folds the body (failed on the teddy).
- **Optional RealityKit check of a clip:** `$M3D_ROOT/viewer/render.sh --anim <dir> <full path to clip.usdz>`
  plays it in the visionOS simulator and writes `<dir>/anim-sheet.png` (three paused
  frames); look at it as in step 6. Report the clip paths (`usdz` per clip) in full.

## Glass (any object with glass or another see-through part)

The 3D engines paint every surface opaque, so glass comes out solid unless you
ask for it. **Whenever the object has glass, crystal, a window, a bottle, a jar,
a lens, a potion, ice or another transparent part, add `--glass <tint>`** to the
command that has `--post` (`m3d make … --post …` or `m3d mesh … --post …`), or
run `m3d post <raw .glb> --preset … --glass <tint>` afterwards. The user does not
have to say "see-through": glass in the description is enough.

- `<tint>` is the glass colour: green, blue, cyan, amber, red, yellow, magenta.
- Glass is found by its colour, so it **must be tinted** in the concept. Put the
  tint in the prompt ("green glass panes", "amber bottle"). If the user wants
  clear glass, use a faint tint such as "pale blue glass" and say why.
- Check the log line `glass '<tint>' …: N/M faces (P%)`. Near 0% means no glass
  was found (wrong tint name, or the concept's glass is not tinted). Compare the
  share with what should be glass: most of a bottle or jar (70-80% is right),
  a sixth or so of a lantern. A large share on an object that is mostly not
  glass means the tint is also the body colour: say so, and do not ship it as glass.
- The model is a hollow shell, so see-through glass shows an empty inside.
  Mention it when the user expects something inside (a flame, a burner, liquid).

## Writing prompts

Describe one object, its materials and, when the shape matters, its construction:
"a six-sided brass ship's lantern with flat green glass panes and a ring on top".
(With only "green glass panes", Z-Image drew globe lanterns: name the parts.) m3d
wraps it with the framing, background and lighting words; adding a scene,
background or lighting yourself makes worse concepts.

## Rules learned the hard way (2026-09-26 to 2026-10-03)

- Never trust `usdchecker` alone: Blender's USDZ passed it and rendered black in
  RealityKit. m3d now builds USDZ with Apple's tools; keep it that way.
- One GPU job at a time on this Mac. After any run, `m3d gpu` and
  `lsof -i :8189` (ComfyUI) must both be clear before saying the GPU is free.
- Every stage stops itself after 15 silent minutes or past its limit; if one
  fails, the error names its log. Do not retry in a loop: read the log.
- Qwen-Image-2.1 (`--image-model qwen-2.1`) is research-licence only: never for
  anything the user sells.
- Report every file the user must open as a full absolute path, copied from the
  JSON, never shortened with `…` (2026-10-03: the clip USDZs were abbreviated).
- If m3d itself misbehaves (a flag ignored, a stage killed), report it to the user
  with the log line; do not work around it with dummy GPU jobs (2026-10-03: four
  throwaway `concept --yield-llm` renders just to free memory) and do not edit your
  notes from an m3d task unless asked.

## From a local model (pi, or any agent whose own model runs on this Mac)

Your own model holds the GPU, so `m3d gpu` lists it and m3d refuses. Pass
**`--yield-llm`** on every GPU command (`make`, `concept`, `mesh`, and `rig`
unless `--engine puppet`; `animate` and `post` never need it), or set
`M3D_YIELD_LLM=1` once. m3d then does what the film rig's `render_and_wait` does:
it unloads your model from llama-swap (or stops a `model <target>` engine and
restarts it afterwards), waits until memory settles, and runs. Your bash call
blocks until the job finishes, and your model reloads on your next request.
Expect your next reply to take extra seconds while it loads.

        m3d make "<object>" --post visionos --yield-llm --json

- Use a bash timeout of at least 1800 seconds (a full run is about 7 minutes; the
  unload, settle and your model's reload add a minute or two). The `[yield]` lines
  on stderr and the `yield` field in the JSON show what was unloaded.
- Measured from pi with qwen38 (2026-10-02): unload under 2 s, reload 8-28 s,
  first reply after a run 22-38 s.
- Only yield your own model. If `m3d gpu` shows ComfyUI or another m3d job,
  `--yield-llm` still refuses: wait or tell the user.
- Since 2026-10-04 every GPU command also takes the Mac's GPU hold (the hold gate
  on :8090, optional: github.com/mthomas100/local-rig; `hold status` shows it). Behind a film render the command waits its
  turn instead of failing (the `hold:` lines on stderr say what it waits for), and
  your model's calls, like every agent's, wait until the m3d run ends. Never
  run `hold on` or `hold off` yourself: those are the user's.
- To judge results, read the PNGs m3d reports (concept, `*.rendercheck.png`)
  with your image tool if your model sees images.

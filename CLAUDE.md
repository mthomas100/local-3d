# local-3d — working rules for agents

`m3d` is a local, offline stand-in for Meshy on Apple Silicon (built on an M5 Max, 128 GB, macOS 27):
prompt → concept image → cutout → textured mesh → game/visionOS asset, and now
**`m3d rig` / `m3d animate`**: skeleton, skin weights, animation clips, USDZ that
RealityKit plays. Read in this order: `README.md` (how it is built, measurements,
what went wrong and what guards it), `m3d.toml` (every model),
`skills/m3d/SKILL.md` (what calling agents read).

## Layout

| path | what |
|---|---|
| `bin/m3d` | the CLI: subcommands, GPU guard, `--yield-llm`, `run_stage`, USDZ conversion |
| `m3d.toml` | the registry: image models, engines, riggers, motion models, post presets, limits |
| `lib/*.py` | one worker per stage, run by `bin/m3d` from a `m3d.toml` row (Blender headless or a venv python) |
| `specs/` | data for deterministic stages (puppet rig specs, motion preset parameters) |
| `viewer/` | the RealityKit acceptance test: `render.sh` renders USDZs in the visionOS simulator |
| `bench/` | `runs.tsv` (every real run: stage, model, wall time, peak memory), `fetch-weights.sh` |
| `skills/` | the `m3d` agent skill, symlinked into `~/.claude/skills` and `~/.pi/agent/skills` |

Outputs go under `$M3D_OUT` (default `~/3d/m3d/`). Weights and venvs live outside the repo (`~/.hy3d`,
`~/.skintokens`, `~/.momask`, `~/.unimate`, the Hugging Face cache).

## Rules that were paid for

1. **`m3d.toml` is the only place a model is named.** A new engine, rigger or motion
   model is a row (command template, licence, role, limits) plus a `lib/` worker.
   Never hard-code a model, path or flag in a script.
2. **Every stage proves itself.** It runs under `run_stage` (`/usr/bin/time -l` →
   `bench/runs.tsv`), has a stall limit, and prints its success marker on the last
   line: `M3D_POST`, `M3D_RIG`, `M3D_ANIM`, `M3D_MOTION`. Blender exits 0 after a
   Python traceback, so the marker is the only proof. `--json` prints one JSON object.
3. **One GPU job at a time; the memory is shared with language models.** Before any
   model load: `m3d gpu` (exit 3 = busy; never `--force` on the shared GPU), a plain
   `GET 127.0.0.1:8090/running` (never touch `/upstream/<model>/…`, it loads the
   model), and a note to any other agent that is using the GPU. When a watchdog/arbiter
   agent is announced, message it model, GiB and minutes before a load and "unloaded" after. Pin PyTorch devices explicitly (`cpu` or
   `mps`): several research repos pick MPS on their own. At most four Blender jobs
   at once. Never SIGKILL a model server (Metal memory can leak until reboot).
   **Every heavy job (trimesh, scipy, Blender, torch, skintokens) runs under
   `bin/memguard CAP_GIB cmd…`**: a Bash tool timeout kills the shell, not a busy
   child; an orphaned trimesh run reached 266 GB and triggered a jetsam kill of 1,247
   processes on 2026-10-02. Never background a heavy job without the guard.
4. **The render is the test, not the checker.** `usdchecker --arkit` passed a USDZ
   that rendered solid black. Acceptance is `viewer/render.sh` (and `--anim` for
   clips) in the visionOS simulator; look at the close-up PNGs yourself. A local
   model judging a full 4K room shot missed defects the close-up showed.
5. **USDZ goes through Apple's tools** (`usdextract` → `usdzip --arkitAsset`), not
   Blender's USD export. That route keeps UsdSkel but only the FIRST glTF animation
   and no clip names: export one GLB per clip; the runtime assembles an
   `AnimationLibraryComponent`.
6. **Upstream stays pristine.** Clones in `~/repos` (Hunyuan3D-Swift, hy3d-mcp,
   ComfyUI, skin-tokens.cpp, momask-codes, UniMate, …) are never edited; refresh with
   `git pull`. Patches live in `lib/`, `comfy_nodes/` or a copied launcher, and the
   README says why.
7. **Licences are data.** Put them in the row (`license`) and say what they exclude in
   `role`: Hunyuan and HY-* community licences exclude the EU, UK and South Korea;
   NVIDIA PartField/PartPacker, Adobe RigAnything and Meta ActionMesh are
   non-commercial; Qwen-Image-2.1 is research-only. Nothing redistributes third-party
   motion libraries (Mixamo, Truebones); the tool consumes a folder the user supplies.
8. **Measure, record, then write it up.** Real runs → `bench/runs.tsv`; live tests →
   `bench/<name>.md` with the raw event stream beside it (kept out of the published repo); findings and decisions →
   the project's design notes, with the raw material kept in full. Update the README table when a number changes.
9. **Git, always.** Commit every working increment with a `local-3d: …` subject and
   the body saying what was verified. Do not commit another agent's uncommitted
   files (`bench/runs.tsv` is appended by every run; commit it with your own runs
   only). Never commit weights, `viewer/.build`, `viewer/Models/*.usdz`, logs.
10. **Keep the agent surface in sync.** A new subcommand or flag changes `m3d --help`,
    `skills/m3d/SKILL.md` and `m3d models` in the same commit. `[yield]` lines must
    stay visible under `--json` (the first live test lost them).

## Rig / animate conventions (2026-10-02)

- Bones carry **roles**, not just names: `root, body, head, arm.L, arm.R, leg.L, leg.R,
  lid, tail, antenna, wheel, spout, other`. Motion presets key on roles so one preset
  animates a puppet rig, a SkinTokens rig or a Mixamo-named humanoid alike.
- Rigid parts get **hard weights** (1.0 on one bone); soft parts (hose, ribbon,
  mushroom) get smooth weights, at most four influences. Squash and stretch are shape
  keys (USD blend shapes), not scale hacks baked into the mesh.
- Every clip **loops** (first pose == last pose) unless its name says otherwise.
- Test fixtures: lamp robot (biped-ish), teapot (lid, spout), kite (sheet, ribbon tail)
  from `$M3D_OUT/roster`, and the knight (humanoid) from
  `$M3D_OUT/spike-2026-09-26/knight-rig/…-visionos/…_LOD0.glb` (outputs are not in the repo).

## Before you stop

Leave the status notes true, the README's numbers current, the design notes written, and the
GPU handed back (`m3d gpu` → "GPU free", a "GPU free" note to the agents you told).

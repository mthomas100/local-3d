// Lays out every USDZ in the bundle's Models folder in a row inside one volume,
// each scaled to fit a 0.25 m cube, turned by M3D_YAW degrees (default 35, a
// three-quarter view; 215 shows the backs). Launch env comes from
// SIMCTL_CHILD_M3D_YAW in viewer/render.sh.
//
// Animation (2026-10-02, for `m3d animate`; driven by render.sh --anim / --lib):
//   M3D_ANIM=1            play each model's clip M3D_ANIM_INDEX (default 0), looping
//   M3D_ANIM_T=<seconds>  seek that clip to t and pause, so a screenshot is deterministic
//   M3D_ANIM_LIB=1        the FIRST file is the character; every other file is a clip
//                         source whose first clip goes into an AnimationLibraryComponent
//                         on the character under the source's basename (without the
//                         "NN-" prefix render.sh adds); M3D_ANIM_CLIP=<name> is played.
//                         Only the character is shown.
// Every launch writes Documents/report.json (files, clips, durations, skeletons,
// materials, what was played); render.sh reads it from the app's data container.
import SwiftUI
import RealityKit

@main
struct ViewerApp: App {
    var body: some SwiftUI.Scene {
        WindowGroup {
            ModelRow()
        }
        .windowStyle(.volumetric)
        .defaultSize(width: 1.3, height: 1.2, depth: 0.5, in: .meters)
    }
}

struct ClipReport: Codable {
    var index: Int
    var name: String
    var duration: Double
    var bindTarget: String
}

struct PlaybackReport: Codable {
    var clip: String
    var clipDuration: Double
    var controllerDuration: Double
    var requestedTime: Double?
    var appliedTime: Double?
    var looping: Bool
    var libraryKeys: [String]?
}

struct FileReport: Codable {
    var path: String
    var loaded = false
    var error: String?
    var displayed = false
    var animations = 0
    var clips: [ClipReport] = []
    var skeletal = false
    var jointCount: Int?
    var jointNames: [String]?
    var skeletalPoseJoints: Int?
    var materialCount = 0
    var modelEntities = 0
    var entityCount = 0
    var playback: PlaybackReport?
}

struct Report: Codable {
    var mode: String
    var env: [String: String]
    var files: [FileReport]
    var notes: [String]
}

struct ModelRow: View {
    var body: some View {
        RealityView { content in
            let env = ProcessInfo.processInfo.environment
            let yaw = Float(env["M3D_YAW"] ?? "35") ?? 35
            // M3D_LIFT raises the row (metres) so a light wall is behind it: the default
            // spot is in front of the room's black TV, which hides transparency (2026-10-02).
            let lift = Float(env["M3D_LIFT"] ?? "0") ?? 0
            // On a device launched from the Home View there is no M3D_* env at all: play the clips
            // (render.sh always sets M3D_YAW, so the simulator's default stays a still frame).
            let anim = env["M3D_ANIM"] == "1" || env["M3D_YAW"] == nil
            let lib = env["M3D_ANIM_LIB"] == "1"
            let animIndex = Int(env["M3D_ANIM_INDEX"] ?? "0") ?? 0
            let seek = env["M3D_ANIM_T"].flatMap { Double($0) }
            var report = Report(mode: lib ? "lib" : (anim ? "anim" : "default"),
                                env: env.filter { $0.key.hasPrefix("M3D_") }, files: [], notes: [])
            defer { writeReport(report) }
            // M3D_HIDE=1: an empty frame of the room, so render.sh can find and crop the models.
            if env["M3D_HIDE"] == "1" { return }
            let dir = Bundle.main.resourceURL!.appendingPathComponent("Models")
            let urls = ((try? FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)) ?? [])
                .filter { $0.pathExtension == "usdz" }
                .sorted { $0.lastPathComponent < $1.lastPathComponent }
            let slot: Float = 0.3
            var character: (Entity, FileReport)?
            var library = AnimationLibraryComponent()
            var libraryKeys: [String] = []
            for (i, url) in urls.enumerated() {
                var fr = FileReport(path: url.lastPathComponent)
                let model: Entity
                do {
                    model = try await Entity(contentsOf: url)
                } catch {
                    fr.error = "\(error)"
                    print("m3d viewer: load failed \(url.lastPathComponent): \(error)")
                    report.files.append(fr)
                    continue
                }
                fr.loaded = true
                inspect(model, into: &fr)
                for (k, a) in model.availableAnimations.enumerated() {
                    fr.clips.append(ClipReport(index: k, name: a.definition.name,
                                               duration: a.definition.duration,
                                               bindTarget: String(describing: a.definition.bindTarget)))
                }
                fr.animations = fr.clips.count
                if lib && i > 0 {
                    // A clip source: its first clip joins the character's library; not shown.
                    if let a = model.availableAnimations.first {
                        library.animations[clipKey(url)] = a
                        libraryKeys.append(clipKey(url))
                    } else {
                        report.notes.append("\(url.lastPathComponent) has no animation to add to the library")
                    }
                    report.files.append(fr)
                    continue
                }
                fr.displayed = true
                let holder = Entity()
                holder.addChild(model)
                let b = model.visualBounds(relativeTo: nil)
                // M3D_FIT: the cube each model is scaled to fit (metres, default 0.25); a burst at 0.9 frames a shoulder.
                let fit = Float(env["M3D_FIT"] ?? "0.25") ?? 0.25
                let s = fit / max(b.extents.x, b.extents.y, b.extents.z)
                model.position = -b.center
                holder.scale = [s, s, s]
                holder.orientation = simd_quatf(angle: yaw * .pi / 180, axis: [0, 1, 0])
                let n = lib ? 1 : urls.count
                holder.position = [(Float(i) - Float(n - 1) / 2) * slot, lift, 0]
                content.add(holder)
                if lib { character = (model, fr); continue }
                if anim {
                    if animIndex < model.availableAnimations.count {
                        let clip = model.availableAnimations[animIndex]
                        fr.playback = play(clip, named: fr.clips[animIndex].name, on: model, seek: seek)
                    } else {
                        report.notes.append("\(url.lastPathComponent): no clip at index \(animIndex)")
                    }
                }
                report.files.append(fr)
            }
            if lib, let (entity, fr0) = character {
                var fr = fr0
                entity.components.set(library)
                let keys = libraryKeys.sorted()
                let wanted = env["M3D_ANIM_CLIP"] ?? keys.first ?? ""
                if let clip = library.animations[wanted] {
                    fr.playback = play(clip, named: wanted, on: entity, seek: seek)
                    fr.playback?.libraryKeys = keys
                } else {
                    report.notes.append("library has no clip '\(wanted)'; keys: \(keys)")
                }
                report.files.insert(fr, at: 0)
            } else if lib {
                report.notes.append("lib mode: no character loaded")
            }
        }
    }
}

/// Library key for a clip source: the file name without extension and without the
/// "NN-" ordering prefix render.sh gives copies ("02-hop.usdz" → "hop").
func clipKey(_ url: URL) -> String {
    let stem = url.deletingPathExtension().lastPathComponent
    if stem.count > 3, stem[stem.index(stem.startIndex, offsetBy: 2)] == "-",
       stem.prefix(2).allSatisfy(\.isNumber) {
        return String(stem.dropFirst(3))
    }
    return stem
}

/// Plays `clip` on `entity`: looping, or seeked to `seek` seconds (clamped to the clip)
/// and paused there so the frame is deterministic.
@MainActor
func play(_ clip: AnimationResource, named name: String, on entity: Entity, seek: Double?) -> PlaybackReport {
    let dur = clip.definition.duration
    var pr = PlaybackReport(clip: name, clipDuration: dur, controllerDuration: 0,
                            requestedTime: seek, appliedTime: nil, looping: seek == nil)
    if let t = seek {
        let ctrl = entity.playAnimation(clip, transitionDuration: 0, startsPaused: false)
        let tt = min(max(0, t), max(0, dur - 0.001))
        ctrl.time = tt
        ctrl.pause()
        pr.controllerDuration = ctrl.duration
        pr.appliedTime = ctrl.time
        print("m3d viewer: \(name) seek \(tt) s of \(dur) s, paused=\(ctrl.isPaused)")
    } else {
        let ctrl = entity.playAnimation(clip.repeat(duration: .infinity), transitionDuration: 0)
        pr.controllerDuration = ctrl.duration
        print("m3d viewer: \(name) playing, clip \(dur) s, looping")
    }
    return pr
}

/// Counts entities, model components, materials and skeletons under `e`.
@MainActor
func inspect(_ e: Entity, into r: inout FileReport) {
    r.entityCount += 1
    if let mc = e.components[ModelComponent.self] {
        r.modelEntities += 1
        r.materialCount += mc.materials.count
        for sk in mc.mesh.contents.skeletons {
            r.skeletal = true
            r.jointCount = (r.jointCount ?? 0) + sk.joints.count
            r.jointNames = sk.joints.map(\.name)
        }
    }
    if let sp = e.components[SkeletalPosesComponent.self] {
        r.skeletal = true
        r.skeletalPoseJoints = sp.poses.default?.jointNames.count
    }
    for c in e.children { inspect(c, into: &r) }
}

func writeReport(_ report: Report) {
    guard let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else { return }
    let enc = JSONEncoder()
    enc.outputFormatting = [.prettyPrinted, .sortedKeys]
    // A looping controller's duration is infinite, which JSON cannot hold (first run, 2026-10-02).
    enc.nonConformingFloatEncodingStrategy = .convertToString(positiveInfinity: "inf", negativeInfinity: "-inf", nan: "nan")
    do {
        try enc.encode(report).write(to: docs.appendingPathComponent("report.json"), options: .atomic)
        print("m3d viewer: report at \(docs.path)/report.json")
    } catch {
        print("m3d viewer: report failed: \(error)")
    }
}

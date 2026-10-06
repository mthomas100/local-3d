// vision-cutout — subject matte with Apple's on-device Vision framework.
// usage: vision-cutout in.png out.png   → RGBA PNG, prints {"png_path","opaque_pct","partial_pct"}
//
// Why (2026-09-26 spike): the colour-key cutout (hy3d-mcp's cutout.py) keys on
// the corner colour. Z-Image Turbo paints a softly graded grey backdrop, so the
// key smeared (7-28% partial alpha; a clean key is < 1%), and Hunyuan built the
// leftover backdrop into a wall behind the knight. VNGenerateForegroundInstanceMaskRequest
// is a learned matte, free, on-device, and licence-clean (RMBG-2.0 is CC BY-NC).
import AppKit
import CoreImage
import Vision

let args = CommandLine.arguments
guard args.count == 3, let src = NSImage(contentsOfFile: args[1]),
      let cg = src.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
    print(#"{"error": "usage: vision-cutout in.png out.png (input unreadable)"}"#); exit(2)
}
let handler = VNImageRequestHandler(cgImage: cg)
let req = VNGenerateForegroundInstanceMaskRequest()
do { try handler.perform([req]) } catch { print(#"{"error": "vision failed: \#(error)"}"#); exit(1) }
guard let obs = req.results?.first, !obs.allInstances.isEmpty else {
    print(#"{"error": "no foreground subject found"}"#); exit(1)
}
// All instances: a knight's sword or a chest's lid may be separate instances.
let masked = try obs.generateMaskedImage(ofInstances: obs.allInstances, from: handler, croppedToInstancesExtent: false)
let ci = CIImage(cvPixelBuffer: masked)
let ctx = CIContext()
guard let outCG = ctx.createCGImage(ci, from: ci.extent) else { print(#"{"error": "render failed"}"#); exit(1) }
let rep = NSBitmapImageRep(cgImage: outCG)
try rep.representation(using: .png, properties: [:])!.write(to: URL(fileURLWithPath: args[2]))

// alpha stats, same contract as cutout.py plus partial_pct (the key-quality measure)
var opaque = 0, partial = 0
let w = rep.pixelsWide, h = rep.pixelsHigh
for y in 0..<h { for x in 0..<w {
    let a = rep.colorAt(x: x, y: y)?.alphaComponent ?? 0
    if a >= 0.999 { opaque += 1 } else if a > 0.001 { partial += 1 }
} }
let n = Double(w * h)
print(String(format: #"{"png_path": "%@", "opaque_pct": %.1f, "partial_pct": %.2f}"#,
             args[2], Double(opaque) / n * 100, Double(partial) / n * 100))

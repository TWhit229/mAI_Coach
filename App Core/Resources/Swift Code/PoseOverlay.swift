import SwiftUI

struct PoseOverlay: View {
    let landmarks: [PoseLandmarkerService.Landmark]
    var mirrorX: Bool = false
    var allowedLandmarkIndices: Set<Int>? = nil

    var body: some View {
        GeometryReader { _ in
            Canvas { ctx, size in
                // Scale normalized (0..1) points to pixels, mirroring X if requested
                let pts = landmarks.map { lm -> CGPoint in
                    let xNorm = mirrorX ? (1 - CGFloat(lm.x)) : CGFloat(lm.x)
                    return CGPoint(x: xNorm * size.width,
                                   y: CGFloat(lm.y) * size.height)
                }

                let allowed = allowedLandmarkIndices ?? Set(landmarks.indices)
                let filteredEdges = edges.filter { allowed.contains($0.0) && allowed.contains($0.1) }

                // Bones (subset)
                for (a, b) in filteredEdges {
                    guard a < pts.count, b < pts.count else { continue }
                    var p = Path()
                    p.move(to: pts[a]); p.addLine(to: pts[b])
                    ctx.stroke(p, with: .color(.green), lineWidth: 2)
                }

                // Joints
                for (idx, p) in pts.enumerated() where allowed.contains(idx) {
                    let r: CGFloat = 3
                    let dot = Path(ellipseIn: CGRect(x: p.x - r, y: p.y - r, width: 2*r, height: 2*r))
                    ctx.fill(dot, with: .color(.yellow))
                }
            }
        }
    }

    // Upper-body edges (MediaPipe Pose indices)
    private let edges: [(Int, Int)] = [
        (11,12),
        (11,13),(13,15),
        (12,14),(14,16)
    ]
}

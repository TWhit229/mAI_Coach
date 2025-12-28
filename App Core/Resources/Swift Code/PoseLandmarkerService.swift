import Foundation
import Combine
import AVFoundation
import UIKit
import MediaPipeTasksVision

final class PoseLandmarkerService: NSObject, ObservableObject {
    struct Landmark: Identifiable {
        let id = UUID()
        let x: CGFloat
        let y: CGFloat
        let z: CGFloat
        let vis: Float
    }

    struct PoseFrame {
        let timestampMs: Int64
        let landmarks: [Landmark]
    }

    @MainActor @Published var landmarks: [Landmark] = []
    @MainActor @Published var lastFrame: PoseFrame?
    @MainActor @Published var statusText: String = "Starting…"

    private var landmarker: PoseLandmarker?
    private var lastTimestampMs: Int64 = 0

    override init() {
        super.init()
        do {
            guard let path = Bundle.main.path(forResource: "pose_landmarker_heavy", ofType: "task") else {
                statusText = "Model not found"
                return
            }

            let options = PoseLandmarkerOptions()
            options.baseOptions.modelAssetPath = path
            options.runningMode = .liveStream
            options.numPoses = 1
            // Keep these a bit lower while testing - Increased to 0.6 for better stability
            options.minPoseDetectionConfidence = 0.6
            options.minPosePresenceConfidence  = 0.6
            options.minTrackingConfidence      = 0.6

            // Live-stream uses a DELEGATE (not a closure listener)
            options.poseLandmarkerLiveStreamDelegate = self

            landmarker = try PoseLandmarker(options: options)
            statusText = "Ready"
        } catch {
            statusText = "Init error: \(error.localizedDescription)"
        }
    }

    /// Feed frames from the camera (CGImagePropertyOrientation in; convert to UIImage.Orientation for MPImage).
    func process(sampleBuffer: CMSampleBuffer, orientation cgOrientation: CGImagePropertyOrientation) {
        guard let landmarker else { return }

        // Strictly increasing timestamp in ms (detectAsync drops frames if not)
        let pts  = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
        let tsMs = Int64(Double(pts.value) * 1000.0 / Double(pts.timescale))
        if tsMs <= lastTimestampMs { return }
        lastTimestampMs = tsMs

        do {
            let uiOri  = Self.uiOrientation(from: cgOrientation)
            let mpImage = try MPImage(sampleBuffer: sampleBuffer, orientation: uiOri)
            try landmarker.detectAsync(image: mpImage, timestampInMilliseconds: Int(tsMs))
        } catch {
            // Ignore occasional single-frame errors to keep the stream smooth.
        }
    }

    private static func uiOrientation(from cg: CGImagePropertyOrientation) -> UIImage.Orientation {
        switch cg {
        case .up: return .up
        case .down: return .down
        case .left: return .left
        case .right: return .right
        case .upMirrored: return .upMirrored
        case .downMirrored: return .downMirrored
        case .leftMirrored: return .leftMirrored
        case .rightMirrored: return .rightMirrored
        @unknown default: return .up
        }
    }

    // MARK: - Smoothing
    private var previousLandmarks: [Landmark]?
    // Alpha: 0.0 = keep old (frozen), 1.0 = use new (no smoothing).
    // 0.5 = 50% new, 50% old. Lower = smoother but more lag.
    private let smoothingAlpha: CGFloat = 0.5

    private func applySmoothing(to newLandmarks: [Landmark]) -> [Landmark] {
        guard let prev = previousLandmarks, prev.count == newLandmarks.count else {
            previousLandmarks = newLandmarks
            return newLandmarks
        }

        var smoothed: [Landmark] = []
        for i in 0..<newLandmarks.count {
            let p = prev[i]
            let n = newLandmarks[i]
            
            // Simple Lerp
            let sX = p.x + (n.x - p.x) * smoothingAlpha
            let sY = p.y + (n.y - p.y) * smoothingAlpha
            let sZ = p.z + (n.z - p.z) * smoothingAlpha
            // Visibility shouldn't really be smoothed, but let's keep it current
            
            smoothed.append(Landmark(x: sX, y: sY, z: sZ, vis: n.vis))
        }
        previousLandmarks = smoothed
        return smoothed
    }
}

// MARK: - Live stream results
extension PoseLandmarkerService: PoseLandmarkerLiveStreamDelegate {
    func poseLandmarker(_ poseLandmarker: PoseLandmarker,
                        didFinishDetection result: PoseLandmarkerResult?,
                        timestampInMilliseconds: Int,
                        error: Error?) {
        if let error {
            Task { @MainActor in
                self.statusText = "Error: \(error.localizedDescription)"
                self.landmarks = []
            }
            return
        }

        guard let result, let first = result.landmarks.first else {
            Task { @MainActor in
                self.statusText = "No person"
                self.landmarks = []
            }
            return
        }

        // Map normalized landmarks (x,y in 0..1). visibility is NSNumber? in this SDK.
        let mapped: [Landmark] = first.map { lm in
            Landmark(
                x: CGFloat(lm.x),
                y: CGFloat(lm.y),
                z: CGFloat(lm.z),
                vis: lm.visibility?.floatValue ?? 0
            )
        }

        Task { @MainActor in
            self.statusText = "Tracking"
            let smoothed = self.applySmoothing(to: mapped)
            self.landmarks = smoothed
            self.lastFrame = PoseFrame(timestampMs: Int64(timestampInMilliseconds), landmarks: smoothed)
        }
    }
}

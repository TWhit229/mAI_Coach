import Foundation
import Combine
import AVFoundation
import UIKit
import MediaPipeTasksVision

@MainActor
final class PoseLandmarkerService: NSObject, ObservableObject {
    struct Landmark: Identifiable {
        let id = UUID()
        let x: CGFloat
        let y: CGFloat
        let vis: Float
    }

    @Published var landmarks: [Landmark] = []
    @Published var statusText: String = "Starting…"

    private var landmarker: PoseLandmarker?
    private var lastTimestampMs: Int64 = 0

    override init() {
        super.init()
        do {
            guard let path = Bundle.main.path(forResource: "pose_landmarker_lite", ofType: "task") else {
                statusText = "Model not found"
                return
            }

            let options = PoseLandmarkerOptions()
            options.baseOptions.modelAssetPath = path
            options.runningMode = .liveStream
            options.numPoses = 1
            // Keep these a bit lower while testing
            options.minPoseDetectionConfidence = 0.3
            options.minPosePresenceConfidence  = 0.3
            options.minTrackingConfidence      = 0.3

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
                vis: lm.visibility?.floatValue ?? 0
            )
        }

        Task { @MainActor in
            self.statusText = "Tracking"
            self.landmarks = mapped
        }
    }
}

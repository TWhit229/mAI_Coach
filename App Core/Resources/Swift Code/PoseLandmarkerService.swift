import Foundation
import Combine
import AVFoundation
import UIKit
import MediaPipeTasksVision

// MARK: - One Euro Filter
/// Adaptive low-pass filter: smooth when slow, responsive when fast.
/// Based on: https://cristal.univ-lille.fr/~casiez/1euro/
private final class OneEuroFilter {
    private let minCutoff: Double
    private let beta: Double
    private let dCutoff: Double
    
    private var xPrev: Double?
    private var dxPrev: Double = 0.0
    private var tPrev: Double?
    
    init(minCutoff: Double = 1.0, beta: Double = 0.007, dCutoff: Double = 1.0) {
        self.minCutoff = minCutoff
        self.beta = beta
        self.dCutoff = dCutoff
    }
    
    func reset() {
        xPrev = nil
        dxPrev = 0.0
        tPrev = nil
    }
    
    private func smoothingFactor(tE: Double, cutoff: Double) -> Double {
        let r = 2.0 * Double.pi * cutoff * tE
        return r / (r + 1.0)
    }
    
    private func exponentialSmoothing(a: Double, x: Double, xPrev: Double) -> Double {
        return a * x + (1.0 - a) * xPrev
    }
    
    func filter(value: Double, timestamp: Double) -> Double {
        guard let xP = xPrev, let tP = tPrev else {
            xPrev = value
            tPrev = timestamp
            return value
        }
        
        let tE = timestamp - tP
        guard tE > 0 else { return xP }
        
        // Derivative estimation
        let aD = smoothingFactor(tE: tE, cutoff: dCutoff)
        let dx = (value - xP) / tE
        let dxSmoothed = exponentialSmoothing(a: aD, x: dx, xPrev: dxPrev)
        dxPrev = dxSmoothed
        
        // Adaptive cutoff
        let cutoff = minCutoff + beta * abs(dxSmoothed)
        let a = smoothingFactor(tE: tE, cutoff: cutoff)
        
        let xFiltered = exponentialSmoothing(a: a, x: value, xPrev: xP)
        xPrev = xFiltered
        tPrev = timestamp
        
        return xFiltered
    }
}

/// Manages One Euro Filters for all landmark coordinates with skeleton constraints
private final class LandmarkSmoother {
    // MediaPipe landmark indices
    private let LEFT_SHOULDER = 11
    private let RIGHT_SHOULDER = 12
    private let LEFT_ELBOW = 13
    private let RIGHT_ELBOW = 14
    
    private var filtersX: [OneEuroFilter] = []
    private var filtersY: [OneEuroFilter] = []
    private var filtersZ: [OneEuroFilter] = []
    private var previousLandmarks: [PoseLandmarkerService.Landmark]?
    private var previousTimestamp: Double = 0
    
    // Outlier rejection: max jump as fraction of shoulder width
    private let maxJumpFraction: CGFloat = 0.15
    
    // Skeleton length calibration
    private var calibratedLeftUpperArm: CGFloat?
    private var calibratedRightUpperArm: CGFloat?
    private var calibrationSamples: [(left: CGFloat, right: CGFloat)] = []
    private var calibrationComplete = false
    private let minCalibrationSamples = 5
    
    func reset() {
        filtersX.removeAll()
        filtersY.removeAll()
        filtersZ.removeAll()
        previousLandmarks = nil
        calibratedLeftUpperArm = nil
        calibratedRightUpperArm = nil
        calibrationSamples.removeAll()
        calibrationComplete = false
    }
    
    private func distance3D(_ p1: PoseLandmarkerService.Landmark, _ p2: PoseLandmarkerService.Landmark) -> CGFloat {
        let dx = p1.x - p2.x
        let dy = p1.y - p2.y
        let dz = p1.z - p2.z
        return sqrt(dx * dx + dy * dy + dz * dz)
    }
    
    private func calibrateSkeleton(_ landmarks: [PoseLandmarkerService.Landmark]) {
        if calibrationComplete { return }
        guard landmarks.count > RIGHT_ELBOW else { return }
        
        let leftShoulder = landmarks[LEFT_SHOULDER]
        let rightShoulder = landmarks[RIGHT_SHOULDER]
        let leftElbow = landmarks[LEFT_ELBOW]
        let rightElbow = landmarks[RIGHT_ELBOW]
        
        // Require high visibility for calibration
        let minVis: Float = 0.8
        if leftShoulder.vis >= minVis && leftElbow.vis >= minVis &&
           rightShoulder.vis >= minVis && rightElbow.vis >= minVis {
            
            let leftLen = distance3D(leftShoulder, leftElbow)
            let rightLen = distance3D(rightShoulder, rightElbow)
            
            // Sanity check: arm lengths should be reasonable (in normalized coords)
            if leftLen > 0.05 && leftLen < 0.5 && rightLen > 0.05 && rightLen < 0.5 {
                calibrationSamples.append((left: leftLen, right: rightLen))
                
                if calibrationSamples.count >= minCalibrationSamples {
                    // Use median for robustness
                    let leftSorted = calibrationSamples.map { $0.left }.sorted()
                    let rightSorted = calibrationSamples.map { $0.right }.sorted()
                    calibratedLeftUpperArm = leftSorted[leftSorted.count / 2]
                    calibratedRightUpperArm = rightSorted[rightSorted.count / 2]
                    calibrationComplete = true
                }
            }
        }
    }
    
    private func applySkeletonConstraints(_ landmarks: [PoseLandmarkerService.Landmark]) -> [PoseLandmarkerService.Landmark] {
        guard calibrationComplete else { return landmarks }
        guard landmarks.count > RIGHT_ELBOW else { return landmarks }
        
        var corrected = landmarks
        
        // Correct left shoulder if visibility is poor but elbow is good
        let leftShoulder = landmarks[LEFT_SHOULDER]
        let leftElbow = landmarks[LEFT_ELBOW]
        
        if leftShoulder.vis < 0.6 && leftElbow.vis >= 0.6, let calibratedLen = calibratedLeftUpperArm {
            let currentLen = distance3D(leftShoulder, leftElbow)
            if currentLen > 0.01 {
                let ratio = calibratedLen / currentLen
                
                // Only correct if arm appears too short (shoulder drifted toward elbow)
                if ratio > 1.1 {
                    // Direction from elbow to shoulder
                    let dirX = leftShoulder.x - leftElbow.x
                    let dirY = leftShoulder.y - leftElbow.y
                    let dirZ = leftShoulder.z - leftElbow.z
                    let norm = sqrt(dirX * dirX + dirY * dirY + dirZ * dirZ)
                    
                    if norm > 0.001 {
                        let unitX = dirX / norm
                        let unitY = dirY / norm
                        let unitZ = dirZ / norm
                        
                        // New shoulder position at calibrated distance from elbow
                        let newShoulder = PoseLandmarkerService.Landmark(
                            x: leftElbow.x + unitX * calibratedLen,
                            y: leftElbow.y + unitY * calibratedLen,
                            z: leftElbow.z + unitZ * calibratedLen,
                            vis: leftShoulder.vis
                        )
                        corrected[LEFT_SHOULDER] = newShoulder
                    }
                }
            }
        }
        
        // Correct right shoulder similarly
        let rightShoulder = landmarks[RIGHT_SHOULDER]
        let rightElbow = landmarks[RIGHT_ELBOW]
        
        if rightShoulder.vis < 0.6 && rightElbow.vis >= 0.6, let calibratedLen = calibratedRightUpperArm {
            let currentLen = distance3D(rightShoulder, rightElbow)
            if currentLen > 0.01 {
                let ratio = calibratedLen / currentLen
                
                if ratio > 1.1 {
                    let dirX = rightShoulder.x - rightElbow.x
                    let dirY = rightShoulder.y - rightElbow.y
                    let dirZ = rightShoulder.z - rightElbow.z
                    let norm = sqrt(dirX * dirX + dirY * dirY + dirZ * dirZ)
                    
                    if norm > 0.001 {
                        let unitX = dirX / norm
                        let unitY = dirY / norm
                        let unitZ = dirZ / norm
                        
                        let newShoulder = PoseLandmarkerService.Landmark(
                            x: rightElbow.x + unitX * calibratedLen,
                            y: rightElbow.y + unitY * calibratedLen,
                            z: rightElbow.z + unitZ * calibratedLen,
                            vis: rightShoulder.vis
                        )
                        corrected[RIGHT_SHOULDER] = newShoulder
                    }
                }
            }
        }
        
        return corrected
    }
    
    func smooth(landmarks: [PoseLandmarkerService.Landmark], timestampMs: Int64) -> [PoseLandmarkerService.Landmark] {
        let timestamp = Double(timestampMs) / 1000.0
        
        // Initialize filters if needed
        if filtersX.count != landmarks.count {
            filtersX = landmarks.map { _ in OneEuroFilter(minCutoff: 1.0, beta: 0.007, dCutoff: 1.0) }
            filtersY = landmarks.map { _ in OneEuroFilter(minCutoff: 1.0, beta: 0.007, dCutoff: 1.0) }
            filtersZ = landmarks.map { _ in OneEuroFilter(minCutoff: 1.0, beta: 0.007, dCutoff: 1.0) }
        }
        
        // Calibrate skeleton lengths during good visibility
        calibrateSkeleton(landmarks)
        
        // Apply skeleton constraints to correct bad detections
        let constrainedLandmarks = applySkeletonConstraints(landmarks)
        
        // Calculate shoulder width for outlier detection
        var shoulderWidth: CGFloat = 0.2 // default fallback
        if constrainedLandmarks.count > 12 {
            let ls = constrainedLandmarks[11] // left shoulder
            let rs = constrainedLandmarks[12] // right shoulder
            let dx = ls.x - rs.x
            let dy = ls.y - rs.y
            shoulderWidth = max(0.05, sqrt(dx * dx + dy * dy))
        }
        let maxJump = shoulderWidth * maxJumpFraction
        
        var smoothed: [PoseLandmarkerService.Landmark] = []
        
        for i in 0..<constrainedLandmarks.count {
            let lm = constrainedLandmarks[i]
            var useX = lm.x
            var useY = lm.y
            var useZ = lm.z
            
            // Outlier rejection: if landmark jumped too far, use previous position
            if let prev = previousLandmarks, i < prev.count {
                let dx = abs(lm.x - prev[i].x)
                let dy = abs(lm.y - prev[i].y)
                let jump = sqrt(dx * dx + dy * dy)
                
                if jump > maxJump && lm.vis < 0.7 {
                    // Large jump with low visibility = likely bad detection, use previous
                    useX = prev[i].x
                    useY = prev[i].y
                    useZ = prev[i].z
                }
            }
            
            // Apply One Euro Filter
            let filteredX = filtersX[i].filter(value: Double(useX), timestamp: timestamp)
            let filteredY = filtersY[i].filter(value: Double(useY), timestamp: timestamp)
            let filteredZ = filtersZ[i].filter(value: Double(useZ), timestamp: timestamp)
            
            // Blend based on visibility: high vis = use filtered, low vis = use more smoothed
            let visWeight = CGFloat(max(0.3, min(1.0, lm.vis)))
            let smoothWeight = 1.0 - (visWeight * 0.5) // 0.5 to 0.85 smoothing
            
            let finalX = CGFloat(filteredX) * (1.0 - smoothWeight) + (previousLandmarks?[i].x ?? CGFloat(filteredX)) * smoothWeight
            let finalY = CGFloat(filteredY) * (1.0 - smoothWeight) + (previousLandmarks?[i].y ?? CGFloat(filteredY)) * smoothWeight
            let finalZ = CGFloat(filteredZ) * (1.0 - smoothWeight) + (previousLandmarks?[i].z ?? CGFloat(filteredZ)) * smoothWeight
            
            // For first frame or high visibility, just use filtered values
            let outputX = previousLandmarks == nil || lm.vis > 0.8 ? CGFloat(filteredX) : finalX
            let outputY = previousLandmarks == nil || lm.vis > 0.8 ? CGFloat(filteredY) : finalY
            let outputZ = previousLandmarks == nil || lm.vis > 0.8 ? CGFloat(filteredZ) : finalZ
            
            smoothed.append(PoseLandmarkerService.Landmark(x: outputX, y: outputY, z: outputZ, vis: lm.vis))
        }
        
        previousLandmarks = smoothed
        previousTimestamp = timestamp
        return smoothed
    }
}

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

    /// Tracking quality states for UI feedback
    enum TrackingState: String {
        case ready = "Ready"
        case tracking = "Tracking"
        case noPersonBrief = "Move into frame"
        case noPersonPersistent = "Check camera angle"
        case poorVisibility = "Poor lighting"
        case error = "Error"
    }

    @MainActor @Published var landmarks: [Landmark] = []
    @MainActor @Published var lastFrame: PoseFrame?
    @MainActor @Published var statusText: String = "Starting…"
    @MainActor @Published var trackingState: TrackingState = .ready

    private var landmarker: PoseLandmarker?
    private var lastTimestampMs: Int64 = 0
    
    // MARK: - Failure Tracking
    /// Count of consecutive frames with no person detected
    private var noPersonFrameCount: Int = 0
    /// Threshold for brief "no person" warning (at 15fps, 30 frames ≈ 2 seconds)
    private let noPersonBriefThreshold: Int = 30
    /// Threshold for persistent "no person" warning (at 15fps, 60 frames ≈ 4 seconds)
    private let noPersonPersistentThreshold: Int = 60
    
    // MARK: - Advanced Smoothing
    private let smoother = LandmarkSmoother()


    override init() {
        super.init()
        do {
            guard let path = Bundle.main.path(forResource: "pose_landmarker_heavy", ofType: "task") else {
                Task { @MainActor in
                    self.statusText = "Model not found"
                }
                return
            }

            let options = PoseLandmarkerOptions()
            options.baseOptions.modelAssetPath = path
            options.runningMode = .liveStream
            options.numPoses = 1
            // Lowered thresholds for better tracking through occlusion (bench press bottom position)
            options.minPoseDetectionConfidence = 0.4
            options.minPosePresenceConfidence  = 0.4
            options.minTrackingConfidence      = 0.3

            // Live-stream uses a DELEGATE (not a closure listener)
            options.poseLandmarkerLiveStreamDelegate = self

            landmarker = try PoseLandmarker(options: options)
            Task { @MainActor in
                self.statusText = "Ready"
            }
        } catch {
            let errorDesc = error.localizedDescription
            Task { @MainActor in
                self.statusText = "Init error: \(errorDesc)"
            }
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
                self.trackingState = .error
                self.landmarks = []
            }
            return
        }

        guard let result, let first = result.landmarks.first else {
            // No person detected - increment failure counter
            noPersonFrameCount += 1
            
            Task { @MainActor in
                if self.noPersonFrameCount >= self.noPersonPersistentThreshold {
                    self.statusText = "Check camera angle"
                    self.trackingState = .noPersonPersistent
                } else if self.noPersonFrameCount >= self.noPersonBriefThreshold {
                    self.statusText = "Move into frame"
                    self.trackingState = .noPersonBrief
                } else {
                    self.statusText = "Searching..."
                    self.trackingState = .noPersonBrief
                }
                self.landmarks = []
            }
            return
        }
        
        // Person detected - reset failure counter
        noPersonFrameCount = 0

        // Map normalized landmarks (x,y in 0..1). visibility is NSNumber? in this SDK.
        let mapped: [Landmark] = first.map { lm in
            Landmark(
                x: CGFloat(lm.x),
                y: CGFloat(lm.y),
                z: CGFloat(lm.z),
                vis: lm.visibility?.floatValue ?? 0
            )
        }
        
        // Check for poor visibility on key landmarks (shoulders, wrists)
        let keyLandmarkIndices = [11, 12, 15, 16]  // L/R shoulder, L/R wrist
        let avgVisibility = keyLandmarkIndices
            .compactMap { i in i < mapped.count ? mapped[i].vis : nil }
            .reduce(0, +) / Float(keyLandmarkIndices.count)

        Task { @MainActor in
            if avgVisibility < 0.4 {
                self.statusText = "Poor visibility"
                self.trackingState = .poorVisibility
            } else {
                self.statusText = "Tracking"
                self.trackingState = .tracking
            }
            // Apply advanced smoothing with One Euro Filter + outlier rejection
            let smoothed = self.smoother.smooth(landmarks: mapped, timestampMs: Int64(timestampInMilliseconds))
            self.landmarks = smoothed
            self.lastFrame = PoseFrame(timestampMs: Int64(timestampInMilliseconds), landmarks: smoothed)
        }
    }
}


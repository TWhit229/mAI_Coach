import Foundation
import Combine

import AVFoundation

/// Small helper to run the bench MLP on-device and gate predictions by rep boundaries.
@MainActor
final class BenchInferenceEngine: ObservableObject {
    struct Prediction {
        let tags: [String]
        let probabilities: [Double]
    }

    @Published var lastPredictionText: String = "Waiting for a full rep…"
    @Published var lastTags: [String] = []
    @Published var repCount: Int = 0
    @Published var isIdle: Bool = true // Exposed for UI Guide

    private var model: BenchMLPModel?
    private var repDetector = RepDetector()
    private let tagList: [String]
    
    // Audio Feedback
    private let synthesizer = AVSpeechSynthesizer()
    private var lastSpokenTime: Date = .distantPast

    init() {
        if let loaded = BenchMLPModel.loadFromBundle() {
            self.model = loaded
            self.tagList = loaded.tags
        } else {
            self.tagList = [
                "no_major_issues",
                "hands_too_wide",
                "hands_too_narrow",
                "grip_uneven",
                "barbell_tilted",
                "bar_depth_insufficient",
                "incomplete_lockout",
            ]
            lastPredictionText = "Model not found"
        }
        
        // Configure audio session for playback even if silent switch is on (optional, but good for fitness apps)
        try? AVAudioSession.sharedInstance().setCategory(.playback, mode: .spokenAudio, options: .mixWithOthers)
        try? AVAudioSession.sharedInstance().setActive(true)
    }

    /// Feed a frame; runs inference when a rep is detected.
    func handle(frame: PoseLandmarkerService.PoseFrame) {
        // Update idle status for UI
        if repDetector.phase == .idle && !isIdle { isIdle = true }
        else if repDetector.phase != .idle && isIdle { isIdle = false }

        guard let model else { return }
        if let repFrames = repDetector.push(frame: frame) {
            guard let features = FeatureExtractor.computeFeatures(frames: repFrames) else {
                lastPredictionText = "Rep skipped (poor tracking)"
                return
            }
            let probs = model.predict(features: features)
            let selected = zip(tagList, probs)
                .filter { $0.1 >= model.threshold }
                .map { $0.0 }
            
            repCount += 1
            lastTags = selected
            
            // Construct feedback string
            var feedback = "Rep \(repCount)."
            
            if selected.isEmpty {
                lastPredictionText = "Rep \(repCount): no major issues"
                // Optionally give positive reinforcement occasionally
                if repCount % 3 == 0 { feedback += " Good form." }
            } else {
                let issues = selected.joined(separator: ", ")
                lastPredictionText = "Rep \(repCount): " + issues
                // Speak the most prominent issue
                if let firstIssue = selected.first {
                    let spokenIssue = firstIssue.replacingOccurrences(of: "_", with: " ")
                    feedback += " \(spokenIssue)."
                }
            }
            
            speak(feedback)
        }
    }
    
    private func speak(_ text: String) {
        // Prevent overlapping speech if needed, or just queue it. 
        // For reps, we probably want to hear it immediately.
        let utterance = AVSpeechUtterance(string: text)
        utterance.rate = 0.55
        synthesizer.speak(utterance)
    }
    
    func resetSession() {
        repCount = 0
        lastTags = []
        lastPredictionText = "Ready for set"
        speak("Session reset. Get ready.")
    }
}

// MARK: - Rep detection (Bench Press: Start High -> Lower (val increases) -> Push (val decreases) -> End High)
private final class RepDetector {
    enum Phase: String {
        case idle // Arms extended (Top)
        case descending // Arms bending
        case bottom // Arms bent (Chest)
        case ascending // Arms straightening
    }

    private(set) var phase: Phase = .idle
    private var currentFrames: [PoseLandmarkerService.PoseFrame] = []
    
    // Config
    private let lockoutAngle = 150.0 // Degrees. >150 is extended.
    private let bottomAngle = 100.0 // Degrees. <100 is bent.
    private let cooldownMs: Int64 = 800
    private var lastRepTime: Int64 = 0

    func push(frame: PoseLandmarkerService.PoseFrame) -> [PoseLandmarkerService.PoseFrame]? {
        // Calculate average elbow angle
        let lms = frame.landmarks
        guard lms.count > 16 else { return nil }
        // Left: 11-13-15
        let leftAngle = angle(first: lms[11], mid: lms[13], last: lms[15])
        // Right: 12-14-16
        let rightAngle = angle(first: lms[12], mid: lms[14], last: lms[16])
        
        let avgAngle = (leftAngle + rightAngle) / 2.0
        let t = frame.timestampMs
        
        currentFrames.append(frame)
        
        // State Machine
        switch phase {
        case .idle:
            // Waiting for descent.
            // If we are at top (> lockout), we are ready.
            // Transition to descending if we break below lockout significantly?
            // Actually, transition to descending when we clearly start bending.
            // Let's us a buffer. If we go below 140, we are descending.
            if avgAngle < (lockoutAngle - 10) {
                phase = .descending
            }
            // Keep buffer limited while idle to save memory?
            if currentFrames.count > 300 { currentFrames.removeFirst(100) }
            
        case .descending:
             // Expecting angle to drop.
             // Transition to bottom if we drop below bottom threshold.
             if avgAngle < bottomAngle {
                 phase = .bottom
             }
             // Abort if we go back up to lockout without hitting bottom?
             if avgAngle > lockoutAngle {
                 phase = .idle
                 currentFrames.removeAll()
             }
             
        case .bottom:
            // At the bottom. Waiting for ascent.
            // Transition to ascending if we start going up.
            // Let's say if we go back above bottomAngle + buffer
            if avgAngle > (bottomAngle + 10) {
                phase = .ascending
            }
            
        case .ascending:
            // Pushing up.
            // Check for completion (Lockout)
            if avgAngle > lockoutAngle {
                // Completed Rep
                if (t - lastRepTime) > cooldownMs {
                    let repFrames = currentFrames
                    reset(t: t)
                    return repFrames
                } else {
                    // Too fast / double count?
                    reset(t: t)
                }
            }
            // Abort if we drop back down?
            if avgAngle < bottomAngle {
                 phase = .bottom
            }
        }
        
        return nil
    }

    private func reset(t: Int64) {
        currentFrames = []
        phase = .idle
        lastRepTime = t
    }

    private func angle(first: PoseLandmarkerService.Landmark,
                       mid: PoseLandmarkerService.Landmark,
                       last: PoseLandmarkerService.Landmark) -> Double {
        // 3D Angle
        let v1x = first.x - mid.x
        let v1y = first.y - mid.y
        let v1z = first.z - mid.z
        
        let v2x = last.x - mid.x
        let v2y = last.y - mid.y
        let v2z = last.z - mid.z
        
        let dot = v1x * v2x + v1y * v2y + v1z * v2z
        let mag1 = sqrt(v1x * v1x + v1y * v1y + v1z * v1z)
        let mag2 = sqrt(v2x * v2x + v2y * v2y + v2z * v2z)
        
        guard mag1 * mag2 > 1e-6 else { return 0.0 }
        let cosVal = max(-1.0, min(1.0, dot / (mag1 * mag2)))
        return acos(cosVal) * 180.0 / .pi
    }
}

// MARK: - Feature extraction (mirror of Dev_tools pipeline)
private enum FeatureExtractor {
    static func computeFeatures(frames: [PoseLandmarkerService.PoseFrame]) -> [Double]? {
        guard !frames.isEmpty else { return nil }

        var gripRatios: [Double] = []
        var gripUnevenVals: [Double] = []
        var barTilts: [Double] = []
        var wristYVals: [Double] = []
        var badFrames = 0

        let totalFrames = frames.count

        for frame in frames {
            let lms = frame.landmarks
            if lms.count <= 16 {
                badFrames += 1
                continue
            }
            guard lms[11].vis >= 0.7,
                  lms[12].vis >= 0.7,
                  lms[15].vis >= 0.7,
                  lms[16].vis >= 0.7 else {
                badFrames += 1
                continue
            }
            let ls3 = lms[11]
            let rs3 = lms[12]
            let lw3 = lms[15]
            let rw3 = lms[16]
            let shoulderWidth3D = distance3D(ls3, rs3)
            if shoulderWidth3D > 1e-6 {
                let gripWidth = distance3D(lw3, rw3)
                gripRatios.append(gripWidth / shoulderWidth3D)
                let shoulderCenterX = 0.5 * Double(ls3.x + rs3.x)
                let leftOffset = abs(Double(lw3.x) - shoulderCenterX)
                let rightOffset = abs(Double(rw3.x) - shoulderCenterX)
                gripUnevenVals.append(abs(leftOffset - rightOffset) / shoulderWidth3D)
            }

            let dx = Double(rw3.x - lw3.x)
            let dy = Double(rw3.y - lw3.y)
            let angle = abs(atan2(dy, dx)) * 180.0 / Double.pi
            barTilts.append(min(angle, abs(180.0 - angle)))

            let shoulderWidth = hypot(Double(ls3.x - rs3.x), Double(ls3.y - rs3.y))
            if shoulderWidth > 1e-6 {
                let chestY = 0.5 * Double(ls3.y + rs3.y)
                let lwY = (Double(lw3.y) - chestY) / shoulderWidth
                let rwY = (Double(rw3.y) - chestY) / shoulderWidth
                wristYVals.append(0.5 * (lwY + rwY))
            }

            // Visibility check for tracking quality
            var visible = 0
            let ids = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26]
            for idx in ids {
                if lms[idx].vis >= 0.7 { visible += 1 }
            }
            let visibleFraction = Double(visible) / Double(ids.count)
            if visibleFraction < 0.7 {
                badFrames += 1
            }
        }

        let gripMedian = median(of: gripRatios)
        let gripRange = (gripRatios.max() ?? 0) - (gripRatios.min() ?? 0)
        let gripUnevenMedian = median(of: gripUnevenVals)
        let gripUnevenMax = gripUnevenVals.max() ?? 0
        let barTiltMedian = median(of: barTilts)
        let barTiltMax = barTilts.max() ?? 0

        let trackingBadRatio = totalFrames > 0 ? Double(badFrames) / Double(totalFrames) : 1.0
        let trackingQuality = max(0.0, min(1.0, 1.0 - trackingBadRatio))

        let wristMin = wristYVals.min() ?? 0
        let wristMax = wristYVals.max() ?? 0
        let wristRange = wristMax - wristMin

        let loadLbs = 0.0 // Not known at runtime; default to zero for demo

        let feats: [Double] = [
            loadLbs,
            gripMedian,
            gripRange,
            gripUnevenMedian,
            gripUnevenMax,
            barTiltMedian,
            barTiltMax,
            trackingBadRatio,
            trackingQuality,
            wristMin,
            wristMax,
            wristRange,
        ]

        return feats
    }

    private static func distance3D(_ a: PoseLandmarkerService.Landmark, _ b: PoseLandmarkerService.Landmark) -> Double {
        let dx = Double(a.x - b.x)
        let dy = Double(a.y - b.y)
        let dz = Double(a.z - b.z)
        return sqrt(dx * dx + dy * dy + dz * dz)
    }

    private static func median(of values: [Double]) -> Double {
        guard !values.isEmpty else { return 0.0 }
        let sorted = values.sorted()
        let mid = sorted.count / 2
        if sorted.count % 2 == 0 {
            return 0.5 * (sorted[mid - 1] + sorted[mid])
        }
        return sorted[mid]
    }
}

// MARK: - MLP (weights loaded from JSON)
private struct BenchMLPModel {
    let w0: [[Double]]
    let b0: [Double]
    let w1: [[Double]]
    let b1: [Double]
    let w2: [[Double]]
    let b2: [Double]
    let mean: [Double]
    let scale: [Double]
    let tags: [String]
    let threshold: Double

    static func loadFromBundle() -> BenchMLPModel? {
        guard let url = Bundle.main.url(forResource: "bench_mlp_v1_weights", withExtension: "json"),
              let data = try? Data(contentsOf: url),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let layers = json["layers"] as? [String: Any],
              let scaler = json["scaler"] as? [String: Any],
              let meta = json["meta"] as? [String: Any],
              let w0 = layers["0.weight"] as? [[Double]],
              let b0 = layers["0.bias"] as? [Double],
              let w1 = layers["2.weight"] as? [[Double]],
              let b1 = layers["2.bias"] as? [Double],
              let w2 = layers["4.weight"] as? [[Double]],
              let b2 = layers["4.bias"] as? [Double],
              let mean = scaler["mean"] as? [Double],
              let scale = scaler["scale"] as? [Double],
              let tags = meta["tags"] as? [String]
        else {
            return nil
        }
        let threshold = (meta["threshold"] as? Double) ?? 0.5
        return BenchMLPModel(
            w0: w0, b0: b0, w1: w1, b1: b1, w2: w2, b2: b2,
            mean: mean, scale: scale, tags: tags, threshold: threshold
        )
    }

    func predict(features: [Double]) -> [Double] {
        let x = zip(features, mean.indices).map { (features[$1] - mean[$1]) / (scale[$1] == 0 ? 1 : scale[$1]) }
        let h1 = relu(matmul(vec: x, weights: w0, bias: b0))
        let h2 = relu(matmul(vec: h1, weights: w1, bias: b1))
        let logits = matmul(vec: h2, weights: w2, bias: b2)
        return logits.map { 1.0 / (1.0 + exp(-$0)) }
    }

    private func matmul(vec: [Double], weights: [[Double]], bias: [Double]) -> [Double] {
        var out = [Double](repeating: 0, count: bias.count)
        for i in 0..<bias.count {
            var sum = bias[i]
            for j in 0..<vec.count {
                sum += weights[i][j] * vec[j]
            }
            out[i] = sum
        }
        return out
    }

    private func relu(_ v: [Double]) -> [Double] {
        v.map { max(0, $0) }
    }
}

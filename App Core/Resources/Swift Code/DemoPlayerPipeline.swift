import AVFoundation
import SwiftUI
import Combine

/// Demo pipeline: plays the demo clip and feeds frames to MediaPipe.
final class DemoPlayerPipeline: ObservableObject {
    @Published var status: String = "Idle"
    @Published var framesRead: Int = 0
    @Published var lastError: String = ""
    @Published var aspectRatio: CGFloat = 9.0 / 16.0
    private(set) var player: AVPlayer?
    private var reader: AVAssetReader?
    private var output: AVAssetReaderVideoCompositionOutput?
    private let queue = DispatchQueue(label: "demo.pipeline.reader")
    private var isCancelled = false

    func start(url: URL, poseService: PoseLandmarkerService) {
        stop()
        DispatchQueue.main.async {
            self.status = "Loading demo..."
            self.framesRead = 0
            self.lastError = ""
        }
        let asset = AVAsset(url: url)
        guard let track = asset.tracks(withMediaType: .video).first else { return }
        let comp = DemoPlayerPipeline.makeComposition(for: track, duration: asset.duration)
        let rs = comp.renderSize
        DispatchQueue.main.async {
            self.aspectRatio = rs.width / max(rs.height, 1)
        }

        let item = AVPlayerItem(asset: asset)
        item.videoComposition = comp
        let player = AVPlayer(playerItem: item)
        self.player = player

        queue.async { [weak self] in
            self?.runReader(asset: asset, track: track, composition: comp, pose: poseService)
        }
        print("[DemoPipeline] player play()")
        player.play()
        DispatchQueue.main.async {
            self.status = "Playing demo..."
        }
    }

    func stop() {
        isCancelled = true
        queue.sync {
            reader?.cancelReading()
            reader = nil
            output = nil
        }
        player?.pause()
        player = nil
        isCancelled = false
        DispatchQueue.main.async {
            self.status = "Stopped"
        }
    }

    private func runReader(asset: AVAsset, track: AVAssetTrack, composition: AVMutableVideoComposition, pose: PoseLandmarkerService) {
        do {
            reader = try AVAssetReader(asset: asset)
        } catch {
            DispatchQueue.main.async {
                self.lastError = "Reader init failed: \(error.localizedDescription)"
                self.status = "Error"
            }
            return
        }
        print("[DemoPipeline] Reader started")
        let settings: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        let output = AVAssetReaderVideoCompositionOutput(videoTracks: [track], videoSettings: settings)
        output.videoComposition = composition
        output.alwaysCopiesSampleData = false
        guard reader!.canAdd(output) else {
            DispatchQueue.main.async {
                self.lastError = "Cannot add reader output"
                self.status = "Error"
            }
            return
        }
        reader!.add(output)
        self.output = output

        reader!.startReading()
        let orientation: CGImagePropertyOrientation = .up
        var startTime: CFAbsoluteTime?
        var firstPTS: CMTime?
        while reader?.status == .reading, !isCancelled {
            guard let sample = output.copyNextSampleBuffer() else { break }
            DispatchQueue.main.async {
                self.framesRead += 1
            }
            let pts = CMSampleBufferGetPresentationTimeStamp(sample)
            if startTime == nil {
                startTime = CFAbsoluteTimeGetCurrent()
                firstPTS = pts
            }
            if let start = startTime, let first = firstPTS {
                let elapsed = CFAbsoluteTimeGetCurrent() - start
                let target = CMTimeGetSeconds(pts - first)
                let sleepSec = target - elapsed
                if sleepSec > 0 {
                    Thread.sleep(forTimeInterval: sleepSec)
                }
            }
            // Feed inference directly on the reader queue (non-blocking main)
            pose.process(sampleBuffer: sample, orientation: orientation)
        }
        print("[DemoPipeline] Reader finished with status \(reader?.status.rawValue ?? -1) error: \(reader?.error?.localizedDescription ?? "nil")")
        DispatchQueue.main.async {
            if self.isCancelled {
                self.status = "Stopped"
            } else if self.reader?.status == .completed {
                self.status = "Completed"
            } else if self.reader?.status == .failed {
                self.status = "Failed"
                self.lastError = self.reader?.error?.localizedDescription ?? "Reader failed"
            }
        }
    }

    static func orientation(for transform: CGAffineTransform) -> CGImagePropertyOrientation {
        if transform.a == 0 && transform.b == 1.0 && transform.c == -1.0 && transform.d == 0 {
            return .right
        } else if transform.a == 0 && transform.b == -1.0 && transform.c == 1.0 && transform.d == 0 {
            return .left
        } else if transform.a == -1.0 && transform.b == 0 && transform.c == 0 && transform.d == -1.0 {
            return .down
        }
        return .up
    }

    static func makeComposition(for track: AVAssetTrack, duration: CMTime) -> AVMutableVideoComposition {
        let composition = AVMutableVideoComposition()
        let instruction = AVMutableVideoCompositionInstruction()
        instruction.timeRange = CMTimeRange(start: .zero, duration: duration)
        let layerInstruction = AVMutableVideoCompositionLayerInstruction(assetTrack: track)
        // Check if we need to rotate (if width > height)
        let naturalSize = track.naturalSize
        if naturalSize.width > naturalSize.height {
            // Rotate -90 degrees (270 degrees) to fix "upside down" issue.
            // Transform: (x, y) -> (y, -x)
            // (0, 0) -> (0, 0)
            // (W, 0) -> (0, -W)
            // (0, H) -> (H, 0)
            // (W, H) -> (H, -W)
            // Bounds X: [0, H], Bounds Y: [-W, 0]
            // Need to translate Y by +W to bring into [0, W]
            
            let t1 = CGAffineTransform(rotationAngle: -.pi / 2)
            let t2 = CGAffineTransform(translationX: 0, y: naturalSize.width)
            let transform = t1.concatenating(t2)
            
            layerInstruction.setTransform(transform, at: .zero)
            composition.renderSize = CGSize(width: naturalSize.height, height: naturalSize.width)
        } else {
            // Ignore embedded rotation; keep upright and render natural size
            layerInstruction.setTransform(track.preferredTransform, at: .zero)
            let transformedSize = track.naturalSize.applying(track.preferredTransform)
            composition.renderSize = CGSize(width: abs(transformedSize.width), height: abs(transformedSize.height))
        }
        
        instruction.layerInstructions = [layerInstruction]
        composition.instructions = [instruction]
        let fps = track.nominalFrameRate > 0 ? track.nominalFrameRate : 30
        composition.frameDuration = CMTime(value: 1, timescale: CMTimeScale(fps))
        // Reverted SDR forcing per user request
        return composition
    }
}

/// SwiftUI wrapper around AVPlayerLayer
struct DemoPlayerLayer: UIViewRepresentable {
    let player: AVPlayer

    func makeUIView(context: Context) -> PlayerView {
        let view = PlayerView()
        view.playerLayer.player = player
        view.playerLayer.videoGravity = .resizeAspect
        return view
    }

    func updateUIView(_ uiView: PlayerView, context: Context) {
        uiView.playerLayer.player = player
    }
}

final class PlayerView: UIView {
    override static var layerClass: AnyClass { AVPlayerLayer.self }
    var playerLayer: AVPlayerLayer { layer as! AVPlayerLayer }
}

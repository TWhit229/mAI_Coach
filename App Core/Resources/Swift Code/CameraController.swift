import Foundation
import Combine
@preconcurrency import AVFoundation
import UIKit

final class CameraController: NSObject, ObservableObject {
    let session = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let queue = DispatchQueue(label: "camera.queue")

    /// Current camera position (published so UI can reflect Front/Back)
    @Published private(set) var position: AVCaptureDevice.Position = .back

    /// Frames go to whoever sets this (BenchSessionView → PoseLandmarkerService)
    var onFrame: ((CMSampleBuffer, CGImagePropertyOrientation) -> Void)?

    // MARK: - Public
    func start() async {
        guard await authorize() else { return }
        configureOnce()

        // Start OFF the main thread
        await withCheckedContinuation { cont in
            queue.async {
                self.session.startRunning()
                cont.resume()
            }
        }

        UIDevice.current.beginGeneratingDeviceOrientationNotifications()
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(orientationDidChange),
            name: UIDevice.orientationDidChangeNotification,
            object: nil
        )
        applyConnectionOrientation(for: UIDevice.current.orientation)
    }

    func stop() {
        NotificationCenter.default.removeObserver(self, name: UIDevice.orientationDidChangeNotification, object: nil)
        UIDevice.current.endGeneratingDeviceOrientationNotifications()
        queue.async { self.session.stopRunning() }
    }

    /// Toggle between front/back cameras.
    func toggleCamera() {
        let next: AVCaptureDevice.Position = (position == .back) ? .front : .back
        setPosition(next)
    }

    // MARK: - Private
    private var configured = false

    private func configureOnce() {
        guard !configured else { return }
        configured = true

        session.beginConfiguration()
        session.sessionPreset = .high

        // Initial input (defaults to back)
        addVideoInput(position: position)

        // Frames for ML
        videoOutput.alwaysDiscardsLateVideoFrames = true
        // MPImage(sampleBuffer:) expects BGRA; make it explicit
        videoOutput.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        videoOutput.setSampleBufferDelegate(self, queue: queue)
        if session.canAddOutput(videoOutput) { session.addOutput(videoOutput) }

        session.commitConfiguration()
    }

    /// Switch the capture input to a specific position (runs on capture queue).
    private func setPosition(_ newPos: AVCaptureDevice.Position) {
        queue.async {
            self.session.beginConfiguration()

            // Remove existing video inputs
            for input in self.session.inputs {
                if let di = input as? AVCaptureDeviceInput,
                   di.device.hasMediaType(.video) {
                    self.session.removeInput(di)
                }
            }

            // Add the new one
            self.addVideoInput(position: newPos)

            self.session.commitConfiguration()

            // Publish position on main for SwiftUI
            DispatchQueue.main.async {
                self.position = newPos
                self.applyConnectionOrientation(for: UIDevice.current.orientation)
            }
        }
    }

    private func addVideoInput(position: AVCaptureDevice.Position) {
        if let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: position),
           let input = try? AVCaptureDeviceInput(device: device),
           session.canAddInput(input) {
            session.addInput(input)
        }
    }

    @objc private func orientationDidChange() {
        applyConnectionOrientation(for: UIDevice.current.orientation)
    }

    private func applyConnectionOrientation(for deviceOrientation: UIDeviceOrientation) {
        guard let conn = videoOutput.connections.first else { return }
        let angle = rotationAngle(for: deviceOrientation)
        if conn.isVideoRotationAngleSupported(angle) {
            conn.videoRotationAngle = angle
        }
    }

    private func rotationAngle(for o: UIDeviceOrientation) -> CGFloat {
        switch o {
        case .landscapeRight:     return 0
        case .portrait:           return 90
        case .landscapeLeft:      return 180
        case .portraitUpsideDown: return 270
        default:                  return 90
        }
    }

    /// CGImage orientation for MediaPipe, accounting for front/back (front is mirrored).
    private func cgImageOrientation(for o: UIDeviceOrientation) -> CGImagePropertyOrientation {
        if position == .back {
            switch o {
            case .portrait:           return .right
            case .portraitUpsideDown: return .left
            case .landscapeLeft:      return .down   // device rotated left (notch right)
            case .landscapeRight:     return .up
            default:                  return .right
            }
        } else {
            // Front camera is mirrored
            switch o {
            case .portrait:           return .leftMirrored
            case .portraitUpsideDown: return .rightMirrored
            case .landscapeLeft:      return .upMirrored
            case .landscapeRight:     return .downMirrored
            default:                  return .leftMirrored
            }
        }
    }

    private func authorize() async -> Bool {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized: return true
        case .notDetermined:
            return await withCheckedContinuation { cont in
                AVCaptureDevice.requestAccess(for: .video) { cont.resume(returning: $0) }
            }
        default: return false
        }
    }
}

// MARK: - AVCaptureVideoDataOutputSampleBufferDelegate
extension CameraController: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput,
                       didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        let cgOri = cgImageOrientation(for: UIDevice.current.orientation)
        // Ensure PoseLandmarkerService (@MainActor) runs on main
        DispatchQueue.main.async { [weak self] in
            self?.onFrame?(sampleBuffer, cgOri)
        }
    }
}

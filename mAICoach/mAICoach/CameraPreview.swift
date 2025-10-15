import SwiftUI
import AVFoundation

struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession
    var mirrored: Bool = false   // <-- NEW

    func makeUIView(context: Context) -> PreviewView {
        let v = PreviewView()
        v.videoPreviewLayer.session = session
        v.videoPreviewLayer.videoGravity = .resizeAspectFill
        applyMirroring(v)
        return v
    }

    func updateUIView(_ uiView: PreviewView, context: Context) {
        if uiView.videoPreviewLayer.session !== session {
            uiView.videoPreviewLayer.session = session
        }
        applyMirroring(uiView)
    }

    private func applyMirroring(_ view: PreviewView) {
        guard let conn = view.videoPreviewLayer.connection else { return }
        conn.automaticallyAdjustsVideoMirroring = false
        conn.isVideoMirrored = mirrored
    }
}

final class PreviewView: UIView {
    override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
    var videoPreviewLayer: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }
}

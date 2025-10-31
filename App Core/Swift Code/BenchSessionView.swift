import SwiftUI
import AVFoundation

struct BenchSessionView: View {
    @StateObject private var camera = CameraController()
    @StateObject private var pose   = PoseLandmarkerService()

    var body: some View {
        ZStack {
            // Camera behind (mirror when using front camera)
            CameraPreview(session: camera.session,
                          mirrored: camera.position == .front)
                .ignoresSafeArea()

            // Overlay in front (mirror X to match preview)
            PoseOverlay(landmarks: pose.landmarks,
                        mirrorX: camera.position == .front)
                .ignoresSafeArea()
                .allowsHitTesting(false)

            // Top bar: status + camera toggle
            VStack {
                HStack {
                    Text(pose.statusText)
                        .font(.headline)
                        .padding(8)
                        .background(.ultraThinMaterial, in: Capsule())

                    Spacer()

                    Button {
                        camera.toggleCamera()
                    } label: {
                        Label(camera.position == .front ? "Front" : "Back",
                              systemImage: "arrow.triangle.2.circlepath.camera")
                            .labelStyle(.iconOnly)
                            .font(.title3)
                            .padding(10)
                            .background(.ultraThinMaterial, in: Circle())
                    }
                    .accessibilityLabel(
                        camera.position == .front ? "Switch to back camera"
                                                  : "Switch to front camera"
                    )
                }
                .padding(.top, 16)
                .padding(.horizontal, 16)

                Spacer()
            }
        }
        .onAppear {
            Task {
                camera.onFrame = { buffer, orientation in
                    pose.process(sampleBuffer: buffer, orientation: orientation)
                }
                await camera.start()
            }
        }
        .onDisappear { camera.stop() }
        .navigationTitle("Bench (Live)")
        .navigationBarTitleDisplayMode(.inline)
    }
}

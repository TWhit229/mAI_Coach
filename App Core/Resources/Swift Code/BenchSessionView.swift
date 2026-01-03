import SwiftUI
import AVFoundation
import Combine

struct BenchSessionView: View {
    enum Mode {
        case live
        case demo
    }
    
    let mode: Mode
    
    @StateObject private var camera = CameraController()
    @StateObject private var pose   = PoseLandmarkerService()
    @StateObject private var inference = BenchInferenceEngine()
    
    // Demo variables
    @State private var demoPlayer: AVPlayer?
    @StateObject private var demoPipeline = DemoPlayerPipeline()
    @State private var showDevData = false
    
    // MARK: - Tracking State Colors
    private var trackingStateColor: Color {
        switch pose.trackingState {
        case .ready, .tracking:
            return .green
        case .noPersonBrief:
            return .yellow
        case .noPersonPersistent, .poorVisibility:
            return .orange
        case .error:
            return .red
        }
    }
    
    private var trackingStateBackground: some ShapeStyle {
        switch pose.trackingState {
        case .tracking:
            return AnyShapeStyle(.ultraThinMaterial)
        case .noPersonBrief, .noPersonPersistent, .poorVisibility:
            return AnyShapeStyle(Color.black.opacity(0.7))
        case .error:
            return AnyShapeStyle(Color.red.opacity(0.3))
        case .ready:
            return AnyShapeStyle(.ultraThinMaterial)
        }
    }

    var body: some View {
        ZStack {
            if mode == .demo, let player = demoPlayer {
                GeometryReader { geo in
                    ZStack {
                        DemoPlayerLayer(player: player)
                            .onDisappear { player.pause() }
                        PoseOverlay(landmarks: pose.landmarks,
                                    mirrorX: false,
                                    allowedLandmarkIndices: upperBodyIds)
                            .allowsHitTesting(false)
                    }
                    .aspectRatio(demoPipeline.aspectRatio, contentMode: .fit)
                    .frame(maxWidth: geo.size.width, maxHeight: geo.size.height)
                    .frame(maxWidth: geo.size.width, maxHeight: geo.size.height)
                }
                .ignoresSafeArea()
            } else if mode == .live {
                CameraPreview(session: camera.session,
                              mirrored: camera.position == .front)
                    .ignoresSafeArea()
            }

            if mode == .live {
                PoseOverlay(landmarks: pose.landmarks,
                            mirrorX: camera.position == .front,
                            allowedLandmarkIndices: upperBodyIds)
                    .ignoresSafeArea()
                    .allowsHitTesting(false)
                
            }
            
            // Visual Calibration Guide (Only when Idle, for BOTH Live and Demo)
            // Disappears once the set starts (not idle) or after the first rep is done.
            if inference.isIdle && inference.repCount == 0 {
                GeometryReader { geo in
                    let w = geo.size.width
                    let h = geo.size.height
                    
                    ZStack {
                        Image("bench_setup_overlay")
                            .resizable()
                            .aspectRatio(contentMode: .fit)
                            .opacity(0.5) // "See through" ghost effect
                            .frame(width: w * 0.9, height: h * 0.9) // Fill most of the screen
                            .allowsHitTesting(false)
                        
                        VStack {
                            Spacer()
                            Text("Align your body with the guide")
                                .font(.headline)
                                .foregroundColor(.white.opacity(0.9))
                                .padding(8)
                                .background(.black.opacity(0.4), in: Capsule())
                                .padding(.bottom, 40)
                        }
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                }
                .allowsHitTesting(false)
            }

            // Overlays
            VStack {
                // Top Bar
                HStack {
                    // Top Left: Dev Data Toggle
                    Button {
                        showDevData.toggle()
                    } label: {
                        Text("Dev Data")
                            .font(.footnote.weight(.semibold))
                            .padding(8)
                            .background(.ultraThinMaterial, in: Capsule())
                    }
                    
                    Spacer()
                    
                    // Top Right: Controls
                    if mode == .live {
                        Button {
                            inference.resetSession()
                        } label: {
                            Label("Reset", systemImage: "arrow.counterclockwise")
                                .labelStyle(.iconOnly)
                                .font(.title3)
                                .padding(10)
                                .background(.ultraThinMaterial, in: Circle())
                        }
                        
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
                    }
                }
                .padding()
                
                Spacer()
                
                // Bottom Area
                ZStack(alignment: .bottom) {
                    // Center: Tracking & Prediction
                    VStack(spacing: 8) {
                        // Color-coded status based on tracking quality
                        HStack(spacing: 6) {
                            Circle()
                                .fill(trackingStateColor)
                                .frame(width: 10, height: 10)
                            Text(pose.statusText)
                                .font(.subheadline.bold())
                        }
                        .padding(8)
                        .background(trackingStateBackground, in: Capsule())
                        
                        Text(inference.lastPredictionText)
                            .font(.headline)
                            .padding(10)
                            .frame(maxWidth: .infinity)
                            .background(.ultraThinMaterial, in: Capsule())
                    }
                    .padding(.horizontal, 40)

                    
                    // Bottom Left: Dev Data Display
                    if showDevData {
                        VStack(alignment: .leading, spacing: 4) {
                            if mode == .demo {
                                Text("Status: \(demoPipeline.status)")
                                Text("Frames: \(demoPipeline.framesRead)")
                                if !demoPipeline.lastError.isEmpty {
                                    Text("Err: \(demoPipeline.lastError)")
                                }
                            } else {
                                Text("Live Stream")
                            }
                        }
                        .font(.caption2)
                        .padding(8)
                        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 8))
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(.bottom, 60)
                    }
                }
                .padding()
            }
        }
        .onAppear {
            startCurrentMode()
        }
        .onDisappear {
            camera.stop()
            demoPipeline.stop()
        }
        .onReceive(pose.$lastFrame.compactMap { $0 }) { frame in
            inference.handle(frame: frame)
        }
        .navigationTitle(mode == .demo ? "Bench (Demo)" : "Bench (Live)")
        .navigationBarTitleDisplayMode(.inline)
    }

    private func startCurrentMode() {
        camera.stop()
        demoPipeline.stop()
        if mode == .demo {
            if let url = Bundle.main.url(forResource: "demo_bench", withExtension: "mp4") {
                demoPipeline.start(url: url, poseService: pose)
                demoPlayer = demoPipeline.player
                pose.statusText = "Demo playing"
            } else {
                pose.statusText = "Demo video missing"
            }
        } else {
            Task {
                camera.onFrame = { buffer, orientation in
                    pose.process(sampleBuffer: buffer, orientation: orientation)
                }
                await camera.start()
            }
        }
    }
}

private let upperBodyIds: Set<Int> = [11, 12, 13, 14, 15, 16]

import SwiftUI

struct CoachView: View {
    var body: some View {
        VStack(spacing: 24) {
            Image("AppLogo")
                .resizable()
                .scaledToFit()
                .frame(width: 100)

            // Tapping this pushes BenchSessionView onto the nav stack
            // Tapping this pushes BenchSessionView onto the nav stack
            NavigationLink {
                BenchSessionView(mode: .live)
            } label: {
                BigRectButton(title: "Bench (Live)", systemImage: "figure.strengthtraining.traditional")
            }
            
            NavigationLink {
                BenchSessionView(mode: .demo)
            } label: {
                BigRectButton(title: "Demo", systemImage: "play.tv")
            }

            Spacer()
        }
        .padding()
        .navigationTitle("Coach")
        .navigationBarTitleDisplayMode(.inline)
        .background(Color.white.ignoresSafeArea())
    }
}

import SwiftUI

struct CoachView: View {
    var body: some View {
        VStack(spacing: 24) {
            Image("AppLogo")
                .resizable()
                .scaledToFit()
                .frame(width: 100)

            // Tapping this pushes BenchSessionView onto the nav stack
            NavigationLink {
                BenchSessionView()
            } label: {
                BigRectButton(title: "Bench", systemImage: "dumbbell")
            }

            Spacer()
        }
        .padding()
        .navigationTitle("Coach")
        .navigationBarTitleDisplayMode(.inline)
        .background(Color.white.ignoresSafeArea())
    }
}

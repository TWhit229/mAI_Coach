import SwiftUI

struct CoachView: View {
    var body: some View {
        VStack(spacing: 24) {
            Image("AppLogo")
                .resizable()
                .scaledToFit()
                .frame(width: 100)

            // Big “Bench” button (hook up action later)
            BigRectButton(title: "Bench", systemImage: "dumbbell")
                .onTapGesture {
                    // TODO: start Bench flow (camera/model) here
                    print("Bench tapped")
                }

            Spacer()
        }
        .padding()
        .navigationTitle("Coach")
        .navigationBarTitleDisplayMode(.inline)
        .background(Color.white.ignoresSafeArea())
    }
}

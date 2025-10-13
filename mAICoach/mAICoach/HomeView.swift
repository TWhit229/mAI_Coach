import SwiftUI

struct HomeView: View {
    @EnvironmentObject var session: AuthSession

    @State private var showSignIn = false
    @State private var showCreate = false

    var body: some View {
        NavigationStack {
            content
                .navigationTitle("mAI Coach")
                // nav to the sign-in/create forms if you're keeping the auth-on-Home pattern
                .navigationDestination(isPresented: $showSignIn) { SignInView(mode: .signIn) }
                .navigationDestination(isPresented: $showCreate) { SignInView(mode: .create) }
        }
    }

    @ViewBuilder
    private var content: some View {
        switch session.state {
        case .signedOut:
            // ---- AUTH OPTIONS ON HOME (same as before) ----
            VStack(spacing: 16) {
                Image("AppLogo").resizable().scaledToFit().frame(width: 120)

                Text("Welcome").font(.title2).padding(.bottom, 8)

                Button("Sign in") { showSignIn = true }
                    .buttonStyle(.borderedProminent)
                    .frame(maxWidth: .infinity)

                Button("Continue without signing in") { session.continueAsGuest() }
                    .buttonStyle(.bordered)
                    .frame(maxWidth: .infinity)

                Button("Create account") { showCreate = true }
                    .frame(maxWidth: .infinity)

                Spacer()
            }
            .padding()

        case .guest, .signedIn:
            // ---- YOUR MAIN HOME CONTENT ----
            VStack(spacing: 24) {
                Image("AppLogo").resizable().scaledToFit().frame(width: 120)

                // Big rectangle button that navigates to the new screen
                NavigationLink {
                    CoachView()   // <-- the new screen below
                } label: {
                    BigRectButton(title: "Coach me!", systemImage: "figure.strengthtraining.traditional")
                }

                // (Optional) sign out while testing
                Button("Sign out") { session.signOut() }
                    .buttonStyle(.bordered)

                Spacer()
            }
            .padding()
        }
    }
}

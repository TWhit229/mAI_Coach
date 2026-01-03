// BootScreen.swift
import SwiftUI

struct BootScreen: View {
    @State private var showLogo = false

    var body: some View {
        ZStack {
            Color.white.ignoresSafeArea()          // white background
            Image("AppLogo")                       // your PDF asset in Assets
                .resizable()
                .scaledToFit()
                .frame(width: 160)                 // tweak size
                .opacity(showLogo ? 1 : 0)         // fade-in effect
                .animation(.easeOut(duration: 0.45), value: showLogo)
        }
        .onAppear { showLogo = true }              // start the fade-in
    }
}

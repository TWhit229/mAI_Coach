// RootView.swift
import SwiftUI

struct RootView: View {
    @State private var showBoot = true   // state that controls which screen is visible

    var body: some View {
        ZStack {
            // Main app underneath
            HomeView()
                .opacity(showBoot ? 0 : 1)   // hidden until boot finishes

            // Boot screen on top at first
            if showBoot {
                BootScreen()
                    .transition(.opacity)    // when it disappears, fade out
                    .task {
                        // keep boot up briefly, then flip the switch
                        try? await Task.sleep(nanoseconds: 1_200_000_000) // ~1.2s
                        withAnimation(.easeInOut(duration: 0.3)) {
                            showBoot = false
                        }
                    }
            }
        }
    }
}


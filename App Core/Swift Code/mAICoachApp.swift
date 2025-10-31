//
//  mAICoachApp.swift
//  mAICoach
//
//  Created by Travis Whitney on 10/12/25.
//

import SwiftUI

@main
struct MAICoachApp: App {
    @StateObject private var session = AuthSession()
    var body: some Scene {
        WindowGroup { RootView().environmentObject(session) }
    }
}


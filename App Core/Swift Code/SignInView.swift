// SignInView.swift
import SwiftUI

struct SignInView: View {
    enum Mode { case signIn, create }
    let mode: Mode

    @EnvironmentObject var session: AuthSession
    @Environment(\.dismiss) private var dismiss
    @State private var email = ""
    @State private var password = ""
    @State private var working = false
    @State private var error: String?

    var body: some View {
        Form {
            Section {
                TextField("Email", text: $email)
                    .textContentType(.emailAddress)
                    .keyboardType(.emailAddress)
                    .autocapitalization(.none)
                SecureField("Password", text: $password)
                    .textContentType(.password)
            }
            if let error { Text(error).foregroundStyle(.red) }
            Button(action: submit) {
                if working { ProgressView() }
                else { Text(mode == .signIn ? "Sign in" : "Create account") }
            }
            .disabled(working)
        }
        .navigationTitle(mode == .signIn ? "Sign in" : "Create account")
    }

    private func submit() {
        working = true
        Task {
            do {
                switch mode {
                case .signIn: try await session.signIn(email: email, password: password)
                case .create: try await session.createAccount(email: email, password: password)
                }
                dismiss() // HomeView will auto-switch to main content
            } catch { self.error = error.localizedDescription }
            working = false
        }
    }
}

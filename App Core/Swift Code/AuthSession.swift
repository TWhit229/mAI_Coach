import Foundation
import Combine

@MainActor
final class AuthSession: ObservableObject {
    enum State { case signedOut, guest, signedIn(userID: String) }

    @Published private(set) var state: State
    private let key = "auth_user_id" // persist only userID

    init() {
        if let uid = UserDefaults.standard.string(forKey: key), !uid.isEmpty {
            state = .signedIn(userID: uid)
        } else {
            state = .signedOut
        }
    }

    // ---- Guest: DO NOT persist ----
    func continueAsGuest() { state = .guest }

    // ---- Persist signed in ----
    private func setSignedIn(uid: String) {
        UserDefaults.standard.set(uid, forKey: key)
        state = .signedIn(userID: uid)
    }

    func signOut() {
        // If using Firebase, also call try? Auth.auth().signOut()
        UserDefaults.standard.removeObject(forKey: key)
        state = .signedOut
    }

    // Placeholder – you’ll wire to Firebase below
    func signIn(email: String, password: String) async throws {
        // Replace with Firebase call; use returned uid
        guard !email.isEmpty, !password.isEmpty else { throw NSError(domain: "auth", code: 1) }
        let uid = "fake-uid" // <- replace with Firebase user.uid
        setSignedIn(uid: uid)
    }

    func createAccount(email: String, password: String) async throws {
        // Replace with Firebase call; use returned uid
        guard !email.isEmpty, password.count >= 6 else { throw NSError(domain: "auth", code: 2) }
        let uid = "fake-uid" // <- replace with Firebase user.uid
        setSignedIn(uid: uid)
    }
}

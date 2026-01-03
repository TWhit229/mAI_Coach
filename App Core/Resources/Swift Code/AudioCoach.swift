import Foundation
import AVFoundation

/// Plays pre-recorded coaching audio files for natural-sounding feedback.
/// Falls back to TTS if audio file is not found.
final class AudioCoach {
    static let shared = AudioCoach()
    
    private var audioPlayer: AVAudioPlayer?
    private let synthesizer = AVSpeechSynthesizer()
    
    /// Audio file categories
    private let positiveFiles = ["positive_1", "positive_2", "positive_3", "positive_4", "positive_5", "positive_6"]
    private let handsWideFiles = ["hands_wide_1", "hands_wide_2", "hands_wide_3", "hands_wide_4"]
    private let handsNarrowFiles = ["hands_narrow_1", "hands_narrow_2", "hands_narrow_3", "hands_narrow_4"]
    private let gripUnevenFiles = ["grip_uneven_1", "grip_uneven_2", "grip_uneven_3", "grip_uneven_4"]
    private let barTiltedFiles = ["bar_tilted_1", "bar_tilted_2", "bar_tilted_3", "bar_tilted_4"]
    private let depthFiles = ["depth_1", "depth_2", "depth_3", "depth_4"]
    private let lockoutFiles = ["lockout_1", "lockout_2", "lockout_3", "lockout_4"]
    
    private init() {
        // Configure audio session
        try? AVAudioSession.sharedInstance().setCategory(.playback, mode: .spokenAudio, options: .mixWithOthers)
        try? AVAudioSession.sharedInstance().setActive(true)
    }
    
    // MARK: - Public API
    
    /// Play rep count audio (e.g., "Rep 1.")
    func playRep(_ number: Int) {
        let fileName = "rep_\(min(number, 12))"  // We only have up to rep_12
        playAudio(fileName) ?? speakFallback("Rep \(number).")
    }
    
    /// Play positive reinforcement
    func playPositive() {
        let fileName = positiveFiles.randomElement()!
        playAudio(fileName) ?? speakFallback("Good form!")
    }
    
    /// Play coaching for a specific form issue
    func playFormFeedback(for tag: String) {
        let files: [String]
        let fallbackText: String
        
        switch tag {
        case "hands_too_wide":
            files = handsWideFiles
            fallbackText = "Narrow your grip."
        case "hands_too_narrow":
            files = handsNarrowFiles
            fallbackText = "Widen your grip."
        case "grip_uneven":
            files = gripUnevenFiles
            fallbackText = "Even out your grip."
        case "barbell_tilted":
            files = barTiltedFiles
            fallbackText = "Keep the bar level."
        case "bar_depth_insufficient":
            files = depthFiles
            fallbackText = "Go deeper."
        case "incomplete_lockout":
            files = lockoutFiles
            fallbackText = "Lock out at the top."
        default:
            speakFallback(tag.replacingOccurrences(of: "_", with: " "))
            return
        }
        
        let fileName = files.randomElement()!
        playAudio(fileName) ?? speakFallback(fallbackText)
    }
    
    /// Play session reset audio
    func playSessionReset() {
        playAudio("session_reset") ?? speakFallback("Session reset. Get ready.")
    }
    
    /// Play ready audio
    func playReady() {
        playAudio("ready") ?? speakFallback("Ready for your set.")
    }
    
    // MARK: - Private
    
    /// Try to play an audio file, returns true if successful
    @discardableResult
    private func playAudio(_ fileName: String) -> Bool? {
        guard let url = Bundle.main.url(forResource: fileName, withExtension: "mp3") else {
            print("⚠️ AudioCoach: File not found: \(fileName).mp3")
            return nil
        }
        
        do {
            audioPlayer = try AVAudioPlayer(contentsOf: url)
            audioPlayer?.prepareToPlay()
            audioPlayer?.play()
            return true
        } catch {
            print("⚠️ AudioCoach: Failed to play \(fileName): \(error)")
            return nil
        }
    }
    
    /// Fallback to TTS if audio file not available
    private func speakFallback(_ text: String) {
        let utterance = AVSpeechUtterance(string: text)
        utterance.rate = 0.52
        utterance.pitchMultiplier = 1.05
        
        if let voice = AVSpeechSynthesisVoice(language: "en-US") {
            utterance.voice = voice
        }
        
        synthesizer.speak(utterance)
    }
}

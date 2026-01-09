#!/usr/bin/env python3
"""
Generate coaching audio files using OpenAI TTS API.

Usage:
    export OPENAI_API_KEY="your-api-key-here"
    python3 generate_coaching_audio.py
    
This will create audio files in the 'coaching_audio' folder.
"""

import os
from pathlib import Path

# Check for openai package
try:
    from openai import OpenAI
except ImportError:
    print("Installing openai package...")
    os.system("pip install openai")
    from openai import OpenAI

# All coaching phrases organized by category
PHRASES = {
    # Positive reinforcement
    "positive_1": "Good form!",
    "positive_2": "Nice rep!",
    "positive_3": "Looking good!",
    "positive_4": "Keep it up!",
    "positive_5": "Great control!",
    "positive_6": "Perfect!",
    
    # Rep announcements
    "rep_1": "Rep 1.",
    "rep_2": "Rep 2.",
    "rep_3": "Rep 3.",
    "rep_4": "Rep 4.",
    "rep_5": "Rep 5.",
    "rep_6": "Rep 6.",
    "rep_7": "Rep 7.",
    "rep_8": "Rep 8.",
    "rep_9": "Rep 9.",
    "rep_10": "Rep 10.",
    "rep_11": "Rep 11.",
    "rep_12": "Rep 12.",
    
    # Hands too wide
    "hands_wide_1": "Try bringing your hands in a bit.",
    "hands_wide_2": "Your grip is a little wide.",
    "hands_wide_3": "Narrow your grip slightly.",
    "hands_wide_4": "Hands could be closer together.",
    
    # Hands too narrow
    "hands_narrow_1": "Try widening your grip a little.",
    "hands_narrow_2": "Your hands are a bit too narrow.",
    "hands_narrow_3": "Move your hands out wider.",
    "hands_narrow_4": "Widen your grip slightly.",
    
    # Grip uneven
    "grip_uneven_1": "Even out your grip.",
    "grip_uneven_2": "Your hands aren't quite even.",
    "grip_uneven_3": "Check your hand placement.",
    "grip_uneven_4": "Balance your grip.",
    
    # Barbell tilted
    "bar_tilted_1": "Keep the bar level.",
    "bar_tilted_2": "The bar is tilting a bit.",
    "bar_tilted_3": "Even out the bar.",
    "bar_tilted_4": "Watch your bar angle.",
    
    # Depth insufficient
    "depth_1": "Go a little deeper.",
    "depth_2": "Touch your chest.",
    "depth_3": "More depth on that rep.",
    "depth_4": "Bring it down further.",
    
    # Incomplete lockout
    "lockout_1": "Lock out at the top.",
    "lockout_2": "Extend your arms fully.",
    "lockout_3": "Push all the way up.",
    "lockout_4": "Complete the lockout.",
    
    # Session messages
    "session_reset": "Session reset. Get ready.",
    "ready": "Ready for your set.",
}

# OpenAI voice options: alloy, echo, fable, onyx, nova, shimmer
# 'nova' is energetic and clear - good for coaching
VOICE = "nova"


def generate_audio():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        print("\nSet it with:")
        print('  export OPENAI_API_KEY="your-api-key-here"')
        return
    
    client = OpenAI(api_key=api_key)
    
    # Output directory
    output_dir = Path(__file__).parent / "coaching_audio"
    output_dir.mkdir(exist_ok=True)
    
    print(f"🎙️ Generating {len(PHRASES)} audio files with voice '{VOICE}'...")
    print(f"📁 Output: {output_dir}\n")
    
    for name, text in PHRASES.items():
        output_file = output_dir / f"{name}.mp3"
        
        if output_file.exists():
            print(f"⏭️  Skipping {name} (already exists)")
            continue
        
        print(f"🔊 Generating: {name} - \"{text}\"")
        
        try:
            response = client.audio.speech.create(
                model="tts-1",  # or "tts-1-hd" for higher quality
                voice=VOICE,
                input=text,
                speed=1.0
            )
            
            response.stream_to_file(str(output_file))
            print(f"   ✅ Saved: {output_file.name}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print(f"\n✅ Done! Audio files saved to: {output_dir}")
    print("\nNext steps:")
    print("1. Copy the 'coaching_audio' folder to your iOS app bundle")
    print("2. I'll update the Swift code to play these files")


if __name__ == "__main__":
    generate_audio()

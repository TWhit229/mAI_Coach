#!/usr/bin/env python3
"""Add audio files and AudioCoach.swift to the Xcode project."""

from pbxproj import XcodeProject
from pathlib import Path
import os

# Paths
project_path = "/Users/whitney/Documents/mAI_Coach/App Core/mAICoach.xcodeproj/project.pbxproj"
audio_dir = Path("/Users/whitney/Documents/mAI_Coach/App Core/Resources/Audio")
audio_coach_path = "/Users/whitney/Documents/mAI_Coach/App Core/Resources/Swift Code/AudioCoach.swift"

# Load project
project = XcodeProject.load(project_path)

# Find the main target
target_name = "mAICoach"

# Add AudioCoach.swift
print("Adding AudioCoach.swift...")
try:
    project.add_file(audio_coach_path, target_name=target_name)
    print("  ✅ Added AudioCoach.swift")
except Exception as e:
    print(f"  ⚠️ {e}")

# Add audio files
print("\nAdding audio files...")
audio_files = list(audio_dir.glob("*.mp3"))
for audio_file in audio_files:
    try:
        project.add_file(str(audio_file), target_name=target_name)
        print(f"  ✅ Added {audio_file.name}")
    except Exception as e:
        print(f"  ⚠️ {audio_file.name}: {e}")

# Save project
project.save()
print(f"\n✅ Done! Added {len(audio_files)} audio files + AudioCoach.swift to Xcode project.")

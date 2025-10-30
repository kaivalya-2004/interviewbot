#!/usr/bin/env python3
"""
Test script to verify virtual audio setup for Google Meet bot.
Run this before starting the interview to ensure audio routing works.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from app.config.audio_config import VirtualAudioConfig, test_virtual_audio_setup

def main():
    print("\n" + "="*60)
    print("Virtual Audio Setup Test")
    print("="*60 + "\n")
    
    # Run comprehensive test
    success = test_virtual_audio_setup()
    
    print("\n" + "="*60)
    if success:
        print("✅ SUCCESS: Virtual audio is configured correctly!")
        print("\nYou can now:")
        print("  1. Run: streamlit run main_meet.py")
        print("  2. Create a Google Meet link")
        print("  3. Start the bot - it will join with working audio")
    else:
        print("❌ FAILED: Virtual audio is NOT configured")
        print("\nPlease install virtual audio drivers:")
        print("  • Windows: VB-Cable from https://vb-audio.com/Cable/")
        print("  • macOS: BlackHole from https://github.com/ExistentialAudio/BlackHole")
        print("  • Linux: Use PulseAudio (run setup_virtual_audio.sh)")
    print("="*60 + "\n")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
#!/usr/bin/env python3
"""
Audio Setup Diagnostic Script
Tests virtual audio cable configuration for Google Meet bot
"""

import sys
import sounddevice as sd
import numpy as np
import time
from pathlib import Path
from typing import Any # --- FIX: Import Any ---

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from app.config.audio_config import VirtualAudioConfig


def print_header(text):
    """Print formatted header"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)


def test_1_list_devices():
    """Test 1: List all audio devices"""
    print_header("TEST 1: Listing All Audio Devices")
    
    try:
        devices = VirtualAudioConfig.list_all_audio_devices()
        print(f"\n✅ Found {len(devices)} audio devices")
        return True
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False


def test_2_find_virtual_devices():
    """Test 2: Find virtual audio cable devices"""
    print_header("TEST 2: Finding Virtual Audio Devices")
    
    output_device = VirtualAudioConfig.get_virtual_output_device()
    input_device = VirtualAudioConfig.get_virtual_input_device()
    
    success = True
    
    if output_device is not None:
        print(f"✅ Virtual OUTPUT device found: {output_device}")
        print(f"   (Bot will play audio to this device)")
    else:
        print("❌ Virtual OUTPUT device NOT found")
        print("   Install VB-Audio Virtual Cable or similar")
        success = False
    
    if input_device is not None:
        print(f"✅ Virtual INPUT device found: {input_device}")
        print(f"   (Bot will record from this device)")
    else:
        print("❌ Virtual INPUT device NOT found")
        print("   Check your virtual cable installation")
        success = False
    
    return success


def test_3_sample_rate_support():
    """Test 3: Check 48kHz support"""
    print_header("TEST 3: Checking 48kHz Sample Rate Support")
    
    output_device = VirtualAudioConfig.get_virtual_output_device()
    
    if output_device is None:
        print("⚠️  Skipped: No output device found")
        return False
    
    supported = VirtualAudioConfig.verify_sample_rate_support(output_device, 48000)
    
    if supported:
        print("✅ Device supports 48kHz (Google Meet compatible)")
        return True
    else:
        print("⚠️  Device may not support 48kHz - check logs above")
        return False


def test_4_play_test_tone():
    """Test 4: Play test tone through virtual device"""
    print_header("TEST 4: Playing Test Tone")
    
    output_device = VirtualAudioConfig.get_virtual_output_device()
    
    if output_device is None:
        print("⚠️  Skipped: No output device found")
        return False
    
    print("\n🔊 Generating test tone (440Hz for 2 seconds)...")
    print("   If Google Meet is open, you should hear this tone!")
    
    try:
        # Generate 440Hz tone at 48kHz
        duration = 2.0
        sample_rate = 48000
        frequency = 440
        
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Fade in/out to prevent clicking
        fade_samples = int(0.1 * sample_rate)  # 100ms fade
        fade_in = np.linspace(0, 1, fade_samples)
        fade_out = np.linspace(1, 0, fade_samples)
        
        audio = np.sin(2 * np.pi * frequency * t) * 0.5
        audio[:fade_samples] *= fade_in
        audio[-fade_samples:] *= fade_out
        
        print(f"   Playing to device {output_device} at {sample_rate}Hz...")
        
        sd.play(audio, sample_rate, device=output_device)
        sd.wait()
        
        print("✅ Test tone played successfully")
        print("\n   Did you hear the tone in Google Meet?")
        response = input("   Enter 'y' if yes, 'n' if no: ").strip().lower()
        
        return response == 'y'
        
    except Exception as e:
        print(f"❌ Error playing test tone: {e}")
        return False


def test_5_check_dependencies():
    """Test 5: Check required dependencies"""
    print_header("TEST 5: Checking Python Dependencies")
    
    # --- FIX: Update dict value type to allow bool or None ---
    dependencies: dict[str, bool | None] = {
        'sounddevice': None,
        'soundfile': None,
        'numpy': None,
        'librosa': None,
    }
    # --- END FIX ---
    
    all_present = True
    
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"✅ {dep:15s} installed")
            dependencies[dep] = True # Mark as present
        except ImportError:
            print(f"❌ {dep:15s} NOT installed")
            dependencies[dep] = False # Mark as missing
            all_present = False
    
    if not all_present:
        print("\n⚠️  Install missing dependencies:")
        print("   pip install sounddevice soundfile numpy librosa")
    
    return all_present


def generate_report(results):
    """Generate final diagnostic report"""
    print_header("DIAGNOSTIC REPORT")
    
    total_tests = len(results)
    passed_tests = sum(1 for r in results.values() if r)
    
    print(f"\nTests passed: {passed_tests}/{total_tests}")
    print("\nResults:")
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}  {test_name}")
    
    print("\n" + "-"*70)
    
    if passed_tests == total_tests:
        print("\n🎉 All tests passed! Your setup is ready for Google Meet.")
        print("\nNext steps:")
        print("1. Open Google Meet")
        print("2. Go to Settings → Audio")
        print("3. Set Microphone to 'CABLE Input' (or your virtual device)")
        print("4. Start your interview bot")
        
    elif passed_tests >= 3:
        print("\n⚠️  Setup is mostly working, but some issues detected.")
        print("Check the failed tests above and refer to GOOGLE_MEET_AUDIO_SETUP.md")
        
    else:
        print("\n❌ Setup has significant issues.")
        print("\nTroubleshooting steps:")
        print("1. Install VB-Audio Virtual Cable (Windows) or BlackHole (Mac)")
        print("2. Restart your computer after installation")
        print("3. Run this script again")
        print("4. See GOOGLE_MEET_AUDIO_SETUP.md for detailed instructions")


def main():
    """Run all diagnostic tests"""
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║                                                                  ║
    ║          Google Meet Audio Setup Diagnostic Tool                ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    
    results = {}
    
    # Run all tests
    results["1. List Devices"] = test_1_list_devices()
    time.sleep(1)
    
    results["2. Find Virtual Devices"] = test_2_find_virtual_devices()
    time.sleep(1)
    
    results["3. Check 48kHz Support"] = test_3_sample_rate_support()
    time.sleep(1)
    
    results["4. Play Test Tone"] = test_4_play_test_tone()
    time.sleep(1)
    
    results["5. Check Dependencies"] = test_5_check_dependencies()
    time.sleep(1)
    
    # Generate report
    generate_report(results)
    
    print("\n" + "="*70)
    print("Diagnostic complete. See GOOGLE_MEET_AUDIO_SETUP.md for help.")
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDiagnostic cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
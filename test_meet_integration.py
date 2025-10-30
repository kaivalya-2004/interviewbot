# test_meet_integration.py
"""
Testing script for Google Meet integration.
Run this to verify your setup before using in production.
"""

import logging
import time
from pathlib import Path
from typing import Any, cast # --- FIX: Import Any and cast ---

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# At the top of test_meet_integration.py
import warnings
import sys
import atexit

# Suppress Windows handle warnings
if sys.platform == 'win32':
    warnings.filterwarnings('ignore', category=ResourceWarning)

# Track all drivers for cleanup
_active_drivers = []

def cleanup_all_drivers():
    """Cleanup all drivers on exit."""
    for driver in _active_drivers:
        try:
            driver.quit()
        except:
            pass
    _active_drivers.clear()

atexit.register(cleanup_all_drivers)


def test_meet_controller():
    """Test 4: Verify MeetController can be imported and initialized."""
    print("\n" + "="*60)
    print("TEST 4: Meet Controller Import")
    print("="*60)
    
    try:
        from app.services.meet_controller import MeetController
        
        # Use context manager for automatic cleanup
        with MeetController(headless=True) as controller:
            print("✅ PASSED: MeetController initialized")
            
            # Test driver setup
            if controller.setup_driver():
                print("✅ PASSED: Chrome driver setup successful")
                # Add driver to cleanup list only if setup is successful
                if controller.driver:
                    _active_drivers.append(controller.driver)
                return True
            else:
                print("❌ FAILED: Chrome driver setup failed")
                return False
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False
    
def test_selenium_setup():
    """Test 1: Verify Selenium and Chrome driver work."""
    print("\n" + "="*60)
    print("TEST 1: Selenium & Chrome Driver")
    print("="*60)
    
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from webdriver_manager.chrome import ChromeDriverManager
        from selenium.webdriver.chrome.service import Service
        
        options = Options()
        options.add_argument('--headless=new')
        
        driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=options
        )
        
        driver.get('https://www.google.com')
        assert 'Google' in driver.title
        
        driver.quit()
        
        print("✅ PASSED: Selenium and Chrome driver working")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


def test_undetected_chromedriver():
    """Test 2: Verify undetected-chromedriver works."""
    print("\n" + "="*60)
    print("TEST 2: Undetected ChromeDriver")
    print("="*60)
    
    try:
        import undetected_chromedriver as uc
        
        options = uc.ChromeOptions()
        options.add_argument('--headless=new')
        
        driver = uc.Chrome(options=options)
        _active_drivers.append(driver) # Register for cleanup
        
        driver.get('https://www.google.com')
        
        time.sleep(2)
        
        # Check if detected
        is_detected = driver.execute_script(
            "return navigator.webdriver"
        )
        
        driver.quit()
        _active_drivers.remove(driver) # Remove from cleanup list
        
        if is_detected:
            print("⚠️  WARNING: Chrome automation detected (might cause issues)")
        else:
            print("✅ PASSED: Undetected ChromeDriver working")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


def test_audio_devices():
    """Test 3: Verify audio devices are accessible."""
    print("\n" + "="*60)
    print("TEST 3: Audio Devices")
    print("="*60)
    
    try:
        import sounddevice as sd
        
        # --- FIX: Cast query_devices() to DeviceList ---
        devices: sd.DeviceList = cast(sd.DeviceList, sd.query_devices())
        
        print(f"\n📊 Found {len(devices)} audio devices:")
        for idx, device_untyped in enumerate(devices):
            # --- FIX: Cast device to dict ---
            device: dict[str, Any] = device_untyped
            if device['max_input_channels'] > 0:
                print(f"  [{idx}] {device['name']} (Input)")
            if device['max_output_channels'] > 0:
                print(f"  [{idx}] {device['name']} (Output)")
        
        # Check for virtual devices
        virtual_devices = [
            # --- FIX: Cast d to dict ---
            cast(dict, d)['name'] for d in devices 
            if any(keyword in cast(dict, d)['name'].lower() for keyword in 
                   ['cable', 'virtual', 'blackhole', 'loopback'])
        ]
        
        if virtual_devices:
            print(f"\n✅ Found virtual audio devices: {', '.join(virtual_devices)}")
        else:
            print("\n⚠️  WARNING: No virtual audio devices found")
            print("   You need to set up VB-Cable, PulseAudio, or BlackHole")
        
        return len(devices) > 0
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


def test_meet_join(test_meet_url=None):
    """Test 5: Attempt to join a Google Meet call."""
    print("\n" + "="*60)
    print("TEST 5: Google Meet Join")
    print("="*60)
    
    if not test_meet_url:
        print("⏭️  SKIPPED: No test Meet URL provided")
        print("   To test, run: python test_meet_integration.py --meet-url YOUR_MEET_LINK")
        return True
    
    controller = None # Initialize controller outside try block for cleanup
    try:
        from app.services.meet_controller import MeetController
        
        print(f"🔗 Attempting to join: {test_meet_url}")
        
        controller = MeetController(headless=False)  # Visible for testing
        if controller.driver:
            _active_drivers.append(controller.driver)
        
        success = controller.join_meeting(test_meet_url, "Test Bot")
        
        if success:
            print("✅ PASSED: Successfully joined Google Meet")
            print("   Check the Meet call to verify bot is visible")
            
            input("\nPress Enter after verifying bot in Meet...")
            
            controller.leave_meeting()
            controller.cleanup()
            return True
        else:
            print("❌ FAILED: Could not join Google Meet")
            if controller:
                controller.cleanup()
            return False
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        if controller:
            controller.cleanup()
        return False


def test_audio_playback():
    """Test 6: Verify audio can be played through virtual device."""
    print("\n" + "="*60)
    print("TEST 6: Audio Playback")
    print("="*60)
    
    try:
        from app.core.services.audio_service import play_audio_blocking
        import sounddevice as sd
        import soundfile as sf
        import numpy as np
        
        # Create test audio file
        test_audio_path = Path("test_audio.wav")
        
        # Generate 440 Hz tone (1 second)
        sample_rate = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio_data = 0.3 * np.sin(2 * np.pi * 440 * t)
        
        sf.write(str(test_audio_path), audio_data, sample_rate)
        
        print("🔊 Playing test tone (440 Hz)...")
        print("   You should hear a beep if audio routing is correct")
        
        play_audio_blocking(str(test_audio_path))
        
        # Cleanup
        test_audio_path.unlink()
        
        response = input("\nDid you hear the test tone? (y/n): ")
        
        if response.lower() == 'y':
            print("✅ PASSED: Audio playback working")
            return True
        else:
            print("❌ FAILED: Audio not heard - check virtual audio setup")
            return False
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        # Ensure test file is deleted even on error
        if 'test_audio_path' in locals() and test_audio_path.exists():
            test_audio_path.unlink()
        return False


def test_session_manager():
    """Test 7: Verify session manager can create sessions."""
    print("\n" + "="*60)
    print("TEST 7: Session Manager")
    print("="*60)
    
    try:
        from app.services.meet_session_manager import MeetSessionManager
        from app.core.services.database_service import DBHandler
        
        db = DBHandler()
        session_mgr = MeetSessionManager(db)
        
        # Test session creation
        session_data = session_mgr.create_interview_session(
            candidate_id="TEST_001",
            session_id="test_session_123",
            job_position="Software Engineer"
        )
        
        assert session_data['candidate_id'] == "TEST_001"
        assert 'meet_link' in session_data
        
        print(f"✅ PASSED: Session created with Meet link: {session_data['meet_link']}")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


def test_full_integration(test_meet_url=None):
    """Test 8: End-to-end integration test."""
    print("\n" + "="*60)
    print("TEST 8: Full Integration (Optional)")
    print("="*60)
    
    if not test_meet_url:
        print("⏭️  SKIPPED: Requires test Meet URL and manual verification")
        return True
    
    session_mgr = None
    session_id = None
    try:
        from app.services.meet_session_manager import MeetSessionManager
        from app.core.services.database_service import DBHandler
        from app.services.interview_service import InterviewService
        
        # Initialize services
        db = DBHandler()
        session_mgr = MeetSessionManager(db)
        interview_svc = InterviewService()
        
        # Create session
        resume_text = "Sample resume for testing purposes"
        session_id = interview_svc.start_new_interview(resume_text, "TEST_FULL_001")
        
        # Start bot session
        print("🤖 Starting bot session...")
        success = session_mgr.start_bot_session(
            session_id=session_id,
            meet_link=test_meet_url,
            candidate_id="TEST_FULL_001",
            audio_device=None,
            enable_video=False,
            headless=False
        )
        
        if not success:
            print("❌ FAILED: Bot could not join meeting")
            if session_mgr and session_id:
                session_mgr.end_session(session_id)
            return False
        
        print("✅ Bot joined meeting")
        print("   Join from another device to test full flow")
        
        input("\nPress Enter after joining from another device...")
        
        # Check if candidate joined
        # --- FIX: Typo 'wait_for_ candidate' -> 'wait_for_candidate' ---
        if session_mgr.wait_for_candidate(session_id, timeout=60):
            print("✅ PASSED: Full integration working!")
            
        else:
            print("⚠️  WARNING: Candidate detection timeout")
        
        # Cleanup
        session_mgr.end_session(session_id)
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        if session_mgr and session_id:
            try:
                session_mgr.end_session(session_id)
            except Exception as ce:
                print(f"Error during cleanup: {ce}")
        return False


def run_all_tests(test_meet_url=None):
    """Run all tests and generate report."""
    print("\n" + "🔧"*30)
    print("GOOGLE MEET INTEGRATION - TEST SUITE")
    print("🔧"*30)
    
    tests = [
        ("Selenium Setup", test_selenium_setup),
        ("Undetected ChromeDriver", test_undetected_chromedriver),
        ("Audio Devices", test_audio_devices),
        ("Meet Controller", test_meet_controller),
        ("Audio Playback", test_audio_playback),
        ("Session Manager", test_session_manager),
    ]
    
    # Optional tests that require Meet URL
    if test_meet_url:
        tests.extend([
            ("Meet Join", lambda: test_meet_join(test_meet_url)),
            # ("Full Integration", lambda: test_full_integration(test_meet_url)), # Disabling full integration by default
        ])
    else:
        print("\nNote: Skipping Meet Join & Full Integration tests.")
        print("   To run them, use: python test_meet_integration.py --meet-url YOUR_MEET_LINK\n")

    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except KeyboardInterrupt:
            print("\n\n⏸️  Tests interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error in {test_name}: {e}")
            results.append((test_name, False))
    
    # Generate report
    print("\n" + "📊"*30)
    print("TEST REPORT")
    print("📊"*30 + "\n")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status} - {test_name}")
    
    print(f"\n{'='*60}")
    if total > 0:
        print(f"TOTAL: {passed}/{total} tests passed ({passed*100//total}%)")
    else:
        print("TOTAL: 0/0 tests passed")
    print(f"{'='*60}\n")
    
    if passed == total:
        print("🎉 All tests passed! Your setup is ready for Google Meet integration.")
    elif passed >= total * 0.7:
        print("⚠️  Most tests passed. Review failed tests before production use.")
    else:
        print("❌ Multiple tests failed. Please review setup instructions.")
    
    # Ensure all drivers are cleaned up
    cleanup_all_drivers()
    
    return passed == total


if __name__ == "__main__":
    import sys
    
    # Check for Meet URL argument
    test_meet_url = None
    if "--meet-url" in sys.argv:
        idx = sys.argv.index("--meet-url")
        if idx + 1 < len(sys.argv):
            test_meet_url = sys.argv[idx + 1]
    
    # Run all tests
    success = run_all_tests(test_meet_url)
    
    sys.exit(0 if success else 1)
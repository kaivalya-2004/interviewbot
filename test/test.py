# test_video_integration.py
"""
Test script to verify video capture and behavioral analysis integration.
Run this script to ensure all components are working correctly before production.
"""

import os
import sys
from pathlib import Path
import logging
import time
import numpy as np
from PIL import Image

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_imports():
    """Test if all required libraries are installed."""
    print("\n" + "="*60)
    print("TEST 1: Checking Required Imports")
    print("="*60)
    
    required_imports = {
        'streamlit': 'streamlit',
        'streamlit_webrtc': 'streamlit-webrtc',
        'cv2': 'opencv-python',
        'av': 'av',
        'PIL': 'pillow',
        'google.generativeai': 'google-generativeai',
        'pymongo': 'pymongo',
        'sounddevice': 'sounddevice',
        'soundfile': 'soundfile',
        'pydub': 'pydub',
        'speech_recognition': 'SpeechRecognition'
    }
    
    missing = []
    for module, package in required_imports.items():
        try:
            __import__(module)
            print(f"✅ {package:30s} - OK")
        except ImportError:
            print(f"❌ {package:30s} - MISSING")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        return False
    
    print("\n✅ All required packages are installed!")
    return True


def test_environment_variables():
    """Test if required environment variables are set."""
    print("\n" + "="*60)
    print("TEST 2: Checking Environment Variables")
    print("="*60)
    
    from dotenv import load_dotenv
    load_dotenv()
    
    required_vars = ['GEMINI_API_KEY', 'MONGO_URI']
    missing = []
    
    for var in required_vars:
        value = os.getenv(var)
        if value:
            masked = value[:8] + "..." + value[-4:] if len(value) > 12 else "***"
            print(f"✅ {var:20s} - Set ({masked})")
        else:
            print(f"❌ {var:20s} - NOT SET")
            missing.append(var)
    
    if missing:
        print(f"\n⚠️  Missing environment variables: {', '.join(missing)}")
        print("Create a .env file with these variables.")
        return False
    
    print("\n✅ All environment variables are set!")
    return True


def test_video_service():
    """Test VideoCapture service."""
    print("\n" + "="*60)
    print("TEST 3: Testing Video Service")
    print("="*60)
    
    try:
        from app.core.services.video_service import VideoCapture
        
        # Create test instance
        test_user_id = "test_user_123"
        test_session_id = "test_session_456"
        
        video_capture = VideoCapture(
            user_id=test_user_id,
            session_id=test_session_id,
            capture_interval=5.0
        )
        
        print("✅ VideoCapture instance created successfully")
        
        # Test directory creation
        expected_dir = Path("data") / test_user_id / test_session_id / "snapshots"
        if expected_dir.exists():
            print(f"✅ Snapshot directory created: {expected_dir}")
        else:
            print(f"❌ Snapshot directory not found: {expected_dir}")
            return False
        
        # Test snapshot saving with dummy image
        dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        video_capture.start_capture()
        snapshot_path = video_capture.save_snapshot(dummy_frame)
        
        if snapshot_path and Path(snapshot_path).exists():
            print(f"✅ Snapshot saved successfully: {snapshot_path}")
            
            # Verify image can be loaded
            img = Image.open(snapshot_path)
            img_size = img.size
            img.close()  # FIX: Close the image before cleanup
            print(f"✅ Snapshot verified (size: {img_size})")
        else:
            print("❌ Failed to save snapshot")
            return False
        
        video_capture.stop_capture()
        
        # Cleanup test files
        import shutil
        import gc
        
        # Force garbage collection to release file handles
        gc.collect()
        time.sleep(0.5)  # Give OS time to release file handles
        
        test_data_dir = Path("data") / test_user_id
        if test_data_dir.exists():
            try:
                shutil.rmtree(test_data_dir)
                print("✅ Test files cleaned up")
            except PermissionError as e:
                print(f"⚠️  Could not delete test files (still in use): {e}")
                print("   This is not critical - files will be cleaned up on next run")
        
        print("\n✅ Video Service tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Video Service test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_behavioral_analyzer():
    """Test BehavioralAnalyzer service."""
    print("\n" + "="*60)
    print("TEST 4: Testing Behavioral Analyzer")
    print("="*60)
    
    try:
        from app.services.behavioral_analyzer import BehavioralAnalyzer
        
        analyzer = BehavioralAnalyzer()
        print("✅ BehavioralAnalyzer instance created successfully")
        
        # Create test snapshots
        test_user_id = "test_user_789"
        test_session_id = "test_session_101"
        snapshot_dir = Path("data") / test_user_id / test_session_id / "snapshots"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        
        # Create 3 dummy snapshot images
        for i in range(1, 4):
            dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            img = Image.fromarray(dummy_frame)
            img.save(snapshot_dir / f"snapshot_{i:03d}.jpg", "JPEG", quality=95)
            img.close()  # FIX: Close images after saving
        
        print(f"✅ Created 3 test snapshots in {snapshot_dir}")
        
        print("\n⚠️  Skipping actual API call to save API quota")
        print("   To test API integration, uncomment the analysis call below")
        
        # Uncomment to test actual API call (will use API quota)
        # print("🔄 Calling Gemini API for analysis...")
        # analysis_result = analyzer.analyze_interview_snapshots(
        #     test_user_id,
        #     test_session_id
        # )
        # 
        # if analysis_result:
        #     print("✅ Analysis completed successfully")
        #     print(f"   Snapshots analyzed: {analysis_result['total_snapshots_analyzed']}")
        # else:
        #     print("❌ Analysis failed")
        #     return False
        
        # Cleanup test files
        import shutil
        import gc
        
        gc.collect()
        time.sleep(0.5)
        
        test_data_dir = Path("data") / test_user_id
        if test_data_dir.exists():
            try:
                shutil.rmtree(test_data_dir)
                print("✅ Test files cleaned up")
            except PermissionError:
                print("⚠️  Could not delete test files (still in use)")
        
        print("\n✅ Behavioral Analyzer tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Behavioral Analyzer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_file_structure():
    """Test if all required files exist."""
    print("\n" + "="*60)
    print("TEST 5: Checking File Structure")
    print("="*60)
    
    required_files = [
        'main.py',
        'app/core/services/audio_service.py',
        'app/core/services/database_service.py',
        'app/core/services/video_service.py',
        'app/services/interview_service.py',
        'app/services/behavioral_analyzer.py',
        'app/utils/pdf_parser.py',
        'app/utils/response_analyzer.py',
        'app/core/log_config.py',
        'requirements.txt',
        '.env'
    ]
    
    missing = []
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path:50s} - Found")
        else:
            print(f"❌ {file_path:50s} - MISSING")
            missing.append(file_path)
    
    if missing:
        print(f"\n⚠️  Missing files: {', '.join(missing)}")
        return False
    
    print("\n✅ All required files exist!")
    return True


def test_opencv_camera():
    """Test if OpenCV can access camera."""
    print("\n" + "="*60)
    print("TEST 6: Testing Camera Access (OpenCV)")
    print("="*60)
    
    try:
        import cv2
        
        print("🔄 Attempting to access camera...")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Cannot access camera")
            print("   - Check if camera is connected")
            print("   - Check if another application is using the camera")
            print("   - Check camera permissions")
            return False
        
        ret, frame = cap.read()
        
        if ret:
            print(f"✅ Camera accessed successfully")
            print(f"   Frame size: {frame.shape}")
        else:
            print("❌ Cannot read frame from camera")
            cap.release()
            return False
        
        cap.release()
        print("\n✅ Camera test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Camera test failed: {e}")
        return False


def run_all_tests():
    """Run all integration tests."""
    print("\n" + "="*60)
    print("VIDEO INTEGRATION TEST SUITE")
    print("="*60)
    
    tests = [
        ("Import Check", test_imports),
        ("Environment Variables", test_environment_variables),
        ("File Structure", test_file_structure),
        ("Video Service", test_video_service),
        ("Behavioral Analyzer", test_behavioral_analyzer),
        ("Camera Access", test_opencv_camera),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"Test '{test_name}' crashed: {e}")
            results[test_name] = False
        
        time.sleep(0.5)  # Small delay between tests
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:30s} : {status}")
    
    total = len(results)
    passed = sum(results.values())
    
    print("\n" + "-"*60)
    print(f"Total: {passed}/{total} tests passed")
    print("-"*60)
    
    if passed == total:
        print("\n🎉 All tests passed! Your integration is ready.")
        return True
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please fix the issues above.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
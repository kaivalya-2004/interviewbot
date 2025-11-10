# tests/test_meet_controller.py
import pytest
from unittest.mock import Mock, patch, MagicMock
import time
from pathlib import Path
from selenium.common.exceptions import TimeoutException, JavascriptException

# --- Setup Path ---
import sys, os
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, project_root)
# --- End Path Setup ---

from app.services.meet_controller import MeetController


@pytest.fixture
def mock_chrome_driver():
    """Mock Chrome driver"""
    with patch('app.services.meet_controller.uc.Chrome') as mock_chrome:
        driver_instance = MagicMock(name="DriverInstance")
        driver_instance.current_url = "https://meet.google.com/abc-defg-hij"
        driver_instance.get_screenshot_as_png.return_value = b'\x89PNG\r\n' + b'\x00' * 1000
        
        # Mock execute_script for various operations
        driver_instance.execute_script.return_value = True
        
        mock_chrome.return_value = driver_instance
        yield driver_instance


@pytest.fixture
def meet_controller(mock_chrome_driver):
    """Create MeetController instance with mocked driver"""
    controller = MeetController(headless=True, use_vb_audio=False)
    # Patch os.makedirs during setup to avoid creating real dirs
    with patch('os.makedirs'):
        controller.setup_driver()
    return controller


class TestMeetController:
    """Test suite for MeetController"""

    def test_initialization(self):
        """Test controller initialization"""
        controller = MeetController(
            headless=True,
            audio_device_index=1,
            use_vb_audio=True
        )
        
        assert controller.headless is True
        assert controller.audio_device_index == 1
        assert controller.use_vb_audio is True
        assert controller.driver is None

    def test_setup_driver_success(self, mock_chrome_driver):
        """Test successful driver setup"""
        controller = MeetController(headless=True)
        with patch('os.makedirs'):
            result = controller.setup_driver()
        
        assert result is True
        assert controller.driver is not None

    def test_setup_driver_with_vb_audio(self, mock_chrome_driver):
        """Test driver setup with VB Audio configuration"""
        controller = MeetController(headless=True, use_vb_audio=True, audio_device_index=2)
        with patch('os.makedirs'):
            result = controller.setup_driver()
        
        assert result is True
        # Verify audio device configuration was attempted
        assert controller.audio_device_index == 2

    def test_setup_driver_failure(self):
        """Test driver setup failure"""
        with patch('app.services.meet_controller.uc.Chrome') as mock_chrome:
            mock_chrome.side_effect = Exception("Driver setup failed")
            
            controller = MeetController(headless=True)
            with patch('os.makedirs'):
                result = controller.setup_driver()
            
            assert result is False

    def test_join_meeting_success_no_captions(self, meet_controller, mock_chrome_driver):
        """Test successful meeting join (and no caption logic)"""
        meet_link = "https://meet.google.com/abc-defg-hij"
        display_name = "Test Bot"
        
        # Mock the WebDriverWait for button finding
        with patch('app.services.meet_controller.WebDriverWait') as mock_wait:
            mock_button = Mock()
            mock_wait.return_value.until.return_value = mock_button
            
            # --- REMOVED patch.object for 'turn_on_captions' ---
            result = meet_controller.join_meeting(meet_link, display_name)
            
            assert result is True
            assert meet_controller.current_meet_link == meet_link
            mock_chrome_driver.get.assert_called_with(meet_link)

    def test_join_meeting_no_driver(self):
        """Test join meeting without initialized driver"""
        controller = MeetController(headless=True)
        
        result = controller.join_meeting("https://meet.google.com/test", "Bot")
        
        assert result is False

    def test_join_meeting_button_not_found(self, meet_controller, mock_chrome_driver):
        """Test join meeting when join button not found"""
        with patch('app.services.meet_controller.WebDriverWait') as mock_wait:
            mock_wait.return_value.until.side_effect = TimeoutException()
            
            result = meet_controller.join_meeting("https://meet.google.com/test", "Bot")
            
            assert result is False

    # --- REMOVED test_turn_on_captions ---

    def test_enable_microphone(self, meet_controller, mock_chrome_driver):
        """Test enabling microphone"""
        mock_chrome_driver.execute_script.return_value = True
        
        meet_controller.enable_microphone()
        
        assert mock_chrome_driver.execute_script.called

    def test_disable_microphone(self, meet_controller, mock_chrome_driver):
        """Test disabling microphone"""
        mock_chrome_driver.execute_script.return_value = True
        
        meet_controller.disable_microphone()
        
        assert mock_chrome_driver.execute_script.called

    def test_capture_candidate_video_js_success(self, meet_controller, mock_chrome_driver):
        """Test successful video capture via JavaScript"""
        mock_chrome_driver.execute_script.return_value = {
            'data': 'data:image/jpeg;base64,/9j/4AAQSkZJRg==',
            'width': 640,
            'height': 480
        }
        
        result = meet_controller.capture_candidate_video_js()
        
        assert result is not None
        image_bytes, width, height = result
        assert isinstance(image_bytes, bytes)
        assert width == 640
        assert height == 480

    def test_capture_candidate_video_js_no_video(self, meet_controller, mock_chrome_driver):
        """Test video capture when no video element found"""
        mock_chrome_driver.execute_script.return_value = None
        
        result = meet_controller.capture_candidate_video_js()
        
        assert result is None

    def test_capture_candidate_video_screenshot_success(self, meet_controller, mock_chrome_driver):
        """Test successful screenshot capture"""
        mock_chrome_driver.get_screenshot_as_png.return_value = b'\x89PNG\r\n' + b'\x00' * 1000
        
        result = meet_controller.capture_candidate_video_screenshot()
        
        assert result is not None
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_capture_candidate_video_screenshot_failure(self, meet_controller, mock_chrome_driver):
        """Test screenshot capture failure"""
        mock_chrome_driver.get_screenshot_as_png.side_effect = Exception("Screenshot failed")
        
        result = meet_controller.capture_candidate_video_screenshot()
        
        assert result is None

    def test_get_participant_count_via_js_unique_ids(self, meet_controller, mock_chrome_driver):
        """Test getting participant count via NEW JavaScript (unique IDs)"""
        mock_chrome_driver.execute_script.return_value = 2
        
        count = meet_controller.get_participant_count()
        
        assert count == 2
        # Check that the new script was called
        script_call = mock_chrome_driver.execute_script.call_args[0][0]
        assert "[data-participant-id]" in script_call
        assert "uniqueIds.add(id)" in script_call

    def test_get_participant_count_fallback_ui(self, meet_controller, mock_chrome_driver):
        """Test participant count fallback to UI button"""
        # --- UPDATED: Use side_effect to test fallback ---
        # First call (JS) returns None, fallback should be used
        mock_chrome_driver.execute_script.side_effect = [
            None, # JS unique IDs fails
            JavascriptException("Other JS error") # JS video count fails (if it got this far)
        ]
        
        with patch('app.services.meet_controller.WebDriverWait') as mock_wait:
            mock_button = Mock()
            mock_button.text = "Show everyone (3)"
            mock_button.get_attribute.return_value = "Show everyone (3)"
            mock_wait.return_value.until.return_value = mock_button
            
            count = meet_controller.get_participant_count()
            
            assert count == 3
            # Ensure the JS ID script was at least tried
            script_call = mock_chrome_driver.execute_script.call_args_list[0][0][0]
            assert "[data-participant-id]" in script_call

    def test_get_participant_count_fallback_video(self, meet_controller, mock_chrome_driver):
        """Test participant count fallback to video count"""
        # --- UPDATED: Use side_effect to test full fallback chain ---
        mock_chrome_driver.execute_script.side_effect = [
            None, # Call 1: JS for unique IDs fails
            2     # Call 2: JS for video count succeeds
        ]
        
        with patch('app.services.meet_controller.WebDriverWait') as mock_wait:
            # Mock UI button method to fail
            mock_wait.return_value.until.side_effect = TimeoutException("Button not found")
            
            count = meet_controller.get_participant_count()
            
            assert count == 2
            # Check that the second JS script (video count) was called
            script_call = mock_chrome_driver.execute_script.call_args[0][0]
            assert "document.querySelectorAll('video')" in script_call

    def test_get_participant_count_error_handling(self, meet_controller, mock_chrome_driver):
        """Test participant count with all methods failing"""
        mock_chrome_driver.execute_script.side_effect = JavascriptException("All JS methods failed")
        
        with patch('app.services.meet_controller.WebDriverWait') as mock_wait:
            mock_wait.return_value.until.side_effect = TimeoutException("Button not found")
            
            count = meet_controller.get_participant_count()
        
        # Should return 1 as default
        assert count == 1

    def test_leave_meeting_success(self, meet_controller, mock_chrome_driver):
        """Test successful meeting leave"""
        mock_chrome_driver.execute_script.return_value = True
        
        meet_controller.leave_meeting()
        
        # Verify script was executed
        assert mock_chrome_driver.execute_script.called
        script_call = mock_chrome_driver.execute_script.call_args[0][0]
        assert "Leave call" in script_call

    def test_leave_meeting_fallback(self, meet_controller, mock_chrome_driver):
        """Test leave meeting with fallback"""
        # JS method fails, fallback to Selenium
        mock_chrome_driver.execute_script.return_value = False
        
        with patch('app.services.meet_controller.WebDriverWait') as mock_wait:
            mock_button = Mock()
            mock_wait.return_value.until.return_value = mock_button
            
            meet_controller.leave_meeting()
            
            # Should attempt JS
            assert mock_chrome_driver.execute_script.called
            # Should attempt Selenium
            mock_wait.return_value.until.assert_called()

    def test_cleanup_success(self, meet_controller, mock_chrome_driver):
        """Test successful cleanup"""
        meet_controller.cleanup()
        
        assert mock_chrome_driver.close.called
        assert mock_chrome_driver.quit.called
        assert meet_controller.driver is None

    def test_cleanup_with_errors(self, meet_controller, mock_chrome_driver):
        """Test cleanup handles errors gracefully"""
        mock_chrome_driver.close.side_effect = Exception("Close failed")
        
        # Should not raise exception
        meet_controller.cleanup()
        
        # Should still try to quit
        assert mock_chrome_driver.quit.called

    def test_context_manager(self, mock_chrome_driver):
        """Test using controller as context manager"""
        with patch('os.makedirs'):
            with MeetController(headless=True) as controller:
                controller.setup_driver()
                assert controller.driver is not None
        
        # Driver should be cleaned up after context
        assert mock_chrome_driver.close.called

    @pytest.mark.parametrize("headless,expected", [
        (True, True),
        (False, False)
    ])
    def test_headless_mode(self, headless, expected):
        """Test headless mode configuration"""
        controller = MeetController(headless=headless)
        
        assert controller.headless == expected

    def test_persistent_profile(self, mock_chrome_driver):
        """Test using persistent Chrome profile"""
        profile_dir = "/path/to/profile"
        
        controller = MeetController(headless=True, user_data_dir=profile_dir)
        with patch('os.makedirs'):
            controller.setup_driver()
        
        assert controller.user_data_dir == profile_dir

    def test_capture_invalid_base64(self, meet_controller, mock_chrome_driver):
        """Test handling of invalid base64 data in capture"""
        # Return invalid base64
        mock_chrome_driver.execute_script.return_value = {
            'data': 'data:image/jpeg;base64,INVALID!!!',
            'width': 640,
            'height': 480
        }
        
        result = meet_controller.capture_candidate_video_js()
        
        assert result is None

    def test_multiple_join_attempts(self, meet_controller, mock_chrome_driver):
        """Test multiple join attempts (rejoin scenario)"""
        meet_link = "https://meet.google.com/test"
        
        with patch('app.services.meet_controller.WebDriverWait') as mock_wait:
            mock_button = Mock()
            mock_wait.return_value.until.return_value = mock_button
            
            # First join
            result1 = meet_controller.join_meeting(meet_link, "Bot 1")
            assert result1 is True
            
            # Second join (same controller)
            result2 = meet_controller.join_meeting(meet_link, "Bot 2")
            assert result2 is True
            
            # Should have called get twice
            assert mock_chrome_driver.get.call_count == 2
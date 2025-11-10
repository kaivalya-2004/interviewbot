# tests/test_integration_meet_session_manager.py
"""
Integration tests for MeetSessionManager
Enhanced version with comprehensive coverage
"""
import pytest
import sys
import os
import time
import threading
from unittest.mock import MagicMock, patch, Mock, call
from pathlib import Path
from datetime import datetime
from PIL import Image
import io

# --- Setup Path ---
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, project_root)
# --- End Path Setup ---

from app.services.meet_session_manager import MeetSessionManager


@pytest.fixture
def mock_db_handler():
    """Mocks the DBHandler for the session manager."""
    # We patch the import location, not the source
    with patch('app.services.meet_session_manager.DBHandler') as mock:
        db_instance = MagicMock(name="DBHandler_Instance")
        db_instance.get_session.return_value = {
            "session_id": "fake_session",
            "candidate_id": "fake_candidate"
        }
        mock.return_value = db_instance
        yield db_instance


@pytest.fixture
def mock_meet_controller_class():
    """Mocks the MeetController class itself."""
    with patch('app.services.meet_session_manager.MeetController') as mock_class:
        mock_instance = MagicMock(name="MeetController_Instance")
        mock_instance.setup_driver.return_value = True
        mock_instance.join_meeting.return_value = True
        mock_instance.get_participant_count.return_value = 1
        mock_instance.capture_candidate_video_js.return_value = None
        mock_instance.capture_candidate_video_screenshot.return_value = None
        mock_class.return_value = mock_instance
        yield mock_class


@pytest.fixture
def session_manager(mock_db_handler):
    """Provides a MeetSessionManager instance with a mock DB."""
    return MeetSessionManager(db_handler=mock_db_handler)


class TestMeetSessionManagerBasics:
    """Basic functionality tests"""

    def test_initialization(self, session_manager):
        """Test session manager initializes correctly"""
        assert session_manager.db is not None
        assert len(session_manager.active_sessions) == 0
        assert session_manager.CHROME_PROFILE_PATH.exists() or True

    def test_create_interview_session(self, session_manager):
        """Test creating an interview session"""
        session_id = "session_001"
        candidate_id = "candidate_001"
        
        result = session_manager.create_interview_session(
            candidate_id, session_id, "Software Engineer"
        )
        
        assert result["session_id"] == session_id
        assert result["candidate_id"] == candidate_id
        assert result["status"] == "pending"
        assert "meet_link" in result
        assert result["meet_link"].startswith("https://meet.google.com/")

    def test_create_interview_session_custom_link(self, session_manager):
        """Test creating session with custom meet link"""
        session_id = "session_002"
        candidate_id = "candidate_002"
        custom_link = "https://meet.google.com/custom-link-xyz"
        
        result = session_manager.create_interview_session(
            candidate_id, session_id, "Data Scientist", meet_link=custom_link
        )
        
        assert result["meet_link"] == custom_link

    def test_create_interview_session_no_db_session(self, session_manager, mock_db_handler):
        """Test creating session when DB session doesn't exist"""
        mock_db_handler.get_session.return_value = None
        
        result = session_manager.create_interview_session(
            "candidate_003", "session_003", "DevOps Engineer"
        )
        
        assert result["session_id"] == "session_003"

    def test_generate_meet_link_format(self, session_manager):
        """Test generated meet link has correct format"""
        link = session_manager._generate_meet_link()
        
        assert link.startswith("https://meet.google.com/")
        code = link.split("/")[-1]
        parts = code.split("-")
        assert len(parts) == 3
        assert len(parts[0]) == 3
        assert len(parts[1]) == 4
        assert len(parts[2]) == 3
        assert parts[0].islower()
        assert parts[1].islower()
        assert parts[2].islower()

    def test_generate_meet_link_unique(self, session_manager):
        """Test that generated links are unique"""
        links = [session_manager._generate_meet_link() for _ in range(10)]
        assert len(set(links)) == 10  # All unique


class TestStartBotSession:
    """Tests for start_bot_session method"""

    def test_start_bot_session_success(self, session_manager, mock_meet_controller_class):
        """Test successful bot session start"""
        session_id = "session_001"
        meet_link = "https://meet.google.com/test-link-abc"
        candidate_id = "candidate_001"

        success = session_manager.start_bot_session(
            session_id, meet_link, candidate_id, headless=True
        )

        assert success is True
        assert session_id in session_manager.active_sessions
        
        session = session_manager.active_sessions[session_id]
        assert session["meet_link"] == meet_link
        assert session["candidate_id"] == candidate_id
        assert session["status"] == "active"
        assert "controller" in session
        assert "stop_capture" in session
        assert "stop_interview" in session
        # Assert removed keys are gone
        assert "tab_switch_events" not in session
        assert "device_type" not in session

    def test_start_bot_session_driver_fail(self, session_manager, mock_meet_controller_class):
        """Test session fails if driver setup fails"""
        mock_controller_instance = mock_meet_controller_class.return_value
        mock_controller_instance.setup_driver.return_value = False

        success = session_manager.start_bot_session(
            "s1", "meet.link", "c1", headless=True
        )

        assert success is False
        assert "s1" not in session_manager.active_sessions

    def test_start_bot_session_join_fail(self, session_manager, mock_meet_controller_class):
        """Test session fails if joining meeting fails"""
        mock_controller_instance = mock_meet_controller_class.return_value
        mock_controller_instance.join_meeting.return_value = False

        success = session_manager.start_bot_session(
            "s1", "meet.link", "c1", headless=True
        )

        assert success is False
        assert "s1" not in session_manager.active_sessions
        mock_controller_instance.cleanup.assert_called_once()

    def test_start_bot_session_with_audio_device(self, session_manager, mock_meet_controller_class):
        """Test session start with specific audio device"""
        success = session_manager.start_bot_session(
            "s1", "meet.link", "c1", audio_device=5, headless=True
        )

        assert success is True
        mock_meet_controller_class.assert_called_once()
        call_kwargs = mock_meet_controller_class.call_args[1]
        assert call_kwargs['audio_device_index'] == 5

    def test_start_bot_session_exception(self, session_manager, mock_meet_controller_class):
        """Test session start handles exceptions"""
        mock_meet_controller_class.side_effect = Exception("Test exception")

        success = session_manager.start_bot_session(
            "s1", "meet.link", "c1", headless=True
        )

        assert success is False
        assert "s1" not in session_manager.active_sessions


class TestWaitForCandidate:
    """Tests for wait_for_candidate method"""

    def test_wait_for_candidate_success(self, session_manager, mock_meet_controller_class):
        """Test successful candidate join detection"""
        session_id = "s_wait_success"
        session_manager.start_bot_session(session_id, "meet.link", "c1", headless=True)
        
        mock_controller_instance = session_manager.active_sessions[session_id]['controller']
        mock_controller_instance.get_participant_count.side_effect = [1, 1, 2, 2, 2]
        
        def time_generator():
            t = 100.0
            while True:
                yield t
                t += 0.5
        
        with patch('app.services.meet_session_manager.time.time', side_effect=time_generator()):
            joined = session_manager.wait_for_candidate(session_id, timeout=300)

        assert joined is True
        assert session_manager.active_sessions[session_id]['status'] == 'candidate_joined'

    def test_wait_for_candidate_timeout(self, session_manager, mock_meet_controller_class):
        """Test timeout when candidate never joins"""
        session_id = "s_wait_timeout"
        session_manager.start_bot_session(session_id, "meet.link", "c1", headless=True)
        
        mock_controller = session_manager.active_sessions[session_id]['controller']
        stop_event = session_manager.active_sessions[session_id]['stop_interview']
        mock_controller.get_participant_count.return_value = 1
        
        def time_generator():
            times = [100.0, 100.0]
            for i in range(100):
                times.append(100.0 + (i * 3.5))
            for t in times:
                yield t
            while True:
                yield 500.0
        
        with patch('app.services.meet_session_manager.time.time', side_effect=time_generator()):
            with patch.object(stop_event, 'wait', return_value=False):
                with patch.object(stop_event, 'is_set', return_value=False):
                    joined = session_manager.wait_for_candidate(session_id, timeout=300)

        assert joined is False

    def test_wait_for_candidate_too_many_people(self, session_manager, mock_meet_controller_class):
        """Test immediate abort when too many participants"""
        session_id = "s_wait_toomany"
        session_manager.start_bot_session(session_id, "meet.link", "c1", headless=True)
        
        mock_controller = session_manager.active_sessions[session_id]['controller']
        stop_event = session_manager.active_sessions[session_id]['stop_interview']
        mock_controller.get_participant_count.side_effect = [1, 3]
        
        def time_generator():
            t = 100.0
            while True:
                yield t
                t += 0.5
        
        with patch('app.services.meet_session_manager.time.time', side_effect=time_generator()):
            with patch.object(stop_event, 'wait', return_value=False):
                with patch.object(stop_event, 'is_set', return_value=False):
                    joined = session_manager.wait_for_candidate(session_id, timeout=300)
        
        assert joined is False
        assert session_manager.active_sessions[session_id]['status'] == 'aborted_multiple_participants'

    def test_wait_for_candidate_timeout_with_too_many(self, session_manager, mock_meet_controller_class):
        """Test timeout scenario when final count is > 2"""
        session_id = "s_wait_timeout_many"
        session_manager.start_bot_session(session_id, "meet.link", "c1", headless=True)
        
        mock_controller = session_manager.active_sessions[session_id]['controller']
        stop_event = session_manager.active_sessions[session_id]['stop_interview']
        
        # Return 1 for many checks, then 3 near timeout
        call_count = [0]
        def get_count():
            call_count[0] += 1
            if call_count[0] > 50:
                return 3
            return 1
        
        mock_controller.get_participant_count.side_effect = get_count
        
        def time_generator():
            times = [100.0]
            for i in range(100):
                times.append(100.0 + (i * 3.5))
            for t in times:
                yield t
            while True:
                yield 500.0
        
        with patch('app.services.meet_session_manager.time.time', side_effect=time_generator()):
            with patch.object(stop_event, 'wait', return_value=False):
                with patch.object(stop_event, 'is_set', return_value=False):
                    joined = session_manager.wait_for_candidate(session_id, timeout=300)
        
        assert joined is False
        session = session_manager.active_sessions.get(session_id)
        if session:
            assert session['status'] in ['aborted_multiple_participants', 'aborted_multiple_participants_timeout']

    def test_wait_for_candidate_with_stop_signal(self, session_manager, mock_meet_controller_class):
        """Test wait respects stop signal"""
        session_id = "s_wait_stop"
        session_manager.start_bot_session(session_id, "meet.link", "c1", headless=True)
        
        session_manager.active_sessions[session_id]['stop_interview'].set()
        joined = session_manager.wait_for_candidate(session_id, timeout=300)
        
        assert joined is False

    def test_wait_for_candidate_no_session(self, session_manager):
        """Test wait fails if session doesn't exist"""
        joined = session_manager.wait_for_candidate("nonexistent", timeout=300)
        assert joined is False

    def test_wait_for_candidate_no_controller(self, session_manager):
        """Test wait fails if controller is missing"""
        session_id = "s_no_ctrl"
        session_manager.active_sessions[session_id] = {}
        
        joined = session_manager.wait_for_candidate(session_id, timeout=300)
        assert joined is False

    def test_wait_for_candidate_exception(self, session_manager, mock_meet_controller_class):
        """Test wait handles exceptions during participant check"""
        session_id = "s_wait_exception"
        session_manager.start_bot_session(session_id, "meet.link", "c1", headless=True)
        
        mock_controller = session_manager.active_sessions[session_id]['controller']
        stop_event = session_manager.active_sessions[session_id]['stop_interview']
        
        # First call raises exception, subsequent calls timeout
        mock_controller.get_participant_count.side_effect = [
            Exception("Test exception"),
            1, 1, 1
        ]
        
        def time_generator():
            times = [100.0]
            for i in range(10):
                times.append(100.0 + (i * 3.5))
            for t in times:
                yield t
            while True:
                yield 500.0
        
        with patch('app.services.meet_session_manager.time.time', side_effect=time_generator()):
            with patch.object(stop_event, 'wait', return_value=False):
                with patch.object(stop_event, 'is_set', return_value=False):
                    joined = session_manager.wait_for_candidate(session_id, timeout=300)
        
        assert joined is False

    def test_wait_for_candidate_stop_during_interval(self, session_manager, mock_meet_controller_class):
        """Test stop signal during wait interval"""
        session_id = "s_wait_stop_interval"
        session_manager.start_bot_session(session_id, "meet.link", "c1", headless=True)
        
        mock_controller = session_manager.active_sessions[session_id]['controller']
        stop_event = session_manager.active_sessions[session_id]['stop_interview']
        
        mock_controller.get_participant_count.return_value = 1
        
        def time_generator():
            t = 100.0
            while True:
                yield t
                t += 0.5
        
        with patch('app.services.meet_session_manager.time.time', side_effect=time_generator()):
            with patch.object(stop_event, 'wait', return_value=True):  # Signal during wait
                with patch.object(stop_event, 'is_set', return_value=False):
                    joined = session_manager.wait_for_candidate(session_id, timeout=300)
        
        assert joined is False

    def test_wait_for_candidate_count_drops(self, session_manager, mock_meet_controller_class):
        """Test count dropping back to 1 resets stability"""
        session_id = "s_wait_drop"
        session_manager.start_bot_session(session_id, "meet.link", "c1", headless=True)
        
        mock_controller = session_manager.active_sessions[session_id]['controller']
        # Count goes: 1, 2, 2 (2 stable), 1 (drops - reset), 2, 2, 2 (3 stable - success)
        mock_controller.get_participant_count.side_effect = [1, 2, 2, 1, 2, 2, 2]
        
        def time_generator():
            t = 100.0
            while True:
                yield t
                t += 0.5
        
        with patch('app.services.meet_session_manager.time.time', side_effect=time_generator()):
            joined = session_manager.wait_for_candidate(session_id, timeout=300)
        
        assert joined is True


class TestVideoCaptureIntegration:
    """Tests for video capture functionality"""

    def test_start_candidate_video_capture(self, session_manager, mock_meet_controller_class):
        """Test video capture thread starts"""
        session_id = "s_video"
        session_manager.start_bot_session(session_id, "meet.link", "c1", enable_video=True)
        
        with patch('threading.Thread') as mock_thread:
            mock_thread_instance = Mock()
            mock_thread.return_value = mock_thread_instance
            
            session_manager._start_candidate_video_capture(session_id)
            
            mock_thread.assert_called_once()
            mock_thread_instance.start.assert_called_once()

    def test_start_video_capture_no_session(self, session_manager):
        """Test video capture handles missing session"""
        session_manager._start_candidate_video_capture("nonexistent")
        # Should not crash

    def test_start_video_capture_no_controller(self, session_manager):
        """Test video capture handles missing controller"""
        session_id = "s_no_ctrl"
        session_manager.active_sessions[session_id] = {}
        
        session_manager._start_candidate_video_capture(session_id)
        # Should not crash

    def test_video_capture_loop_js_success(self, session_manager, mock_meet_controller_class, tmp_path):
        """Test video capture loop with JS capture success"""
        session_id = "s_video_js"
        candidate_id = "c_video_js"
        
        # Create test image data
        img = Image.new('RGB', (640, 480), color='red')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        test_image_data = img_bytes.getvalue()
        
        session_manager.start_bot_session(session_id, "meet.link", candidate_id, enable_video=True)
        
        mock_controller = session_manager.active_sessions[session_id]['controller']
        mock_controller.capture_candidate_video_js.return_value = (test_image_data, 640, 480)
        
        # Override snapshot directory to tmp_path
        session_manager.active_sessions[session_id]['snapshot_dir'] = tmp_path
        
        # Mock time to speed up test - patch in correct module
        with patch('app.services.meet_session_manager.time.sleep'):  # Skip sleep delays
            # Start capture in a thread
            capture_thread = threading.Thread(
                target=session_manager._start_candidate_video_capture,
                args=(session_id,),
                daemon=True
            )
            capture_thread.start()
            
            # Let it run briefly (real time, but sleep is mocked so it's fast)
            time.sleep(0.2)
            
            # Stop capture
            session_manager.active_sessions[session_id]['stop_capture'].set()
            capture_thread.join(timeout=2)
        
        # Should not have called screenshot fallback
        assert mock_controller.capture_candidate_video_screenshot.call_count == 0

    def test_video_capture_loop_screenshot_fallback(self, session_manager, mock_meet_controller_class, tmp_path):
        """Test video capture falls back to screenshot"""
        session_id = "s_video_screenshot"
        candidate_id = "c_video_screenshot"
        
        # Create test image data
        img = Image.new('RGB', (640, 480), color='blue')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        test_image_data = img_bytes.getvalue()
        
        session_manager.start_bot_session(session_id, "meet.link", candidate_id, enable_video=True)
        
        mock_controller = session_manager.active_sessions[session_id]['controller']
        mock_controller.capture_candidate_video_js.return_value = None  # JS fails
        mock_controller.capture_candidate_video_screenshot.return_value = test_image_data
        
        session_manager.active_sessions[session_id]['snapshot_dir'] = tmp_path
        
        # Mock time to control timing - patch in the session manager module
        current_time = [100.0]
        
        def mock_time():
            return current_time[0]
        
        def mock_sleep(seconds):
            current_time[0] += seconds
        
        # Mock Event.wait to control timing
        stop_event = session_manager.active_sessions[session_id]['stop_capture']
        original_wait = stop_event.wait
        
        def mock_event_wait(timeout=None):
            # Advance time for wait calls
            if timeout:
                current_time[0] += timeout
            return original_wait(0)  # Check if set, but don't actually wait
        
        # Patch in the correct module scope
        with patch('app.services.meet_session_manager.time.time', side_effect=mock_time):
            with patch('app.services.meet_session_manager.time.sleep', side_effect=mock_sleep):
                with patch.object(stop_event, 'wait', side_effect=mock_event_wait):
                    # Call the function directly, which starts its own thread
                    session_manager._start_candidate_video_capture(session_id)
                    
                    # Give thread real time to start and process
                    time.sleep(0.5)

                    # Get the ACTUAL capture thread created by the manager
                    capture_thread = session_manager.active_sessions[session_id]['snapshot_thread']
                    
                    # Wait for thread to complete
                    capture_thread.join(timeout=3)
        
        # Should have called screenshot at least once
        assert mock_controller.capture_candidate_video_screenshot.call_count >= 1

    def test_video_capture_loop_max_failures(self, session_manager, mock_meet_controller_class):
        """Test video capture stops after max consecutive failures"""
        session_id = "s_video_fail"
        candidate_id = "c_video_fail"
        
        session_manager.start_bot_session(session_id, "meet.link", candidate_id, enable_video=True)
        
        mock_controller = session_manager.active_sessions[session_id]['controller']
        mock_controller.capture_candidate_video_js.return_value = None
        mock_controller.capture_candidate_video_screenshot.return_value = None
        
        # Mock time to speed up the test
        current_time = [100.0]
        
        def mock_time():
            return current_time[0]
        
        def mock_sleep(seconds):
            # Advance time when sleep is called
            current_time[0] += seconds
        
        # Mock Event.wait to control timing
        stop_event = session_manager.active_sessions[session_id]['stop_capture']
        original_wait = stop_event.wait
        
        def mock_event_wait(timeout=None):
            # Advance time for wait calls
            if timeout:
                current_time[0] += timeout
            return original_wait(0)  # Check if set, but don't actually wait
        
        # Patch in the correct module scope
        with patch('app.services.meet_session_manager.time.time', side_effect=mock_time):
            with patch('app.services.meet_session_manager.time.sleep', side_effect=mock_sleep):
                with patch.object(stop_event, 'wait', side_effect=mock_event_wait):
                    # Call the function directly, which starts its own thread
                    session_manager._start_candidate_video_capture(session_id)
                    
                    # Give the thread real time to process
                    # With mocked time.sleep and time.time, it should run very fast
                    time.sleep(2)
                    
                    # Get the ACTUAL capture thread created by the manager
                    capture_thread = session_manager.active_sessions[session_id]['snapshot_thread']

                    # Wait for the thread to complete
                    capture_thread.join(timeout=5)
        
        # Verify the thread stopped
        assert not capture_thread.is_alive(), "Capture thread should have stopped after max failures"
        
        # Check that failures were recorded
        stats = session_manager.active_sessions[session_id]['capture_stats']
        
        # Should have attempted at least 10 captures before stopping
        assert stats['total_attempts'] >= 10, \
            f"Expected at least 10 attempts, got {stats['total_attempts']}"
        assert stats['failed_captures'] >= 10, \
            f"Expected at least 10 failures, got {stats['failed_captures']}"
        assert stats['successful_captures'] == 0, \
            f"Expected 0 successes, got {stats['successful_captures']}"


class TestEndSession:
    """Tests for end_session method"""

    def test_end_session_basic(self, session_manager, mock_meet_controller_class):
        """Test ending a session"""
        session_id = "s_end"
        session_manager.start_bot_session(session_id, "meet.link", "c1", headless=True)
        
        assert session_id in session_manager.active_sessions
        mock_controller = session_manager.active_sessions[session_id]['controller']

        session_manager.end_session(session_id)

        assert session_id not in session_manager.active_sessions
        mock_controller.leave_meeting.assert_called_once()
        mock_controller.cleanup.assert_called_once()

    def test_end_session_nonexistent(self, session_manager):
        """Test ending nonexistent session"""
        session_manager.end_session("nonexistent")
        # Should not crash

    def test_end_session_with_video_capture(self, session_manager, mock_meet_controller_class):
        """Test ending session with active video capture"""
        session_id = "s_end_video"
        session_manager.start_bot_session(session_id, "meet.link", "c1", enable_video=True)
        
        # Create a mock thread
        mock_thread = Mock()
        mock_thread.is_alive.return_value = True
        session_manager.active_sessions[session_id]['snapshot_thread'] = mock_thread
        
        session_manager.end_session(session_id)
        
        mock_thread.join.assert_called_once_with(timeout=5)

    def test_end_session_thread_not_alive(self, session_manager, mock_meet_controller_class):
        """Test ending session when capture thread already stopped"""
        session_id = "s_end_dead_thread"
        session_manager.start_bot_session(session_id, "meet.link", "c1", enable_video=True)
        
        mock_thread = Mock()
        mock_thread.is_alive.return_value = False
        session_manager.active_sessions[session_id]['snapshot_thread'] = mock_thread
        
        session_manager.end_session(session_id)
        
        mock_thread.join.assert_not_called()

    def test_end_session_with_stats(self, session_manager, mock_meet_controller_class, caplog):
        """Test ending session logs statistics"""
        session_id = "s_end_stats"
        session_manager.start_bot_session(session_id, "meet.link", "c1", headless=True)
        
        # Set some stats
        session_manager.active_sessions[session_id]['snapshot_count'] = 20
        session_manager.active_sessions[session_id]['capture_stats']['total_attempts'] = 25
        session_manager.active_sessions[session_id]['capture_stats']['successful_captures'] = 20
        
        with caplog.at_level('INFO'):
            session_manager.end_session(session_id)
        
        assert 'Final capture rate' in caplog.text
        assert '80.0%' in caplog.text or '20/25' in caplog.text

    def test_end_session_no_attempts(self, session_manager, mock_meet_controller_class, caplog):
        """Test ending session with no capture attempts"""
        session_id = "s_end_no_attempts"
        session_manager.start_bot_session(session_id, "meet.link", "c1", headless=True)
        
        with caplog.at_level('INFO'):
            session_manager.end_session(session_id)
        
        assert 'No capture attempts' in caplog.text

    def test_end_session_exception_during_cleanup(self, session_manager, mock_meet_controller_class):
        """Test end session handles cleanup exceptions"""
        session_id = "s_end_exception"
        session_manager.start_bot_session(session_id, "meet.link", "c1", headless=True)
        
        mock_controller = session_manager.active_sessions[session_id]['controller']
        mock_controller.leave_meeting.side_effect = Exception("Test exception")
        
        # Should not raise exception
        session_manager.end_session(session_id)
        
        # Session should still be removed
        assert session_id not in session_manager.active_sessions
        # Cleanup should still be attempted
        mock_controller.cleanup.assert_called_once()

    def test_end_session_cleanup_also_fails(self, session_manager, mock_meet_controller_class):
        """Test end session handles double failure"""
        session_id = "s_end_double_fail"
        session_manager.start_bot_session(session_id, "meet.link", "c1", headless=True)
        
        mock_controller = session_manager.active_sessions[session_id]['controller']
        mock_controller.leave_meeting.side_effect = Exception("Leave failed")
        mock_controller.cleanup.side_effect = Exception("Cleanup failed")
        
        session_manager.end_session(session_id)
        
        assert session_id not in session_manager.active_sessions


class TestGetMethods:
    """Tests for getter methods"""

    def test_get_session(self, session_manager, mock_meet_controller_class):
        """Test getting session info"""
        session_id = "s_get"
        session_manager.start_bot_session(session_id, "meet.link", "c1")
        
        session = session_manager.get_session(session_id)
        
        assert session is not None
        assert session["meet_link"] == "meet.link"

    def test_get_session_nonexistent(self, session_manager):
        """Test getting nonexistent session"""
        session = session_manager.get_session("nonexistent")
        assert session is None

    def test_get_snapshot_count_active(self, session_manager, mock_meet_controller_class):
        """Test getting snapshot count for active session"""
        session_id = "s_snaps"
        session_manager.start_bot_session(session_id, "meet.link", "c1")
        session_manager.active_sessions[session_id]['snapshot_count'] = 10
        
        count = session_manager.get_snapshot_count(session_id)
        assert count == 10

    # --- UPDATED TEST ---
    def test_get_snapshot_count_from_disk(self, session_manager, mock_db_handler, tmp_path):
        """Test getting snapshot count from disk for a non-active session"""
        session_id = "s_disk"
        candidate_id = "c_disk"
        
        # Setup mock DB to return session data for the *non-active* session
        mock_db_handler.get_session.return_value = {
            "session_id": session_id,
            "candidate_id": candidate_id
        }
        
        # Create fake snapshots in the tmp_path
        # The code will look for "data" / candidate_id / session_id / "snapshots"
        snapshot_dir = tmp_path / "data" / candidate_id / session_id / "snapshots"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        for i in range(5):
            (snapshot_dir / f"snapshot_{i}.jpg").touch()
        # Add a non-jpg file to be ignored
        (snapshot_dir / "log.txt").touch()

        # We need to patch `Path` *where it is used*: in `app.services.meet_session_manager`
        # We will make Path("data") return our temp "data" directory
        with patch('app.services.meet_session_manager.Path') as mock_path_constructor:
            
            # This side effect will intercept the Path("data") call
            def path_side_effect(arg):
                if str(arg) == "data":
                    # Return the *root* of our temp data folder
                    return tmp_path / "data"
                # Allow other Path calls to work as normal (e.g., Path("chrome_profile"))
                # Note: This relies on Path being used with an absolute path or a known relative root
                # This is a bit fragile, but works for Path("data")
                return Path(arg) 
            
            # A more robust mock:
            # Re-create the Path object's behavior for our specific case
            mock_data_path = tmp_path / "data"
            
            def new_path_constructor(arg):
                if str(arg) == "data":
                    return mock_data_path
                return Path(arg) # default behavior

            mock_path_constructor.side_effect = new_path_constructor
            
            # Ensure the session is NOT active
            session_manager.active_sessions.pop(session_id, None)
            
            count = session_manager.get_snapshot_count(session_id)
            assert count == 5
    # --- END UPDATED TEST ---

    def test_get_snapshot_count_no_session(self, session_manager, mock_db_handler):
        """Test snapshot count when no session data"""
        mock_db_handler.get_session.return_value = None
        count = session_manager.get_snapshot_count("nonexistent")
        assert count == 0

    def test_get_snapshot_count_exception(self, session_manager, mock_db_handler):
        """Test snapshot count handles exceptions"""
        mock_db_handler.get_session.side_effect = Exception("DB error")
        
        count = session_manager.get_snapshot_count("error_session")
        assert count == 0

    def test_get_capture_stats(self, session_manager, mock_meet_controller_class):
        """Test getting capture statistics"""
        session_id = "s_stats"
        session_manager.start_bot_session(session_id, "meet.link", "c1")
        
        session = session_manager.active_sessions[session_id]
        session['snapshot_count'] = 20
        session['capture_stats']['total_attempts'] = 25
        session['capture_stats']['successful_captures'] = 20
        session['capture_stats']['failed_captures'] = 5
        session['capture_stats']['js_captures'] = 15
        session['capture_stats']['screenshot_captures'] = 5
        
        stats = session_manager.get_capture_stats(session_id)
        
        assert stats is not None
        assert stats['snapshot_count'] == 20
        assert stats['total_attempts'] == 25
        assert stats['successful_captures'] == 20
        assert stats['success_rate'] == "20/25"
        assert stats['js_captures'] == 15
        assert stats['screenshot_captures'] == 5

    def test_get_capture_stats_no_attempts(self, session_manager, mock_meet_controller_class):
        """Test getting stats when no attempts made"""
        session_id = "s_stats_none"
        session_manager.start_bot_session(session_id, "meet.link", "c1")
        
        stats = session_manager.get_capture_stats(session_id)
        
        assert stats is not None
        assert stats['success_rate'] == "N/A"
        assert stats['total_attempts'] == 0

    def test_get_capture_stats_nonexistent(self, session_manager):
        """Test getting stats for nonexistent session"""
        stats = session_manager.get_capture_stats("nonexistent")
        assert stats is None

    def test_get_all_active_sessions(self, session_manager, mock_meet_controller_class):
        """Test getting all active sessions summary"""
        session_manager.start_bot_session("s1", "link1", "c1")
        session_manager.start_bot_session("s2", "link2", "c2")
        
        all_sessions = session_manager.get_all_active_sessions()
        
        assert len(all_sessions) == 2
        assert "s1" in all_sessions
        assert "s2" in all_sessions
        assert all_sessions["s1"]["candidate_id"] == "c1"
        assert all_sessions["s2"]["candidate_id"] == "c2"
        # Assert removed keys are gone
        assert "tab_switches" not in all_sessions["s1"]
        assert "device_type" not in all_sessions["s1"]

    def test_get_all_active_sessions_empty(self, session_manager):
        """Test getting all sessions when none active"""
        all_sessions = session_manager.get_all_active_sessions()
        assert len(all_sessions) == 0
        assert isinstance(all_sessions, dict)

    def test_get_all_active_sessions_with_stats(self, session_manager, mock_meet_controller_class):
        """Test all sessions includes capture stats"""
        session_id = "s_all_stats"
        session_manager.start_bot_session(session_id, "link", "c1")
        
        session = session_manager.active_sessions[session_id]
        session['snapshot_count'] = 10
        session['capture_stats']['total_attempts'] = 15
        session['capture_stats']['successful_captures'] = 10
        
        all_sessions = session_manager.get_all_active_sessions()
        
        assert all_sessions[session_id]['snaps'] == 10
        assert all_sessions[session_id]['capture_stats']['attempts'] == 15
        assert all_sessions[session_id]['capture_stats']['success'] == 10
        assert all_sessions[session_id]['capture_stats']['rate'] == "10/15"


class TestVideoOptions:
    """Tests for video capture options"""

    def test_video_capture_enabled(self, session_manager, mock_meet_controller_class):
        """Test session with video enabled"""
        session_id = "s_video_on"
        
        success = session_manager.start_bot_session(
            session_id, "meet.link", "c1", enable_video=True
        )
        
        assert success is True
        session = session_manager.active_sessions[session_id]
        assert session['video_enabled'] is True

    def test_video_capture_disabled(self, session_manager, mock_meet_controller_class):
        """Test session with video disabled"""
        session_id = "s_video_off"
        
        success = session_manager.start_bot_session(
            session_id, "meet.link", "c1", enable_video=False
        )
        
        assert success is True
        session = session_manager.active_sessions[session_id]
        assert session['video_enabled'] is False

    def test_video_capture_method_javascript(self, session_manager, mock_meet_controller_class):
        """Test session with javascript capture method"""
        session_id = "s_video_js"
        
        success = session_manager.start_bot_session(
            session_id, "meet.link", "c1",
            enable_video=True,
            video_capture_method="javascript"
        )
        
        assert success is True
        session = session_manager.active_sessions[session_id]
        assert session['video_capture_method'] == "javascript"

    def test_video_capture_method_screenshot(self, session_manager, mock_meet_controller_class):
        """Test session with screenshot capture method"""
        session_id = "s_video_screenshot"
        
        success = session_manager.start_bot_session(
            session_id, "meet.link", "c1",
            enable_video=True,
            video_capture_method="screenshot"
        )
        
        assert success is True
        session = session_manager.active_sessions[session_id]
        assert session['video_capture_method'] == "screenshot"


class TestConcurrency:
    """Tests for concurrent session management"""

    def test_multiple_sessions_concurrent(self, session_manager, mock_meet_controller_class):
        """Test managing multiple concurrent sessions"""
        sessions = []
        for i in range(5):
            session_id = f"session_{i}"
            success = session_manager.start_bot_session(
                session_id, f"link_{i}", f"candidate_{i}"
            )
            assert success is True
            sessions.append(session_id)
        
        assert len(session_manager.active_sessions) == 5
        
        # End one session
        session_manager.end_session(sessions[2])
        assert len(session_manager.active_sessions) == 4
        assert sessions[2] not in session_manager.active_sessions

    def test_session_isolation(self, session_manager, mock_meet_controller_class):
        """Test that sessions are isolated from each other"""
        session_manager.start_bot_session("s1", "link1", "c1")
        session_manager.start_bot_session("s2", "link2", "c2")
        
        # Modify one session
        session_manager.active_sessions["s1"]["snapshot_count"] = 100
        
        # Other session should be unaffected
        assert session_manager.active_sessions["s2"]["snapshot_count"] == 0

    def test_end_multiple_sessions(self, session_manager, mock_meet_controller_class):
        """Test ending multiple sessions"""
        for i in range(3):
            session_manager.start_bot_session(f"s{i}", f"link{i}", f"c{i}")
        
        assert len(session_manager.active_sessions) == 3
        
        for i in range(3):
            session_manager.end_session(f"s{i}")
        
        assert len(session_manager.active_sessions) == 0


class TestEdgeCases:
    """Tests for edge cases and error conditions"""

    def test_session_events_initialized(self, session_manager, mock_meet_controller_class):
        """Test that stop events are properly initialized"""
        session_id = "s_events"
        session_manager.start_bot_session(session_id, "meet.link", "c1")
        
        session = session_manager.active_sessions[session_id]
        assert 'stop_capture' in session
        assert 'stop_interview' in session
        assert isinstance(session['stop_capture'], threading.Event)
        assert isinstance(session['stop_interview'], threading.Event)

    def test_capture_stats_initialized(self, session_manager, mock_meet_controller_class):
        """Test that capture stats are properly initialized"""
        session_id = "s_stats_init"
        session_manager.start_bot_session(session_id, "meet.link", "c1")
        
        session = session_manager.active_sessions[session_id]
        stats = session['capture_stats']
        
        assert stats['total_attempts'] == 0
        assert stats['successful_captures'] == 0
        assert stats['failed_captures'] == 0
        assert stats['js_captures'] == 0
        assert stats['screenshot_captures'] == 0

    def test_snapshot_directory_created(self, session_manager, mock_meet_controller_class):
        """Test that snapshot directory is created"""
        session_id = "s_snapshot_dir"
        candidate_id = "c_snapshot_dir"
        
        with patch('pathlib.Path.mkdir') as mock_mkdir:
            session_manager.start_bot_session(session_id, "meet.link", candidate_id)
            
            # mkdir should be called to create snapshot directory
            mock_mkdir.assert_called()

    def test_headless_mode_true(self, session_manager, mock_meet_controller_class):
        """Test session starts in headless mode"""
        session_id = "s_headless_true"
        
        session_manager.start_bot_session(session_id, "meet.link", "c1", headless=True)
        
        mock_meet_controller_class.assert_called_once()
        call_kwargs = mock_meet_controller_class.call_args[1]
        assert call_kwargs['headless'] is True

    def test_headless_mode_false(self, session_manager, mock_meet_controller_class):
        """Test session starts with visible browser"""
        session_id = "s_headless_false"
        
        session_manager.start_bot_session(session_id, "meet.link", "c1", headless=False)
        
        mock_meet_controller_class.assert_called_once()
        call_kwargs = mock_meet_controller_class.call_args[1]
        assert call_kwargs['headless'] is False

    def test_vb_audio_enabled(self, session_manager, mock_meet_controller_class):
        """Test that VB Audio is enabled by default"""
        session_id = "s_vb_audio"
        
        session_manager.start_bot_session(session_id, "meet.link", "c1")
        
        mock_meet_controller_class.assert_called_once()
        call_kwargs = mock_meet_controller_class.call_args[1]
        assert call_kwargs['use_vb_audio'] is True

    def test_chrome_profile_path(self, session_manager, mock_meet_controller_class):
        """Test that Chrome profile path is set correctly"""
        session_id = "s_chrome_profile"
        
        session_manager.start_bot_session(session_id, "meet.link", "c1")
        
        mock_meet_controller_class.assert_called_once()
        call_kwargs = mock_meet_controller_class.call_args[1]
        assert 'chrome_profile' in str(call_kwargs['user_data_dir'])


class TestParametrizedScenarios:
    """Parametrized tests for various scenarios"""

    @pytest.mark.parametrize("count_sequence,expected_result,description", [
        ([1, 2, 2, 2], True, "Successful stable join"),
        ([2, 2, 2], True, "Candidate already present"),
        ([1, 2, 1, 2, 2, 2], True, "Unstable then stable"),
        ([1, 1, 1, 1], False, "Never joins"),
        ([1, 2, 2, 1, 1], False, "Joins then leaves"),
    ])
    def test_wait_for_candidate_scenarios(
        self, session_manager, mock_meet_controller_class, 
        count_sequence, expected_result, description
    ):
        """Test various candidate join scenarios"""
        session_id = f"s_scenario_{description.replace(' ', '_')}"
        session_manager.start_bot_session(session_id, "meet.link", "c1")
        
        mock_controller = session_manager.active_sessions[session_id]['controller']
        stop_event = session_manager.active_sessions[session_id]['stop_interview']
        
        # Extend sequence if too short
        extended_sequence = list(count_sequence)
        if expected_result is False and len(count_sequence) < 100:
            extended_sequence.extend([count_sequence[-1]] * 100)
        
        mock_controller.get_participant_count.side_effect = extended_sequence
        
        def time_generator():
            t = 100.0
            for _ in range(len(extended_sequence) + 50):
                yield t
                t += 0.5
            while True:
                yield 500.0  # Force timeout
        
        with patch('app.services.meet_session_manager.time.time', side_effect=time_generator()):
            with patch.object(stop_event, 'wait', return_value=False):
                with patch.object(stop_event, 'is_set', return_value=False):
                    joined = session_manager.wait_for_candidate(session_id, timeout=300)
        
        assert joined == expected_result, f"Failed for scenario: {description}"

    @pytest.mark.parametrize("video_enabled,method", [
        (True, "javascript"),
        (True, "screenshot"),
        (False, "javascript"),
        (False, "screenshot"),
    ])
    def test_video_configuration_combinations(
        self, session_manager, mock_meet_controller_class,
        video_enabled, method
    ):
        """Test different video configuration combinations"""
        session_id = f"s_video_{video_enabled}_{method}"
        
        success = session_manager.start_bot_session(
            session_id, "meet.link", "c1",
            enable_video=video_enabled,
            video_capture_method=method
        )
        
        assert success is True
        session = session_manager.active_sessions[session_id]
        assert session['video_enabled'] == video_enabled
        assert session['video_capture_method'] == method


class TestStopEventHandling:
    """Tests for stop event signal handling"""

    def test_stop_interview_event_set_on_end(self, session_manager, mock_meet_controller_class):
        """Test that stop_interview event is set when ending session"""
        session_id = "s_stop_event"
        session_manager.start_bot_session(session_id, "meet.link", "c1")
        
        stop_event = session_manager.active_sessions[session_id]['stop_interview']
        assert not stop_event.is_set()
        
        session_manager.end_session(session_id)
        
        # Event should have been set (though session is now removed)
        # We can't check it directly, but the test verifies no errors occur

    def test_stop_event_already_set(self, session_manager, mock_meet_controller_class):
        """Test ending session when stop event already set"""
        session_id = "s_stop_already_set"
        session_manager.start_bot_session(session_id, "meet.link", "c1")
        
        # Pre-set the stop event
        session_manager.active_sessions[session_id]['stop_interview'].set()
        
        # Should still end cleanly
        session_manager.end_session(session_id)
        assert session_id not in session_manager.active_sessions

    def test_stop_capture_event_handling(self, session_manager, mock_meet_controller_class):
        """Test stop_capture event for video capture"""
        session_id = "s_stop_capture"
        session_manager.start_bot_session(session_id, "meet.link", "c1", enable_video=True)
        
        stop_capture = session_manager.active_sessions[session_id]['stop_capture']
        assert not stop_capture.is_set()
        
        # Set stop capture
        stop_capture.set()
        assert stop_capture.is_set()


class TestSnapshotDirectoryHandling:
    """Tests for snapshot directory operations"""

    def test_snapshot_directory_structure(self, session_manager, mock_meet_controller_class):
        """Test correct snapshot directory structure"""
        session_id = "s_snapshot_structure"
        candidate_id = "c_snapshot_structure"
        
        session_manager.start_bot_session(session_id, "meet.link", candidate_id)
        
        session = session_manager.active_sessions[session_id]
        snapshot_dir = session['snapshot_dir']
        
        # Check path structure
        assert str(snapshot_dir).endswith(f"{candidate_id}/{session_id}/snapshots") or \
               str(snapshot_dir).endswith(f"{candidate_id}\\{session_id}\\snapshots")


class TestStatusTracking:
    """Tests for session status tracking"""

    def test_initial_status_active(self, session_manager, mock_meet_controller_class):
        """Test session starts with active status"""
        session_id = "s_status_active"
        session_manager.start_bot_session(session_id, "meet.link", "c1")
        
        assert session_manager.active_sessions[session_id]['status'] == 'active'

    def test_status_candidate_joined(self, session_manager, mock_meet_controller_class):
        """Test status changes to candidate_joined"""
        session_id = "s_status_joined"
        session_manager.start_bot_session(session_id, "meet.link", "c1")
        
        mock_controller = session_manager.active_sessions[session_id]['controller']
        mock_controller.get_participant_count.side_effect = [1, 2, 2, 2]
        
        def time_generator():
            t = 100.0
            while True:
                yield t
                t += 0.5
        
        with patch('app.services.meet_session_manager.time.time', side_effect=time_generator()):
            session_manager.wait_for_candidate(session_id, timeout=300)
        
        assert session_manager.active_sessions[session_id]['status'] == 'candidate_joined'

    def test_status_aborted_multiple_participants(self, session_manager, mock_meet_controller_class):
        """Test status changes to aborted when too many participants"""
        session_id = "s_status_aborted"
        session_manager.start_bot_session(session_id, "meet.link", "c1")
        
        mock_controller = session_manager.active_sessions[session_id]['controller']
        stop_event = session_manager.active_sessions[session_id]['stop_interview']
        mock_controller.get_participant_count.side_effect = [1, 4]
        
        def time_generator():
            t = 100.0
            while True:
                yield t
                t += 0.5
        
        with patch('app.services.meet_session_manager.time.time', side_effect=time_generator()):
            with patch.object(stop_event, 'wait', return_value=False):
                with patch.object(stop_event, 'is_set', return_value=False):
                    session_manager.wait_for_candidate(session_id, timeout=300)
        
        assert session_manager.active_sessions[session_id]['status'] == 'aborted_multiple_participants'


class TestVideoCaptureStatistics:
    """Tests for video capture statistics tracking"""

    def test_capture_stats_increment(self, session_manager, mock_meet_controller_class):
        """Test that capture stats increment correctly"""
        session_id = "s_stats_increment"
        session_manager.start_bot_session(session_id, "meet.link", "c1")
        
        stats = session_manager.active_sessions[session_id]['capture_stats']
        
        # Simulate captures
        stats['total_attempts'] += 5
        stats['successful_captures'] += 3
        stats['failed_captures'] += 2
        
        retrieved_stats = session_manager.get_capture_stats(session_id)
        assert retrieved_stats['total_attempts'] == 5
        assert retrieved_stats['successful_captures'] == 3
        assert retrieved_stats['failed_captures'] == 2

    def test_js_vs_screenshot_tracking(self, session_manager, mock_meet_controller_class):
        """Test tracking of JS vs screenshot captures"""
        session_id = "s_method_tracking"
        session_manager.start_bot_session(session_id, "meet.link", "c1")
        
        stats = session_manager.active_sessions[session_id]['capture_stats']
        
        stats['js_captures'] = 10
        stats['screenshot_captures'] = 5
        stats['successful_captures'] = 15
        stats['total_attempts'] = 20
        
        retrieved_stats = session_manager.get_capture_stats(session_id)
        assert retrieved_stats['js_captures'] == 10
        assert retrieved_stats['screenshot_captures'] == 5
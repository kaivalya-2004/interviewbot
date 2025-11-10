import pytest
import sys
import os
import io
import json  # Added import
from pathlib import Path  # Added import
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch, Mock
from datetime import datetime

# --- Setup Path ---
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, project_root)
# --- End Path Setup ---

# Import the FastAPI app object from your main_api.py
from main_api import app

# Fixtures

@pytest.fixture
def sample_resume_text():
    """Sample resume text for testing"""
    return "Software Engineer with 5 years of experience in Python, FastAPI, and cloud technologies. Strong background in AI/ML and distributed systems."


@pytest.fixture
def client():
    """
    Create a FastAPI TestClient for the application.
    This allows us to send HTTP requests to the app.
    """
    # Mock all services during TestClient startup to prevent initialization
    with patch('main_api.db_handler', MagicMock()), \
         patch('main_api.interview_service', MagicMock()), \
         patch('main_api.transcript_parser', MagicMock()), \
         patch('main_api.transcript_analyzer', MagicMock()), \
         patch('main_api.meet_session_mgr', MagicMock()), \
         patch('main_api.meet_orchestrator', MagicMock()), \
         patch('main_api.combined_analyzer', MagicMock()):
        with TestClient(app) as test_client:
            yield test_client


# --- Integration Tests for main_api.py ---

class TestHealthEndpoint:
    """Tests for health check endpoint"""

    def test_health_check_endpoint(self, client):
        """Tests the /health endpoint returns correct structure"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "services" in data
        assert data["services"]["db"] is True
        assert data["services"]["interview"] is True
        assert data["services"]["combined_analyzer"] is True
        assert data["services"]["transcript_parser"] is True
        assert data["services"]["transcript_analyzer"] is True
        assert data["services"]["meet_mgr"] is True
        assert data["services"]["orchestrator"] is True
        # REMOVED: assert data["services"]["device_detector"] is True


class TestStartGoogleMeetEndpoint:
    """Tests for /interview/start-google-meet endpoint"""

    def test_start_google_meet_endpoint_success(self, client, sample_resume_text, mocker):
        """
        Tests successful interview start with all required fields.
        Verifies:
        1. Resume parsing works
        2. Questionnaire JSON parsing works
        3. Background task is scheduled with correct args
        4. Correct session ID is returned
        """
        
        # Mock all the services
        mock_add_task = mocker.patch('main_api.BackgroundTasks.add_task')
        mock_pdf_parser = mocker.patch('main_api.parse_resume_pdf', return_value=sample_resume_text)
        mock_start_interview = mocker.patch(
            'main_api.interview_service.start_new_interview',
            return_value="mock_session_123"
        )
        mock_update_status = mocker.patch('main_api.db_handler.update_session_status')
        
        # Create fake resume file
        fake_resume_file = ("test_resume.pdf", io.BytesIO(b"fake-pdf-content"), "application/pdf")
        
        # Define form data
        form_data = {
            "candidate_id": "test_candidate_001",
            "meet_link": "https://meet.google.com/fake-link",
            "questionnaire_json": '["Question 1?", "Question 2?"]'
        }

        # Send HTTP POST request
        response = client.post(
            "/interview/start-google-meet",
            data=form_data,
            files={"resume": fake_resume_file}
        )

        # Verify response
        assert response.status_code == 202  # 202 Accepted
        response_data = response.json()
        assert response_data["status"] == "pending"
        assert response_data["session_id"] == "mock_session_123"
        assert response_data["questions_in_session"] == 2

        # Verify file parser was called
        mock_pdf_parser.assert_called_once()
        
        # Verify interview service was called with correct params
        mock_start_interview.assert_called_once_with(
            sample_resume_text,
            "test_candidate_001",
            ["Question 1?", "Question 2?"]
        )

        # Verify status update was called
        mock_update_status.assert_called_once_with("mock_session_123", "active_scheduled")

        # Verify background task was scheduled
        mock_add_task.assert_called_once()
        call_args = mock_add_task.call_args
        assert call_args[1]['session_id'] == "mock_session_123"
        assert call_args[1]['candidate_id'] == "test_candidate_001"
        assert call_args[1]['meet_link'] == "https://meet.google.com/fake-link"
        assert call_args[1]['enable_video'] is True
        assert call_args[1]['video_capture_method'] == "javascript"
        # UPDATED: Verify resume_content is passed to the task
        assert call_args[1]['resume_content'] == sample_resume_text

    def test_start_google_meet_txt_resume(self, client, mocker):
        """Tests that TXT resume files are properly handled"""
        mock_add_task = mocker.patch('main_api.BackgroundTasks.add_task')
        mock_start_interview = mocker.patch(
            'main_api.interview_service.start_new_interview',
            return_value="mock_session_456"
        )
        mocker.patch('main_api.db_handler.update_session_status')
        
        # Create fake TXT resume
        resume_text = "Plain text resume content"
        fake_resume_file = ("resume.txt", io.BytesIO(resume_text.encode('utf-8')), "text/plain")
        
        form_data = {
            "candidate_id": "test_candidate_002",
            "meet_link": "https://meet.google.com/another-link"
        }

        response = client.post(
            "/interview/start-google-meet",
            data=form_data,
            files={"resume": fake_resume_file}
        )

        assert response.status_code == 202
        mock_start_interview.assert_called_once()
        # Verify the resume text was properly decoded
        call_args = mock_start_interview.call_args[0]
        assert call_args[0] == resume_text

    def test_start_google_meet_no_questionnaire(self, client, sample_resume_text, mocker):
        """Tests interview start without questionnaire (should use empty list)"""
        mocker.patch('main_api.BackgroundTasks.add_task')
        mocker.patch('main_api.parse_resume_pdf', return_value=sample_resume_text)
        mock_start_interview = mocker.patch(
            'main_api.interview_service.start_new_interview',
            return_value="mock_session_789"
        )
        mocker.patch('main_api.db_handler.update_session_status')
        
        fake_resume_file = ("resume.pdf", io.BytesIO(b"fake-pdf"), "application/pdf")
        
        form_data = {
            "candidate_id": "test_candidate_003",
            "meet_link": "https://meet.google.com/test-link"
            # No questionnaire_json field
        }

        response = client.post(
            "/interview/start-google-meet",
            data=form_data,
            files={"resume": fake_resume_file}
        )

        assert response.status_code == 202
        # Verify empty questionnaire was passed
        mock_start_interview.assert_called_once()
        call_args = mock_start_interview.call_args[0]
        assert call_args[2] == []  # questionnaire should be empty list

    def test_start_google_meet_custom_audio_device(self, client, sample_resume_text, mocker):
        """Tests specifying custom audio device"""
        mock_add_task = mocker.patch('main_api.BackgroundTasks.add_task')
        mocker.patch('main_api.parse_resume_pdf', return_value=sample_resume_text)
        mocker.patch('main_api.interview_service.start_new_interview', return_value="mock_session")
        mocker.patch('main_api.db_handler.update_session_status')
        
        fake_resume_file = ("resume.pdf", io.BytesIO(b"fake-pdf"), "application/pdf")
        
        form_data = {
            "candidate_id": "test_candidate",
            "meet_link": "https://meet.google.com/test",
            "audio_device": "5"  # Custom device index
        }

        response = client.post(
            "/interview/start-google-meet",
            data=form_data,
            files={"resume": fake_resume_file}
        )

        assert response.status_code == 202
        # Verify audio_device was passed to background task
        call_args = mock_add_task.call_args[1]
        assert call_args['audio_device'] == 5

    def test_start_google_meet_video_disabled(self, client, sample_resume_text, mocker):
        """Tests starting interview with video capture disabled"""
        mock_add_task = mocker.patch('main_api.BackgroundTasks.add_task')
        mocker.patch('main_api.parse_resume_pdf', return_value=sample_resume_text)
        mocker.patch('main_api.interview_service.start_new_interview', return_value="mock_session")
        mocker.patch('main_api.db_handler.update_session_status')
        
        fake_resume_file = ("resume.pdf", io.BytesIO(b"fake-pdf"), "application/pdf")
        
        form_data = {
            "candidate_id": "test_candidate",
            "meet_link": "https://meet.google.com/test",
            "enable_video": "false"
        }

        response = client.post(
            "/interview/start-google-meet",
            data=form_data,
            files={"resume": fake_resume_file}
        )

        assert response.status_code == 202
        call_args = mock_add_task.call_args[1]
        assert call_args['enable_video'] is False

    def test_start_google_meet_screenshot_capture(self, client, sample_resume_text, mocker):
        """Tests using screenshot capture method instead of javascript"""
        mock_add_task = mocker.patch('main_api.BackgroundTasks.add_task')
        mocker.patch('main_api.parse_resume_pdf', return_value=sample_resume_text)
        mocker.patch('main_api.interview_service.start_new_interview', return_value="mock_session")
        mocker.patch('main_api.db_handler.update_session_status')
        
        fake_resume_file = ("resume.pdf", io.BytesIO(b"fake-pdf"), "application/pdf")
        
        form_data = {
            "candidate_id": "test_candidate",
            "meet_link": "https://meet.google.com/test",
            "video_capture_method": "screenshot"
        }

        response = client.post(
            "/interview/start-google-meet",
            data=form_data,
            files={"resume": fake_resume_file}
        )

        assert response.status_code == 202
        call_args = mock_add_task.call_args[1]
        assert call_args['video_capture_method'] == "screenshot"


class TestStartGoogleMeetEndpointErrors:
    """Tests for error handling in start endpoint"""

    def test_start_google_meet_invalid_resume_type(self, client):
        """Tests that invalid resume file type returns 400"""
        fake_resume_file = ("resume.zip", io.BytesIO(b"fake-zip"), "application/zip")
        
        form_data = {
            "candidate_id": "test_candidate",
            "meet_link": "https://meet.google.com/test"
        }

        response = client.post(
            "/interview/start-google-meet",
            data=form_data,
            files={"resume": fake_resume_file}
        )

        assert response.status_code == 400
        assert "Invalid resume file type" in response.json()["detail"]

    def test_start_google_meet_invalid_questionnaire_json(self, client, mocker):
        """Tests that malformed questionnaire JSON returns 400"""
        mocker.patch('main_api.parse_resume_pdf', return_value="Resume content")
        
        fake_resume_file = ("resume.pdf", io.BytesIO(b"fake-pdf"), "application/pdf")
        
        form_data = {
            "candidate_id": "test_candidate",
            "meet_link": "https://meet.google.com/test",
            "questionnaire_json": "Not valid JSON at all"
        }

        response = client.post(
            "/interview/start-google-meet",
            data=form_data,
            files={"resume": fake_resume_file}
        )
        
        assert response.status_code == 400
        assert "Invalid questionnaire_json" in response.json()["detail"]

    def test_start_google_meet_questionnaire_wrong_type(self, client, mocker):
        """Tests that questionnaire must be a list of strings"""
        mocker.patch('main_api.parse_resume_pdf', return_value="Resume")
        
        fake_resume_file = ("resume.pdf", io.BytesIO(b"fake-pdf"), "application/pdf")
        
        # Valid JSON but wrong structure (object instead of list)
        form_data = {
            "candidate_id": "test_candidate",
            "meet_link": "https://meet.google.com/test",
            "questionnaire_json": '{"question": "What is your name?"}'
        }

        response = client.post(
            "/interview/start-google-meet",
            data=form_data,
            files={"resume": fake_resume_file}
        )
        
        assert response.status_code == 400
        assert "questionnaire_json" in response.json()["detail"].lower()

    def test_start_google_meet_invalid_video_capture_method(self, client, sample_resume_text, mocker):
        """Tests that invalid video capture method returns 400"""
        mocker.patch('main_api.parse_resume_pdf', return_value=sample_resume_text)
        
        fake_resume_file = ("resume.pdf", io.BytesIO(b"fake-pdf"), "application/pdf")
        
        form_data = {
            "candidate_id": "test_candidate",
            "meet_link": "https://meet.google.com/test",
            "video_capture_method": "invalid_method"
        }

        response = client.post(
            "/interview/start-google-meet",
            data=form_data,
            files={"resume": fake_resume_file}
        )
        
        assert response.status_code == 400
        assert "Invalid video_capture_method" in response.json()["detail"]

    def test_start_google_meet_empty_resume(self, client, mocker):
        """Tests handling of empty resume file"""
        mocker.patch('main_api.parse_resume_pdf', return_value="")
        
        fake_resume_file = ("resume.pdf", io.BytesIO(b""), "application/pdf")
        
        form_data = {
            "candidate_id": "test_candidate",
            "meet_link": "https://meet.google.com/test"
        }

        response = client.post(
            "/interview/start-google-meet",
            data=form_data,
            files={"resume": fake_resume_file}
        )
        
        assert response.status_code == 400
        assert "Failed to extract text" in response.json()["detail"]


class TestGetInterviewStatus:
    """Tests for /interview/{session_id}/status endpoint"""

    def test_get_interview_status_active_session(self, client, mocker):
        """Tests getting status of an active interview session"""
        mock_db_data = {
            "candidate_id": "test_candidate",
            "created_at": datetime(2025, 1, 1, 12, 0, 0),
            "questionnaire": ["Q1", "Q2", "Q3"],
            "conversation": [
                {"role": "assistant", "text": "Question 1"},
                {"role": "user", "text": "Answer 1"}
            ],
            "status": "active_interviewing"
        }
        mock_active_session = {
            "status": "active",
            "video_enabled": True,
            "video_capture_method": "javascript"
        }
        mock_capture_stats = {
            "total_attempts": 15,
            "successful_captures": 15,
            "failed_captures": 0
        }
        
        mocker.patch('main_api.db_handler.get_session', return_value=mock_db_data)
        mocker.patch('main_api.meet_session_mgr.get_session', return_value=mock_active_session)
        mocker.patch('main_api.meet_session_mgr.get_snapshot_count', return_value=15)
        mocker.patch('main_api.meet_session_mgr.get_capture_stats', return_value=mock_capture_stats)

        response = client.get("/interview/test_session_123/status")

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "test_session_123"
        assert data["candidate_id"] == "test_candidate"
        assert data["session_status_db"] == "active_interviewing"
        assert data["bot_status_live"] == "active"
        assert data["snapshots_captured"] == 15
        assert data["video_enabled"] is True
        assert data["video_capture_method"] == "javascript"
        assert data["questionnaire_length"] == 3
        # UPDATED: Check for new fields
        assert data["conversation_length"] == 2
        assert data["capture_stats"] == mock_capture_stats

    def test_get_interview_status_completed_session(self, client, mocker):
        """Tests getting status of completed session (no active bot)"""
        mock_db_data = {
            "candidate_id": "test_candidate",
            "created_at": datetime(2025, 1, 1, 12, 0, 0),
            "questionnaire": ["Q1"],
            "conversation": [{"role": "assistant", "text": "Q1"}, {"role": "user", "text": "A1"}],
            "status": "completed"
        }
        
        mocker.patch('main_api.db_handler.get_session', return_value=mock_db_data)
        mocker.patch('main_api.meet_session_mgr.get_session', return_value=None)  # No active session
        mocker.patch('main_api.meet_session_mgr.get_snapshot_count', return_value=25)

        response = client.get("/interview/test_session_456/status")

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "test_session_456"
        assert data["session_status_db"] == "completed"
        assert data["bot_status_live"] == "ended_or_unknown"
        assert data["snapshots_captured"] == 25

    def test_get_interview_status_not_found(self, client, mocker):
        """Tests getting status of non-existent session"""
        mocker.patch('main_api.db_handler.get_session', return_value=None)

        response = client.get("/interview/nonexistent_session/status")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()


class TestEndInterviewEndpoint:
    """Tests for /interview/{session_id}/end endpoint"""

    def test_end_interview_success(self, client, mocker):
        """Tests successfully ending an active interview"""
        mock_end_session = mocker.patch('main_api.meet_session_mgr.end_session')
        mock_end_interview = mocker.patch('main_api.interview_service.end_interview_session')
        mock_update_status = mocker.patch('main_api.db_handler.update_session_status')
        mock_capture_stats = {
            "total_attempts": 20,
            "successful_captures": 18
        }
        mocker.patch('main_api.meet_session_mgr.get_snapshot_count', return_value=20)
        mocker.patch('main_api.meet_session_mgr.get_capture_stats', return_value=mock_capture_stats)

        response = client.post("/interview/test_session_123/end")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ending_triggered"
        assert data["snapshots"] == 20
        assert data["capture_stats"] == mock_capture_stats
        
        # Verify all cleanup methods were called
        mock_end_session.assert_called_once_with("test_session_123")
        mock_end_interview.assert_called_once_with("test_session_123")
        mock_update_status.assert_called_once_with("test_session_123", "terminated_by_user")

    def test_end_interview_error_handling(self, client, mocker):
        """Tests that errors during end are handled gracefully"""
        mocker.patch('main_api.meet_session_mgr.end_session', side_effect=Exception("End error"))
        mocker.patch('main_api.interview_service.end_interview_session')
        mocker.patch('main_api.db_handler.update_session_status')
        mocker.patch('main_api.meet_session_mgr.get_snapshot_count', return_value=0)
        mocker.patch('main_api.meet_session_mgr.get_capture_stats', return_value={})

        # Should not raise, should return response
        response = client.post("/interview/test_session_789/end")

        assert response.status_code == 200
        assert response.json()["status"] == "ending_triggered"


class TestSnapshotsEndpoint:
    """Tests for /interview/{session_id}/snapshots endpoint"""

    def test_get_snapshot_info_active_session(self, client, mocker):
        """Tests getting snapshot info for active session"""
        mock_session = {
            "status": "active",
            "video_enabled": True,
            "video_capture_method": "javascript"
        }
        mock_capture_stats = {
            "total_attempts": 30,
            "successful_captures": 28,
            "failed_captures": 2
        }
        
        mocker.patch('main_api.meet_session_mgr.get_session', return_value=mock_session)
        mocker.patch('main_api.meet_session_mgr.get_snapshot_count', return_value=30)
        mocker.patch('main_api.meet_session_mgr.get_capture_stats', return_value=mock_capture_stats)

        response = client.get("/interview/test_session_123/snapshots")

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "test_session_123"
        assert data["status"] == "active"
        assert data["video_enabled"] is True
        assert data["snapshots"] == 30
        assert data["capture_stats"] == mock_capture_stats

    def test_get_snapshot_info_completed_session(self, client, mocker, tmp_path):
        """
        Tests getting snapshot info for completed session
        This test now uses tmp_path to create a real directory structure
        and patches main_api.Path to use this temp path.
        """
        candidate_id = "test_candidate_completed"
        session_id = "test_session_456"
        
        # Create mock snapshot directory and files within tmp_path
        snapshot_dir = tmp_path / "data" / candidate_id / session_id / "snapshots"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        
        # Create 10 fake snapshot files
        for i in range(10):
            (snapshot_dir / f"snapshot_{i:04d}.jpg").write_bytes(b"fake_image")
        
        # Create one non-jpg file to ensure it's ignored
        (snapshot_dir / "log.txt").write_bytes(b"log data")
        
        mock_session_data = {
            "candidate_id": candidate_id
        }
        
        mocker.patch('main_api.meet_session_mgr.get_session', return_value=None)
        mocker.patch('main_api.db_handler.get_session', return_value=mock_session_data)
        
        # UPDATED: Correctly patch main_api.Path to use tmp_path as its root
        mocker.patch('main_api.Path', lambda *args: Path(tmp_path).joinpath(*args))

        response = client.get(f"/interview/{session_id}/snapshots")

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == session_id
        assert data["status"] == "completed_or_unknown"
        # UPDATED: Assert the exact count of .jpg files
        assert data["snapshots"] == 10
        assert data["dir"] == str(snapshot_dir)

    def test_get_snapshot_info_session_not_found(self, client, mocker):
        """Tests snapshot endpoint with non-existent session"""
        mocker.patch('main_api.meet_session_mgr.get_session', return_value=None)
        mocker.patch('main_api.db_handler.get_session', return_value=None)

        response = client.get("/interview/nonexistent/snapshots")

        assert response.status_code == 404


class TestProductionReadiness:
    """Production-specific integration tests"""
    
    def test_concurrent_start_requests(self, client, sample_resume_text, mocker):
        """Tests handling multiple concurrent start requests"""
        import threading
        
        mocker.patch('main_api.BackgroundTasks.add_task')
        mocker.patch('main_api.parse_resume_pdf', return_value=sample_resume_text)
        mocker.patch('main_api.interview_service.start_new_interview', 
                    side_effect=lambda r, c, q: f"session_{c}")
        mocker.patch('main_api.db_handler.update_session_status')
        
        results = []
        
        def make_request(candidate_num):
            fake_resume = ("resume.pdf", io.BytesIO(b"fake-pdf"), "application/pdf")
            form_data = {
                "candidate_id": f"candidate_{candidate_num}",
                "meet_link": "https://meet.google.com/test"
            }
            response = client.post(
                "/interview/start-google-meet",
                data=form_data,
                files={"resume": fake_resume}
            )
            results.append(response)
        
        threads = [threading.Thread(target=make_request, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # All requests should succeed
        assert all(r.status_code == 202 for r in results)
        # All should have unique session IDs
        session_ids = [r.json()["session_id"] for r in results]
        assert len(set(session_ids)) == 5
    
    def test_large_questionnaire_handling(self, client, sample_resume_text, mocker):
        """Tests handling of large questionnaire (100+ questions)"""
        mocker.patch('main_api.BackgroundTasks.add_task')
        mocker.patch('main_api.parse_resume_pdf', return_value=sample_resume_text)
        mock_start = mocker.patch('main_api.interview_service.start_new_interview', 
                                 return_value="session_large")
        mocker.patch('main_api.db_handler.update_session_status')
        
        large_questionnaire = [f"Question {i}?" for i in range(100)]
        
        fake_resume = ("resume.pdf", io.BytesIO(b"fake-pdf"), "application/pdf")
        form_data = {
            "candidate_id": "test_candidate",
            "meet_link": "https://meet.google.com/test",
            "questionnaire_json": json.dumps(large_questionnaire)
        }
        
        response = client.post(
            "/interview/start-google-meet",
            data=form_data,
            files={"resume": fake_resume}
        )
        
        assert response.status_code == 202
        assert response.json()["questions_in_session"] == 100
        # Verify all questions were passed
        call_args = mock_start.call_args[0]
        assert len(call_args[2]) == 100
    
    def test_unicode_handling_in_fields(self, client, mocker):
        """Tests handling of unicode characters in all text fields"""
        sample_text = "Software Engineer with émojis 🚀 and special çhars"
        mock_add_task = mocker.patch('main_api.BackgroundTasks.add_task')
        mocker.patch('main_api.parse_resume_pdf', return_value=sample_text)
        mock_start = mocker.patch('main_api.interview_service.start_new_interview',
                                 return_value="session_unicode")
        mocker.patch('main_api.db_handler.update_session_status')
        
        fake_resume = ("resume.pdf", io.BytesIO(b"fake-pdf"), "application/pdf")
        form_data = {
            "candidate_id": "候選人_001",  # Unicode candidate ID
            "meet_link": "https://meet.google.com/test",
            "questionnaire_json": '["¿Cuál es tu experiencia?", "What about 日本語?"]'
        }
        
        response = client.post(
            "/interview/start-google-meet",
            data=form_data,
            files={"resume": fake_resume}
        )
        
        assert response.status_code == 202
        
        # Verify unicode was preserved in DB/Interview service call
        call_args = mock_start.call_args[0]
        assert "候選人_001" in str(call_args)
        assert "日本語" in str(call_args)

        # UPDATED: Verify unicode was preserved in background task call
        add_task_args = mock_add_task.call_args[1]
        assert add_task_args['candidate_id'] == "候選人_001"
        assert add_task_args['resume_content'] == sample_text
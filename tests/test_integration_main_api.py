import pytest
import sys
import os
import io
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

# --- Setup Path ---
# This adds your project root (one level up) to the path
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, project_root)
# --- End Path Setup ---

# Import the FastAPI app object from your main_api.py
from main_api import app

# Create a client fixture to interact with the app
@pytest.fixture
def client():
    """
    Create a FastAPI TestClient for the application.
    This allows us to send HTTP requests to the app.
    """
    # --- FIX: Use patch to mock all services during TestClient startup ---
    # This prevents the app from trying to initialize real services
    with patch('main_api.db_handler', MagicMock()), \
         patch('main_api.interview_service', MagicMock()), \
         patch('main_api.transcript_parser', MagicMock()), \
         patch('main_api.transcript_analyzer', MagicMock()), \
         patch('main_api.meet_session_mgr', MagicMock()), \
         patch('main_api.meet_orchestrator', MagicMock()), \
         patch('main_api.combined_analyzer', MagicMock()):
        with TestClient(app) as test_client:
            yield test_client
    # --- END FIX ---

# --- Integration Tests for main_api.py ---

class TestMainApiIntegration:

    def test_health_check_endpoint(self, client):
        """
        Tests the /health endpoint.
        This is a simple integration test to ensure the app is running.
        """
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["services"]["db"] is True
        assert data["services"]["interview"] is True
        assert data["services"]["combined_analyzer"] is True

    def test_start_google_meet_endpoint_success(self, client, sample_resume_text, mocker):
        """
        Tests the /interview/start-google-meet endpoint.
        This test will:
        1. Mock the services called by the endpoint.
        2. Send a real HTTP POST request with form data and a file.
        3. Verify the endpoint logic (parsing, service calls) is correct.
        4. Verify the background task is correctly scheduled.
        """
        
        # 1. ARRANGE: Mock all the services that main_api.py calls
        
        # --- FIX: Patch path is 'main_api....', not 'app.main_api....' ---
        mock_add_task = mocker.patch('main_api.BackgroundTasks.add_task')
        mock_pdf_parser = mocker.patch('main_api.parse_resume_pdf', return_value=sample_resume_text)
        mock_start_interview = mocker.patch(
            'main_api.interview_service.start_new_interview',
            return_value="mock_session_123" # Return a fake session ID
        )
        # --- END FIX ---
        
        # Create a fake resume file to upload
        fake_resume_file = ("test_resume.pdf", io.BytesIO(b"fake-pdf-content"), "application/pdf")
        
        # Define the form data
        form_data = {
            "candidate_id": "test_candidate_001",
            "meet_link": "https://meet.google.com/fake-link",
            "questionnaire_json": '["Question 1?", "Question 2?"]'
        }

        # 2. ACT: Send the HTTP POST request
        response = client.post(
            "/interview/start-google-meet",
            data=form_data,
            files={"resume": fake_resume_file}
        )

        # 3. ASSERT: Check that the endpoint behaved correctly
        assert response.status_code == 202 # 202 Accepted
        response_data = response.json()
        assert response_data["status"] == "pending"
        assert response_data["session_id"] == "mock_session_123"
        assert response_data["questions_in_session"] == 2

        # Verify the file parser was called correctly
        mock_pdf_parser.assert_called_once()
        
        # Verify the interview service was called correctly
        mock_start_interview.assert_called_once_with(
            sample_resume_text,
            "test_candidate_001",
            ["Question 1?", "Question 2?"] # Check that JSON parsing worked
        )

        # Verify the background task was scheduled with the correct arguments
        mock_add_task.assert_called_once_with(
            mocker.ANY, # The function itself (start_and_conduct_interview_task)
            session_id="mock_session_123",
            candidate_id="test_candidate_001",
            meet_link="https://meet.google.com/fake-link",
            audio_device=None,
            enable_video=True,
            video_capture_method="javascript",
            resume_content=sample_resume_text
        )

    def test_start_google_meet_endpoint_bad_resume(self, client):
        """
        Tests that the endpoint fails if a non-PDF/TXT file is uploaded.
        """
        # Create a fake resume file with a bad extension
        fake_resume_file = ("resume.zip", io.BytesIO(b"fake-zip-content"), "application/zip")
        
        form_data = {
            "candidate_id": "test_candidate_002",
            "meet_link": "https://meet.google.com/fake-link",
        }

        # Act
        response = client.post(
            "/interview/start-google-meet",
            data=form_data,
            files={"resume": fake_resume_file}
        )

        # Assert
        assert response.status_code == 400 # Bad Request
        assert "Invalid resume file type" in response.json()["detail"]

    def test_start_google_meet_endpoint_bad_json(self, client, mocker): # <-- FIX: Added mocker
        """
        Tests that the endpoint fails if the questionnaire JSON is malformed.
        """
        
        # --- FIX: Mock the PDF parser so the code doesn't fail early ---
        mocker.patch('main_api.parse_resume_pdf', return_value="fake resume content")
        # --- END FIX ---

        fake_resume_file = ("resume.pdf", io.BytesIO(b"fake-pdf-content"), "application/pdf")
        
        form_data = {
            "candidate_id": "test_candidate_003",
            "meet_link": "https://meet.google.com/fake-link",
            "questionnaire_json": 'This is not json' # Malformed JSON
        }

        # Act
        response = client.post(
            "/interview/start-google-meet",
            data=form_data,
            files={"resume": fake_resume_file}
        )
        
        # Assert
        assert response.status_code == 400
        assert "Invalid questionnaire_json" in response.json()["detail"]

    def test_get_interview_status(self, client, mocker):
        """
        Tests the /interview/{session_id}/status endpoint.
        """
        
        # Arrange: Mock the DB and Session Manager
        mock_db_data = {
            "candidate_id": "test_candidate",
            "created_at": "2025-01-01T12:00:00",
            "questionnaire": ["Q1"],
            "conversation": ["Msg1"]
        }
        mock_active_session = {
            "status": "active",
            "video_enabled": True,
            "video_capture_method": "javascript"
        }
        
        # --- FIX: Patch path is 'main_api....', not 'app.main_api....' ---
        mocker.patch('main_api.db_handler.get_session', return_value=mock_db_data)
        mocker.patch('main_api.meet_session_mgr.get_session', return_value=mock_active_session)
        mocker.patch('main_api.meet_session_mgr.get_snapshot_count', return_value=5)
        mocker.patch('main_api.meet_session_mgr.get_capture_stats', return_value={"total_attempts": 5, "successful_captures": 5})
        # --- END FIX ---

        # Act
        response = client.get("/interview/test_session_123/status")

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "test_session_123"
        assert data["bot_status"] == "active"
        assert data["snapshots_captured"] == 5
        assert data["conversation_length"] == 1
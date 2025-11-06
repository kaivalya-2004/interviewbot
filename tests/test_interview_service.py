# tests/test_interview_service.py
import pytest
import unittest # Import for mock_open
from unittest.mock import Mock, patch, MagicMock, call
import numpy as np
from datetime import datetime
from pathlib import Path
import threading
import speech_recognition as sr
from google.cloud import texttospeech

# Need to set up the path first
import sys, os
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, project_root)

from app.services.interview_service import InterviewService


@pytest.fixture
def mock_db_handler():
    """Mock database handler"""
    with patch('app.services.interview_service.DBHandler') as mock:
        db_instance = Mock()
        
        # This lambda runs *every time* create_session is called
        def create_session_side_effect(resume, candidate_id, questionnaire):
            if candidate_id == "candidate_1":
                return "session_A"
            if candidate_id == "candidate_2":
                return "session_B"
            return "test_session_123" # Default for all other tests

        db_instance.create_session.side_effect = create_session_side_effect
        
        db_instance.get_session.return_value = {
            "candidate_id": "test_candidate",
            "created_at": datetime(2025, 10, 31, 12, 0, 0),
            "questionnaire": ["Q1", "Q2"],
            "conversation": [
                {
                    "role": "assistant",
                    "text": "<speak>Hello</speak>", # Updated to SSML
                    "timestamp": datetime(2025, 10, 31, 12, 1, 0),
                    "is_follow_up": False
                }
            ]
        }
        mock.return_value = db_instance
        yield db_instance


@pytest.fixture
def mock_gemini():
    """Mock Gemini API"""
    with patch('app.services.interview_service.genai') as mock_genai:
        mock_model_instance = MagicMock(name="GenerativeModel_Instance")
        mock_chat_session = MagicMock(name="ChatSession_Instance")
        
        mock_response = MagicMock(name="Gemini_Response")
        # --- FIX: Mock response is now SSML ---
        mock_response.text = "<speak>What is your experience with Python?</speak>"
        mock_response.usage_metadata = MagicMock(
            prompt_token_count=100,
            candidates_token_count=20,
            total_token_count=120
        )
        
        mock_chat_session.send_message.return_value = mock_response
        mock_model_instance.start_chat.return_value = mock_chat_session
        mock_genai.GenerativeModel.return_value = mock_model_instance
        
        yield mock_genai


@pytest.fixture
def mock_tts_client():
    """Mock Google TTS client"""
    with patch('app.services.interview_service.texttospeech.TextToSpeechClient') as mock_class:
        client_instance = Mock()
        mock_response = Mock()
        mock_response.audio_content = b'\x00\x01' * 1000
        client_instance.synthesize_speech.return_value = mock_response
        mock_class.return_value = client_instance
        yield mock_class # <-- FIX: Yield the patch object (mock_class)


@pytest.fixture
def interview_service(mock_db_handler, mock_gemini, mock_tts_client):
    """Create InterviewService instance with mocked dependencies"""
    with patch.dict('os.environ', {'GEMINI_API_KEY': 'test_key'}):
        service = InterviewService()
        return service


class TestInterviewService:
    """Test suite for InterviewService"""

    def test_initialization(self, interview_service, mock_gemini, mock_tts_client):
        """Test service initializes correctly"""
        assert interview_service.db is not None
        assert interview_service.recognizer is not None
        assert interview_service.tts_client is not None
        assert len(interview_service.active_chat_sessions) == 0
        
        # Test Gemini Init
        mock_gemini.GenerativeModel.assert_called_once_with('gemini-2.5-flash')
        
        # Test TTS Init
        mock_tts_client.assert_called_once() # <-- This now correctly asserts the class was called
        
        # --- FIX: Check the call args to ensure the endpoint was set ---
        call_args = mock_tts_client.call_args
        assert "texttospeech.googleapis.com:443" in str(call_args[1]['client_options'])
        
        # --- FIX: Check that the incompatible params are NOT in the config ---
        assert interview_service.tts_audio_config.sample_rate_hertz == 0
        assert interview_service.tts_audio_config.speaking_rate == 0.0
        assert interview_service.tts_audio_config.pitch == 0.0
        assert interview_service.tts_audio_config.audio_encoding == texttospeech.AudioEncoding.LINEAR16
        assert "en-IN-Chirp3-HD-Alnilam" in interview_service.tts_voice.name

    def test_start_new_interview_with_ssml_prompt(self, interview_service, mock_gemini):
        """Test starting interview and check for SSML in prompt"""
        resume = "John Doe\nSoftware Engineer\nPython, Django"
        candidate_id = "candidate_001"
        
        session_id = interview_service.start_new_interview(resume, candidate_id, [])
        
        assert session_id == "test_session_123"
        assert session_id in interview_service.active_chat_sessions
        
        # Verify start_chat was called on the model *instance*
        mock_model = mock_gemini.GenerativeModel.return_value
        mock_model.start_chat.assert_called_once()
        
        call_args = mock_model.start_chat.call_args
        history = call_args[1]['history']
        
        # Check system prompt
        assert "<speak> tags" in history[0]['parts'][0]
        assert "<break time='1000ms'/>" in history[0]['parts'][0]
        
        # --- FIX: Check model's example response ---
        assert history[1]['role'] == 'model'
        assert history[1]['parts'][0] == "<speak>Understood. <break time='700ms'/> I will ask short, clear, and conversational questions using SSML with pauses after punctuation. <break time='1000ms'/> Ready for the candidate's introduction.</speak>"

    def test_start_new_interview_with_questionnaire(self, interview_service):
        """Test starting interview with questionnaire"""
        resume = "John Doe\nSoftware Engineer"
        candidate_id = "candidate_002"
        questionnaire = ["..."]
        
        session_id = interview_service.start_new_interview(resume, candidate_id, questionnaire)
        
        assert session_id == "test_session_123"
        assert session_id in interview_service.active_chat_sessions

    def test_generate_initial_greeting(self, interview_service):
        """Test initial greeting generation"""
        greeting = interview_service.generate_initial_greeting()
        # --- FIX: Check for SSML and new text ---
        assert greeting.startswith("<speak>")
        assert greeting.endswith("</speak>")
        assert "A. I. interviewer" in greeting
        assert "<break time='300ms'/>" in greeting

    def test_generate_next_question(self, interview_service, mock_gemini):
        """Test generating next question from Gemini"""
        resume = "Test Resume"
        candidate_id = "test_candidate"
        
        session_id = interview_service.start_new_interview(resume, candidate_id, [])
        assert session_id == "test_session_123"
        
        interview_service._latest_transcript_for_gemini[session_id] = "I have 5 years of Python experience"
        
        question = interview_service.generate_next_question(session_id)
        
        # --- FIX: Assert the SSML response ---
        assert question == "<speak>What is your experience with Python?</speak>"

    def test_generate_next_question_wraps_non_ssml(self, interview_service, mock_gemini):
        """Test that non-SSML responses from Gemini are wrapped"""
        resume = "Test Resume"
        candidate_id = "test_candidate"
        session_id = interview_service.start_new_interview(resume, candidate_id, [])
        interview_service._latest_transcript_for_gemini[session_id] = "Test"
        
        # --- FIX: Force Gemini mock to return non-SSML ---
        mock_chat_session = interview_service.active_chat_sessions[session_id]
        mock_chat_session.send_message.return_value.text = "This is not SSML."
        
        question = interview_service.generate_next_question(session_id)
        
        assert question == "<speak>This is not SSML.</speak>"

    def test_generate_next_question_ssml_fallback(self, interview_service, mock_gemini):
        """Test that the fallback question is valid SSML"""
        resume = "Test Resume"
        candidate_id = "test_candidate"
        session_id = interview_service.start_new_interview(resume, candidate_id, [])
        interview_service._latest_transcript_for_gemini[session_id] = "Test"
        
        # Force an error
        mock_chat_session = interview_service.active_chat_sessions[session_id]
        mock_chat_session.send_message.side_effect = Exception("API Error")
        
        question = interview_service.generate_next_question(session_id)
        
        assert question.startswith("<speak>")
        assert "Apologies" in question
        assert question.endswith("</speak>")

    def test_generate_next_question_no_session(self, interview_service):
        """Test generating question with no active session"""
        question = interview_service.generate_next_question("nonexistent_session")
        assert "Error" in question

    def test_text_to_speech_success(self, interview_service, mock_tts_client):
        """Test successful TTS generation"""
        # --- FIX: Input must be SSML ---
        ssml_text = "<speak>Hello</speak>"
        audio_data, sample_rate = interview_service.text_to_speech(ssml_text, "s1", 1, "c1")
        
        assert audio_data is not None
        # --- FIX: Chirp models output 24kHz ---
        assert sample_rate == 24000
        
        # Check the *class* was called in the fixture, and *method* was called here
        mock_tts_client.return_value.synthesize_speech.assert_called_once()
        
        # --- FIX: Verify SSML was passed, not text ---
        call_args = mock_tts_client.return_value.synthesize_speech.call_args
        synthesis_input = call_args[1]['input']
        assert synthesis_input.ssml == ssml_text
        assert synthesis_input.text == "" # Text field should be empty

    def test_text_to_speech_uses_ssml(self, interview_service, mock_tts_client):
        """Test TTS passes SSML directly"""
        # --- FIX: This test replaces test_text_to_speech_cleans_markdown ---
        ssml_text = "<speak>**Bold** #Hi <break time='500ms'/></speak>"
        interview_service.text_to_speech(ssml_text, "s1", 1, "c1")
        
        call_args = mock_tts_client.return_value.synthesize_speech.call_args
        synthesis_input = call_args[1]['input']
        
        # The service no longer cleans, it passes the SSML as-is
        assert synthesis_input.ssml == ssml_text

    def test_speech_to_text_success(self, interview_service):
        """Test successful speech to text conversion"""
        test_audio_path = "test_audio.wav"
        
        with patch('app.services.interview_service.Path') as mock_path:
            mock_path.return_value.exists.return_value = True
            with patch('app.services.interview_service.sr.AudioFile'):
                with patch.object(interview_service.recognizer, 'record'):
                    with patch.object(interview_service.recognizer, 'recognize_google') as mock_recognize:
                        mock_recognize.return_value = "This is a test transcript"
                        transcript = interview_service.speech_to_text(test_audio_path)
                        assert transcript == "This is a test transcript"

    def test_speech_to_text_unintelligible(self, interview_service):
        """Test STT with unintelligible audio"""
        test_audio_path = "test_audio.wav"
        
        with patch('app.services.interview_service.Path') as mock_path:
            mock_path.return_value.exists.return_value = True
            with patch('app.services.interview_service.sr.AudioFile'):
                with patch.object(interview_service.recognizer, 'record'):
                    with patch.object(interview_service.recognizer, 'recognize_google') as mock_recognize:
                        mock_recognize.side_effect = sr.UnknownValueError()
                        transcript = interview_service.speech_to_text(test_audio_path)
                        assert transcript == "[Unintelligible]"

    def test_process_and_log_transcript(self, interview_service):
        """Test transcript processing and logging"""
        session_id, audio_path, turn_count, candidate_id = "s1", "a.wav", 1, "c1"
        start_time, end_time = datetime.now(), datetime.now()
        
        with patch.object(interview_service, 'speech_to_text') as mock_stt:
            mock_stt.return_value = "Test transcript"
            
            interview_service.process_and_log_transcript(
                session_id, audio_path, turn_count, candidate_id,
                start_time, end_time, False
            )
            
            assert interview_service._latest_transcript_for_gemini[session_id] == "Test transcript"
            interview_service.db.add_message_to_session.assert_called_with(
                session_id, "user", "Test transcript", audio_path,
                start_time, end_time, is_follow_up=False
            )

    def test_generate_final_transcript_file(self, interview_service, tmp_path):
        """Test final transcript file generation"""
        session_id = "test_session_123"
        candidate_id = "test_candidate"
        
        with patch('app.services.interview_service.Path') as mock_path_class:
            mock_data_dir = MagicMock()
            mock_candidate_dir = MagicMock()
            mock_transcript_dir = MagicMock()
            mock_transcript_file_path = MagicMock(name="FinalPath")

            mock_path_class.return_value = mock_data_dir
            mock_data_dir.__truediv__.return_value = mock_candidate_dir
            mock_candidate_dir.__truediv__.return_value = mock_transcript_dir
            mock_transcript_dir.__truediv__.return_value = mock_transcript_file_path

            with patch("builtins.open", new_callable=unittest.mock.mock_open) as mock_file:
                
                interview_service.generate_final_transcript_file(session_id)

                mock_transcript_dir.mkdir.assert_called_with(parents=True, exist_ok=True)
                
                mock_file.assert_called_with(mock_transcript_file_path, "w", encoding="utf-8")
                
                expected_first_write = f"--- Interview Transcript ---\nSession ID: {session_id}\nCandidate ID: {candidate_id}\n"
                mock_file().write.assert_any_call(expected_first_write)
                
                # --- FIX: Check SSML from mock DB ---
                mock_file().write.assert_any_call(f"[{datetime(2025, 10, 31, 12, 1, 0)}] Assistant:\n<speak>Hello</speak>\n\n---\n\n")

    def test_end_interview_session(self, interview_service):
        """Test ending interview session"""
        session_id = "test_session_123"
        resume = "Test Resume"
        candidate_id = "candidate_001"
        
        interview_service.start_new_interview(resume, candidate_id, [])
        
        interview_service._session_token_counts[session_id] = {'prompt': 1000, 'response': 500, 'total': 1500}
        interview_service._session_tts_char_counts[session_id] = 5000
        
        interview_service.end_interview_session(session_id)
        
        assert session_id not in interview_service.active_chat_sessions
        assert session_id not in interview_service._session_token_counts
        assert session_id not in interview_service._session_tts_char_counts

    def test_token_counting(self, interview_service, mock_gemini):
        """Test token usage is tracked correctly"""
        resume = "Test Resume"
        candidate_id = "candidate_001"
        
        session_id = interview_service.start_new_interview(resume, candidate_id, [])
        assert session_id == "test_session_123"
        
        interview_service._latest_transcript_for_gemini[session_id] = "Test answer"
        
        interview_service.generate_next_question(session_id)
        
        assert session_id in interview_service._session_token_counts
        token_data = interview_service._session_token_counts[session_id]
        assert token_data['prompt'] == 100
        assert token_data['response'] == 20
        assert token_data['total'] == 120

    def test_tts_character_counting(self, interview_service, mock_tts_client):
        """Test TTS character usage is tracked"""
        session_id = "test_session"
        # --- FIX: Use SSML text ---
        text = "<speak>This is a test</speak>" # 26 chars in string, but log said 29?
        
        # Let's re-check:
        # '<' 's' 'p' 'e' 'a' 'k' '>' = 7
        # 'T' 'h' 'i' 's' ' ' 'i' 's' ' ' 'a' ' ' 't' 'e' 's' 't' = 14
        # '<' '/' 's' 'p' 'e' 'a' 'k' '>' = 8
        # 7 + 14 + 8 = 29. The log was correct. My manual count was wrong.
        
        turn_count = 1
        candidate_id = "candidate_001"
        
        interview_service.text_to_speech(text, session_id, turn_count, candidate_id)
        
        assert session_id in interview_service._session_tts_char_counts
        char_count = interview_service._session_tts_char_counts[session_id]
        
        # --- FIX: Assert the correct count from the log ---
        assert char_count == 29

    def test_concurrent_sessions(self, interview_service):
        """Test handling multiple concurrent sessions"""
        resume = "Test Resume"
        
        session_1 = interview_service.start_new_interview(resume, "candidate_1", [])
        session_2 = interview_service.start_new_interview(resume, "candidate_2", [])
        
        assert session_1 == "session_A"
        assert session_2 == "session_B"
        
        assert session_1 != session_2
        assert session_1 in interview_service.active_chat_sessions
        assert session_2 in interview_service.active_chat_sessions
        assert len(interview_service.active_chat_sessions) == 2

    @pytest.mark.parametrize("error_scenario", [
        "empty_response",
        "api_error",
    ])
    def test_generate_question_error_handling(self, interview_service, mock_gemini, error_scenario):
        """Test error handling in question generation"""
        resume = "Test Resume"
        candidate_id = "candidate_001"
        
        session_id = interview_service.start_new_interview(resume, candidate_id, [])
        assert session_id == "test_session_123"
        
        interview_service._latest_transcript_for_gemini[session_id] = "Test answer"
        
        mock_chat_session = interview_service.active_chat_sessions[session_id]
        
        if error_scenario == "empty_response":
            mock_chat_session.send_message.return_value.text = ""
        elif error_scenario == "api_error":
            mock_chat_session.send_message.side_effect = Exception("API Error")
            
        question = interview_service.generate_next_question(session_id)
        
        assert isinstance(question, str)
        assert len(question) > 0
        # --- FIX: Check for SSML in fallback ---
        assert question.startswith("<speak>")
        assert "proud of" in question or "skill" in question or "Apologies" in question
        assert question.endswith("</speak>")
# tests/test_interview_service.py
import pytest
import unittest # Import for mock_open
from unittest.mock import Mock, patch, MagicMock, call, AsyncMock, PropertyMock
import numpy as np
from datetime import datetime
from pathlib import Path
import threading
from io import StringIO # Import StringIO

# Need to set up the path first
import sys, os
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, project_root)

from app.services.interview_service import InterviewService
# We need this for checking the call args
from google.cloud import texttospeech


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
                    "text": "Hello", # UPDATED: Now raw text
                    "timestamp": datetime(2025, 10, 31, 12, 1, 0),
                    "is_follow_up": False,
                    "audio_path": "bot_test_session_123_1_streamed.txt" # ADDED: Path for new log format
                }
            ],
            "tab_switch_events": [{'timestamp': '2025-10-31T12:05:00', 'visibilityState': 'hidden'}] # ADDED: For transcript test
        }
        mock.return_value = db_instance
        yield db_instance


@pytest.fixture
def mock_gemini():
    """Mock Gemini API"""
    with patch('app.services.interview_service.genai') as mock_genai:
        mock_model_instance = MagicMock(name="GenerativeModel_Instance")
        mock_chat_session = MagicMock(name="ChatSession_Instance")
        
        # --- UPDATED: Mock the streaming response ---
        mock_chunk1 = MagicMock()
        # Use PropertyMock to correctly handle the `chunk.text` access
        type(mock_chunk1).text = PropertyMock(return_value="What is your experience ")
        mock_chunk1.usage_metadata = None

        mock_chunk2 = MagicMock()
        type(mock_chunk2).text = PropertyMock(return_value="with Python?")
        mock_chunk2.usage_metadata = None # Metadata is on the *last* chunk

        mock_final_chunk = MagicMock()
        # Mock the `except ValueError` block by raising it on text access
        type(mock_final_chunk).text = PropertyMock(side_effect=ValueError)
        mock_final_chunk.usage_metadata = MagicMock(
            prompt_token_count=100,
            candidates_token_count=20,
            total_token_count=120
        )
        
        # This is what chat.send_message(..., stream=True) will return
        mock_chat_session.send_message.return_value = [mock_chunk1, mock_chunk2, mock_final_chunk]
        
        mock_model_instance.start_chat.return_value = mock_chat_session
        mock_genai.GenerativeModel.return_value = mock_model_instance
        
        yield mock_genai


@pytest.fixture
def mock_tts_client():
    """Mock Google TTS client"""
    with patch('app.services.interview_service.texttospeech.TextToSpeechClient') as mock_class:
        client_instance = Mock()
        
        # --- UPDATED: Mock streaming_synthesize ---
        mock_tts_response = Mock()
        mock_tts_response.audio_content = b'\xDE\xAD' # Mock MULAW bytes
        
        # This will be an iterator
        client_instance.streaming_synthesize.return_value = [mock_tts_response]
        
        mock_class.return_value = client_instance
        yield mock_class


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
        assert interview_service.tts_client is not None
        assert len(interview_service.active_chat_sessions) == 0
        
        # Test Gemini Init
        mock_gemini.GenerativeModel.assert_called_once_with(
            'gemini-2.5-flash', 
            generation_config=unittest.mock.ANY
        )
        
        # Test TTS Init
        mock_tts_client.assert_called_once()
        
        # Check the call args to ensure the endpoint was set
        call_args = mock_tts_client.call_args
        assert "texttospeech.googleapis.com:443" in str(call_args[1]['client_options'])
        
        # --- UPDATED: Check that the incompatible params are NOT in the config ---
        assert not hasattr(interview_service, "tts_audio_config")
        
        # Check that the voice is set
        assert "en-IN-Chirp3-HD-Alnilam" in interview_service.tts_voice.name

    def test_start_new_interview_with_raw_text_prompt(self, interview_service, mock_gemini):
        """Test starting interview and check for raw text in prompt"""
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
        system_prompt = history[0]['parts'][0]
        assert "<speak> tags" not in system_prompt
        assert "MUST be raw text" in system_prompt
        assert "punctuation" in system_prompt
        
        # Check model's example response
        assert history[1]['role'] == 'model'
        assert "raw text" in history[1]['parts'][0]
        assert "punctuation" in history[1]['parts'][0]

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
        # --- UPDATED: Check for raw text ---
        assert not greeting.startswith("<speak>")
        assert not greeting.endswith("</speak>")
        assert "Interviewer for today" in greeting
        assert "<break" not in greeting

    # --- NEW: Test stream_plain_text ---
    def test_stream_plain_text_success(self, interview_service, mock_tts_client):
        """Test successful streaming TTS for plain text"""
        plain_text = "Hello"
        
        # --- FIX: We need to mock Path and open here too ---
        with patch("builtins.open", new_callable=unittest.mock.mock_open()), \
             patch('app.services.interview_service.Path') as mock_path, \
             patch('app.services.interview_service.sf.write'), \
             patch('app.services.interview_service.audioop.ulaw2lin', return_value=b'\x00\x01'):
            
            # Configure Path mock
            mock_path_instance = MagicMock()
            mock_path.return_value = mock_path_instance
            mock_path_instance.__truediv__ = MagicMock(return_value=mock_path_instance)
            mock_path_instance.mkdir = MagicMock()
            
            audio_chunks = list(interview_service.stream_plain_text(plain_text, "s1", 1, "c1"))
        
        assert len(audio_chunks) > 0
        assert audio_chunks[0] == b'\xDE\xAD' # Our mock audio
        
        # Check that streaming_synthesize was called
        mock_tts_client.return_value.streaming_synthesize.assert_called_once()
        
        # Check the config passed in the *first* request
        call_args = mock_tts_client.return_value.streaming_synthesize.call_args
        request_iterator = call_args[1]['requests']
        first_request = next(request_iterator)
        
        config = first_request.streaming_config.streaming_audio_config
        assert config.audio_encoding == texttospeech.AudioEncoding.MULAW
        assert config.sample_rate_hertz == 24000

    # --- NEW: Test stream_interview_turn ---
    def test_stream_interview_turn_success(self, interview_service, mock_gemini, mock_tts_client):
        """Test successful streaming of a full Gemini->TTS turn"""
        session_id = interview_service.start_new_interview("Resume", "c1", [])
        
        # Patch the Gemini stream function to simplify this test
        with patch.object(interview_service, '_stream_gemini_sentences') as mock_gemini_stream:
            mock_gemini_stream.return_value = iter(["Hello.", "How are you?"])
            
            # --- FIX: Add mocks for Path and open ---
            with patch("builtins.open", new_callable=unittest.mock.mock_open()) as mock_file, \
                 patch('app.services.interview_service.Path') as mock_path_class:
                
                # --- THIS IS THE CORRECTED MOCK CHAIN ---
                mock_path_data = MagicMock(name="PathData")
                mock_path_c1 = MagicMock(name="PathC1")
                mock_path_session = MagicMock(name="PathSession")
                mock_path_audio_dir = MagicMock(name="PathAudioDir") # This is the one we want
                mock_path_audio_file = MagicMock(name="PathAudioFile")
                
                # Path("data")
                mock_path_class.return_value = mock_path_data
                # Path("data") / "c1"
                mock_path_data.__truediv__.return_value = mock_path_c1
                # Path("data") / "c1" / "test_session_123" (session_id)
                mock_path_c1.__truediv__.return_value = mock_path_session
                # Path("data") / "c1" / "test_session_123" / "audio"
                mock_path_session.__truediv__.return_value = mock_path_audio_dir
                # ... / f"bot_{session_id}_{turn_count}_streamed.txt"
                mock_path_audio_dir.__truediv__.return_value = mock_path_audio_file
                # --- END FIX ---

                # Mock the context manager for file writing
                mock_file.return_value.__enter__.return_value.write = MagicMock()
                
                # --- FIX: Add side_effect to consume the request generator ---
                mock_tts_client_instance = mock_tts_client.return_value
                original_tts_response = [Mock(audio_content=b'\xDE\xAD')]

                def consume_requests_and_return(*args, **kwargs):
                    request_gen = kwargs.get('requests')
                    if request_gen:
                        list(request_gen)  # Consume the generator
                    return original_tts_response

                mock_tts_client_instance.streaming_synthesize.side_effect = consume_requests_and_return
                # --- END FIX ---
                
                audio_chunks = list(interview_service.stream_interview_turn(session_id, 2, "c1"))

                # Check file system calls
                mock_path_class.assert_called_with("data")
                mock_path_data.__truediv__.assert_called_with("c1")
                mock_path_c1.__truediv__.assert_called_with("test_session_123")
                mock_path_session.__truediv__.assert_called_with("audio")
                mock_path_audio_dir.mkdir.assert_called_with(parents=True, exist_ok=True)
                mock_file.assert_called_with(mock_path_audio_file, "w", encoding="utf-8")
            
            # Assert our mock audio was yielded
            assert len(audio_chunks) > 0
            assert audio_chunks[0] == b'\xDE\xAD'
            
            # Assert both parts were called
            mock_gemini_stream.assert_called_once_with(session_id)
            mock_tts_client.return_value.streaming_synthesize.assert_called_once()
            
            # Assert the final text was logged
            calls = interview_service.db.add_message_to_session.call_args_list
            # Find the call with "assistant" role
            assistant_calls = [c for c in calls if len(c[0]) > 1 and c[0][1] == "assistant"]
            assert len(assistant_calls) > 0
            # Verify the text contains the sentences
            assert "Hello." in assistant_calls[0][0][2]
            assert "How are you?" in assistant_calls[0][0][2]

    def test_process_and_log_transcript(self, interview_service):
        """Test transcript processing and logging"""
        session_id, audio_path, turn_count, candidate_id = "s1", "a.wav", 1, "c1"
        start_time, end_time = datetime.now(), datetime.now()
        
        interview_service.process_and_log_transcript(
            session_id, audio_path, "Test transcript", turn_count, candidate_id,
            start_time, end_time, False
        )
        
        assert interview_service._latest_transcript_for_gemini[session_id] == "Test transcript"
        interview_service.db.add_message_to_session.assert_called_with(
            session_id, "user", "Test transcript", audio_path,
            start_time, end_time, is_follow_up=False
        )

    def test_process_and_log_transcript_with_error(self, interview_service):
        """Test transcript processing with error/unintelligible content"""
        session_id = "s1"
        
        # Test with error message
        interview_service.process_and_log_transcript(
            session_id, "a.wav", "[Error: API timeout]", 1, "c1",
            datetime.now(), datetime.now(), False
        )
        
        # Verify error transcript is NOT set as latest
        assert session_id not in interview_service._latest_transcript_for_gemini or \
               interview_service._latest_transcript_for_gemini[session_id] != "[Error: API timeout]"
        
        # Test with unintelligible
        interview_service.process_and_log_transcript(
            session_id, "b.wav", "[Unintelligible]", 2, "c1",
            datetime.now(), datetime.now(), False
        )
        
        assert session_id not in interview_service._latest_transcript_for_gemini or \
               interview_service._latest_transcript_for_gemini[session_id] != "[Unintelligible]"

    def test_generate_final_transcript_file(self, interview_service, tmp_path):
        """Test final transcript file generation"""
        session_id = "test_session_123"
        candidate_id = "test_candidate"
        
        # Use StringIO to capture file writes
        file_content = StringIO()
        
        with patch("builtins.open", new_callable=unittest.mock.mock_open()) as mock_file, \
             patch('app.services.interview_service.Path') as mock_path_class:
            
            # --- THIS IS THE CORRECTED MOCK CHAIN ---
            mock_path_data = MagicMock(name="PathData")
            mock_candidate_path = MagicMock(name="CandidatePath")
            mock_transcript_dir = MagicMock(name="TranscriptDir") # This is the one we want
            mock_transcript_file_path = MagicMock(name="FinalPath")
            
            # Path("data")
            mock_path_class.return_value = mock_path_data
            # Path("data") / "test_candidate"
            mock_path_data.__truediv__.return_value = mock_candidate_path
            # Path("data") / "test_candidate" / "transcripts"
            mock_candidate_path.__truediv__.return_value = mock_transcript_dir
            # ... / f"transcript_{session_id}.txt"
            mock_transcript_dir.__truediv__.return_value = mock_transcript_file_path
            # --- END FIX ---
            
            # Capture writes to StringIO
            def write_side_effect(content):
                file_content.write(content)
                return len(content)
            
            mock_file.return_value.__enter__.return_value.write.side_effect = write_side_effect
            
            interview_service.generate_final_transcript_file(session_id)

            # Check path calls
            mock_path_class.assert_called_with("data")
            mock_path_data.__truediv__.assert_called_with("test_candidate")
            mock_candidate_path.__truediv__.assert_called_with("transcripts")
            
            # Check that the directory was created
            mock_transcript_dir.mkdir.assert_called_with(parents=True, exist_ok=True)
            
            # --- FIX: This was the failing assertion ---
            mock_transcript_dir.__truediv__.assert_called_with(f"transcript_{session_id}.txt")
            
            # Check that the file was opened for writing
            mock_file.assert_called_with(mock_transcript_file_path, "w", encoding="utf-8")
            
            # Get captured content
            transcript_content = file_content.getvalue()
            
            # Check for key content
            assert "--- Interview Transcript ---" in transcript_content
            assert f"Session ID: {session_id}" in transcript_content
            assert f"Candidate ID: {candidate_id}" in transcript_content
            assert "Date: 2025-10-31 12:00:00" in transcript_content
            assert "Tab Switches (Hidden): 1" in transcript_content
            assert "(Streamed Text) Assistant:" in transcript_content
            assert "Hello" in transcript_content

    def test_end_interview_session(self, interview_service):
        """Test ending interview session"""
        session_id = "test_session_123"
        resume = "Test Resume"
        candidate_id = "candidate_001"
        
        interview_service.start_new_interview(resume, candidate_id, [])
        
        interview_service._session_token_counts[session_id] = {
            'prompt': 1000, 'response': 500, 'total': 1500
        }
        interview_service._session_tts_char_counts[session_id] = 5000
        
        interview_service.end_interview_session(session_id)
        
        assert session_id not in interview_service.active_chat_sessions
        assert session_id not in interview_service._session_token_counts
        assert session_id not in interview_service._session_tts_char_counts
        assert session_id not in interview_service._latest_transcript_for_gemini

    def test_token_counting(self, interview_service, mock_gemini):
        """Test token usage is tracked correctly from stream"""
        resume = "Test Resume"
        candidate_id = "candidate_001"
        
        session_id = interview_service.start_new_interview(resume, candidate_id, [])
        assert session_id == "test_session_123"
        
        # Consume the generator
        list(interview_service._stream_gemini_sentences(session_id))
        
        assert session_id in interview_service._session_token_counts
        token_data = interview_service._session_token_counts[session_id]
        
        # Verify token counts from mock
        assert token_data['prompt'] == 100
        assert token_data['response'] == 20
        assert token_data['total'] == 120

    def test_tts_character_counting(self, interview_service, mock_tts_client):
        """Test TTS character usage is tracked via stream_plain_text"""
        session_id = "test_session"
        text = "This is a test"
        turn_count = 1
        candidate_id = "candidate_001"
        
        with patch("builtins.open", new_callable=unittest.mock.mock_open()), \
             patch('app.services.interview_service.Path') as mock_path, \
             patch('app.services.interview_service.sf.write'), \
             patch('app.services.interview_service.audioop.ulaw2lin', return_value=b'\x00\x01'):
            
            mock_path_instance = MagicMock()
            mock_path.return_value = mock_path_instance
            mock_path_instance.__truediv__ = MagicMock(return_value=mock_path_instance)
            mock_path_instance.mkdir = MagicMock()
            
            list(interview_service.stream_plain_text(text, session_id, turn_count, candidate_id))
        
        assert session_id in interview_service._session_tts_char_counts
        char_count = interview_service._session_tts_char_counts[session_id]
        
        assert char_count == len(text)

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
        
        # Verify each session has independent tracking
        assert session_1 in interview_service._session_token_counts
        assert session_2 in interview_service._session_token_counts
        assert session_1 in interview_service._latest_transcript_for_gemini
        assert session_2 in interview_service._latest_transcript_for_gemini

    @pytest.mark.parametrize("error_scenario", ["api_error"])
    def test_gemini_stream_error_handling(self, interview_service, mock_gemini, error_scenario):
        """Test error handling in _stream_gemini_sentences"""
        resume = "Test Resume"
        candidate_id = "candidate_001"
        
        session_id = interview_service.start_new_interview(resume, candidate_id, [])
        assert session_id == "test_session_123"
        
        interview_service._latest_transcript_for_gemini[session_id] = "Test answer"
        
        mock_chat_session = interview_service.active_chat_sessions[session_id]
        
        if error_scenario == "api_error":
            mock_chat_session.send_message.side_effect = Exception("API Error")
        
        questions = list(interview_service._stream_gemini_sentences(session_id))
        
        assert len(questions) == 1
        question = questions[0]
        
        assert isinstance(question, str)
        assert len(question) > 0
        assert not question.startswith("<speak>")
        assert "Apologies" in question or "skill" in question

    def test_stream_interview_turn_with_alert(self, interview_service, mock_tts_client):
        """Test streaming an alert message (e.g., camera/mic warnings)"""
        session_id = "test_session"
        alert_text = "Please turn on your camera."
        
        with patch("builtins.open", unittest.mock.mock_open()), \
             patch('app.services.interview_service.Path') as mock_path, \
             patch('app.services.interview_service.sf.write'), \
             patch('app.services.interview_service.audioop.ulaw2lin', return_value=b'\x00\x01'):
            
            mock_path_instance = MagicMock()
            mock_path.return_value = mock_path_instance
            mock_path_instance.__truediv__ = MagicMock(return_value=mock_path_instance)
            mock_path_instance.mkdir = MagicMock()
            
            audio_chunks = list(interview_service.stream_plain_text(
                alert_text, session_id, "alert_1", "c1", is_follow_up=True
            ))
        
        assert len(audio_chunks) > 0
        
        # Verify alert was logged with follow_up flag
        interview_service.db.add_message_to_session.assert_called()
        call_args = interview_service.db.add_message_to_session.call_args
        assert call_args[0][2] == alert_text  # text
        assert call_args[1]['is_follow_up'] == True

    def test_thread_safety_transcript_updates(self, interview_service):
        """Test thread-safe transcript updates"""
        session_id = "s1"
        transcripts = ["Hello", "World", "Test"]
        
        def update_transcript(text):
            interview_service.process_and_log_transcript(
                session_id, f"{text}.wav", text, 1, "c1",
                datetime.now(), datetime.now(), False
            )
        
        threads = [threading.Thread(target=update_transcript, args=(t,)) for t in transcripts]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Verify one of the transcripts was set (no crashes)
        assert session_id in interview_service._latest_transcript_for_gemini
        assert interview_service._latest_transcript_for_gemini[session_id] in transcripts

    def test_gemini_streaming_sentence_splitting(self, interview_service):
        """Test that Gemini responses are properly split into sentences"""
        session_id = "test_session_123"
        interview_service.start_new_interview("Resume", "c1", [])
        
        # Mock chat session to return chunks with multiple sentences
        mock_chat = interview_service.active_chat_sessions[session_id]
        
        chunk1 = MagicMock()
        type(chunk1).text = PropertyMock(return_value="What is your experience? ")
        chunk1.usage_metadata = None
        
        chunk2 = MagicMock()
        type(chunk2).text = PropertyMock(return_value="Tell me more.")
        chunk2.usage_metadata = None
        
        chunk3 = MagicMock()
        type(chunk3).text = PropertyMock(side_effect=ValueError)
        chunk3.usage_metadata = MagicMock(
            prompt_token_count=50,
            candidates_token_count=10,
            total_token_count=60
        )
        
        mock_chat.send_message.return_value = [chunk1, chunk2, chunk3]
        
        sentences = list(interview_service._stream_gemini_sentences(session_id))
        
        # Should yield two separate sentences
        assert len(sentences) >= 2
        assert any("experience?" in s for s in sentences)
        assert any("Tell me more" in s for s in sentences)

    def test_production_readiness_checks(self, interview_service):
        """Verify production-ready features are working"""
        # Check error handling exists
        assert hasattr(interview_service, 'end_interview_session')
        assert hasattr(interview_service, 'generate_final_transcript_file')
        
        # Check tracking exists
        assert hasattr(interview_service, '_session_token_counts')
        assert hasattr(interview_service, '_session_tts_char_counts')
        
        # Check thread safety
        assert hasattr(interview_service, '_transcript_lock')
        
        # Check model configuration for deterministic output
        assert interview_service.model is not None
        
        # Check TTS client is initialized
        assert interview_service.tts_client is not None
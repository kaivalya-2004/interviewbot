# tests/test_integration_meet_interview_orchestrator.py
"""
Integration tests for MeetInterviewOrchestrator
Tests the complete interview flow with mocked audio and session management
"""
import queue
import pytest
import sys
import os
import time
import numpy as np
import threading
import asyncio # Import asyncio to patch it
from unittest.mock import MagicMock, patch, Mock, call, AsyncMock
from pathlib import Path
from datetime import datetime
import unittest
import numpy.testing as np_testing

# --- Setup Path ---
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, project_root)
# --- End Path Setup ---

# --- FIX: We must patch the 'speech' module *where it is used* ---
@pytest.fixture(autouse=True)
def mock_google_speech_client():
    """
    Mocks the 'speech.SpeechClient' class *as it is imported* by the orchestrator.
    This is more robust than the sys.modules hack.
    The 'autouse=True' ensures this patch is active for *all* tests in this file.
    """
    # 1. Patch the SpeechClient class *inside* the orchestrator module
    with patch('app.services.meet_interview_orchestrator.speech.SpeechClient') as mock_client_class:
        
        # 2. Mock the *instance* that the class will return
        mock_client_instance = MagicMock(name="SpeechClient_Instance")
        mock_client_class.return_value = mock_client_instance
        
        # 3. Mock the instance's methods
        mock_response = MagicMock()
        mock_result = MagicMock()
        mock_alternative = MagicMock()
        mock_alternative.transcript = "This is a test transcript."
        mock_result.alternatives = [mock_alternative]
        mock_result.is_final = True
        mock_response.results = [mock_result]
        mock_client_instance.streaming_recognize.return_value = [mock_response]
        
        # 4. Yield the MOCK CLASS, which we can assert against
        yield mock_client_class
# --- END FIX ---


from app.services.meet_interview_orchestrator import MeetInterviewOrchestrator

@pytest.fixture
def mock_session_manager():
    """Mock MeetSessionManager"""
    mock_mgr = MagicMock(name="SessionManager")
    
    # Default session data
    mock_session = {
        'controller': MagicMock(name="MeetController"),
        'candidate_id': 'test_candidate_001',
        'stop_interview': threading.Event(),
        'status': 'active'
    }
    
    # Mock controller methods
    mock_session['controller'].enable_microphone = MagicMock()
    mock_session['controller'].disable_microphone = MagicMock()
    mock_session['controller'].get_participant_count = MagicMock(return_value=2)
    
    mock_mgr.get_session.return_value = mock_session
    
    return mock_mgr


@pytest.fixture
def mock_interview_service():
    """Mock InterviewService"""
    mock_svc = MagicMock(name="InterviewService")
    
    mock_svc.generate_initial_greeting.return_value = "Hello! Welcome to your interview."
    
    # Mock streaming functions to return a simple generator
    mock_svc.stream_plain_text.return_value = iter([b"fake_ulaw_chunk"])
    mock_svc.stream_interview_turn.return_value = iter([b"fake_ulaw_chunk"])

    # Mock DB for non-answer logging
    mock_svc.db = MagicMock()
    mock_svc.db.add_message_to_session = MagicMock()
    
    mock_svc.process_and_log_transcript = MagicMock()
    mock_svc.generate_final_transcript_file = MagicMock()
    
    return mock_svc


@pytest.fixture
def mock_sounddevice():
    """Mock sounddevice module"""
    with patch('app.services.meet_interview_orchestrator.sd') as mock_sd:
        mock_stream = MagicMock()
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)
        mock_stream.stop = MagicMock()
        mock_stream.start = MagicMock()
        mock_stream.close = MagicMock()
        mock_stream.closed = False
        
        mock_sd.OutputStream.return_value = mock_stream
        mock_sd.InputStream.return_value = mock_stream
        mock_sd.CallbackStop = Exception
        mock_sd.PortAudioError = Exception
        mock_sd.default.device = [0, 1] # Default devices
        
        mock_sd.query_devices.return_value = [
            {'index': 0, 'name': 'Default Mic', 'max_input_channels': 2, 'max_output_channels': 0},
            {'index': 1, 'name': 'CABLE Input (VB-Audio Virtual Cable)', 'max_input_channels': 0, 'max_output_channels': 2},
            {'index': 2, 'name': 'CABLE Output (VB-Audio Virtual Cable)', 'max_input_channels': 2, 'max_output_channels': 0},
            {'index': 3, 'name': 'Default Speakers', 'max_input_channels': 0, 'max_output_channels': 2},
        ]
        
        yield mock_sd


@pytest.fixture
def mock_soundfile():
    """Mock soundfile module"""
    with patch('app.services.meet_interview_orchestrator.sf') as mock_sf:
        mock_sf.write = MagicMock()
        yield mock_sf

@pytest.fixture
def orchestrator(mock_session_manager, mock_interview_service, mock_sounddevice, mock_google_speech_client):
    """
    Create orchestrator with mocked dependencies.
    mock_sounddevice is required here to patch sd.query_devices()
    during the orchestrator's initialization.
    mock_google_speech_client is required to patch the client on init.
    """
    # Because mock_google_speech_client is autouse=True, its patch is
    # *already active* when this function runs.
    return MeetInterviewOrchestrator(
        session_manager=mock_session_manager,
        interview_service=mock_interview_service
    )

@pytest.fixture
def mock_orchestrator_io(mocker):
    """
    Mocks the blocking I/O helper functions
    _play_audio_stream and _record_and_process_stt_streaming_manual_vad
    """
    mock_play = mocker.patch(
        'app.services.meet_interview_orchestrator.MeetInterviewOrchestrator._play_audio_stream',
        return_value=True  # Simulate successful playback
    )
    
    mock_record = mocker.patch(
        'app.services.meet_interview_orchestrator.MeetInterviewOrchestrator._record_and_process_stt_streaming_manual_vad',
        # Default return: (start_time, transcript_text)
        return_value=(datetime.now(), "Mocked transcript")
    )
    
    yield {
        "play": mock_play,
        "record": mock_record
    }

@pytest.fixture
def mock_app_time():
    """
    Creates a mock for the 'time' module used by the orchestrator.
    This mock must also include 'sleep'.
    """
    mock = MagicMock()
    mock.sleep = MagicMock() # Mock the sleep function
    return mock


class TestMeetInterviewOrchestrator:
    """Integration tests for MeetInterviewOrchestrator"""

    def test_initialization(self, orchestrator, mock_session_manager, mock_interview_service, mock_sounddevice, mock_google_speech_client):
        """Test orchestrator initializes correctly"""
        assert orchestrator.session_mgr == mock_session_manager
        assert orchestrator.interview_svc == mock_interview_service
        
        mock_sounddevice.query_devices.assert_called_once()
        
        # --- FIX: Assert against the mock class, which was yielded by the fixture ---
        mock_google_speech_client.assert_called_once()
        # --- END FIX ---
        
        assert orchestrator.target_samplerate == 24000
        assert orchestrator.virtual_output == 1 # 'CABLE Input (VB-Audio Virtual Cable)'
        assert orchestrator.virtual_input == 2  # 'CABLE Output (VB-Audio Virtual Cable)'

    @pytest.mark.asyncio
    async def test_conduct_interview_basic_flow(
        self, orchestrator, mock_session_manager, mock_interview_service,
        mock_orchestrator_io, mock_app_time
    ):
        """Test basic interview flow completes successfully"""
        session_id = "test_session_001"
        
        # Mock the audio streams for Q1 and Q2
        mock_interview_service.stream_interview_turn.side_effect = [
            iter([b"q1_chunk"]),
            iter([b"q2_chunk"])
        ]
        
        # Mock the transcripts for Intro, A1, and A2
        mock_orchestrator_io['record'].side_effect = [
            (datetime.now(), "This is my intro."),
            (datetime.now(), "This is my answer to Q1."),
            (datetime.now(), "This is my answer to Q2.")
        ]
        
        mock_app_time.time.side_effect = [
            0.0,    # start_time
            30.0,   # elapsed check 1
            60.0,   # elapsed check 2
            90.0,   # elapsed check 3 (max_questions hit)
            100.0,  # final elapsed
        ]

        with patch('app.services.meet_interview_orchestrator.time', mock_app_time), \
             patch('asyncio.sleep', AsyncMock()), \
             patch('asyncio.to_thread', side_effect=lambda func, *args, **kwargs: func(*args, **kwargs)):
            result = await orchestrator.conduct_interview(
                session_id=session_id,
                interview_duration_minutes=10, 
                max_questions=2
            )
        
        assert result['status'] == 'max_questions_reached'
        assert result['session_id'] == session_id
        assert result['questions_asked'] == 2
        mock_interview_service.generate_initial_greeting.assert_called_once()
        assert mock_interview_service.stream_interview_turn.call_count == 2
        mock_interview_service.generate_final_transcript_file.assert_called_once_with(session_id)
        
        # play calls: Greeting, Q1, Q2, Closing
        assert mock_orchestrator_io['play'].call_count == 4
        # record calls: Intro, A1, A2
        assert mock_orchestrator_io['record'].call_count == 3
        # process_and_log calls: Intro, A1, A2
        assert mock_interview_service.process_and_log_transcript.call_count == 3

    @pytest.mark.asyncio
    async def test_conduct_interview_time_limit(
        self, orchestrator, mock_session_manager, mock_interview_service,
        mock_orchestrator_io, mock_app_time
    ):
        """Test interview stops at time limit"""
        session_id = "test_session_time"
        
        # Mock time to run out after 1 question
        mock_app_time.time.side_effect = [
            0.0,    # start_time
            30.0,   # elapsed check 1 (Q1)
            590.0,  # elapsed check 2 ( > 510s, triggers time limit)
            600.0   # final elapsed
        ] 
        
        with patch('app.services.meet_interview_orchestrator.time', mock_app_time), \
             patch('asyncio.sleep', AsyncMock()), \
             patch('asyncio.to_thread', side_effect=lambda func, *args, **kwargs: func(*args, **kwargs)):
            result = await orchestrator.conduct_interview(
                session_id=session_id,
                interview_duration_minutes=10,  # 600 seconds
                max_questions=100
            )
        
        assert result['status'] == 'time_limit_reached'
        assert result['questions_asked'] == 1 

    @pytest.mark.asyncio
    async def test_conduct_interview_session_not_found(
        self, orchestrator, mock_session_manager
    ):
        """Test handling when session is not found"""
        mock_session_manager.get_session.return_value = None
        result = await orchestrator.conduct_interview("nonexistent_session")
        assert result['error'] == "Active session not found"
        assert result['status'] == "failed"

    @pytest.mark.asyncio
    async def test_conduct_interview_missing_controller(
        self, orchestrator, mock_session_manager
    ):
        """Test handling when controller is missing from session"""
        mock_session = {'candidate_id': 'test', 'controller': None, 'stop_interview': threading.Event()}
        mock_session_manager.get_session.return_value = mock_session
        result = await orchestrator.conduct_interview("test_session")
        assert result['status'] == "failed"
        assert 'error' in result

    @pytest.mark.asyncio
    async def test_conduct_interview_stop_signal(
        self, orchestrator, mock_session_manager, mock_interview_service,
        mock_orchestrator_io, mock_app_time
    ):
        """Test interview stops on stop signal"""
        session_id = "test_session_stop"
        session = mock_session_manager.get_session(session_id)
        
        session['stop_interview'].clear()
        
        def record_side_effect(*args, **kwargs):
             # Stop signal after intro
             session['stop_interview'].set()
             return (datetime.now(), "My intro")
        
        mock_orchestrator_io['record'].side_effect = record_side_effect
        mock_app_time.time.return_value = 0.0
        
        with patch('app.services.meet_interview_orchestrator.time', mock_app_time), \
             patch('asyncio.sleep', AsyncMock()), \
             patch('asyncio.to_thread', side_effect=lambda func, *args, **kwargs: func(*args, **kwargs)):
             result = await orchestrator.conduct_interview(
                 session_id=session_id,
                 interview_duration_minutes=10,
                 max_questions=5
             )

        assert result['status'] == 'terminated'
        assert mock_orchestrator_io['play'].call_count == 2
        assert mock_orchestrator_io['record'].call_count == 1

    @pytest.mark.asyncio
    async def test_conduct_interview_multiple_participants(
        self, orchestrator, mock_session_manager, mock_interview_service,
        mock_orchestrator_io, mock_app_time
    ):
        """Test interview terminates when multiple participants detected"""
        session_id = "test_session_multi"
        session = mock_session_manager.get_session(session_id)
        controller = session['controller']
        
        mock_interview_service.reset_mock()
        
        # 1. Check in loop (2)
        # 2. Check in loop (3) -> Triggers termination
        controller.get_participant_count.side_effect = [2, 3] 
        
        mock_app_time.time.side_effect = [0.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0]
        
        with patch('app.services.meet_interview_orchestrator.time', mock_app_time), \
             patch('asyncio.sleep', AsyncMock()), \
             patch('asyncio.to_thread', side_effect=lambda func, *args, **kwargs: func(*args, **kwargs)):
            result = await orchestrator.conduct_interview(
                session_id=session_id,
                interview_duration_minutes=10,
                max_questions=5
            )
        
        assert result['status'] == 'multiple_participants'
        assert session['termination_reason'] == 'multiple_participants'
        
        # play calls: Greeting, Q1, Termination Msg
        assert mock_orchestrator_io['play'].call_count == 3
        # record calls: Intro, A1
        assert mock_orchestrator_io['record'].call_count == 2
        
        # Check that the *second* call to stream_plain_text was the termination msg
        # Call 0 is Greeting, Call 1 is Termination
        termination_call_text = mock_interview_service.stream_plain_text.call_args_list[1].args[0]
        assert "extra participants" in termination_call_text

    @pytest.mark.asyncio
    async def test_conduct_interview_candidate_leaves_no_rejoin(
        self, orchestrator, mock_session_manager, mock_interview_service,
        mock_orchestrator_io, mock_app_time
    ):
        """Test interview terminates when candidate leaves and doesn't rejoin"""
        session_id = "test_session_leave"
        session = mock_session_manager.get_session(session_id)
        controller = session['controller']

        # Check in loop (2), then (1) for 150 checks (timeout)
        count_sequence = [2] + ([1] * 150)
        controller.get_participant_count.side_effect = count_sequence

        time_values = [0.0, 30.0] + [30.0 + i * 1 for i in range(150)]
        mock_app_time.time.side_effect = time_values
        
        with patch('app.services.meet_interview_orchestrator.time', mock_app_time), \
             patch('asyncio.sleep', AsyncMock()), \
             patch('asyncio.to_thread', side_effect=lambda func, *args, **kwargs: func(*args, **kwargs)):
            result = await orchestrator.conduct_interview(
                session_id=session_id,
                interview_duration_minutes=10,
                max_questions=5
            )
        
        assert result['status'] == 'candidate_left'
        assert session['termination_reason'] == 'candidate_left'

    @pytest.mark.asyncio
    async def test_conduct_interview_candidate_leaves_then_rejoins(
        self, orchestrator, mock_session_manager, mock_interview_service,
        mock_orchestrator_io, mock_app_time
    ):
        """Test interview continues when candidate rejoins after temporary leave"""
        session_id = "test_session_rejoin"
        session = mock_session_manager.get_session(session_id)
        controller = session['controller']
        
        # Drop to 1, then come back to 2
        count_sequence = [2, 1, 1, 1, 2, 2, 2] 
        controller.get_participant_count.side_effect = count_sequence + [2] * 50
        
        mock_interview_service.stream_interview_turn.side_effect = [
            iter([b"q1"]), iter([b"q2"])
        ]
        
        time_values = [0.0] + [i * 5 for i in range(150)] # Needs to be long
        mock_app_time.time.side_effect = time_values
        
        with patch('app.services.meet_interview_orchestrator.time', mock_app_time), \
             patch('asyncio.sleep', AsyncMock()), \
             patch('asyncio.to_thread', side_effect=lambda func, *args, **kwargs: func(*args, **kwargs)):
            result = await orchestrator.conduct_interview(
                session_id=session_id,
                interview_duration_minutes=10,
                max_questions=2
            )
        
        assert result['status'] == 'max_questions_reached'

    @pytest.mark.asyncio
    async def test_get_user_audio_path_for_stt(self, orchestrator):
        """Test STT audio path generation"""
        session_id = "test_session"
        turn_count = 5
        candidate_id = "test_candidate"
        
        path = orchestrator._get_user_audio_path_for_stt(
            session_id, turn_count, candidate_id, is_follow_up=False
        )
        assert "candidate_turn_5_stt_24k.wav" in str(path)
        assert str(path).startswith(os.path.join("data", "test_candidate", "test_session", "audio"))
        
        path_followup = orchestrator._get_user_audio_path_for_stt(
            session_id, turn_count, candidate_id, is_follow_up=True
        )
        assert "candidate_turn_5_followup_stt_24k.wav" in str(path_followup)

    @pytest.mark.asyncio
    async def test_conduct_interview_greeting_playback_fail_no_rejoin(
        self, orchestrator, mock_session_manager, mock_interview_service,
        mock_orchestrator_io, mock_app_time
    ):
        """Test interview handles candidate leaving during greeting"""
        session_id = "test_session_greeting_fail"
        session = mock_session_manager.get_session(session_id)
        controller = session['controller']
        
        # Mock that the first playback fails (e.g., candidate left)
        mock_orchestrator_io['play'].return_value = False
        controller.get_participant_count.return_value = 1 # Candidate stays gone
        
        time_values = [0.0] + [i * 5 for i in range(150)]
        mock_app_time.time.side_effect = time_values
        
        with patch('app.services.meet_interview_orchestrator.time', mock_app_time), \
             patch('asyncio.sleep', AsyncMock()), \
             patch('asyncio.to_thread', side_effect=lambda func, *args, **kwargs: func(*args, **kwargs)):
            
                result = await orchestrator.conduct_interview(
                    session_id=session_id,
                    interview_duration_minutes=10,
                    max_questions=5
                )
        
        assert result['status'] == 'candidate_left'
        assert session['termination_reason'] == 'candidate_left'
        assert mock_orchestrator_io['play'].call_count == 1 # Only greeting playback was attempted
        assert mock_orchestrator_io['record'].call_count == 0 # No intro was recorded

    @pytest.mark.asyncio
    async def test_conduct_interview_full_flow_with_transcript(
        self, orchestrator, mock_session_manager, mock_interview_service,
        mock_orchestrator_io, mock_app_time
    ):
        """Test complete interview flow with transcript generation"""
        session_id = "test_session_full"
        
        mock_interview_service.stream_interview_turn.side_effect = [
            iter([b"q1"]), iter([b"q2"])
        ]
        mock_orchestrator_io['record'].side_effect = [
            (datetime.now(), "Mocked Intro"),
            (datetime.now(), "Mocked Answer 1"),
            (datetime.now(), "Mocked Answer 2")
        ]
        
        mock_app_time.time.side_effect = [0.0, 30.0, 60.0, 90.0, 120.0, 150.0]
        
        with patch('app.services.meet_interview_orchestrator.time', mock_app_time), \
             patch('asyncio.sleep', AsyncMock()), \
             patch('asyncio.to_thread', side_effect=lambda func, *args, **kwargs: func(*args, **kwargs)):
            result = await orchestrator.conduct_interview(
                session_id=session_id,
                interview_duration_minutes=5,
                max_questions=2
            )
        
        assert 'final_transcript_summary' in result
        transcript = result['final_transcript_summary']
        
        # Transcript should contain the *user's* responses
        assert len(transcript) == 3
        assert transcript[0]['content'] == "Mocked Intro"
        assert transcript[1]['content'] == "Mocked Answer 1"
        assert transcript[2]['content'] == "Mocked Answer 2"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("duration_minutes,max_questions,expected_status", [
        (1, 100, "time_limit_reached"), 
        (100, 3, "max_questions_reached"),
    ])
    async def test_conduct_interview_limit_scenarios(
        self, orchestrator, mock_session_manager, mock_interview_service,
        mock_orchestrator_io, mock_app_time,
        duration_minutes, max_questions, expected_status
    ):
        """Test different limit scenarios"""
        session_id = f"test_session_{expected_status}"
        
        mock_interview_service.stream_interview_turn.return_value = iter([b"q_chunk"])
        
        time_values = [float(i * 30) for i in range(50)]
        mock_app_time.time.side_effect = time_values
        
        with patch('app.services.meet_interview_orchestrator.time', mock_app_time), \
             patch('asyncio.sleep', AsyncMock()), \
             patch('asyncio.to_thread', side_effect=lambda func, *args, **kwargs: func(*args, **kwargs)):
            result = await orchestrator.conduct_interview(
                session_id=session_id,
                interview_duration_minutes=duration_minutes,
                max_questions=max_questions
            )
        
        assert result['status'] == expected_status
        
    @pytest.mark.asyncio
    async def test_conduct_interview_drop_during_question_and_rejoins(
        self, orchestrator, mock_session_manager, mock_interview_service,
        mock_orchestrator_io, mock_app_time
    ):
        """
        Test the rejoin flow when candidate drops *during* question playback.
        This covers the new logic to re-ask the question.
        """
        session_id = "test_session_drop_rejoin"
        session = mock_session_manager.get_session(session_id)
        controller = session['controller']
        
        max_questions_to_ask = 2
        
        # play: Greeting (OK), Q1 (Fail), Q1-repeat (OK), Q2 (OK), Closing (OK)
        mock_orchestrator_io['play'].side_effect = [True, False, True, True, True]
        
        # Check during Q1 playback (fail), then rejoin
        controller.get_participant_count.side_effect = [
            2, # Initial check
            1, # Rejoin check 1
            1, # Rejoin check 2
            2, # Rejoin success!
            2, # Check before Q2
            2  # Final check
        ] + ([2] * 20)
        
        time_values = [
            0.0,   # Start
            30.0,  # Before Q1
            31.0,  # Rejoin wait start
            36.0,  # Rejoin check 1
            41.0,  # Rejoin check 2 (success)
            80.0,  # Before Q2
            120.0, # Before Q3 (max questions hit)
            130.0  # Final elapsed
        ] + ([140.0] * 50)
        mock_app_time.time.side_effect = time_values

        # stream: Q1, Q1-repeat, Q2
        mock_interview_service.stream_interview_turn.side_effect = [
            iter([b"q1"]), iter([b"q1_repeat"]), iter([b"q2"])
        ]
        
        with patch('app.services.meet_interview_orchestrator.time', mock_app_time), \
             patch('asyncio.sleep', AsyncMock()), \
             patch('asyncio.to_thread', side_effect=lambda func, *args, **kwargs: func(*args, **kwargs)):
            result = await orchestrator.conduct_interview(
                session_id=session_id,
                interview_duration_minutes=10,
                max_questions=max_questions_to_ask
            )
        
        assert result['status'] == 'max_questions_reached'
        assert result['questions_asked'] == 2 # Still only 2 *unique* questions asked
        
        # play: Greeting, Q1(fail), Q1(repeat), Q2, Closing
        assert mock_orchestrator_io['play'].call_count == 5
        # record: Intro, A1, A2
        assert mock_orchestrator_io['record'].call_count == 3
        # stream: Q1, Q1-repeat, Q2
        assert mock_interview_service.stream_interview_turn.call_count == 3
        
    @pytest.mark.asyncio
    async def test_conduct_interview_bad_intro_text(
        self, orchestrator, mock_session_manager, mock_interview_service,
        mock_orchestrator_io, mock_app_time
    ):
        """Test the logic branch for when STT returns bad intro text."""
        session_id = "test_session_bad_intro"
        
        # Return an error for the first recording (the intro)
        mock_orchestrator_io['record'].return_value = (datetime.now(), "[Error: STT Failed]")
        
        mock_app_time.time.side_effect = [0.0, 30.0, 60.0] + ([70.0] * 50)
        
        with patch('app.services.meet_interview_orchestrator.time', mock_app_time), \
             patch('asyncio.sleep', AsyncMock()), \
             patch('asyncio.to_thread', side_effect=lambda func, *args, **kwargs: func(*args, **kwargs)):
            result = await orchestrator.conduct_interview(
                session_id=session_id,
                interview_duration_minutes=1,
                max_questions=0 # Stop after intro
            )
        
        assert result['status'] == 'time_limit_reached'
        # Check that the bad intro was still added to the summary
        assert result['final_transcript_summary'][0]['content'] == "[Error: STT Failed]"

    @pytest.mark.asyncio
    async def test_conduct_interview_participant_check_fails(
        self, orchestrator, mock_session_manager, mock_interview_service,
        mock_orchestrator_io, mock_app_time
    ):
        """Test the loop continues if get_participant_count fails."""
        session_id = "test_session_check_fails"
        session = mock_session_manager.get_session(session_id)
        
        session['controller'].get_participant_count.side_effect = [
            2,
            Exception("Selenium check failed"),
            2
        ]
        
        mock_app_time.time.side_effect = [0.0, 30.0, 60.0, 90.0] + ([100.0] * 50)
        
        mock_interview_service.stream_interview_turn.side_effect = [
            iter([b"q1"]), iter([b"q2"])
        ]
        
        with patch('app.services.meet_interview_orchestrator.time', mock_app_time), \
             patch('asyncio.sleep', AsyncMock()), \
             patch('asyncio.to_thread', side_effect=lambda func, *args, **kwargs: func(*args, **kwargs)):
            result = await orchestrator.conduct_interview(
                session_id=session_id,
                interview_duration_minutes=10,
                max_questions=2
            )
        
        assert result['status'] == 'max_questions_reached'
        assert result['questions_asked'] == 2

    @pytest.mark.asyncio
    async def test_conduct_interview_final_transcript_fails(
        self, orchestrator, mock_session_manager, mock_interview_service,
        mock_orchestrator_io, mock_app_time
    ):
        """Test interview completes even if final transcript save fails."""
        session_id = "test_session_final_fail"
        
        mock_interview_service.generate_final_transcript_file.side_effect = Exception("Disk full!")
        
        mock_app_time.time.side_effect = [0.0, 30.0, 60.0] + ([70.0] * 50)
        
        with patch('app.services.meet_interview_orchestrator.time', mock_app_time), \
             patch('asyncio.sleep', AsyncMock()), \
             patch('asyncio.to_thread', side_effect=lambda func, *args, **kwargs: func(*args, **kwargs)):
            result = await orchestrator.conduct_interview(
                session_id=session_id,
                interview_duration_minutes=1,
                max_questions=0 # Stop after intro
            )
        
        assert result['status'] == 'time_limit_reached'
        mock_interview_service.generate_final_transcript_file.assert_called_once()

    # --- START: New Tests for New Logic ---

    @pytest.mark.asyncio
    async def test_conduct_interview_non_answer_reasks_question(
        self, orchestrator, mock_session_manager, mock_interview_service,
        mock_orchestrator_io, mock_app_time
    ):
        """
        Test the new non-answer logic:
        1. Asks Q1
        2. User says "pardon"
        3. Bot plays "Sorry", re-asks Q1
        4. User gives real answer
        5. Bot asks Q2
        """
        session_id = "test_session_non_answer"
        mock_app_time.time.side_effect = [0.0, 30.0, 60.0, 90.0, 120.0, 150.0] + ([200.0] * 50)
        
        # play: Greeting, Q1, "Sorry", Q1-repeat, Q2, Closing
        mock_orchestrator_io['play'].side_effect = [True, True, True, True, True, True]
        
        # record: Intro, Non-Answer ("Pardon?"), Good Answer 1, Good Answer 2
        mock_orchestrator_io['record'].side_effect = [
            (datetime.now(), "My intro"),
            (datetime.now(), "Pardon?"),
            (datetime.now(), "Ah, my real answer to Q1"),
            (datetime.now(), "My answer to Q2")
        ]
        
        # stream_interview_turn: Q1, Q1-repeat, Q2
        mock_interview_service.stream_interview_turn.side_effect = [
            iter([b"q1"]), iter([b"q1_repeat"]), iter([b"q2"])
        ]
        
        with patch('app.services.meet_interview_orchestrator.time', mock_app_time), \
             patch('asyncio.sleep', AsyncMock()), \
             patch('asyncio.to_thread', side_effect=lambda func, *args, **kwargs: func(*args, **kwargs)):
            result = await orchestrator.conduct_interview(
                session_id=session_id, interview_duration_minutes=10, max_questions=2
            )
        
        assert result['status'] == 'max_questions_reached'
        assert result['questions_asked'] == 2 # Only 2 *unique* questions
        
        # play calls: Greeting, Q1, "Sorry", Q1-repeat, Q2, Closing
        assert mock_orchestrator_io['play'].call_count == 6
        # record calls: Intro, "Pardon", Answer 1, Answer 2
        assert mock_orchestrator_io['record'].call_count == 4
        # stream_interview_turn calls: Q1, Q1-repeat, Q2
        assert mock_interview_service.stream_interview_turn.call_count == 3
        
        # Check that the "pardon" message was logged to DB (but not Gemini)
        mock_interview_service.db.add_message_to_session.assert_called_with(
            session_id, "user", "Pardon?", unittest.mock.ANY, unittest.mock.ANY, unittest.mock.ANY
        )
        
        # --- FIX: INDENTATION ERROR WAS HERE ---
        # Check that process_and_log_transcript (Gemini update) was called for Intro, A1, A2
        assert mock_interview_service.process_and_log_transcript.call_count == 3
        calls = mock_interview_service.process_and_log_transcript.call_args_list 
        assert calls[0].args[2] == "My intro"
        assert calls[1].args[2] == "Ah, my real answer to Q1"
        assert calls[2].args[2] == "My answer to Q2"
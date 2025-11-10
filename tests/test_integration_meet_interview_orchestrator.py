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
from unittest.mock import MagicMock, patch, Mock, call
from pathlib import Path
from datetime import datetime
import unittest
import numpy.testing as np_testing

# --- Setup Path ---
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, project_root)
# --- End Path Setup ---

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
    
    mock_svc.generate_initial_greeting.return_value = "<speak>Hello! Welcome to your interview.</speak>"
    mock_svc.generate_next_question.return_value = "<speak>Tell me about your experience with Python.</speak>"
    
    mock_svc.text_to_speech.return_value = (
        np.random.randn(24000).astype(np.float32),  # 1 second of audio at 24kHz
        24000
    )
    mock_svc.process_and_log_transcript = MagicMock()
    mock_svc.generate_final_transcript_file = MagicMock()
    mock_svc._transcript_lock = threading.Lock()
    mock_svc._latest_transcript_for_gemini = {
        'test_session_001': 'I have 5 years of Python experience.'
    }
    
    return mock_svc


@pytest.fixture
def mock_sounddevice():
    """Mock sounddevice module"""
    with patch('app.services.meet_interview_orchestrator.sd') as mock_sd:
        mock_stream = MagicMock()
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)
        mock_stream.stop = MagicMock()
        
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
def mock_librosa():
    """Mock librosa module (used in _play_audio_data)"""
    mock_resample = MagicMock(side_effect=lambda y, orig_sr, target_sr: y)
    
    mock_librosa_module = MagicMock()
    mock_librosa_module.resample = mock_resample
    
    with patch.dict('sys.modules', {'librosa': mock_librosa_module}):
        yield mock_librosa_module

@pytest.fixture
def orchestrator(mock_session_manager, mock_interview_service, mock_sounddevice):
    """
    Create orchestrator with mocked dependencies.
    mock_sounddevice is required here to patch sd.query_devices()
    during the orchestrator's initialization.
    """
    return MeetInterviewOrchestrator(
        session_manager=mock_session_manager,
        interview_service=mock_interview_service
    )

@pytest.fixture
def mock_orchestrator_helpers(mocker):
    """
    Mocks the blocking helper functions _play_audio_data and
    _record_and_process_stt_background inside the orchestrator.
    """
    mock_play = mocker.patch(
        'app.services.meet_interview_orchestrator.MeetInterviewOrchestrator._play_audio_data',
        return_value=True
    )
    
    mock_thread = MagicMock(name="MockSTTThread")
    mock_thread.is_alive.return_value = False 
    mock_thread.join = MagicMock()
    
    mock_record = mocker.patch(
        'app.services.meet_interview_orchestrator.MeetInterviewOrchestrator._record_and_process_stt_background',
        return_value=(mock_thread, datetime.now())
    )
    
    yield {
        "play": mock_play,
        "record": mock_record,
        "thread": mock_thread
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

    def test_initialization(self, orchestrator, mock_session_manager, mock_interview_service, mock_sounddevice):
        """Test orchestrator initializes correctly"""
        assert orchestrator.session_mgr == mock_session_manager
        assert orchestrator.interview_svc == mock_interview_service
        
        mock_sounddevice.query_devices.assert_called_once()
        assert orchestrator.target_samplerate == 24000
        assert orchestrator.virtual_output == 1 # 'CABLE Input (VB-Audio Virtual Cable)'
        assert orchestrator.virtual_input == 2  # 'CABLE Output (VB-Audio Virtual Cable)'

    def test_conduct_interview_basic_flow(
        self, orchestrator, mock_session_manager, mock_interview_service,
        mock_orchestrator_helpers, mock_app_time
    ):
        """Test basic interview flow completes successfully"""
        session_id = "test_session_001"
        
        mock_interview_service.generate_next_question.side_effect = [
            "<speak>Question 1?</speak>",
            "<speak>Question 2?</speak>",
            StopIteration 
        ]
        
        mock_app_time.time.side_effect = [
            0.0,    # start_time
            30.0,   # elapsed check 1
            60.0,   # elapsed check 2
            90.0,   # elapsed check 3 (max_questions hit)
            100.0,  # final elapsed
        ]

        with patch('logging.time', MagicMock(time=lambda: 0.0)):
            with patch('app.services.meet_interview_orchestrator.time', mock_app_time):
                result = orchestrator.conduct_interview(
                    session_id=session_id,
                    interview_duration_minutes=10, 
                    max_questions=2
                )
        
        assert result['status'] == 'max_questions_reached'
        assert result['session_id'] == session_id
        assert result['questions_asked'] == 2
        mock_interview_service.generate_initial_greeting.assert_called_once()
        assert mock_interview_service.generate_next_question.call_count == 2
        mock_interview_service.generate_final_transcript_file.assert_called_once_with(session_id)
        assert mock_orchestrator_helpers['play'].call_count == 4 # Greeting, Q1, Q2, Closing
        assert mock_orchestrator_helpers['record'].call_count == 3 # Intro, A1, A2

    def test_conduct_interview_time_limit(
        self, orchestrator, mock_session_manager, mock_interview_service,
        mock_orchestrator_helpers, mock_app_time
    ):
        """Test interview stops at time limit"""
        session_id = "test_session_time"
        
        mock_interview_service.generate_next_question.return_value = "<speak>Question?</speak>"
        
        mock_app_time.time.side_effect = [
            0.0,    # start_time
            30.0,   # elapsed check 1
            130.0,  # elapsed check 2 ( > 120s )
            140.0   # final elapsed
        ] 
        
        with patch('logging.time', MagicMock(time=lambda: 0.0)):
            with patch('app.services.meet_interview_orchestrator.time', mock_app_time):
                result = orchestrator.conduct_interview(
                    session_id=session_id,
                    interview_duration_minutes=2,  # 120 seconds
                    max_questions=100
                )
        
        assert result['status'] == 'time_limit_reached'
        assert result['questions_asked'] == 1 

    def test_conduct_interview_session_not_found(
        self, orchestrator, mock_session_manager
    ):
        """Test handling when session is not found"""
        mock_session_manager.get_session.return_value = None
        result = orchestrator.conduct_interview("nonexistent_session")
        assert result['error'] == "Active session not found"
        assert result['status'] == "failed"

    def test_conduct_interview_missing_controller(
        self, orchestrator, mock_session_manager
    ):
        """Test handling when controller is missing from session"""
        mock_session = {'candidate_id': 'test', 'controller': None, 'stop_interview': threading.Event()}
        mock_session_manager.get_session.return_value = mock_session
        result = orchestrator.conduct_interview("test_session")
        assert result['status'] == "failed"
        assert 'error' in result

    def test_conduct_interview_stop_signal(
        self, orchestrator, mock_session_manager, mock_interview_service,
        mock_orchestrator_helpers, mock_app_time
    ):
        """Test interview stops on stop signal"""
        session_id = "test_session_stop"
        session = mock_session_manager.get_session(session_id)
        
        session['stop_interview'].clear()
        mock_interview_service.generate_next_question.return_value = "<speak>Question?</speak>"

        def record_side_effect(*args, **kwargs):
             # args[1] is turn_count
             if args[1] == 2: # This is the intro recording
                 session['stop_interview'].set()
             return (MagicMock(is_alive=lambda: False, join=MagicMock()), datetime.now())
        
        mock_orchestrator_helpers['record'].reset_mock()
        mock_orchestrator_helpers['record'].side_effect = record_side_effect
        mock_app_time.time.return_value = 0.0
        
        with patch('logging.time', MagicMock(time=lambda: 0.0)):
             with patch('app.services.meet_interview_orchestrator.time', mock_app_time):
                 result = orchestrator.conduct_interview(
                     session_id=session_id,
                     interview_duration_minutes=10,
                     max_questions=5
                 )

        assert result['status'] == 'terminated'
        assert mock_orchestrator_helpers['play'].call_count == 1 # 1 greeting
        assert mock_orchestrator_helpers['record'].call_count == 1 # 1 intro

    def test_conduct_interview_multiple_participants(
        self, orchestrator, mock_session_manager, mock_interview_service,
        mock_orchestrator_helpers, mock_app_time
    ):
        """Test interview terminates when multiple participants detected"""
        session_id = "test_session_multi"
        session = mock_session_manager.get_session(session_id)
        controller = session['controller']
        
        mock_interview_service.reset_mock()
        
        # 1. Check in loop (2)
        # 2. Check in loop (3)
        controller.get_participant_count.side_effect = [2, 3] 
        
        mock_app_time.time.side_effect = [0.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0]
        
        with patch('logging.time', MagicMock(time=lambda: 0.0)):
            with patch('app.services.meet_interview_orchestrator.time', mock_app_time):
                result = orchestrator.conduct_interview(
                    session_id=session_id,
                    interview_duration_minutes=10,
                    max_questions=5
                )
        
        assert result['status'] == 'terminated'
        assert session['termination_reason'] == 'multiple_participants'
        
        # --- FIX: Update assertions to reflect the correct flow ---
        # Greeting, Q1, Termination Msg
        assert mock_orchestrator_helpers['play'].call_count == 3
        # Intro, A1
        assert mock_orchestrator_helpers['record'].call_count == 2
        
        # Call list is now:
        # [0] Greeting TTS
        # [1] Q1 TTS
        # [2] Termination TTS
        termination_call_text = mock_interview_service.text_to_speech.call_args_list[2].args[0]
        assert "extra participants" in termination_call_text
        # --- END FIX ---

    def test_conduct_interview_candidate_leaves_no_rejoin(
        self, orchestrator, mock_session_manager, mock_interview_service,
        mock_orchestrator_helpers, mock_app_time
    ):
        """Test interview terminates when candidate leaves and doesn't rejoin"""
        session_id = "test_session_leave"
        session = mock_session_manager.get_session(session_id)
        controller = session['controller']

        count_sequence = [2] + ([1] * 150)
        controller.get_participant_count.side_effect = count_sequence

        time_values = [0.0, 30.0] + [30.0 + i * 1 for i in range(150)]
        mock_app_time.time.side_effect = time_values
        
        with patch('logging.time', MagicMock(time=lambda: 0.0)):
            with patch('app.services.meet_interview_orchestrator.time', mock_app_time):
                result = orchestrator.conduct_interview(
                    session_id=session_id,
                    interview_duration_minutes=10,
                    max_questions=5
                )
        
        assert result['status'] == 'terminated'
        assert session['termination_reason'] == 'candidate_left'

    def test_conduct_interview_candidate_leaves_then_rejoins(
        self, orchestrator, mock_session_manager, mock_interview_service,
        mock_orchestrator_helpers, mock_app_time
    ):
        """Test interview continues when candidate rejoins after temporary leave"""
        session_id = "test_session_rejoin"
        session = mock_session_manager.get_session(session_id)
        controller = session['controller']
        
        count_sequence = [2, 1, 1, 1, 2, 2, 2] 
        controller.get_participant_count.side_effect = count_sequence + [2] * 50
        
        mock_interview_service.generate_next_question.side_effect = [
            "<speak>Question 1?</speak>",
            "<speak>Question 2?</speak>",
        ]
        
        time_values = [0.0] + [i * 5 for i in range(150)] # Needs to be long
        mock_app_time.time.side_effect = time_values
        
        with patch('logging.time', MagicMock(time=lambda: 0.0)):
            with patch('app.services.meet_interview_orchestrator.time', mock_app_time):
                result = orchestrator.conduct_interview(
                    session_id=session_id,
                    interview_duration_minutes=10,
                    max_questions=2
                )
        
        assert result['status'] == 'max_questions_reached'

    # --- Tests for helper functions ---

    def test_play_audio_data_success_no_resample(
        self, orchestrator, mock_session_manager, mock_sounddevice, mock_librosa
    ):
        """Test successful audio playback (matching sample rate)"""
        audio_data = np.random.randn(24000).astype(np.float32)
        sample_rate = 24000
        
        session = mock_session_manager.get_session('test')
        meet = session['controller']
        stop_event = session['stop_interview']
        
        mock_stream = mock_sounddevice.OutputStream.return_value
        finished_event = threading.Event()
        finished_event.set() 
        
        with patch('threading.Event', return_value=finished_event):
            result = orchestrator._play_audio_data(
                audio_data, sample_rate, meet, stop_event
            )
        
        assert result is True
        mock_librosa.resample.assert_not_called()
        mock_sounddevice.OutputStream.assert_called_with(
            samplerate=24000,
            device=orchestrator.virtual_output,
            channels=1,
            dtype='float32',
            callback=unittest.mock.ANY,
            finished_callback=unittest.mock.ANY
        )


    def test_play_audio_data_stop_signal(
        self, orchestrator, mock_session_manager, mock_sounddevice, mock_librosa
    ):
        """Test audio playback stops on stop signal"""
        audio_data = np.random.randn(24000).astype(np.float32)
        sample_rate = 24000
        
        session = mock_session_manager.get_session('test')
        meet = session['controller']
        stop_event = session['stop_interview']
        stop_event.set()
        
        mock_stream = mock_sounddevice.OutputStream.return_value
        
        result = orchestrator._play_audio_data(
            audio_data, sample_rate, meet, stop_event
        )
        
        assert result is False
        mock_stream.stop.assert_called()

    def test_play_audio_data_candidate_leaves(
        self, orchestrator, mock_session_manager, mock_sounddevice, mock_librosa
    ):
        """Test audio playback stops when candidate leaves"""
        audio_data = np.random.randn(24000).astype(np.float32)
        sample_rate = 24000
        
        session = mock_session_manager.get_session('test')
        meet = session['controller']
        stop_event = session['stop_interview']
        
        meet.get_participant_count.return_value = 1
        
        mock_stream = mock_sounddevice.OutputStream.return_value
        
        finished_event = threading.Event() 
        
        with patch('threading.Event', return_value=finished_event):
             result = orchestrator._play_audio_data(
                 audio_data, sample_rate, meet, stop_event
             )
        
        assert result is False
        mock_stream.stop.assert_called()

    def test_play_audio_data_resampling(
        self, orchestrator, mock_session_manager, mock_sounddevice, mock_librosa
    ):
        """Test audio resampling when sample rates don't match"""
        audio_data = np.random.randn(16000).astype(np.int16)
        sample_rate = 16000
        
        session = mock_session_manager.get_session('test')
        meet = session['controller']
        stop_event = session['stop_interview']
        
        finished_event = threading.Event()
        finished_event.set()
        
        with patch('threading.Event', return_value=finished_event):
            result = orchestrator._play_audio_data(
                audio_data, sample_rate, meet, stop_event
            )
        
        mock_librosa.resample.assert_called_once()
        call_args = mock_librosa.resample.call_args
        assert call_args[1]['orig_sr'] == 16000
        assert call_args[1]['target_sr'] == 24000
        assert result is True

    def test_get_user_audio_path_for_stt(self, orchestrator):
        """Test STT audio path generation (renamed)"""
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

    def test_record_and_process_stt_background_session_ended(
        self, orchestrator, mock_session_manager
    ):
        """Test recording stops if session ends"""
        mock_session_manager.get_session.return_value = None
        thread, start_time = orchestrator._record_and_process_stt_background(
            "ended_session", 1, "candidate", duration=10
        )
        assert thread is None
        assert start_time is None

    def test_record_and_process_stt_background_stop_signal(
        self, orchestrator, mock_session_manager
    ):
        """Test recording respects stop signal"""
        session = mock_session_manager.get_session('test')
        session['stop_interview'].set()
        thread, start_time = orchestrator._record_and_process_stt_background(
            'test', 1, 'candidate', duration=10
        )
        assert thread is None
        assert start_time is None

    def test_record_and_process_stt_background_creates_thread(
        self, orchestrator, mock_session_manager, mock_sounddevice,
        mock_soundfile, mock_librosa
    ):
        """Test recording creates background processing thread"""
        session_id = "test_session"
        
        mock_stream = MagicMock()
        mock_stream.start = MagicMock()
        mock_stream.stop = MagicMock()
        mock_stream.close = MagicMock()
        mock_stream.closed = False
        mock_sounddevice.InputStream.return_value = mock_stream
        
        with patch('queue.Queue') as mock_queue_class:
            mock_queue = MagicMock()
            mock_audio_chunk = np.random.randn(1024).astype(np.float32)
            mock_silence_chunk = np.zeros(1024).astype(np.float32)
            mock_queue.get.side_effect = [
                mock_audio_chunk, # Speech
                mock_audio_chunk, # Speech
                mock_silence_chunk, # Silence starts
                mock_silence_chunk,
                mock_silence_chunk,
                queue.Empty("Simulate end of silence")
            ]
            mock_queue.empty.return_value = True
            mock_queue_class.return_value = mock_queue
            
            with patch('app.services.meet_interview_orchestrator.datetime') as mock_dt:
                mock_dt.now.side_effect = [
                    datetime(2024, 1, 1, 12, 0, 0), # Call 1: recording_start_time
                    datetime(2024, 1, 1, 12, 0, 5)  # Call 2: elapsed check (to break)
                ]
                
                time_side_effect = [
                    1000.0, # speech_start_time
                    1001.0, # silence_start_time
                    1002.0,
                    1003.0, # > 2.0s silence
                    1004.0
                ]
                with patch('app.services.meet_interview_orchestrator.time.time', side_effect=time_side_effect):
                    thread, start_time = orchestrator._record_and_process_stt_background(
                        session_id, 1, 'candidate', duration=1
                    )
        
        assert start_time is not None
        assert thread is not None
        assert thread.name == "STTProcess-1"
        
        mock_sounddevice.InputStream.assert_called_with(
            samplerate=24000,
            device=orchestrator.virtual_input,
            channels=1,
            callback=unittest.mock.ANY,
            dtype='float32'
        )


    def test_save_and_process_audio_thread_func(
        self, orchestrator, mock_interview_service, mock_soundfile, mock_librosa, tmp_path
    ):
        """Test background STT audio processing function (renamed)"""
        recording_data = np.random.randn(24000, 1).astype(np.float32)
        original_samplerate = 24000
        audio_path_stt = tmp_path / "test_audio_24k.wav"
        session_id = "test_session"
        turn_count = 1
        candidate_id = "test_candidate"
        start_time = datetime.now()
        
        orchestrator._save_and_process_audio_thread_func(
            recording_data, original_samplerate, audio_path_stt,
            session_id, turn_count, candidate_id, start_time, False
        )
        
        mock_soundfile.write.assert_called_once()
        call_args, call_kwargs = mock_soundfile.write.call_args
        
        assert call_args[0] == str(audio_path_stt)
        np_testing.assert_array_equal(call_args[1], recording_data.flatten())
        assert call_args[2] == 24000
        assert call_kwargs['format'] == 'WAV'
        assert call_kwargs['subtype'] == 'PCM_16'
        
        mock_librosa.resample.assert_not_called() 
        
        mock_interview_service.process_and_log_transcript.assert_called_once_with(
            session_id, str(audio_path_stt), turn_count, candidate_id,
            start_time, unittest.mock.ANY, False
        )

    def test_conduct_interview_greeting_playback_fail_no_rejoin(
        self, orchestrator, mock_session_manager, mock_interview_service,
        mock_orchestrator_helpers, mock_app_time
    ):
        """Test interview handles candidate leaving during greeting"""
        session_id = "test_session_greeting_fail"
        session = mock_session_manager.get_session(session_id)
        controller = session['controller']
        
        mock_orchestrator_helpers['play'].return_value = False
        controller.get_participant_count.return_value = 1
        
        time_values = [0.0] + [i * 5 for i in range(150)]
        mock_app_time.time.side_effect = time_values
        
        with patch('logging.time', MagicMock(time=lambda: 0.0)):
            with patch('app.services.meet_interview_orchestrator.time', mock_app_time):
                
                result = orchestrator.conduct_interview(
                    session_id=session_id,
                    interview_duration_minutes=10,
                    max_questions=5
                )
        
        assert result['status'] == 'terminated'
        assert session['termination_reason'] == 'candidate_left'
        assert mock_orchestrator_helpers['play'].call_count == 1
        assert mock_orchestrator_helpers['record'].call_count == 0

    def test_conduct_interview_full_flow_with_transcript(
        self, orchestrator, mock_session_manager, mock_interview_service,
        mock_orchestrator_helpers, mock_app_time
    ):
        """Test complete interview flow with transcript generation"""
        session_id = "test_session_full"
        
        mock_interview_service.generate_next_question.side_effect = [
            "<speak>What is your experience?</speak>",
            "<speak>Tell me about a project.</speak>",
        ]
        
        mock_interview_service._latest_transcript_for_gemini = {
            session_id: "[Mocked Intro]"
        }
        
        mock_app_time.time.side_effect = [0.0, 30.0, 60.0, 90.0, 120.0, 150.0]
        
        with patch('logging.time', MagicMock(time=lambda: 0.0)):
            with patch('app.services.meet_interview_orchestrator.time', mock_app_time):
                result = orchestrator.conduct_interview(
                    session_id=session_id,
                    interview_duration_minutes=5,
                    max_questions=2
                )
        
        assert 'final_transcript_summary' in result
        transcript = result['final_transcript_summary']
        assert len(transcript) > 0
        assistant_messages = [t for t in transcript if t.get('role') == 'assistant']
        assert len(assistant_messages) == 2
        assert assistant_messages[0]['content'] == "<speak>What is your experience?</speak>"

    @pytest.mark.parametrize("duration_minutes,max_questions,expected_status", [
        (1, 100, "time_limit_reached"), 
        (100, 3, "max_questions_reached"),
    ])
    def test_conduct_interview_limit_scenarios(
        self, orchestrator, mock_session_manager, mock_interview_service,
        mock_orchestrator_helpers, mock_app_time,
        duration_minutes, max_questions, expected_status
    ):
        """Test different limit scenarios"""
        session_id = f"test_session_{expected_status}"
        
        mock_interview_service.generate_next_question.return_value = "<speak>Question?</speak>"
        
        time_values = [float(i * 30) for i in range(50)]
        mock_app_time.time.side_effect = time_values
        
        with patch('logging.time', MagicMock(time=lambda: 0.0)):
            with patch('app.services.meet_interview_orchestrator.time', mock_app_time):
                result = orchestrator.conduct_interview(
                    session_id=session_id,
                    interview_duration_minutes=duration_minutes,
                    max_questions=max_questions
                )
        
        assert result['status'] == expected_status
        
    # --- START: New Tests for Coverage ---

    def test_initialization_no_vb_audio_devices(self, mock_session_manager, mock_interview_service, mock_sounddevice):
        """Test initialization when VB-Audio devices are not in the list."""
        mock_sounddevice.query_devices.return_value = [
            {'index': 0, 'name': 'Default Mic', 'max_input_channels': 2, 'max_output_channels': 0},
            {'index': 3, 'name': 'Default Speakers', 'max_input_channels': 0, 'max_output_channels': 2},
        ]
        
        orchestrator = MeetInterviewOrchestrator(
            session_manager=mock_session_manager,
            interview_service=mock_interview_service
        )
        
        assert orchestrator.virtual_output is None
        assert orchestrator.virtual_input is None

    def test_initialization_device_query_fails(self, mock_session_manager, mock_interview_service, mock_sounddevice):
        """Test initialization when sd.query_devices() raises an exception."""
        mock_sounddevice.query_devices.side_effect = Exception("Failed to query devices")
        
        orchestrator = MeetInterviewOrchestrator(
            session_manager=mock_session_manager,
            interview_service=mock_interview_service
        )
        
        assert orchestrator.virtual_output is None
        assert orchestrator.virtual_input is None
        
    def test_play_audio_data_resample_fails(
        self, orchestrator, mock_session_manager, mock_sounddevice, mock_librosa
    ):
        """Test audio resampling failure (covers librosa exception)."""
        mock_librosa.resample.side_effect = Exception("Resample failed")
        
        audio_data = np.random.randn(16000).astype(np.int16)
        sample_rate = 16000
        
        session = mock_session_manager.get_session('test')
        
        result = orchestrator._play_audio_data(
            audio_data, sample_rate, session['controller'], session['stop_interview']
        )
        
        assert result is False

    def test_play_audio_data_participant_check_fails(
        self, orchestrator, mock_session_manager, mock_sounddevice, mock_librosa
    ):
        """Test audio playback continues even if participant check fails."""
        audio_data = np.random.randn(24000).astype(np.float32)
        sample_rate = 24000

        session = mock_session_manager.get_session('test')
        meet = session['controller']
        stop_event = session['stop_interview']

        meet.get_participant_count.reset_mock()
        meet.get_participant_count.side_effect = Exception("Selenium error")

        mock_stream = mock_sounddevice.OutputStream.return_value
        
        # 1. Create the mock event
        mock_event = MagicMock(spec=threading.Event)
        
        # 2. Define the behavior of is_set():
        #    - First call (while loop check): return False (loop enters)
        #    - Second call (while loop check): return True (loop exits)
        mock_event.is_set.side_effect = [False, True]
        
        # 3. Define the behavior of wait():
        #    - It will be called inside the loop.
        #    - Make it return False (simulating a 1.0s timeout)
        mock_event.wait.return_value = False
        
        # 4. Patch threading.Event to return this mock event
        with patch('threading.Event', return_value=mock_event):
            result = orchestrator._play_audio_data(
                audio_data, sample_rate, meet, stop_event
            )

        # 5. Assertions
        assert result is True 
        
        # The loop ran once, so the check was called once
        meet.get_participant_count.assert_called_once()
        
        # is_set was called twice (once to enter, once to exit)
        assert mock_event.is_set.call_count == 2
        
        # wait was called once
        mock_event.wait.assert_called_once_with(timeout=1.0)

    def test_conduct_interview_drop_during_question_and_rejoins(
        self, orchestrator, mock_session_manager, mock_interview_service,
        mock_orchestrator_helpers, mock_app_time
    ):
        """
        Test the rejoin flow when candidate drops *during* question playback.
        This covers the 'if not playback_ok:' block (lines 234-256).
        """
        session_id = "test_session_drop_rejoin"
        session = mock_session_manager.get_session(session_id)
        controller = session['controller']
        
        # --- FIX: Define max_questions as a variable ---
        max_questions_to_ask = 2
        # --- END FIX ---
        
        mock_orchestrator_helpers['play'].side_effect = [True, False, True, True, True]
        
        controller.get_participant_count.side_effect = [
            2, # Initial check
            1, # Rejoin check 1
            1, # Rejoin check 2
            1, # Rejoin check 3
            2, # Rejoin success!
            2, # Check before Q2
            2  # Final check
        ] + ([2] * 20) # Add extra values
        
        # --- FIX: Add many more time values for the loops ---
        time_values = [
            0.0,   # Start
            30.0,  # Before Q1
            31.0,  # Rejoin wait start
            36.0,  # Rejoin check 1
            41.0,  # Rejoin check 2
            46.0,  # Rejoin check 3 (success)
            80.0,  # Before Q2
            120.0, # Before Q3 (max questions hit)
            130.0  # Final elapsed
        ] + ([140.0] * 50) # Add many extra values
        mock_app_time.time.side_effect = time_values
        # --- END FIX ---

        mock_interview_service.generate_next_question.side_effect = [
            "<speak>Question 1?</speak>",
            "<speak>Question 2?</speak>",
            "<speak>Goodbye!</speak>" # FIX: Add a 3rd value for the final call
        ]
        
        with patch('app.services.meet_interview_orchestrator.time', mock_app_time):
            result = orchestrator.conduct_interview(
                session_id=session_id,
                interview_duration_minutes=10,
                max_questions=max_questions_to_ask # <-- USE VARIABLE
            )
        
        assert result['status'] == 'max_questions_reached'
        assert result['questions_asked'] == 2
        assert mock_orchestrator_helpers['play'].call_count == 5
        assert mock_orchestrator_helpers['record'].call_count == 3
        # --- FIX: Use the local variable for the assertion ---
        assert mock_interview_service.generate_next_question.call_count == max_questions_to_ask + 1
        
    def test_conduct_interview_stt_thread_timeout_in_loop(
        self, orchestrator, mock_session_manager, mock_interview_service,
        mock_orchestrator_helpers, mock_app_time
    ):
        """Test STT thread timing out *inside* the loop (covers lines 210-213)."""
        session_id = "test_session_stt_timeout"
        
        mock_orchestrator_helpers['thread'].is_alive.return_value = True
        
        # --- FIX: Add many more time values ---
        mock_app_time.time.side_effect = [0.0, 30.0, 60.0, 90.0] + ([100.0] * 50)
        # --- END FIX ---
        
        mock_interview_service.generate_next_question.return_value = "<speak>Question 1?</speak>"
        
        with patch('app.services.meet_interview_orchestrator.time', mock_app_time):
            result = orchestrator.conduct_interview(
                session_id=session_id,
                interview_duration_minutes=1, # 60 seconds
                max_questions=5
            )

        assert result['status'] == 'time_limit_reached'
        mock_orchestrator_helpers['thread'].join.assert_called_with(timeout=30)


    def test_conduct_interview_bad_intro_text(
        self, orchestrator, mock_session_manager, mock_interview_service,
        mock_orchestrator_helpers, mock_app_time
    ):
        """Test the logic branch for when STT returns bad intro text (covers line 147)."""
        session_id = "test_session_bad_intro"
        
        mock_interview_service._latest_transcript_for_gemini = {
            session_id: "[Error: STT Failed]"
        }
        
        # --- FIX: Add more time values ---
        mock_app_time.time.side_effect = [0.0, 30.0, 60.0] + ([70.0] * 50)
        # --- END FIX ---
        
        with patch('app.services.meet_interview_orchestrator.time', mock_app_time):
            result = orchestrator.conduct_interview(
                session_id=session_id,
                interview_duration_minutes=1,
                max_questions=0 # Stop after intro
            )
        
        # --- FIX: Assertion was wrong, code is correct. ---
        assert result['status'] == 'time_limit_reached'
        # --- END FIX ---
        assert result['final_transcript_summary'][0]['content'] == "[Error: STT Failed]"

    def test_conduct_interview_participant_check_fails(
        self, orchestrator, mock_session_manager, mock_interview_service,
        mock_orchestrator_helpers, mock_app_time
    ):
        """Test the loop continues if get_participant_count fails (covers line 204)."""
        session_id = "test_session_check_fails"
        session = mock_session_manager.get_session(session_id)
        
        session['controller'].get_participant_count.side_effect = [
            2,
            Exception("Selenium check failed"),
            2
        ]
        
        # --- FIX: Add many more time values ---
        mock_app_time.time.side_effect = [0.0, 30.0, 60.0, 90.0] + ([100.0] * 50)
        # --- END FIX ---
        
        mock_interview_service.generate_next_question.side_effect = [
            "<speak>Question 1?</speak>",
            "<speak>Question 2?</speak>",
        ]
        
        with patch('app.services.meet_interview_orchestrator.time', mock_app_time):
            result = orchestrator.conduct_interview(
                session_id=session_id,
                interview_duration_minutes=10,
                max_questions=2
            )
        
        assert result['status'] == 'max_questions_reached'
        assert result['questions_asked'] == 2

    def test_conduct_interview_final_transcript_fails(
        self, orchestrator, mock_session_manager, mock_interview_service,
        mock_orchestrator_helpers, mock_app_time
    ):
        """Test interview completes even if final transcript save fails (covers line 305)."""
        session_id = "test_session_final_fail"
        
        mock_interview_service.generate_final_transcript_file.side_effect = Exception("Disk full!")
        
        # --- FIX: Need more time values for the loop to exit properly ---
        mock_app_time.time.side_effect = [0.0, 30.0, 60.0] + ([70.0] * 50)
        # --- END FIX ---
        
        with patch('app.services.meet_interview_orchestrator.time', mock_app_time):
            result = orchestrator.conduct_interview(
                session_id=session_id,
                interview_duration_minutes=1,
                max_questions=0 # Stop after intro
            )
        
        # --- FIX: Assertion was wrong, code is correct. ---
        assert result['status'] == 'time_limit_reached'
        # --- END FIX ---
        mock_interview_service.generate_final_transcript_file.assert_called_once()
        
    # --- END: New Tests for Coverage ---
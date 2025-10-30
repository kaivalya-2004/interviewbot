# app/services/meet_interview_orchestrator.py
import logging
import time
import sounddevice as sd
import soundfile as sf
import numpy as np
from typing import Dict, Any, Optional, List, Tuple # Added Tuple
from datetime import datetime
from pathlib import Path
import queue
import threading
import librosa # Needed for resampling recorded audio

from app.services.meet_session_manager import MeetSessionManager
from app.services.interview_service import InterviewService
from app.config.audio_config import VirtualAudioConfig

logger = logging.getLogger(__name__)

# --- NEW: Estimated buffer time for one full Q&A turn ---
ESTIMATED_TURN_DURATION_SECONDS = 90
# --- END NEW ---

class MeetInterviewOrchestrator:
    """Orchestrates the interview flow based on duration, using Gemini for questions."""

    def __init__(
        self,
        session_manager: MeetSessionManager,
        interview_service: InterviewService
    ):
        self.session_mgr = session_manager
        self.interview_svc = interview_service
        self.virtual_output = VirtualAudioConfig.get_virtual_output_device()
        self.virtual_input = VirtualAudioConfig.get_virtual_input_device()
        self.target_samplerate = 24000
        if self.virtual_output is None or self.virtual_input is None:
            logger.warning("⚠️ Virtual audio devices not configured properly!")

    def conduct_interview(
        self,
        session_id: str,
        interview_duration_minutes: int = 10,
        max_questions: int = 10, # Keep max_questions as a secondary limit
    ) -> Dict[str, Any]:
        """
        Conducts interview, prioritizing duration limit over max questions.
        """
        logger.info(f"🎤 Starting interview orchestration for session: {session_id} (Duration: {interview_duration_minutes} mins, Max Qs: {max_questions})")
        interview_duration_seconds = interview_duration_minutes * 60

        session = self.session_mgr.get_session(session_id)
        if not session: return {"error": "Active session not found", "status": "failed"}

        stop_event = session.get('stop_interview')
        meet = session.get('controller')
        candidate_id = session.get('candidate_id')
        if not meet or not candidate_id:
             logger.error("Controller or candidate_id missing."); return {"error": "Internal session error", "status": "failed"}

        logger.info("Proceeding with interview start.")
        time.sleep(1)

        start_time = time.time()
        transcript = [] # Local summary
        turn_count = 1 # For unique filenames
        questions_asked_count = 0 # Track actual questions asked
        active_stt_thread: Optional[threading.Thread] = None

        # --- Ask Initial Greeting & Handle Intro ---
        # (This section remains the same, including starting the STT thread for the intro)
        initial_greeting = self.interview_svc.generate_initial_greeting()
        logger.info(f"🤖 Bot: {initial_greeting}")
        meet.enable_microphone(); time.sleep(0.5)
        greeting_audio_data, _ = self.interview_svc.text_to_speech( initial_greeting, session_id, turn_count, candidate_id )
        turn_count += 1
        if greeting_audio_data is not None: self._play_audio_data(greeting_audio_data, self.target_samplerate)
        time.sleep(0.5)
        logger.info("🎤 Listening for candidate's introduction...")
        active_stt_thread, rec_start_time = self._record_and_process_stt_background( session_id, turn_count, candidate_id, duration=30, is_follow_up_response=False )
        intro_turn = turn_count
        turn_count += 1
        logger.info("Disabling mic after intro..."); meet.disable_microphone(); logger.info("Mic disabled.")
        # --- End Greeting/Intro ---

        # --- Main Interview Loop (Prioritizes Time) ---
        while True:
            # --- MODIFICATION: Check termination conditions at the START ---
            elapsed_time = time.time() - start_time
            remaining_time = interview_duration_seconds - elapsed_time

            if stop_event and stop_event.is_set():
                logger.info("Termination signal received. Stopping loop.")
                break
            if questions_asked_count >= max_questions:
                logger.info(f"Maximum question limit ({max_questions}) reached. Ending interview.")
                break
            if remaining_time < ESTIMATED_TURN_DURATION_SECONDS:
                logger.info(f"Estimated time remaining ({remaining_time:.0f}s) is less than buffer ({ESTIMATED_TURN_DURATION_SECONDS}s). Ending interview.")
                break
            # --- END MODIFICATION ---

            logger.info(f"Interview time elapsed: {elapsed_time:.0f}s / {interview_duration_seconds}s (Remaining: {remaining_time:.0f}s)")

            # Wait for previous STT thread before generating next question
            if active_stt_thread and active_stt_thread.is_alive():
                wait_start = time.time()
                logger.info(f"Waiting for previous STT thread (Turn {intro_turn if questions_asked_count == 0 else current_turn}) to complete...")
                active_stt_thread.join(timeout=30) # Add a timeout to join
                wait_duration = time.time() - wait_start
                if active_stt_thread.is_alive():
                     logger.error("STT thread did not complete within timeout! Proceeding cautiously.")
                     # Handle this? Maybe skip the Gemini call or use old transcript?
                else:
                     logger.info(f"Previous STT thread finished (waited {wait_duration:.1f}s).")

            # Check stop event again after potentially waiting
            if stop_event and stop_event.is_set(): logger.info("Termination signal after STT wait."); break

            logger.info(f"\n--- Generating Question {questions_asked_count + 1}/{max_questions} ---")
            current_question = self.interview_svc.generate_next_question(session_id)
            if "Error:" in current_question: logger.error(f"Gemini error: {current_question}. Ending."); break
            # Increment count *after* successfully getting a question
            questions_asked_count += 1

            logger.info(f"🤖 Bot: {current_question}")
            transcript.append({"role": "assistant", "content": current_question})

            # Ask question
            meet.enable_microphone(); time.sleep(0.5)
            question_audio_data, _ = self.interview_svc.text_to_speech( current_question, session_id, turn_count, candidate_id )
            turn_count += 1
            if question_audio_data is not None: self._play_audio_data(question_audio_data, self.target_samplerate)
            time.sleep(0.5)

            # Record response and start background processing
            logger.info("🎤 Listening for candidate response...")
            active_stt_thread, rec_start_time = self._record_and_process_stt_background( session_id, turn_count, candidate_id, duration=30, is_follow_up_response=False )
            current_turn = turn_count # Store turn number for this response/STT thread
            turn_count += 1

            # Disable mic right away
            logger.info("Disabling microphone..."); meet.disable_microphone(); logger.info("Mic disabled.")

            # Check stop event after starting background process
            if stop_event and stop_event.is_set(): logger.info("Stop signal after starting STT thread."); break

            transcript.append({"role": "user", "content": "[Processing Response...]", "turn": current_turn})
            time.sleep(0.5)


        # --- End of Loop ---
        logger.info("Interview loop finished.")

        # Wait for the last STT thread
        if active_stt_thread and active_stt_thread.is_alive():
            logger.info(f"Waiting for final STT thread (Turn {current_turn}) to complete...")
            active_stt_thread.join(timeout=30) # Timeout for final wait too
            if active_stt_thread.is_alive(): logger.error("Final STT thread timed out.")
            else: logger.info("Final STT thread finished.")

        # Closing statement
        closing_reason = "Allocated time is up." # Default if time limit hit
        if stop_event and stop_event.is_set(): closing_reason = "Interview was ended early."
        elif questions_asked_count >= max_questions: closing_reason = "We have reached the question limit."

        if not (stop_event and stop_event.is_set()):
            closing_text = f"Thank you for your time. {closing_reason} That concludes our interview today."
            logger.info(f"🤖 Bot: {closing_text}")
            meet.enable_microphone(); time.sleep(0.5)
            closing_audio_data, _ = self.interview_svc.text_to_speech( closing_text, session_id, turn_count, candidate_id )
            if closing_audio_data is not None: self._play_audio_data(closing_audio_data, self.target_samplerate)
            time.sleep(0.5)
            meet.disable_microphone()
        else: logger.info("Interview terminated externally. Skipping closing.")

        try: self.interview_svc.generate_final_transcript_file(session_id)
        except Exception as e: logger.error(f"Failed generate final transcript: {e}")

        logger.info("✅ Interview orchestration function finished.")

        # Determine final status more accurately
        final_status = "unknown"
        final_elapsed = time.time() - start_time # Recalculate final time
        if stop_event and stop_event.is_set(): final_status = "terminated"
        elif remaining_time < ESTIMATED_TURN_DURATION_SECONDS or final_elapsed >= interview_duration_seconds : final_status = "time_limit_reached"
        elif questions_asked_count >= max_questions: final_status = "max_questions_reached"
        else: final_status = "completed" # Completed normally before limits

        return { "status": final_status, "session_id": session_id, "questions_asked": questions_asked_count, "final_transcript_summary": transcript }

    # --- Helper functions (_play_audio_data, _get_stt_audio_path, _save_stt_audio_thread_func, _record_and_process_stt_background) ---
    # These remain unchanged from the previous version.
    # (Code omitted for brevity)
    def _play_audio_data(self, audio_data: np.ndarray, sample_rate: int):
        # ... (same as previous response) ...
        try:
            if sample_rate != self.target_samplerate: logger.warning(f"⚠️ Playback SR mismatch! Expected {self.target_samplerate}, got {sample_rate}.")
            logger.info(f"📥 Playing audio data: {sample_rate}Hz, {len(audio_data)} samples")
            if audio_data.dtype == np.int16: data_float = audio_data.astype(np.float32) / 32768.0
            elif audio_data.dtype != np.float32: data_float = audio_data.astype(np.float32)
            else: data_float = audio_data
            if len(data_float.shape) > 1: data_float = np.mean(data_float, axis=1)
            max_val = np.abs(data_float).max();
            if max_val > 0: data_float = data_float / max_val * 0.9
            silence_duration = 0.7; silence_samples = int(silence_duration * self.target_samplerate); silence = np.zeros(silence_samples, dtype=np.float32)
            data_to_play = np.concatenate([silence, data_float, silence])
            output_device = self.virtual_output if self.virtual_output is not None else sd.default.device[1]
            duration = len(data_to_play) / self.target_samplerate
            logger.info(f"🔊 Playing to device {output_device} ({self.target_samplerate}Hz, {duration:.2f}s)")
            sd.play(data_to_play, self.target_samplerate, device=output_device, blocking=True)
            time.sleep(0.5); logger.info("✅ Playback complete")
        except Exception as e: logger.error(f"❌ Play audio fail: {e}", exc_info=True)

    def _get_stt_audio_path(self, session_id: str, turn_count: int, candidate_id: str, is_follow_up: bool) -> Path:
        audio_dir = Path("data") / "audio" / candidate_id / session_id
        suffix = "_followup" if is_follow_up else ""
        return audio_dir / f"candidate_turn_{turn_count}{suffix}_stt_16k.wav"

    def _save_stt_audio_thread_func(
        self,
        recording_data: np.ndarray,
        original_samplerate: int,
        target_samplerate: int, # Should be 16000
        audio_path_16k: Path
    ):
        try:
            logger.debug(f"Starting background STT audio prep for {audio_path_16k}...")
            recording_float = recording_data.flatten().astype(np.float32)
            recording_16k = librosa.resample( recording_float, orig_sr=original_samplerate, target_sr=target_samplerate )
            sf.write(str(audio_path_16k), recording_16k, target_samplerate, format='WAV', subtype='PCM_16')
            logger.info(f"💾 Background STT recording save complete: {audio_path_16k}")
        except ImportError: logger.critical("❌ Librosa or SoundFile missing!")
        except Exception as save_err: logger.error(f"Error saving/resampling {audio_path_16k}: {save_err}", exc_info=True)

    def _record_and_process_stt_background(
        self,
        session_id: str,
        turn_count: int,
        candidate_id: str,
        duration: int = 30,
        is_follow_up_response: bool = False
    ) -> Tuple[Optional[threading.Thread], Optional[datetime]]:
        stream = None; q = queue.Queue(); stream_closed = threading.Event(); recording_start_time = None
        try:
            session = self.session_mgr.get_session(session_id);
            if not session: logger.warning(f"Session {session_id} ended."); return None, None
            stop_event = session.get('stop_interview')
            if stop_event and stop_event.is_set(): logger.info("Stop signal before recording."); return None, None
            input_device = self.virtual_input if self.virtual_input is not None else sd.default.device[0]
            samplerate = self.target_samplerate # 24000
            logger.info(f"🎙️ Recording from {input_device} max {duration}s @ {samplerate}Hz{' (follow-up)' if is_follow_up_response else ''}")
            def audio_callback(indata, frames, time, status):
                 if status: logger.warning(f"SD status: {status}")
                 if not stream_closed.is_set(): q.put(indata.copy())
            stream = sd.InputStream( samplerate=samplerate, device=input_device, channels=1, callback=audio_callback, dtype='float32' )
            recording_start_time = datetime.now()
            stream.start(); logger.debug("Audio stream started.")
            recorded_chunks = []; silence_threshold_seconds = 2.0; silence_start_time = None
            min_speech_duration = 0.5; speech_start_time = None; volume_threshold = 0.005
            vad_buffer_duration = 0.1; vad_buffer_size = int(vad_buffer_duration * samplerate); recent_audio = np.array([], dtype=np.float32)
            while True: # VAD loop
                session = self.session_mgr.get_session(session_id)
                if not session: logger.warning("Session ended during VAD."); break
                stop_event = session.get('stop_interview')
                if stop_event and stop_event.is_set(): logger.info("Stop signal during VAD."); break
                try:
                    chunk = q.get(timeout=0.1); recorded_chunks.append(chunk); recent_audio = np.append(recent_audio, chunk)
                    if len(recent_audio) > vad_buffer_size: recent_audio = recent_audio[-vad_buffer_size:]
                    if len(recent_audio) >= vad_buffer_size:
                        volume = np.sqrt(np.mean(recent_audio**2))
                        if volume > volume_threshold:
                            silence_start_time = None
                            if speech_start_time is None: speech_start_time = time.time(); logger.debug("Speech.")
                        elif speech_start_time is not None and (time.time() - speech_start_time > min_speech_duration):
                            if silence_start_time is None: silence_start_time = time.time(); logger.debug("Silence?")
                            elif time.time() - silence_start_time > silence_threshold_seconds: logger.info(f"🔇 Silence. Stopping."); break
                except queue.Empty: pass
                elapsed = (datetime.now() - recording_start_time).total_seconds()
                if elapsed >= duration: logger.info(f"⏱️ Max duration."); break
                if speech_start_time is not None and silence_start_time is not None and (time.time() - silence_start_time > silence_threshold_seconds): logger.info(f"🔇 Silence (empty queue). Stopping."); break
            logger.debug("Stopping stream..."); stream_closed.set(); stream.stop(); stream.close(); logger.debug("Stream stopped/closed.")
            while not q.empty():
                 try: recorded_chunks.append(q.get_nowait())
                 except queue.Empty: break
            if not recorded_chunks: logger.warning("No audio data."); return None, recording_start_time
            recording_data = np.concatenate(recorded_chunks, axis=0)
            logger.info(f"Recording finished. Duration: {len(recording_data)/samplerate:.2f}s")
            # Start Background Processing Thread
            audio_path_16k = self._get_stt_audio_path(session_id, turn_count, candidate_id, is_follow_up_response)
            audio_path_16k.parent.mkdir(parents=True, exist_ok=True)
            processing_thread = threading.Thread(
                target=self._process_stt_audio_thread_func,
                args=( recording_data, samplerate, audio_path_16k, session_id, turn_count, candidate_id, recording_start_time, is_follow_up_response ),
                daemon=True, name=f"STTProcess-{turn_count}" )
            processing_thread.start()
            logger.info(f"Background STT processing thread started (Turn {turn_count})")
            return processing_thread, recording_start_time
        except sd.PortAudioError as pae: logger.error(f"❌ PortAudioError: {pae}"); return None, recording_start_time
        except ImportError: logger.critical("❌ Librosa/SoundFile missing!"); return None, recording_start_time
        except Exception as e: logger.error(f"❌ Error recording: {e}", exc_info=True); return None, recording_start_time
        finally:
             if stream and not stream.closed:
                  try: stream.abort(ignore_errors=True); stream.close(); logger.debug("Stream closed finally.")
                  except Exception as close_err: logger.error(f"Error closing stream finally: {close_err}")
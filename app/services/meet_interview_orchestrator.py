# app/services/meet_interview_orchestrator.py
import logging
import time
from time import sleep as sleep
import sounddevice as sd
import soundfile as sf
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from pathlib import Path
import queue
import threading
# import librosa # --- REMOVED ---

from app.services.meet_session_manager import MeetSessionManager
from app.services.interview_service import InterviewService
from app.config.audio_config import VirtualAudioConfig
from app.services.meet_controller import MeetController # Import MeetController for type hinting

logger = logging.getLogger(__name__)

# Estimated buffer time for one full Q&A turn
ESTIMATED_TURN_DURATION_SECONDS = 90

class MeetInterviewOrchestrator:
    """Orchestrates the interview flow, checks for extra participants and drops."""

    def __init__(
        self,
        session_manager: MeetSessionManager,
        interview_service: InterviewService
    ):
        self.session_mgr = session_manager
        self.interview_svc = interview_service
        
        # --- START MODIFICATION: Manually find audio devices ---
        # Manually define the device names
        self.BOT_PLAYBACK_DEVICE_NAME = "CABLE Input (VB-Audio Virtual Cable)" 
        self.BOT_RECORDING_DEVICE_NAME = "CABLE Output (VB-Audio Virtual Cable)"
        self.target_samplerate = 24000
        
        try:
            devices = sd.query_devices()
            # Find the device to PLAY audio TO (Bot's Speaker)
            self.virtual_output = next(
                d['index'] for d in devices 
                if self.BOT_PLAYBACK_DEVICE_NAME in d['name'] and d['max_output_channels'] > 0
            )
            # Find the device to RECORD audio FROM (Bot's Microphone)
            self.virtual_input = next(
                d['index'] for d in devices 
                if self.BOT_RECORDING_DEVICE_NAME in d['name'] and d['max_input_channels'] > 0
            )
            logger.info(f"✅ Found VB-Audio Playback (Output): {self.virtual_output} ('{self.BOT_PLAYBACK_DEVICE_NAME}')")
            logger.info(f"✅ Found VB-Audio Recording (Input): {self.virtual_input} ('{self.BOT_RECORDING_DEVICE_NAME}')")
        
        except StopIteration:
            logger.critical("❌ CRITICAL: Could not find VB-Audio devices by name.")
            logger.critical("Bot will play to speakers and will not hear the candidate.")
            self.virtual_output = None # This will cause playback on default speakers
            self.virtual_input = None  # This will record from default mic
        except Exception as e:
            logger.error(f"Error finding audio devices: {e}")
            self.virtual_output = None
            self.virtual_input = None
        # --- END MODIFICATION ---

    def conduct_interview(
        self,
        session_id: str,
        interview_duration_minutes: int = 10,
        max_questions: int = 10, # Keep max_questions as a secondary limit
    ) -> Dict[str, Any]:
        """
        Conducts interview, prioritizing duration, checking for >2 participants,
        and handling candidate drops (<2) with a 2-minute rejoin window.
        """
        logger.info(f"🎤 Starting interview orchestration for session: {session_id} (Duration: {interview_duration_minutes} mins, Max Qs: {max_questions})")
        interview_duration_seconds = interview_duration_minutes * 60

        session = self.session_mgr.get_session(session_id)
        if not session: return {"error": "Active session not found", "status": "failed"}

        stop_event = session.get('stop_interview')
        meet: MeetController = session.get('controller') # Get controller
        candidate_id = session.get('candidate_id')
        if not meet or not candidate_id:
             logger.error("Controller or candidate_id missing."); return {"error": "Internal session error", "status": "failed"}
        
        session['termination_reason'] = None # Initialize termination reason

        logger.info("Proceeding with interview start.")
        time.sleep(1)

        start_time = time.time()
        transcript = [] # Local summary
        turn_count = 1 # For unique filenames
        questions_asked_count = 0 # Track actual questions asked
        active_stt_thread: Optional[threading.Thread] = None

        # --- Ask Initial Greeting & Handle Intro ---
        initial_greeting = self.interview_svc.generate_initial_greeting()
        logger.info(f"🤖 Bot: {initial_greeting}")
        meet.enable_microphone(); time.sleep(0.5)
        greeting_audio_data, _ = self.interview_svc.text_to_speech( initial_greeting, session_id, turn_count, candidate_id )
        turn_count += 1
        
        playback_ok = True
        if greeting_audio_data is not None:
            # --- MODIFICATION: Need to pass librosa resample here if TTS is not 24k ---
            # Assuming TTS service provides self.target_samplerate (24k)
            # If not, we need to resample the *playback* audio
            playback_ok = self._play_audio_data(greeting_audio_data, self.target_samplerate, meet, stop_event)
        
        # --- Handle drop during greeting ---
        if not playback_ok:
            logger.warning("Playback stopped early during greeting (candidate left or terminated).")
            # Enter rejoin loop
            logger.warning(f"⚠️ Candidate drop during greeting. Waiting 2 minutes to rejoin...")
            rejoin_wait_start = time.time(); rejoin_timeout = 120; rejoined = False
            while time.time() - rejoin_wait_start < rejoin_timeout:
                if stop_event and stop_event.is_set(): logger.info("Terminated manually during rejoin wait."); break
                rejoin_count = meet.get_participant_count()
                if rejoin_count >= 2: logger.info(f"✅ Candidate rejoined! (Count: {rejoin_count}). Resuming."); rejoined = True; break
                logger.debug(f"Waiting rejoin... ({int(time.time() - rejoin_wait_start)}s / {rejoin_timeout}s)"); time.sleep(5)
            
            if not rejoined: # Failed to rejoin
                if stop_event and stop_event.is_set(): logger.info("Manually terminated during rejoin wait.")
                else: logger.error(f"❌ Candidate did not rejoin. Terminating.");
                if stop_event: stop_event.set(); session['termination_reason'] = "candidate_left";
            else:
                playback_ok = True # Reset flag to allow main loop to start
        
        time.sleep(0.5)

        intro_text = "[Recording skipped due to early exit]"
        if playback_ok and not (stop_event and stop_event.is_set()): # Only listen if playback was ok
            logger.info("🎤 Listening for candidate's introduction...")
            active_stt_thread, rec_start_time = self._record_and_process_stt_background( session_id, turn_count, candidate_id, duration=60, is_follow_up_response=False )
            intro_turn = turn_count
            turn_count += 1
            logger.info("Disabling mic after intro..."); meet.disable_microphone(); logger.info("Mic disabled.")
        
        if stop_event and stop_event.is_set():
             logger.info("Stop signal detected after intro recording. Ending early.")
             if active_stt_thread and active_stt_thread.is_alive(): active_stt_thread.join(timeout=10)
             try: self.interview_svc.generate_final_transcript_file(session_id)
             except Exception as e: logger.error(f"Failed to generate final transcript after early exit: {e}")
             return {"status": "terminated", "session_id": session_id, "final_transcript_summary": transcript}

        if active_stt_thread:
            logger.info("Waiting for intro STT file preparation...")
            active_stt_thread.join()
            logger.info("Intro STT thread finished.")
            with self.interview_svc._transcript_lock:
                 intro_text = self.interview_svc._latest_transcript_for_gemini.get(session_id, "[Transcript not updated after intro]")
        
        if intro_text and not intro_text.startswith("[Error") and intro_text != "[Unintelligible]" and intro_text != "[No response]" and intro_text != "[Recording cancelled]" and intro_text != "[Transcript not updated after intro]":
            logger.info(f"👤 Candidate Intro: {intro_text}")
            transcript.append({"role": "user", "content": intro_text, "is_follow_up": False})
        else:
             logger.warning(f"Could not cleanly capture intro: {intro_text}")
             transcript.append({"role": "user", "content": intro_text or "[Introduction Unclear]", "is_follow_up": False})


        # --- Main Interview Loop (Prioritizes Time) ---
        while playback_ok: # Loop continues as long as playback is successful
            elapsed_time = time.time() - start_time
            remaining_time = interview_duration_seconds - elapsed_time

            # Primary Termination Checks
            if stop_event and stop_event.is_set(): logger.info("Termination signal received."); break
            if questions_asked_count >= max_questions: logger.info(f"Max questions ({max_questions}) reached."); break
            if remaining_time < ESTIMATED_TURN_DURATION_SECONDS and questions_asked_count > 0:
                logger.info(f"Time limit approaching ({remaining_time:.0f}s left)."); break

            # Participant Count Check
            try:
                current_participant_count = meet.get_participant_count()
                
                if current_participant_count > 2:
                    logger.error(f"❌ Participant count increased to {current_participant_count}. Terminating.")
                    if stop_event: stop_event.set()
                    session['termination_reason'] = "multiple_participants"
                    termination_msg = ("<speak>It appears there are extra participants in the call. <break time='1000ms'/> "
                                       "For interview integrity, <break time='700ms'/> we must end the session now. <break time='1000ms'/> "
                                       "Thank you.</speak>")
                    logger.info(f"🤖 Bot: {termination_msg}")
                    meet.enable_microphone(); time.sleep(0.5)
                    term_audio_data, _ = self.interview_svc.text_to_speech( termination_msg, session_id, turn_count, candidate_id )
                    if term_audio_data is not None: self._play_audio_data(term_audio_data, self.target_samplerate, meet, stop_event)
                    time.sleep(0.5); meet.disable_microphone()
                    break

                elif current_participant_count < 2:
                    logger.warning(f"⚠️ Participant count dropped to {current_participant_count}. Waiting 2 minutes to rejoin...")
                    rejoin_wait_start = time.time(); rejoin_timeout = 120; rejoined = False
                    while time.time() - rejoin_wait_start < rejoin_timeout:
                        if stop_event and stop_event.is_set(): logger.info("Terminated manually during rejoin wait."); break
                        rejoin_count = meet.get_participant_count()
                        if rejoin_count >= 2: logger.info(f"✅ Candidate rejoined! (Count: {rejoin_count}). Resuming."); rejoined = True; break
                        logger.debug(f"Waiting rejoin... ({int(time.time() - rejoin_wait_start)}s / {rejoin_timeout}s)"); time.sleep(5)
                    if not rejoined:
                        if stop_event and stop_event.is_set(): logger.info("Manually terminated during rejoin wait."); break
                        else: logger.error(f"❌ Candidate did not rejoin. Terminating.");
                        if stop_event: stop_event.set(); session['termination_reason'] = "candidate_left"; break
            except Exception as check_e: logger.error(f"Error during participant check: {check_e}")

            logger.info(f"Time elapsed: {elapsed_time:.0f}s / {interview_duration_seconds}s (Rem: {remaining_time:.0f}s)")

            # Wait for previous STT thread
            if active_stt_thread and active_stt_thread.is_alive():
                wait_start = time.time(); logger.info(f"Waiting STT (Turn {intro_turn if questions_asked_count == 0 else current_turn})...")
                active_stt_thread.join(timeout=30); wait_duration = time.time() - wait_start
                if active_stt_thread.is_alive(): logger.error("STT thread timeout!")
                else: logger.info(f"STT thread finished (waited {wait_duration:.1f}s).")
            if stop_event and stop_event.is_set(): logger.info("Stop signal after STT wait."); break

            logger.info(f"\n--- Generating Question {questions_asked_count + 1}/{max_questions} ---")
            current_question = self.interview_svc.generate_next_question(session_id)
            if "Error:" in current_question: logger.error(f"Gemini error: {current_question}. Ending."); break
            questions_asked_count += 1

            logger.info(f"🤖 Bot: {current_question}"); transcript.append({"role": "assistant", "content": current_question})

            # Ask question
            meet.enable_microphone(); time.sleep(0.5)
            question_audio_data, _ = self.interview_svc.text_to_speech( current_question, session_id, turn_count, candidate_id )
            turn_count += 1
            
            playback_ok = True # Reset for this turn
            if question_audio_data is not None:
                playback_ok = self._play_audio_data(question_audio_data, self.target_samplerate, meet, stop_event)
            
            # --- Handle drop during question playback ---
            if not playback_ok:
                 logger.warning("Playback stopped early (candidate left or terminated).")
                 
                 # Enter rejoin loop
                 logger.warning(f"⚠️ Candidate drop during question. Waiting 2 minutes to rejoin...")
                 rejoin_wait_start = time.time(); rejoin_timeout = 120; rejoined = False
                 while time.time() - rejoin_wait_start < rejoin_timeout:
                     if stop_event and stop_event.is_set(): logger.info("Terminated manually during rejoin wait."); break
                     rejoin_count = meet.get_participant_count()
                     if rejoin_count >= 2: logger.info(f"✅ Candidate rejoined! (Count: {rejoin_count}). Re-asking question."); rejoined = True; break
                     logger.debug(f"Waiting rejoin... ({int(time.time() - rejoin_wait_start)}s / {rejoin_timeout}s)"); time.sleep(5)

                 if not rejoined: # Failed to rejoin
                    if stop_event and stop_event.is_set(): logger.info("Manually terminated during rejoin wait.")
                    else: logger.error(f"❌ Candidate did not rejoin. Terminating.");
                    if stop_event: stop_event.set(); session['termination_reason'] = "candidate_left";
                    break # Exit main interview loop
                 else:
                    # Candidate rejoined, restart the loop to re-ask the same question
                    questions_asked_count -= 1 # Decrement since question wasn't answered
                    transcript.pop() # Remove the question we just added to the local transcript
                    turn_count -= 1 # Decrement turn count for TTS filename
                    playback_ok = True # Reset flag
                    continue # Restart main while loop
            
            time.sleep(0.5)

            # Record response & start background processing
            logger.info("🎤 Listening..."); active_stt_thread, rec_start_time = self._record_and_process_stt_background( session_id, turn_count, candidate_id, duration=60, is_follow_up_response=False )
            current_turn = turn_count; turn_count += 1
            logger.info("Disabling mic..."); meet.disable_microphone(); logger.info("Mic disabled.")
            if stop_event and stop_event.is_set(): logger.info("Stop signal after starting STT."); break
            transcript.append({"role": "user", "content": "[Processing Response...]", "turn": current_turn})
            time.sleep(0.5)


        # --- End of Loop ---
        logger.info("Interview loop finished.")

        # Wait for the last STT thread
        if active_stt_thread and active_stt_thread.is_alive():
            logger.info(f"Waiting final STT (Turn {current_turn if 'current_turn' in locals() else 'N/A'})..."); active_stt_thread.join(timeout=30)
            if active_stt_thread.is_alive(): logger.error("Final STT thread timed out.")
            else: logger.info("Final STT finished.")

        # Closing statement
        closing_reason = "Allocated time is up."; termination_reason = session.get('termination_reason', None)
        final_elapsed = time.time() - start_time # Define final_elapsed here
        try: last_participant_count = meet.get_participant_count()
        except: last_participant_count = 2 # Assume normal if check fails
        
        if stop_event and stop_event.is_set():
             if termination_reason == "multiple_participants": closing_reason = "Multiple participants were detected."
             elif termination_reason == "candidate_left": closing_reason = "The candidate left the meeting."
             else: closing_reason = "The interview was ended early."
        elif questions_asked_count >= max_questions: closing_reason = "We have reached the question limit."
        elif 'remaining_time' in locals() and not (remaining_time < ESTIMATED_TURN_DURATION_SECONDS or final_elapsed >= interview_duration_seconds):
             # This condition might be met if loop broke for other reasons but time wasn't up
             closing_reason = "We have completed the interview."

        if termination_reason == "multiple_participants": logger.info("Skipping generic closing (multi-participant).")
        elif termination_reason == "candidate_left": logger.info("Skipping closing statement (candidate left).")
        else:
             closing_text = (f"<speak>Thank you for your time. <break time='1000ms'/> {closing_reason} "
                             f"<break time='1000ms'/> That concludes our interview today.</speak>")
             logger.info(f"🤖 Bot: {closing_text}")
             meet.enable_microphone(); time.sleep(0.5)
             closing_audio_data, _ = self.interview_svc.text_to_speech( closing_text, session_id, turn_count, candidate_id )
             if closing_audio_data is not None: self._play_audio_data(closing_audio_data, self.target_samplerate, meet, stop_event) # Pass args
             time.sleep(0.5); meet.disable_microphone()


        try: self.interview_svc.generate_final_transcript_file(session_id)
        except Exception as e: logger.error(f"Failed generate final transcript: {e}")

        logger.info("✅ Interview orchestration function finished.")

        final_status = "unknown"
        if stop_event and stop_event.is_set(): final_status = "terminated"
        elif 'remaining_time' in locals() and remaining_time < ESTIMATED_TURN_DURATION_SECONDS: final_status = "time_limit_reached"
        elif final_elapsed >= interview_duration_seconds : final_status = "time_limit_reached"
        elif questions_asked_count >= max_questions: final_status = "max_questions_reached"
        else: final_status = "completed"

        return { "status": final_status, "session_id": session_id, "questions_asked": questions_asked_count, "final_transcript_summary": transcript }

    def _play_audio_data(self, audio_data: np.ndarray, sample_rate: int, meet: MeetController, stop_event: threading.Event) -> bool:
        """Play audio data using non-blocking stream, monitoring for drops/stop."""
        
        # --- MODIFICATION: Need to ensure playback is at target_samplerate ---
        # We need librosa for this one part.
        try:
            import librosa
        except ImportError:
            logger.error("Librosa not installed. Cannot resample playback audio!")
            # Fallback: Try to play as-is, which might fail or be wrong speed
            pass

        try:
            data_float = audio_data.astype(np.float32)
            if audio_data.dtype == np.int16:
                data_float /= 32768.0
            
            if sample_rate != self.target_samplerate:
                logger.warning(f"⚠️ Playback SR mismatch! Expected {self.target_samplerate}, got {sample_rate}. Resampling...")
                try:
                    data_float = librosa.resample(data_float, orig_sr=sample_rate, target_sr=self.target_samplerate)
                except Exception as resample_e:
                    logger.error(f"On-the-fly resampling failed: {resample_e}. Playing as-is.")
                    # Cannot proceed if samplerates don't match and resample fails
                    return False # Or play as-is and hope
            
            logger.info(f"📥 Playing audio data: {self.target_samplerate}Hz, {len(data_float)} samples")
            
            if len(data_float.shape) > 1: data_float = np.mean(data_float, axis=1)
            max_val = np.abs(data_float).max();
            if max_val > 0: data_float = data_float / max_val * 0.9
            
            silence_duration = 0.7; silence_samples = int(silence_duration * self.target_samplerate); silence = np.zeros(silence_samples, dtype=np.float32)
            data_to_play = np.concatenate([silence, data_float, silence])
            
            output_device = self.virtual_output if self.virtual_output is not None else sd.default.device[1]
            duration = len(data_to_play) / self.target_samplerate
            logger.info(f"🔊 Playing to device {output_device} ({self.target_samplerate}Hz, {duration:.2f}s)")

            stream_finished_event = threading.Event()
            current_sample = 0
            
            def playback_callback(outdata: np.ndarray, frames: int, time_info, status):
                nonlocal current_sample
                if status: logger.warning(f"Playback status: {status}")
                chunk_len = len(data_to_play) - current_sample
                if frames >= chunk_len:
                    outdata[:chunk_len] = data_to_play[current_sample:].reshape(-1, 1)
                    outdata[chunk_len:] = 0
                    current_sample += chunk_len
                    raise sd.CallbackStop
                else:
                    chunk_end = current_sample + frames
                    outdata[:] = data_to_play[current_sample:chunk_end].reshape(-1, 1)
                    current_sample += frames

            stream = sd.OutputStream( samplerate=self.target_samplerate, device=output_device, channels=1, dtype='float32', callback=playback_callback, finished_callback=stream_finished_event.set )

            with stream:
                check_interval = 1.0
                while not stream_finished_event.is_set():
                    if stream_finished_event.wait(timeout=check_interval):
                         break # Stream finished
                    if stop_event.is_set():
                        logger.warning("Playback interrupted by external stop signal.")
                        stream.stop()
                        return False
                    try:
                        count = meet.get_participant_count()
                        if count < 2:
                            logger.error("❌ Candidate left during audio playback! Stopping playback.")
                            stream.stop()
                            return False
                    except Exception as e: logger.error(f"Error checking participant count during playback: {e}")
            
            logger.info("✅ Playback complete (monitored)")
            time.sleep(0.5)
            return True
        except Exception as e:
            logger.error(f"❌ Failed to play audio data: {e}", exc_info=True)
            return False

    # --- START MODIFICATION: Renamed and simplified ---
    def _get_user_audio_path_for_stt(self, session_id: str, turn_count: int, candidate_id: str, is_follow_up: bool) -> Path:
        """Gets the path for the 24k (stt) user audio."""
        # Standardized path: data/<candidate_id>/<session_id>/audio/
        audio_dir = Path("data") / candidate_id / session_id / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        suffix = "_followup" if is_follow_up else ""
        
        # Save as 24k, since that's what we recorded
        path_24k_stt = audio_dir / f"candidate_turn_{turn_count}{suffix}_stt_24k.wav"
        
        return path_24k_stt
    # --- END MODIFICATION ---

    # --- START MODIFICATION: Simplified save function (no librosa) ---
    def _save_and_process_audio_thread_func(
        self,
        recording_data: np.ndarray,
        original_samplerate: int, # This will be 24000
        audio_path_for_stt: Path,  # This is the 24k path
        session_id: str,
        turn_count: int,
        candidate_id: str,
        start_time: Optional[datetime],
        is_follow_up_response: bool
    ):
        """Saves 24k audio AND transcribes/logs it in background."""
        try:
            # --- Save the original 24k audio ---
            logger.debug(f"(BG Thread {turn_count}) Saving 24k audio for STT to {audio_path_for_stt}...")
            sf.write(str(audio_path_for_stt), recording_data.flatten(), original_samplerate, format='WAV', subtype='PCM_16')
            logger.info(f"(BG Thread {turn_count}) 💾 24k (STT) recording saved: {audio_path_for_stt}")
            
            # --- Process transcript using the 24k file ---
            self.interview_svc.process_and_log_transcript(
                 session_id, str(audio_path_for_stt), # <-- Use 24k path
                 turn_count, candidate_id,
                 start_time, datetime.now(),
                 is_follow_up_response
            )
        except ImportError: 
            logger.critical("❌ SoundFile missing!") # No longer need librosa
        except Exception as err: 
            logger.error(f"(BG Thread {turn_count}) Error processing STT audio: {err}", exc_info=True)
    # --- END MODIFICATION ---


    def _record_and_process_stt_background(
        self,
        session_id: str,
        turn_count: int,
        candidate_id: str,
        duration: int = 120,
        is_follow_up_response: bool = False
    ) -> Tuple[Optional[threading.Thread], Optional[datetime]]:
        """
        Records audio, starts background save/transcribe, returns thread & start time.
        """
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
            
            # --- START MODIFICATION: Get 24k STT path ---
            audio_path_stt = self._get_user_audio_path_for_stt(
                session_id, turn_count, candidate_id, is_follow_up_response
            )
            # --- END MODIFICATION ---

            processing_thread = threading.Thread(
                target=self._save_and_process_audio_thread_func, # <-- Renamed
                 # --- MODIFICATION: Pass 24k STT path ---
                 args=( 
                     recording_data, samplerate, 
                     audio_path_stt, #<-- Updated arg
                     session_id, turn_count, candidate_id, 
                     recording_start_time, is_follow_up_response 
                 ),
                 daemon=True, name=f"STTProcess-{turn_count}" )

            processing_thread.start()
            logger.info(f"Background STT processing thread started (Turn {turn_count})")
            return processing_thread, recording_start_time
        except sd.PortAudioError as pae: logger.error(f"❌ PortAudioError: {pae}"); return None, recording_start_time
        except ImportError: 
            logger.critical("❌ SoundFile missing!") # No longer need librosa
            return None, recording_start_time
        except Exception as e: 
            logger.error(f"❌ Error recording: {e}", exc_info=True); return None, recording_start_time
        finally:
             if stream and not stream.closed:
                  try: stream.abort(ignore_errors=True); stream.close(); logger.debug("Stream closed finally.")
                  except Exception as close_err: logger.error(f"Error closing stream finally: {close_err}")
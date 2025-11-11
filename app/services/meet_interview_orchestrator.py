# app/services/meet_interview_orchestrator.py
import logging
import time
import asyncio 
import sounddevice as sd
import soundfile as sf
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from pathlib import Path
import queue
import threading
import audioop 
from google.cloud import speech 

from app.services.meet_session_manager import MeetSessionManager
from app.services.interview_service import InterviewService
from app.config.audio_config import VirtualAudioConfig
from app.services.meet_controller import MeetController # Import MeetController for type hinting

logger = logging.getLogger(__name__)

ESTIMATED_TURN_DURATION_SECONDS = 90

class MeetInterviewOrchestrator:
    """
    Orchestrates the interview flow asynchronously,
    checking for extra participants and drops.
    """

    def __init__(
        self,
        session_manager: MeetSessionManager,
        interview_service: InterviewService
    ):
        self.session_mgr = session_manager
        self.interview_svc = interview_service
        self.speech_client = speech.SpeechClient() 
        
        # --- START OF NEW CODE ---
        # Keywords to detect non-answers
        self.NON_ANSWER_KEYWORDS = [
            "pardon", "repeat", "sorry", "what", 
            "didn't hear", "could you say", "didn't catch"
        ]
        # --- END OF NEW CODE ---
        
        # --- Manually find audio devices ---
        self.BOT_PLAYBACK_DEVICE_NAME = "CABLE Input (VB-Audio Virtual Cable)" 
        self.BOT_RECORDING_DEVICE_NAME = "CABLE Output (VB-Audio Virtual Cable)"
        self.target_samplerate = 24000
        
        try:
            devices = sd.query_devices()
            self.virtual_output = next(
                d['index'] for d in devices 
                if self.BOT_PLAYBACK_DEVICE_NAME in d['name'] and d['max_output_channels'] > 0
            )
            self.virtual_input = next(
                d['index'] for d in devices 
                if self.BOT_RECORDING_DEVICE_NAME in d['name'] and d['max_input_channels'] > 0
            )
            logger.info(f"✅ Found VB-Audio Playback (Output): {self.virtual_output} ('{self.BOT_PLAYBACK_DEVICE_NAME}')")
            logger.info(f"✅ Found VB-Audio Recording (Input): {self.virtual_input} ('{self.BOT_RECORDING_DEVICE_NAME}')")
        
        except StopIteration:
            logger.critical("❌ CRITICAL: Could not find VB-Audio devices by name.")
            self.virtual_output = None
            self.virtual_input = None
        except Exception as e:
            logger.error(f"Error finding audio devices: {e}")
            self.virtual_output = None
            self.virtual_input = None

    async def conduct_interview(
        self,
        session_id: str,
        interview_duration_minutes: int = 10,
        # --- MODIFICATION: max_questions parameter removed ---
    ) -> Dict[str, Any]:
        """
        Conducts interview asynchronously, prioritizing duration,
        checking for >2 participants, and handling candidate drops.
        """
        # --- MODIFICATION: Removed max_questions from log ---
        logger.info(f"🎤 Starting ASYNC interview orchestration for session: {session_id} (Duration: {interview_duration_minutes} mins)")
        interview_duration_seconds = interview_duration_minutes * 60

        session = self.session_mgr.get_session(session_id)
        if not session: return {"error": "Active session not found", "status": "failed"}

        stop_event = session.get('stop_interview')
        meet: MeetController = session.get('controller') 
        candidate_id = session.get('candidate_id')
        if not meet or not candidate_id:
             logger.error("Controller or candidate_id missing."); return {"error": "Internal session error", "status": "failed"}
        
        session['termination_reason'] = None
        
        start_time = time.time() 
        transcript = [] 
        questions_asked_count = 0 
        
        try: 
            logger.info("Proceeding with interview start.")
            await asyncio.sleep(1) 

            turn_count = 1 

            # --- Ask Initial Greeting & Handle Intro ---
            # --- MODIFICATION: Pass session_id to get candidate name ---
            initial_greeting = self.interview_svc.generate_initial_greeting(session_id)
            logger.info(f"🤖 Bot: {initial_greeting}")
            
            await asyncio.to_thread(meet.enable_microphone); await asyncio.sleep(0.5)
            
            greeting_audio_stream = self.interview_svc.stream_plain_text(
                initial_greeting, session_id, turn_count, candidate_id
            )
            turn_count += 1
            
            playback_ok = True
            if greeting_audio_stream:
                playback_ok = await asyncio.to_thread(
                    self._play_audio_stream, greeting_audio_stream, meet, stop_event
                )
            
            # --- Handle drop during greeting ---
            if not playback_ok:
                logger.warning("Playback stopped early during greeting (candidate left or terminated).")
                logger.warning(f"⚠️ Candidate drop during greeting. Waiting 2 minutes to rejoin...")
                rejoin_wait_start = time.time(); rejoin_timeout = 120; rejoined = False
                while time.time() - rejoin_wait_start < rejoin_timeout:
                    if stop_event and stop_event.is_set(): logger.info("Terminated manually during rejoin wait."); break
                    rejoin_count = await asyncio.to_thread(meet.get_participant_count)
                    if rejoin_count >= 2: logger.info(f"✅ Candidate rejoined! (Count: {rejoin_count}). Resuming."); rejoined = True; break
                    logger.debug(f"Waiting rejoin... ({int(time.time() - rejoin_wait_start)}s / {rejoin_timeout}s)"); await asyncio.sleep(5)
                
                if not rejoined: 
                    if stop_event and stop_event.is_set(): logger.info("Manually terminated during rejoin wait.")
                    else: logger.error(f"❌ Candidate did not rejoin. Terminating.");
                    # --- FIX: Changed break to playback_ok = False ---
                    if stop_event: stop_event.set(); session['termination_reason'] = "candidate_left"
                    playback_ok = False # This skips the intro and while loop
                    # --- END FIX ---
                else:
                    playback_ok = True 
            
            await asyncio.sleep(0.5)

            # --- MODIFICATION: Handle intro response logic here ---
            intro_text = "[Recording skipped due to early exit]"
            if playback_ok and not (stop_event and stop_event.is_set()): 
                logger.info("🎤 Listening for candidate's introduction...")
                
                # 1. Record audio and get transcript
                rec_start_time, intro_text = await asyncio.to_thread(
                    self._record_and_process_stt_streaming_manual_vad, 
                    session_id, turn_count, candidate_id, 
                    is_follow_up=False
                )
                rec_end_time = datetime.now()
                audio_path_stt = self._get_user_audio_path_for_stt(
                    session_id, turn_count, candidate_id, 
                    is_follow_up=False
                )
                
                # 2. Log this first response (we accept any intro)
                self.interview_svc.process_and_log_transcript(
                    session_id, str(audio_path_stt), intro_text,
                    turn_count, candidate_id,
                    rec_start_time, rec_end_time,
                    is_follow_up_response=False
                )

                turn_count += 1
                logger.info("Disabling mic after intro..."); 
                await asyncio.to_thread(meet.disable_microphone); logger.info("Mic disabled.")
            
            if stop_event and stop_event.is_set():
                 logger.info("Stop signal detected after intro recording. Ending early.")
                 playback_ok = False # Skip main loop

            # 3. Add to local transcript
            if intro_text and not intro_text.startswith("[Error") and intro_text != "[Unintelligible]" and intro_text != "[No response]" and intro_text != "[Recording cancelled]":
                logger.info(f"👤 Candidate Intro: {intro_text}")
                transcript.append({"role": "user", "content": intro_text, "is_follow_up": False})
            else:
                 logger.warning(f"Could not cleanly capture intro: {intro_text}")
                 transcript.append({"role": "user", "content": intro_text or "[Introduction Unclear]", "is_follow_up": False})
            # --- END OF INTRO MODIFICATION ---


            # --- Main Interview Loop (Prioritizes Time) ---
            while playback_ok: 
                elapsed_time = time.time() - start_time
                remaining_time = interview_duration_seconds - elapsed_time

                # Primary Termination Checks
                if stop_event and stop_event.is_set(): logger.info("Termination signal received."); break
                
                # --- MODIFICATION: Removed max_questions check ---
                
                if remaining_time < ESTIMATED_TURN_DURATION_SECONDS and questions_asked_count > 0:
                    logger.info(f"Time limit approaching ({remaining_time:.0f}s left)."); 
                    session['termination_reason'] = "time_limit_reached"; # Set reason before break
                    break

                # Participant Count Check
                try:
                    current_participant_count = await asyncio.to_thread(meet.get_participant_count)
                    
                    if current_participant_count > 2:
                        logger.error(f"❌ Participant count increased to {current_participant_count}. Terminating.")
                        # --- START OF FIX ---
                        # DO NOT set stop_event yet, play audio first
                        # if stop_event: stop_event.set() # <--- MOVED
                        session['termination_reason'] = "multiple_participants"
                        
                        termination_msg = ("It appears there are extra participants in the call. "
                                           "For interview integrity, we must end the session now. "
                                           "Thank you.")
                        logger.info(f"🤖 Bot: {termination_msg}")
                        await asyncio.to_thread(meet.enable_microphone); await asyncio.sleep(0.5)
                        term_audio_stream = self.interview_svc.stream_plain_text(
                             termination_msg, session_id, turn_count, candidate_id 
                        )
                        if term_audio_stream: 
                            # This will now play because stop_event is not set
                            await asyncio.to_thread(
                                self._play_audio_stream, term_audio_stream, meet, stop_event
                            )
                        await asyncio.sleep(0.5); await asyncio.to_thread(meet.disable_microphone)
                        
                        # Now set the stop_event and break
                        if stop_event: stop_event.set()
                        break
                        # --- END OF FIX ---

                    elif current_participant_count < 2:
                        logger.warning(f"⚠️ Participant count dropped to {current_participant_count}. Waiting 2 minutes to rejoin...")
                        rejoin_wait_start = time.time(); rejoin_timeout = 120; rejoined = False
                        while time.time() - rejoin_wait_start < rejoin_timeout:
                            if stop_event and stop_event.is_set(): logger.info("Terminated manually during rejoin wait."); break
                            rejoin_count = await asyncio.to_thread(meet.get_participant_count)
                            if rejoin_count >= 2: logger.info(f"✅ Candidate rejoined! (Count: {rejoin_count}). Resuming."); rejoined = True; break
                            logger.debug(f"Waiting rejoin... ({int(time.time() - rejoin_wait_start)}s / {rejoin_timeout}s)"); await asyncio.sleep(5)
                        if not rejoined:
                            if stop_event and stop_event.is_set(): logger.info("Manually terminated during rejoin wait."); break
                            else: logger.error(f"❌ Candidate did not rejoin. Terminating.");
                            if stop_event: stop_event.set(); session['termination_reason'] = "candidate_left"; break
                except Exception as check_e: logger.error(f"Error during participant check: {check_e}")

                logger.info(f"Time elapsed: {elapsed_time:.0f}s / {interview_duration_seconds}s (Rem: {remaining_time:.0f}s)")
                
                if stop_event and stop_event.is_set(): logger.info("Stop signal after STT wait."); break

                # --- MODIFICATION: Removed max_questions from log ---
                logger.info(f"\n--- Generating Question {questions_asked_count + 1} ---")
                
                # --- MODIFICATION: Store turn_count *before* incrementing ---
                current_question_turn = turn_count
                turn_count += 1
                questions_asked_count += 1

                await asyncio.to_thread(meet.enable_microphone); await asyncio.sleep(0.5)
                
                audio_stream_generator = self.interview_svc.stream_interview_turn(
                    session_id, current_question_turn, candidate_id # Use current_question_turn
                )
                
                playback_ok = True 
                if audio_stream_generator:
                    playback_ok = await asyncio.to_thread(
                        self._play_audio_stream, audio_stream_generator, meet, stop_event
                    )
                else:
                    logger.error("Failed to create audio stream generator.")
                    playback_ok = False

                # --- Handle drop during question playback ---
                if not playback_ok:
                     logger.warning("Playback stopped early (candidate left or terminated).")
                     logger.warning(f"⚠️ Candidate drop during question. Waiting 2 minutes to rejoin...")
                     rejoin_wait_start = time.time(); rejoin_timeout = 120; rejoined = False
                     while time.time() - rejoin_wait_start < rejoin_timeout:
                         if stop_event and stop_event.is_set(): logger.info("Terminated manually during rejoin wait."); break
                         rejoin_count = await asyncio.to_thread(meet.get_participant_count)
                         if rejoin_count >= 2: logger.info(f"✅ Candidate rejoined! (Count: {rejoin_count}). Re-asking question."); rejoined = True; break
                         logger.debug(f"Waiting rejoin... ({int(time.time() - rejoin_wait_start)}s / {rejoin_timeout}s)"); await asyncio.sleep(5)

                     if not rejoined: 
                        if stop_event and stop_event.is_set(): logger.info("Manually terminated during rejoin wait.")
                        else: logger.error(f"❌ Candidate did not rejoin. Terminating.");
                        if stop_event: stop_event.set(); session['termination_reason'] = "candidate_left";
                        break 
                     else:
                        # --- MODIFICATION: Decrement counters to repeat turn ---
                        questions_asked_count -= 1 
                        turn_count -= 1 
                        # --- END MODIFICATION ---
                        playback_ok = True 
                        continue 
                
                await asyncio.sleep(0.5)

                # Record response & start background processing
                logger.info("🎤 Listening..."); 
                
                # --- MODIFICATION: Get transcript back from function ---
                current_response_turn = turn_count # This is the *response* turn number
                turn_count += 1
                
                rec_start_time, transcript_text = await asyncio.to_thread(
                    self._record_and_process_stt_streaming_manual_vad, 
                    session_id, current_response_turn, candidate_id,
                    is_follow_up=False
                )
                rec_end_time = datetime.now()
                audio_path_stt = self._get_user_audio_path_for_stt(
                    session_id, current_response_turn, candidate_id,
                    is_follow_up=False
                )
                # --- END MODIFICATION ---

                logger.info("Disabling mic..."); 
                await asyncio.to_thread(meet.disable_microphone); logger.info("Mic disabled.")
                if stop_event and stop_event.is_set(): logger.info("Stop signal after starting STT."); break
                
                # --- START OF NEW LOGIC (Step 3) ---
                transcript_text_lower = transcript_text.lower()
                is_non_answer = any(keyword in transcript_text_lower for keyword in self.NON_ANSWER_KEYWORDS)
                
                # Also check for very short, likely non-answers (e.g., "yes", "no", "ok")
                if not is_non_answer and len(transcript_text) < 5:
                     logger.info(f"Short response detected ('{transcript_text}'), treating as non-answer.")
                     is_non_answer = True
                
                if is_non_answer:
                    logger.warning(f"Candidate non-answer detected: '{transcript_text}'. Repeating last question.")
                    # Log to DB but do NOT update Gemini state
                    self.interview_svc.db.add_message_to_session(
                        session_id, "user", transcript_text, str(audio_path_stt), 
                        rec_start_time, rec_end_time
                    )
                    
                    await asyncio.to_thread(meet.enable_microphone); await asyncio.sleep(0.5)
                    repeat_audio_stream = self.interview_svc.stream_plain_text(
                         "Sorry, I'll repeat that.", session_id, f"repeat_{current_response_turn}", candidate_id 
                    )
                    if repeat_audio_stream:
                        await asyncio.to_thread(self._play_audio_stream, repeat_audio_stream, meet, stop_event)
                    await asyncio.to_thread(meet.disable_microphone)
                    
                    # Decrement counters to repeat the turn
                    questions_asked_count -= 1
                    turn_count -= 2 # Decrement by 2 to offset both increments
                    
                    # This continues to the top of the while loop, re-running the *same* question
                    continue 
                else:
                    # This is a valid answer. Log it and update Gemini state.
                    self.interview_svc.process_and_log_transcript(
                        session_id, str(audio_path_stt), transcript_text,
                        current_response_turn, candidate_id,
                        rec_start_time, rec_end_time,
                        is_follow_up_response=False
                    )
                    transcript.append({"role": "user", "content": transcript_text, "turn": current_response_turn})
                # --- END OF NEW LOGIC ---

                await asyncio.sleep(0.5)

            # --- End of Loop ---
            logger.info("Interview loop finished.")
            
        except Exception as e:
            logger.error(f"❌ FATAL ERROR in conduct_interview: {e}", exc_info=True)
            if stop_event: stop_event.set() # Ensure cleanup
        
        finally:
            logger.info("Final STT work already completed in loop.")

            # Closing statement
            closing_reason = "Allocated time is up."; termination_reason = session.get('termination_reason', None)
            final_elapsed = time.time() - start_time 
            
            try: 
                last_participant_count = await asyncio.to_thread(meet.get_participant_count)
            except: last_participant_count = 2 
            
            if stop_event and stop_event.is_set():
                 if termination_reason == "multiple_participants": closing_reason = "Multiple participants were detected."
                 elif termination_reason == "candidate_left": closing_reason = "The candidate left the meeting."
                 else: closing_reason = "The interview was ended early."
            
            # --- MODIFICATION: Removed max_questions check ---
            
            elif 'remaining_time' in locals() and not (remaining_time < ESTIMATED_TURN_DURATION_SECONDS or final_elapsed >= interview_duration_seconds):
                 closing_reason = "We have completed the interview."
            elif final_elapsed >= interview_duration_seconds:
                 closing_reason = "Allocated time is up." # Corrected from "closing_warning"
            elif questions_asked_count == 0 and not termination_reason:
                closing_reason = "Interview did not start."

            if termination_reason in ["multiple_participants", "candidate_left"]:
                 logger.info(f"Skipping generic closing statement (Reason: {termination_reason}).")
            else:
                 closing_text = (f"Thank you for your time. {closing_reason} "
                                 f"That concludes our interview today.")
                 logger.info(f"🤖 Bot: {closing_text}")
                 await asyncio.to_thread(meet.enable_microphone); await asyncio.sleep(0.5)
                 
                 closing_audio_stream = self.interview_svc.stream_plain_text(
                     closing_text, session_id, turn_count, candidate_id
                 )
                 if closing_audio_stream:
                     await asyncio.to_thread(
                         self._play_audio_stream, closing_audio_stream, meet, stop_event
                     )
                 
                 await asyncio.sleep(0.5); await asyncio.to_thread(meet.disable_microphone)

            try: self.interview_svc.generate_final_transcript_file(session_id)
            except Exception as e: logger.error(f"Failed generate final transcript: {e}")

            logger.info("✅ Interview orchestration function finished.")

        final_status = "unknown"
        if stop_event and stop_event.is_set(): 
            final_status = session.get('termination_reason') or 'terminated'
        elif 'remaining_time' in locals() and remaining_time < ESTIMATED_TURN_DURATION_SECONDS and questions_asked_count > 0: 
            final_status = "time_limit_reached"
        elif final_elapsed >= interview_duration_seconds : 
            final_status = "time_limit_reached"
        
        # --- MODIFICATION: Removed max_questions check ---
        
        else: 
            final_status = "completed"

        return { "status": final_status, "session_id": session_id, "questions_asked": questions_asked_count, "final_transcript_summary": transcript }

    def _play_audio_stream(self, audio_chunk_iterator: iter, meet: MeetController, stop_event: threading.Event) -> bool:
        # ... (This function remains identical) ...
        audio_queue = queue.Queue(maxsize=100) 
        stream_finished_event = threading.Event() 
        generator_finished_event = threading.Event() 
        
        def feeder_thread_func():
            try:
                for audio_chunk_bytes in audio_chunk_iterator:
                    if stop_event.is_set() or stream_finished_event.is_set():
                        break
                    
                    linear_audio = audioop.ulaw2lin(audio_chunk_bytes, 2)
                    audio_data = np.frombuffer(linear_audio, dtype=np.int16).astype(np.float32) / 32768.0
                    
                    if len(audio_data) > 0:
                        audio_queue.put(audio_data)
                        
            except Exception as e:
                logger.error(f"Audio feeder thread error: {e}", exc_info=True)
            finally:
                audio_queue.put(None) 
                generator_finished_event.set()

        internal_buffer = np.array([], dtype=np.float32)
        
        def playback_callback(outdata: np.ndarray, frames: int, time_info, status):
            nonlocal internal_buffer
            if status: logger.warning(f"Playback status: {status}")
            
            buffer_len = len(internal_buffer)
            
            if buffer_len >= frames:
                outdata[:] = internal_buffer[:frames].reshape(-1, 1)
                internal_buffer = internal_buffer[frames:]
                return
            
            while buffer_len < frames:
                try:
                    chunk = audio_queue.get_nowait()
                    
                    if chunk is None:
                        outdata[:buffer_len] = internal_buffer.reshape(-1, 1)
                        outdata[buffer_len:] = 0 
                        internal_buffer = np.array([], dtype=np.float32)
                        raise sd.CallbackStop 
                    
                    internal_buffer = np.concatenate([internal_buffer, chunk])
                    buffer_len = len(internal_buffer)
                
                except queue.Empty:
                    outdata[:buffer_len] = internal_buffer.reshape(-1, 1)
                    outdata[buffer_len:] = 0
                    internal_buffer = np.array([], dtype=np.float32)
                    return 

            outdata[:] = internal_buffer[:frames].reshape(-1, 1)
            internal_buffer = internal_buffer[frames:]

        try:
            feeder_thread = threading.Thread(target=feeder_thread_func, daemon=True, name="AudioFeeder")
            feeder_thread.start()
            
            output_device = self.virtual_output if self.virtual_output is not None else sd.default.device[1]
            logger.info(f"🔊 Playing audio stream to device {output_device}...")

            stream = sd.OutputStream(
                samplerate=self.target_samplerate,
                device=output_device,
                channels=1,
                dtype='float32',
                callback=playback_callback,
                finished_callback=stream_finished_event.set
            )

            with stream:
                check_interval = 0.5
                while not stream_finished_event.is_set():
                    if stream_finished_event.wait(timeout=check_interval):
                        break 
                    
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
                    except Exception as e:
                        logger.error(f"Error checking participant count during playback: {e}")
            
            logger.info("✅ Playback stream finished.")
            feeder_thread.join(timeout=2) 
            return True

        except Exception as e:
            logger.error(f"❌ Failed to play audio stream: {e}", exc_info=True)
            return False

    def _get_user_audio_path_for_stt(self, session_id: str, turn_count: int, candidate_id: str, is_follow_up: bool) -> Path:
        """Gets the path for the 24k (stt) user audio."""
        audio_dir = Path("data") / candidate_id / session_id / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        suffix = "_followup" if is_follow_up else ""
        
        path_24k_stt = audio_dir / f"candidate_turn_{turn_count}{suffix}_stt_24k.wav"
        
        return path_24k_stt
    
    # --- MODIFICATION: This function now returns the transcript ---
    def _record_and_process_stt_streaming_manual_vad(
        self,
        session_id: str,
        turn_count: int,
        candidate_id: str,
        is_follow_up: bool = False
    ) -> Tuple[Optional[datetime], str]:
        """
        Records audio using a manual 2-second VAD, while simultaneously
        streaming to Google STT.
        Returns the (start_time, final_transcript_string).
        """
        
        audio_queue = queue.Queue() 
        recorded_chunks = [] 
        recording_start_time = None
        stream = None
        stt_thread = None
        final_transcript = "[Unintelligible]" 
        
        try:
            session = self.session_mgr.get_session(session_id);
            if not session: 
                logger.warning(f"Session {session_id} ended."); 
                return None, "[Error]"
            stop_event = session.get('stop_interview')
            if stop_event and stop_event.is_set(): 
                logger.info("Stop signal before recording."); 
                return None, "[Error]"
            
            input_device = self.virtual_input if self.virtual_input is not None else sd.default.device[0]
            samplerate = self.target_samplerate 
            
            def audio_callback(indata, frames, time, status):
                if status: logger.warning(f"SD status: {status}")
                audio_bytes = (indata * 32767).astype(np.int16).tobytes()
                audio_queue.put(audio_bytes)
                recorded_chunks.append(indata.copy())

            stream = sd.InputStream(
                samplerate=samplerate,
                device=input_device,
                channels=1,
                callback=audio_callback,
                dtype='float32' 
            )

            def request_generator():
                try:
                    while True:
                        chunk = audio_queue.get()
                        if chunk is None:
                            break
                        yield speech.StreamingRecognizeRequest(audio_content=chunk)
                except Exception as e:
                    logger.error(f"STT request_generator error: {e}")
            
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=samplerate,
                language_code="en-IN", 
                enable_automatic_punctuation=True
            )
            streaming_config = speech.StreamingRecognitionConfig(config=config)

            def stt_processor_thread():
                nonlocal final_transcript
                try:
                    api_requests = request_generator()
                    responses = self.speech_client.streaming_recognize(
                        config=streaming_config,
                        requests=api_requests
                    )
                    
                    for response in responses:
                        if response.results and response.results[0].is_final:
                            final_transcript = response.results[0].alternatives[0].transcript
                            logger.info(f"STT (final interim): '{final_transcript}'")
                
                except Exception as e:
                    logger.error(f"Google STT streaming error: {e}", exc_info=True)
                    final_transcript = "[Speech service error]"

            logger.info(f"🎙️ Streaming recording from {input_device} @ {samplerate}Hz (Manual VAD 2.0s)...")
            stream.start()
            recording_start_time = datetime.now()
            
            stt_thread = threading.Thread(target=stt_processor_thread, daemon=True)
            stt_thread.start()

            silence_threshold_seconds = 2.0
            silence_start_time = None
            min_speech_duration = 0.5; speech_start_time = None; volume_threshold = 0.005
            
            while True:
                session = self.session_mgr.get_session(session_id)
                if not session: logger.warning("Session ended during VAD."); break
                stop_event = session.get('stop_interview')
                if stop_event and stop_event.is_set(): logger.info("Stop signal during VAD."); break
                
                if len(recorded_chunks) > 10: 
                    recent_audio = np.concatenate(recorded_chunks[-10:], axis=0)
                    volume = np.sqrt(np.mean(recent_audio**2))
                    
                    if volume > volume_threshold:
                        silence_start_time = None
                        if speech_start_time is None: 
                            speech_start_time = time.time(); logger.debug("Speech detected.")
                    elif speech_start_time is not None and (time.time() - speech_start_time > min_speech_duration):
                        if silence_start_time is None: 
                            silence_start_time = time.time(); logger.debug("Silence detected?")
                        elif time.time() - silence_start_time > silence_threshold_seconds: 
                            logger.info(f"🔇 Manual VAD detected 2.0s silence. Stopping."); 
                            break
                
                elapsed = (datetime.now() - recording_start_time).total_seconds()
                if elapsed >= 120: 
                    logger.info(f"⏱️ Max duration (120s) reached."); 
                    break
                
                time.sleep(0.1) 

            logger.debug("Stopping stream..."); 
            stream.stop(); stream.close(); 
            logger.debug("Stream stopped/closed.")
            
            audio_queue.put(None) 
            logger.debug("Waiting for STT thread to join...")
            stt_thread.join(timeout=5) 
            if stt_thread.is_alive():
                logger.error("STT thread timed out.")

            logger.info(f"Final STT result: '{final_transcript}'")

            if recorded_chunks:
                recording_data = np.concatenate(recorded_chunks, axis=0)
                # recording_end_time = datetime.now() # Moved to main loop
                logger.info(f"Recording finished. Duration: {len(recording_data)/samplerate:.2f}s")
                
                audio_path_stt = self._get_user_audio_path_for_stt(
                    session_id, turn_count, candidate_id, 
                    is_follow_up
                )
                
                try:
                    sf.write(str(audio_path_stt), recording_data.flatten(), samplerate, format='WAV', subtype='PCM_16')
                    logger.info(f"💾 (Stream) 24k recording saved: {audio_path_stt}")
                    
                    # --- MODIFICATION: REMOVED process_and_log_transcript call ---
                    # This is now handled in the main conduct_interview loop
                    
                except Exception as save_e:
                    logger.error(f"Failed to save streamed audio: {save_e}", exc_info=True)
            
            else:
                logger.warning("No audio data recorded.")

            return recording_start_time, final_transcript

        except sd.PortAudioError as pae: 
            logger.error(f"❌ PortAudioError: {pae}"); 
            return None, "[Error]"
        except Exception as e: 
            logger.error(f"❌ Error in streaming record: {e}", exc_info=True); 
            return None, "[Error]"
        finally:
             if stream and not stream.closed:
                  try: stream.abort(ignore_errors=True); stream.close(); logger.debug("Stream closed finally.")
                  except Exception as close_err: logger.error(f"Error closing stream finally: {close_err}")
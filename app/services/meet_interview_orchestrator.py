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
import os
import re
import string

# --- V1 API IMPORTS ---
try:
    from google.cloud import speech_v1
    V1_AVAILABLE = True
except ImportError as v1_err:
    V1_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(f"Speech V1 API not available: {v1_err}")

# --- V2 API IMPORTS ---
try:
    from google.cloud.speech_v2 import SpeechClient
    from google.cloud.speech_v2.types import cloud_speech
    from google.api_core.client_options import ClientOptions
    from google.api_core import exceptions as google_exceptions
    V2_AVAILABLE = True
except ImportError as v2_err:
    V2_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(f"Speech V2 API not available: {v2_err}")

from app.services.meet_session_manager import MeetSessionManager
from app.services.interview_service import InterviewService
from app.config.audio_config import VirtualAudioConfig
from app.services.meet_controller import MeetController

logger = logging.getLogger(__name__)

ESTIMATED_TURN_DURATION_SECONDS = 90

class MeetInterviewOrchestrator:
    """
    Orchestrates the interview flow asynchronously.
    HYBRID VERSION: Supports both V1 and V2 Speech APIs
    """

    def __init__(
        self,
        session_manager: MeetSessionManager,
        interview_service: InterviewService
    ):
        self.session_mgr = session_manager
        self.interview_svc = interview_service
        
        # --- CONFIGURATION: CHOOSE API VERSION ---
        self.stt_api_version = os.getenv("SPEECH_API_VERSION", "v2").lower()
        
        if self.stt_api_version not in ["v1", "v2"]:
            logger.error(f"Invalid SPEECH_API_VERSION: {self.stt_api_version}. Using v2.")
            self.stt_api_version = "v2"
        
        logger.info(f"🎙️ Speech API Version: {self.stt_api_version.upper()}")
        
        # Initialize based on chosen version
        if self.stt_api_version == "v2":
            self._init_v2_client()
        else:
            self._init_v1_client()
        
        # Keywords to detect non-answers
        self.NON_ANSWER_KEYWORDS = [
            "pardon",
            "sorry what",
            "sorry can you",
            "didn't hear",
            "didn't catch",
            "could you say",
            "could you repeat",
            "can you repeat",
            "say that again",
            "come again",
            "one more time",
            "repeat that"
        ]
        
        # UPDATED: More specific exit keywords (complete phrases only)
        self.EXIT_KEYWORDS = [
            # Meeting end phrases
            "end the meeting",
            "end this meeting",
            "end the interview",
            "end this interview",
            
            # Leave phrases with intent
            "i want to leave",
            "i need to leave", 
            "i have to leave",
            "i'm going to leave",
            "i am going to leave",
            "can i leave",
            "may i leave",
            "let me leave",
            
            # Stop phrases
            "stop the interview",
            "stop this interview",
            "i want to stop",
            "i need to stop",
            
            # Call-specific
            "leave the call",
            "exit the call",
            "quit the call",
            
            # General end with context
            "can we end",
            "i want to end",
            "let's end"
        ]
        
        # --- Audio device configuration ---
        self.BOT_PLAYBACK_DEVICE_NAME = "CABLE Input (VB-Audio Virtual Cable)" 
        self.BOT_RECORDING_DEVICE_NAME = "CABLE Output (VB-Audio Virtual Cable)"
        self.target_samplerate = 24000
        
        try:
            devices = sd.query_devices()
            self.virtual_output = next(
                (d['index'] for d in devices 
                if self.BOT_PLAYBACK_DEVICE_NAME in d['name'] and d['max_output_channels'] > 0),
                None
            )
            self.virtual_input = next(
                (d['index'] for d in devices 
                if self.BOT_RECORDING_DEVICE_NAME in d['name'] and d['max_input_channels'] > 0),
                None
            )
            if self.virtual_output is not None:
                logger.info(f"✅ Found VB-Audio Playback: {self.virtual_output}")
            else:
                 logger.warning("❌ Could not find VB-Audio Playback device")
            
            if self.virtual_input is not None:
                logger.info(f"✅ Found VB-Audio Recording: {self.virtual_input}")
            else:
                 logger.warning("❌ Could not find VB-Audio Recording device")

        except Exception as e:
            logger.error(f"Error finding audio devices: {e}")
            self.virtual_output = None
            self.virtual_input = None

    def _init_v1_client(self):
        """Initialize Speech V1 client"""
        if not V1_AVAILABLE:
            logger.error("❌ Speech V1 library not installed!")
            self.speech_client = None
            self.recognizer = None 
            return
        
        try:
            self.speech_client = speech_v1.SpeechClient()
            self.recognizer = None 
            logger.info("✅ Speech V1 Client initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Speech V1: {e}", exc_info=True)
            self.speech_client = None
            self.recognizer = None

    def _init_v2_client(self):
        """Initialize Speech V2 client"""
        if not V2_AVAILABLE:
            logger.error("❌ Speech V2 library not installed!")
            self.speech_client = None
            self.recognizer = None
            return
        
        try:
            self.project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
            if not self.project_id:
                raise ValueError("GOOGLE_CLOUD_PROJECT environment variable must be set for V2 API")
            
            self.location = os.getenv("GOOGLE_CLOUD_REGION", "us")
            
            valid_locations = ["us", "eu", "global"]
            if self.location not in valid_locations:
                logger.warning(f"Invalid location '{self.location}'. Using 'us' instead.")
                self.location = "us"
            
            api_endpoint = f"{self.location}-speech.googleapis.com"
            client_options = ClientOptions(api_endpoint=api_endpoint)
            
            self.speech_client = SpeechClient(client_options=client_options)
            self.recognizer = f"projects/{self.project_id}/locations/{self.location}/recognizers/_"
            
            logger.info("✅ Speech V2 Client initialized with Chirp 3")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Speech V2: {e}", exc_info=True)
            self.speech_client = None
            self.recognizer = None
            
    def _check_non_answer(self, transcript: str) -> bool:
        """
        Check if transcript is a non-answer (asking for repetition).
        Uses improved logic to avoid false positives.
        """
        # Remove punctuation for cleaner matching
        transcript_clean = transcript.lower().translate(str.maketrans('', '', string.punctuation)).strip()
        
        word_count = len(transcript_clean.split())
        
        # Very short responses
        if word_count <= 3:
            for keyword in self.NON_ANSWER_KEYWORDS:
                # Strip punctuation from keyword for comparison too, just in case
                key_clean = keyword.translate(str.maketrans('', '', string.punctuation))
                if key_clean in transcript_clean:
                    logger.debug(f"Short non-answer detected: '{transcript_clean}' (matched: '{key_clean}')")
                    return True
        
        # For longer responses (up to 10 words)
        if word_count <= 10:
            for keyword in self.NON_ANSWER_KEYWORDS:
                key_clean = keyword.translate(str.maketrans('', '', string.punctuation))
                
                # Special handling for "pardon":
                # "pardon" is a common word in regular sentences (e.g. "The governor granted a pardon").
                # If the keyword is just "pardon" and the sentence is > 3 words, ignore it unless it's "beg your pardon".
                if key_clean == "pardon" and word_count > 3:
                    continue

                if key_clean in transcript_clean:
                    # Check content words
                    content_indicators = [
                        "project", "experience", "worked", "used", "developed",
                        "implemented", "designed", "built", "created", "system",
                        "application", "feature", "code", "data", "team", "years"
                    ]
                    
                    has_content = any(indicator in transcript_clean for indicator in content_indicators)
                    
                    if not has_content:
                        logger.debug(f"Non-answer detected: '{transcript_clean}' (matched: '{key_clean}')")
                        return True
        
        return False

    def _check_exit_intent(self, transcript: str) -> tuple[bool, str]:
        transcript_lower = transcript.lower()
        
        # Check specific complete phrases first
        for keyword in self.EXIT_KEYWORDS:
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, transcript_lower):
                logger.info(f"🚪 Exit intent detected with phrase: '{keyword}'")
                return True, keyword
        
        # Check intent combinations
        exit_verbs = ["leave", "quit", "exit", "stop", "end"]
        intent_words = ["want to", "going to", "need to", "have to", "can i", "may i", "let me"]
        
        for verb in exit_verbs:
            for intent in intent_words:
                # Match intent...verb with up to 3 intervening words
                pattern = r'\b' + re.escape(intent) + r'\s+(\w+\s+){0,3}' + re.escape(verb) + r'\b'
                if re.search(pattern, transcript_lower):
                    combined = f"{intent} ... {verb}"
                    logger.info(f"🚪 Exit intent detected with pattern: '{combined}'")
                    return True, combined
        
        return False, ""
       
    async def conduct_interview(
            self,
            session_id: str,
            interview_duration_minutes: int = 10,
        ) -> Dict[str, Any]:
            """
            Conducts interview asynchronously.
            """
            logger.info(f"🎤 Starting ASYNC interview orchestration for session: {session_id} (Duration: {interview_duration_minutes} mins)")
            interview_duration_seconds = interview_duration_minutes * 60

            session = self.session_mgr.get_session(session_id)
            if not session: 
                return {"error": "Active session not found", "status": "failed"}

            stop_event = session.get('stop_interview')
            
            # IMMEDIATE STOP CHECK
            if stop_event and stop_event.is_set():
                logger.info("Stop signal detected at start of interview.")
                return {"status": "terminated", "session_id": session_id, "questions_asked": 0, "final_transcript_summary": []}

            meet: MeetController = session.get('controller') 
            candidate_id = session.get('candidate_id')
            if not meet or not candidate_id:
                logger.error("Controller or candidate_id missing.")
                return {"error": "Internal session error", "status": "failed"}
            
            session['termination_reason'] = None
            
            start_time = time.time() 
            transcript = [] 
            questions_asked_count = 0 
            
            try: 
                logger.info("Proceeding with interview start.")
                await asyncio.sleep(1) 

                turn_count = 1 

                # --- Ask Initial Greeting & Handle Intro ---
                initial_greeting = self.interview_svc.generate_initial_greeting(session_id)
                logger.info(f"🤖 Bot: {initial_greeting}")
                
                await asyncio.to_thread(meet.enable_microphone)
                await asyncio.sleep(0.5)
                
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
                    rejoin_wait_start = time.time()
                    rejoin_timeout = 120
                    rejoined = False
                    
                    while time.time() - rejoin_wait_start < rejoin_timeout:
                        if stop_event and stop_event.is_set():
                            logger.info("Terminated manually during rejoin wait.")
                            break
                            
                        rejoin_count = await asyncio.to_thread(meet.get_participant_count)
                        if rejoin_count >= 2:
                            logger.info(f"✅ Candidate rejoined! (Count: {rejoin_count}). Resuming.")
                            rejoined = True
                            break
                            
                        logger.debug(f"Waiting rejoin... ({int(time.time() - rejoin_wait_start)}s / {rejoin_timeout}s)")
                        await asyncio.sleep(5)
                    
                    if not rejoined: 
                        if stop_event and stop_event.is_set():
                            logger.info("Manually terminated during rejoin wait.")
                        else:
                            logger.error(f"❌ Candidate did not rejoin. Terminating.")
                            
                        if stop_event:
                            stop_event.set()
                            session['termination_reason'] = "candidate_left"
                        playback_ok = False
                    else:
                        playback_ok = True 
                
                await asyncio.sleep(0.5)

                # --- Handle intro response logic here ---
                intro_text = "[Recording skipped due to early exit]"
                if playback_ok and not (stop_event and stop_event.is_set()): 
                    logger.info("🎤 Listening for candidate's introduction...")
                    
                    rec_start_time, intro_text = await self._record_and_process_stt_streaming( 
                        session_id, turn_count, candidate_id, 
                        is_follow_up=False
                    )
                    rec_end_time = datetime.now()
                    audio_path_stt = self._get_user_audio_path_for_stt(
                        session_id, turn_count, candidate_id, 
                        is_follow_up=False
                    )
                    
                    self.interview_svc.process_and_log_transcript(
                        session_id, str(audio_path_stt), intro_text,
                        turn_count, candidate_id,
                        rec_start_time, rec_end_time,
                        is_follow_up_response=False
                    )

                    turn_count += 1
                    logger.info("Disabling mic after intro...")
                    await asyncio.to_thread(meet.disable_microphone)
                    logger.info("Mic disabled.")
                
                if stop_event and stop_event.is_set():
                    logger.info("Stop signal detected after intro recording. Ending early.")
                    playback_ok = False

                if intro_text and not intro_text.startswith("[Error") and intro_text != "[Unintelligible]" and intro_text != "[No response]" and intro_text != "[Recording cancelled]":
                    logger.info(f"👤 Candidate Intro: {intro_text}")
                    transcript.append({"role": "user", "content": intro_text, "is_follow_up": False})
                else:
                    logger.warning(f"Could not cleanly capture intro: {intro_text}")
                    transcript.append({"role": "user", "content": intro_text or "[Introduction Unclear]", "is_follow_up": False})


                # --- Main Interview Loop (Prioritizes Time) ---
                while playback_ok: 
                    elapsed_time = time.time() - start_time
                    remaining_time = interview_duration_seconds - elapsed_time

                    # Primary Termination Checks
                    if stop_event and stop_event.is_set():
                        logger.info("Termination signal received.")
                        break
                    
                    if remaining_time < ESTIMATED_TURN_DURATION_SECONDS and questions_asked_count > 0:
                        logger.info(f"Time limit approaching ({remaining_time:.0f}s left).")
                        session['termination_reason'] = "time_limit_reached"
                        break

                    # Participant Count Check
                    try:
                        current_participant_count = await asyncio.to_thread(meet.get_participant_count)
                        
                        if current_participant_count > 2:
                            logger.error(f"❌ Participant count increased to {current_participant_count}. Terminating.")
                            session['termination_reason'] = "multiple_participants"
                            
                            termination_msg = ("It appears there are extra participants in the call. "
                                            "For interview integrity, we must end the session now. "
                                            "Thank you.")
                            logger.info(f"🤖 Bot: {termination_msg}")
                            await asyncio.to_thread(meet.enable_microphone)
                            await asyncio.sleep(0.5)
                            
                            term_audio_stream = self.interview_svc.stream_plain_text(
                                termination_msg, session_id, turn_count, candidate_id 
                            )
                            if term_audio_stream: 
                                await asyncio.to_thread(
                                    self._play_audio_stream, term_audio_stream, meet, stop_event
                                )
                            await asyncio.sleep(0.5)
                            await asyncio.to_thread(meet.disable_microphone)
                            
                            if stop_event:
                                stop_event.set()
                            break

                        elif current_participant_count < 2:
                            logger.warning(f"⚠️ Participant count dropped to {current_participant_count}. Waiting 2 minutes to rejoin...")
                            rejoin_wait_start = time.time()
                            rejoin_timeout = 120
                            rejoined = False
                            
                            while time.time() - rejoin_wait_start < rejoin_timeout:
                                if stop_event and stop_event.is_set():
                                    logger.info("Terminated manually during rejoin wait.")
                                    break
                                    
                                rejoin_count = await asyncio.to_thread(meet.get_participant_count)
                                if rejoin_count >= 2:
                                    logger.info(f"✅ Candidate rejoined! (Count: {rejoin_count}). Resuming.")
                                    rejoined = True
                                    break
                                    
                                logger.debug(f"Waiting rejoin... ({int(time.time() - rejoin_wait_start)}s / {rejoin_timeout}s)")
                                await asyncio.sleep(5)
                                
                            if not rejoined:
                                if stop_event and stop_event.is_set():
                                    logger.info("Manually terminated during rejoin wait.")
                                    break
                                else:
                                    logger.error(f"❌ Candidate did not rejoin. Terminating.")
                                    
                                if stop_event:
                                    stop_event.set()
                                    session['termination_reason'] = "candidate_left"
                                break
                                
                    except Exception as check_e:
                        logger.error(f"Error during participant check: {check_e}")

                    logger.info(f"Time elapsed: {elapsed_time:.0f}s / {interview_duration_seconds}s (Rem: {remaining_time:.0f}s)")
                    
                    if stop_event and stop_event.is_set():
                        logger.info("Stop signal after STT wait.")
                        break

                    logger.info(f"\n--- Generating Question {questions_asked_count + 1} ---")
                    
                    current_question_turn = turn_count
                    turn_count += 1
                    questions_asked_count += 1

                    await asyncio.to_thread(meet.enable_microphone)
                    await asyncio.sleep(0.5)
                    
                    audio_stream_generator = self.interview_svc.stream_interview_turn(
                        session_id, current_question_turn, candidate_id
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
                        rejoin_wait_start = time.time()
                        rejoin_timeout = 120
                        rejoined = False
                        
                        while time.time() - rejoin_wait_start < rejoin_timeout:
                            if stop_event and stop_event.is_set():
                                logger.info("Terminated manually during rejoin wait.")
                                break
                                
                            rejoin_count = await asyncio.to_thread(meet.get_participant_count)
                            if rejoin_count >= 2:
                                logger.info(f"✅ Candidate rejoined! (Count: {rejoin_count}). Re-asking question.")
                                rejoined = True
                                break
                                
                            logger.debug(f"Waiting rejoin... ({int(time.time() - rejoin_wait_start)}s / {rejoin_timeout}s)")
                            await asyncio.sleep(5)

                        if not rejoined: 
                            if stop_event and stop_event.is_set():
                                logger.info("Manually terminated during rejoin wait.")
                            else:
                                logger.error(f"❌ Candidate did not rejoin. Terminating.")
                                
                            if stop_event:
                                stop_event.set()
                                session['termination_reason'] = "candidate_left"
                            break 
                        else:
                            questions_asked_count -= 1 
                            turn_count -= 1 
                            playback_ok = True 
                            continue 
                    
                    await asyncio.sleep(0.5)

                    # Record response & start background processing
                    logger.info("🎤 Listening...")
                    
                    current_response_turn = turn_count
                    turn_count += 1
                    
                    rec_start_time, transcript_text = await self._record_and_process_stt_streaming( 
                        session_id, current_response_turn, candidate_id,
                        is_follow_up=False
                    )
                    rec_end_time = datetime.now()
                    audio_path_stt = self._get_user_audio_path_for_stt(
                        session_id, current_response_turn, candidate_id,
                        is_follow_up=False
                    )

                    logger.info("Disabling mic...")
                    await asyncio.to_thread(meet.disable_microphone)
                    logger.info("Mic disabled.")
                    
                    if stop_event and stop_event.is_set():
                        logger.info("Stop signal after starting STT.")
                        break
                    
                    logger.debug(f"Checking exit intent in: '{transcript_text[:100]}...'")
                    is_exit_request, matched_keyword = self._check_exit_intent(transcript_text)
                    
                    if is_exit_request:
                        logger.warning(f"🚪 Candidate exit request detected: '{transcript_text[:100]}...'")
                        # Log the user's attempt
                        self.interview_svc.db.add_message_to_session(
                            session_id, "user", transcript_text, str(audio_path_stt), 
                            rec_start_time, rec_end_time
                        )
                        
                        await asyncio.to_thread(meet.enable_microphone)
                        await asyncio.sleep(0.5)
                        
                        # Polite redirect message
                        redirect_text = "We still have a few more questions to cover. I'll repeat the last question for you."
                        
                        redirect_audio_stream = self.interview_svc.stream_plain_text(
                            redirect_text, session_id, f"exit_ack_{current_response_turn}", candidate_id 
                        )
                        if redirect_audio_stream:
                            await asyncio.to_thread(self._play_audio_stream, redirect_audio_stream, meet, stop_event)
                        
                        await asyncio.to_thread(meet.disable_microphone)
                        
                        # Re-ask the question
                        questions_asked_count -= 1
                        turn_count -= 2
                        
                        continue
                    
                    # ===== NON-ANSWER DETECTION =====
                    is_non_answer = self._check_non_answer(transcript_text)

                    if is_non_answer:
                        logger.warning(f"Candidate non-answer detected: '{transcript_text}'. Repeating last question.")
                        self.interview_svc.db.add_message_to_session(
                            session_id, "user", transcript_text, str(audio_path_stt), 
                            rec_start_time, rec_end_time
                        )
                        
                        await asyncio.to_thread(meet.enable_microphone)
                        await asyncio.sleep(0.5)
                        
                        # Say we'll repeat, then actually repeat
                        repeat_audio_stream = self.interview_svc.stream_plain_text(
                            "Sorry, I'll repeat that.", session_id, f"repeat_{current_response_turn}", candidate_id 
                        )
                        if repeat_audio_stream:
                            await asyncio.to_thread(self._play_audio_stream, repeat_audio_stream, meet, stop_event)
                        
                        await asyncio.to_thread(meet.disable_microphone)
                        
                        # CRITICAL: Decrement counters to re-ask the SAME question
                        questions_asked_count -= 1
                        turn_count -= 2
                        
                        continue 
                    else:
                        # Normal answer processing
                        logger.debug("Processing normal answer (no exit/non-answer detected)")
                        self.interview_svc.process_and_log_transcript(
                            session_id, str(audio_path_stt), transcript_text,
                            current_response_turn, candidate_id,
                            rec_start_time, rec_end_time,
                            is_follow_up_response=False
                        )
                        transcript.append({"role": "user", "content": transcript_text, "turn": current_response_turn})

                    await asyncio.sleep(0.5)

                # --- End of Loop ---
                logger.info("Interview loop finished.")
                
            except Exception as e:
                logger.error(f"❌ FATAL ERROR in conduct_interview: {e}", exc_info=True)
                if stop_event:
                    stop_event.set()
            
            finally:
                logger.info("Final STT work already completed in loop.")

                # Closing statement
                closing_reason = "It was great talking to you." 
                termination_reason = session.get('termination_reason', None)
                final_elapsed = time.time() - start_time 
                
                try: 
                    last_participant_count = await asyncio.to_thread(meet.get_participant_count)
                except:
                    last_participant_count = 2 
                
                if stop_event and stop_event.is_set():
                    if termination_reason == "multiple_participants":
                        closing_reason = "Multiple participants were detected."
                    elif termination_reason == "candidate_left":
                        closing_reason = "The candidate left the meeting."
                    else:
                        closing_reason = "The interview was ended early."
                
                elif 'remaining_time' in locals() and not (remaining_time < ESTIMATED_TURN_DURATION_SECONDS or final_elapsed >= interview_duration_seconds):
                    closing_reason = "We have completed the interview."
                elif final_elapsed >= interview_duration_seconds:
                    closing_reason = "Allocated time is up."
                elif questions_asked_count == 0 and not termination_reason:
                    closing_reason = "Interview did not start."

                if termination_reason in ["multiple_participants", "candidate_left"]:
                    logger.info(f"Skipping generic closing statement (Reason: {termination_reason}).")
                else:
                    closing_text = (f"Thank you for your time. {closing_reason} "
                                    f"That concludes our interview today.")
                    logger.info(f"🤖 Bot: {closing_text}")
                    await asyncio.to_thread(meet.enable_microphone)
                    await asyncio.sleep(0.5)
                    
                    closing_audio_stream = self.interview_svc.stream_plain_text(
                        closing_text, session_id, turn_count, candidate_id
                    )
                    if closing_audio_stream:
                        await asyncio.to_thread(
                            self._play_audio_stream, closing_audio_stream, meet, stop_event
                        )
                    
                    await asyncio.sleep(0.5)
                    await asyncio.to_thread(meet.disable_microphone)

                try:
                    self.interview_svc.generate_final_transcript_file(session_id)
                except Exception as e:
                    logger.error(f"Failed generate final transcript: {e}")

                logger.info("✅ Interview orchestration function finished.")

            final_status = "unknown"
            if stop_event and stop_event.is_set(): 
                final_status = session.get('termination_reason') or 'terminated'
            elif 'remaining_time' in locals() and remaining_time < ESTIMATED_TURN_DURATION_SECONDS and questions_asked_count > 0: 
                final_status = "time_limit_reached"
            elif final_elapsed >= interview_duration_seconds: 
                final_status = "time_limit_reached"
            else: 
                final_status = "completed"

            return {
                "status": final_status,
                "session_id": session_id,
                "questions_asked": questions_asked_count,
                "final_transcript_summary": transcript
            }

    def _play_audio_stream(self, audio_chunk_iterator: iter, meet: MeetController, stop_event: threading.Event) -> bool:
        """Play audio stream"""
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
        audio_dir = Path("data") / candidate_id / session_id / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        suffix = "_followup" if is_follow_up else ""
        path_24k_stt = audio_dir / f"candidate_turn_{turn_count}{suffix}_stt_24k.wav"
        return path_24k_stt

    # --- STT V1 IMPLEMENTATION ---
    def _record_and_process_stt_streaming_v1(self, session_id: str, turn_count: int, candidate_id: str, is_follow_up: bool = False) -> Tuple[Optional[datetime], str]:
        if not self.speech_client or not V1_AVAILABLE:
            return None, "[Speech service unavailable]"
        
        audio_queue = queue.Queue() 
        recorded_chunks = [] 
        recording_start_time = None
        stream = None
        stt_thread = None
        final_transcript = "[No response]"
        
        try:
            session = self.session_mgr.get_session(session_id)
            if not session: return None, "[Error]"
            stop_event = session.get('stop_interview')
            if stop_event and stop_event.is_set(): return None, "[Error]"
            
            input_device = self.virtual_input if self.virtual_input is not None else sd.default.device[0]
            samplerate = self.target_samplerate 
            
            def audio_callback(indata, frames, time, status):
                audio_bytes = (indata * 32767).astype(np.int16).tobytes()
                audio_queue.put(audio_bytes)
                recorded_chunks.append(indata.copy())

            stream = sd.InputStream(samplerate=samplerate, device=input_device, channels=1, callback=audio_callback, dtype='float32')

            def request_generator():
                try:
                    while True:
                        chunk = audio_queue.get()
                        if chunk is None: break
                        yield speech_v1.StreamingRecognizeRequest(audio_content=chunk)
                except Exception as e: logger.error(f"STT V1 gen error: {e}")
            
            def stt_processor_thread():
                nonlocal final_transcript
                try:
                    config_v1 = speech_v1.RecognitionConfig(
                        encoding=speech_v1.RecognitionConfig.AudioEncoding.LINEAR16,
                        sample_rate_hertz=samplerate, language_code="en-IN",
                        enable_automatic_punctuation=True, model='default'
                    )
                    streaming_config_v1 = speech_v1.StreamingRecognitionConfig(config=config_v1, interim_results=True)
                    responses = self.speech_client.streaming_recognize(config=streaming_config_v1, requests=request_generator())
                    for response in responses:
                        if not response.results: continue
                        result = response.results[0]
                        if not result.alternatives: continue
                        transcript_text = result.alternatives[0].transcript
                        if result.is_final: final_transcript = transcript_text
                except Exception as e:
                    logger.error(f"STT V1 error: {e}")
                    final_transcript = "[Speech service error]"

            stream.start()
            recording_start_time = datetime.now()
            stt_thread = threading.Thread(target=stt_processor_thread, daemon=True, name="STT-V1-Thread")
            stt_thread.start()

            # VAD Loop 
            silence_threshold_seconds = 2.0
            silence_start_time = None
            min_speech_duration = 0.5
            speech_start_time = None
            volume_threshold = 0.005
            
            while True:
                session = self.session_mgr.get_session(session_id)
                if not session or (session.get('stop_interview') and session.get('stop_interview').is_set()): break
                
                if len(recorded_chunks) > 10: 
                    recent_audio = np.concatenate(recorded_chunks[-10:], axis=0)
                    volume = np.sqrt(np.mean(recent_audio**2))
                    if volume > volume_threshold:
                        silence_start_time = None
                        if speech_start_time is None: speech_start_time = time.time()
                    elif speech_start_time is not None and (time.time() - speech_start_time > min_speech_duration):
                        if silence_start_time is None: silence_start_time = time.time()
                        elif time.time() - silence_start_time > silence_threshold_seconds: break
                
                if (datetime.now() - recording_start_time).total_seconds() >= 120: break
                time.sleep(0.1) 

            stream.stop()
            stream.close()
            audio_queue.put(None)
            stt_thread.join(timeout=10)

            if recorded_chunks:
                recording_data = np.concatenate(recorded_chunks, axis=0)
                duration = len(recording_data) / samplerate
                self.interview_svc.db.update_stt_usage(session_id, duration)
                path = self._get_user_audio_path_for_stt(session_id, turn_count, candidate_id, is_follow_up)
                try: sf.write(str(path), recording_data.flatten(), samplerate, format='WAV', subtype='PCM_16')
                except: pass

            return recording_start_time, final_transcript
        except Exception as e:
            logger.error(f"V1 Error: {e}")
            return None, "[Error]"
        finally:
            if stream and not stream.closed: stream.close()

    # --- STT V2 IMPLEMENTATION ---
    def _record_and_process_stt_streaming_v2(self, session_id: str, turn_count: int, candidate_id: str, is_follow_up: bool = False) -> Tuple[Optional[datetime], str]:
        if not self.speech_client or not self.recognizer: return None, "[Speech service unavailable]"
        
        audio_queue = queue.Queue() 
        recorded_chunks = [] 
        recording_start_time = None
        stream = None
        stt_thread = None
        final_transcript = "[No response]"
        
        try:
            session = self.session_mgr.get_session(session_id)
            if not session or (session.get('stop_interview') and session.get('stop_interview').is_set()): return None, "[Error]"
            
            input_device = self.virtual_input if self.virtual_input is not None else sd.default.device[0]
            samplerate = self.target_samplerate 
            
            def audio_callback(indata, frames, time, status):
                audio_bytes = (indata * 32767).astype(np.int16).tobytes()
                audio_queue.put(audio_bytes)
                recorded_chunks.append(indata.copy())

            stream = sd.InputStream(samplerate=samplerate, device=input_device, channels=1, callback=audio_callback, dtype='float32')

            def request_generator():
                try:
                    config = cloud_speech.RecognitionConfig(
                        explicit_decoding_config=cloud_speech.ExplicitDecodingConfig(
                            encoding=cloud_speech.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
                            sample_rate_hertz=samplerate, audio_channel_count=1,
                        ),
                        language_codes=["en-IN"], model="chirp_3",
                        features=cloud_speech.RecognitionFeatures(enable_automatic_punctuation=True),
                    )
                    streaming_config = cloud_speech.StreamingRecognitionConfig(config=config, streaming_features=cloud_speech.StreamingRecognitionFeatures(interim_results=True))
                    yield cloud_speech.StreamingRecognizeRequest(recognizer=self.recognizer, streaming_config=streaming_config)
                    
                    MAX_CHUNK_SIZE = 25600
                    while True:
                        chunk = audio_queue.get()
                        if chunk is None: break
                        if len(chunk) > MAX_CHUNK_SIZE:
                            for i in range(0, len(chunk), MAX_CHUNK_SIZE):
                                yield cloud_speech.StreamingRecognizeRequest(audio=chunk[i:i + MAX_CHUNK_SIZE])
                        else:
                            yield cloud_speech.StreamingRecognizeRequest(audio=chunk)
                except Exception: pass
            
            def stt_processor_thread():
                nonlocal final_transcript
                try:
                    responses = self.speech_client.streaming_recognize(requests=request_generator())
                    for response in responses:
                        if not response.results: continue
                        for result in response.results:
                            if result.is_final and result.alternatives:
                                final_transcript = result.alternatives[0].transcript
                except Exception as e:
                    logger.error(f"STT V2 error: {e}")
                    final_transcript = "[Speech service error]"

            stream.start()
            recording_start_time = datetime.now()
            stt_thread = threading.Thread(target=stt_processor_thread, daemon=True, name="STT-V2-Thread")
            stt_thread.start()

            # VAD logic
            silence_threshold_seconds = 2.0
            silence_start_time = None
            min_speech_duration = 0.5
            speech_start_time = None
            volume_threshold = 0.005
            
            while True:
                session = self.session_mgr.get_session(session_id)
                if not session or (session.get('stop_interview') and session.get('stop_interview').is_set()): break
                
                if len(recorded_chunks) > 10: 
                    recent_audio = np.concatenate(recorded_chunks[-10:], axis=0)
                    volume = np.sqrt(np.mean(recent_audio**2))
                    if volume > volume_threshold:
                        silence_start_time = None
                        if speech_start_time is None: speech_start_time = time.time()
                    elif speech_start_time is not None and (time.time() - speech_start_time > min_speech_duration):
                        if silence_start_time is None: silence_start_time = time.time()
                        elif time.time() - silence_start_time > silence_threshold_seconds: break
                
                if (datetime.now() - recording_start_time).total_seconds() >= 120: break
                time.sleep(0.1) 

            stream.stop()
            stream.close()
            audio_queue.put(None)
            stt_thread.join(timeout=10)

            if recorded_chunks:
                recording_data = np.concatenate(recorded_chunks, axis=0)
                duration = len(recording_data) / samplerate
                self.interview_svc.db.update_stt_usage(session_id, duration)
                path = self._get_user_audio_path_for_stt(session_id, turn_count, candidate_id, is_follow_up)
                try: sf.write(str(path), recording_data.flatten(), samplerate, format='WAV', subtype='PCM_16')
                except: pass

            return recording_start_time, final_transcript
        except Exception as e:
            logger.error(f"V2 Error: {e}")
            return None, "[Error]"
        finally:
            if stream and not stream.closed:
                try: stream.abort(ignore_errors=True); stream.close()
                except: pass

    async def _record_and_process_stt_streaming(self, session_id: str, turn_count: int, candidate_id: str, is_follow_up: bool = False) -> Tuple[Optional[datetime], str]:
        if self.stt_api_version == "v2":
            return await asyncio.to_thread(self._record_and_process_stt_streaming_v2, session_id, turn_count, candidate_id, is_follow_up)
        else:
            return await asyncio.to_thread(self._record_and_process_stt_streaming_v1, session_id, turn_count, candidate_id, is_follow_up)
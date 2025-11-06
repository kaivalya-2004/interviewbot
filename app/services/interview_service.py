# app/services/interview_service.py
import os
import google.generativeai as genai
from dotenv import load_dotenv
import speech_recognition as sr
import logging
from datetime import datetime
from typing import Optional, Dict, Any, Tuple, List
from pathlib import Path
from google.cloud import texttospeech
from google.api_core.client_options import ClientOptions
import numpy as np
import soundfile as sf
import threading
from collections import defaultdict
import re
import audioop

from app.core.services.database_service import DBHandler

load_dotenv()
logger = logging.getLogger(__name__)

class InterviewService:
    def __init__(self):
        self.db = DBHandler()
        self.active_chat_sessions: Dict[str, Any] = {}
        self._session_token_counts = defaultdict(lambda: {'prompt': 0, 'response': 0, 'total': 0})
        self._session_tts_char_counts = defaultdict(int)
        self.recognizer = sr.Recognizer()
        self._latest_transcript_for_gemini: Dict[str, Optional[str]] = {}
        self._transcript_lock = threading.Lock()

        # Initialize Gemini model
        try:
            gemini_api_key = os.getenv("GEMINI_API_KEY")
            if not gemini_api_key: raise ValueError("GEMINI_API_KEY not found.")
            genai.configure(api_key=gemini_api_key)
            self.model = genai.GenerativeModel('gemini-2.5-flash') 
            logger.info("✅ Gemini API configured.")
        except Exception as e:
            logger.error(f"❌ Failed to configure Gemini API: {e}")
            self.model = None

        # Initialize TTS client
        logger.info("Initializing Google Cloud TTS client...")
        try:
            custom_endpoint = "texttospeech.googleapis.com:443"
            self.tts_client = texttospeech.TextToSpeechClient(client_options=ClientOptions(api_endpoint=custom_endpoint))
            logger.info(f"✅ Google Cloud TTS client loaded (Endpoint: {custom_endpoint}).")
            
            # Define voice parameters - MUST use Journey/Chirp 3 voice for streaming
            # Available Journey voices: en-US-Journey-D, en-US-Journey-F, en-US-Journey-O
            self.tts_voice = texttospeech.VoiceSelectionParams(
                language_code="en-IN", 
                name="en-IN-Chirp3-HD-Alnilam"  # Male voice, Journey/Chirp 3
            )
            
            logger.info(f"Set TTS voice for Journey (Chirp 3) model - streaming compatible.")

        except Exception as e:
            logger.error(f"❌ Failed load TTS client: {e}")
            self.tts_client = None

    def start_new_interview(self, resume_text: str, candidate_id: str, questionnaire: List[str]) -> str:
        """Initializes Gemini chat and all tracking."""
        session_id = self.db.create_session(resume_text, candidate_id, questionnaire or [])
        
        if not self.model: 
            raise RuntimeError("Gemini chat model failed.")
        
        logger.info(f"Initializing Gemini chat for session {session_id}")
        
        questionnaire_context = ""
        if questionnaire: 
            questionnaire_str = "\n".join([f"- {q}" for q in questionnaire])
            questionnaire_context = f"\n**Reference Questionnaire:**\n---\n{questionnaire_str}\n---"
        else: 
            questionnaire_context = "\n**Reference Questionnaire:** Not provided."

        system_prompt = f"""
You are an AI Interviewer. Your role is to conduct a professional, conversational interview.
Your responses MUST be raw text. Do not use SSML, Markdown, or any other formatting.

**IMPORTANT:** To sound natural, you MUST use proper punctuation (commas, periods, question marks). 
The Text-to-Speech engine will use this punctuation to create natural pauses.
Your questions must be **very short and direct, ideally under 15 words.** They must be concise, clear, and conversational. Ask only one question at a time.

Your primary goal is to assess the candidate's skills, experience, and suitability for a role. You must be professional, polite, and curious. You will be provided with the candidate's resume and an optional list of topics to guide the conversation.

**Candidate Resume:**
---
{resume_text}
---
{questionnaire_context}

**Task:**
1.  You will be given the transcript of the candidate's last answer.
2.  Your job is to generate the *next* logical question as raw text.
3.  This question **must** be relevant to their previous answer AND the context from their **Resume**.
4.  Use the **Reference Questionnaire** as a high-level guide for key topics to cover, but do not just ask the questions verbatim.
5.  Ask open-ended questions (e.g., "Why did you choose that approach?", "What was the main challenge?", "How did that project turn out?").
6.  Actively listen and ask relevant follow-up questions. For example, if they mention a project from their resume, ask about a specific challenge they faced in that project.
7.  Do not repeat questions.
8.  When the interview is concluding, your final response should be a polite closing remark.
"""
        
        try:
            self._session_token_counts[session_id] = {'prompt': 0, 'response': 0, 'total': 0}
            self._session_tts_char_counts[session_id] = 0
            with self._transcript_lock: 
                self._latest_transcript_for_gemini[session_id] = "[Candidate introduction pending]"
            
            self.active_chat_sessions[session_id] = self.model.start_chat(
                history=[
                    {'role': 'user', 'parts': [system_prompt]},
                    {'role': 'model', 'parts': ["Understood. I will ask short, clear, and conversational questions using raw text with proper punctuation. Ready for the candidate's introduction."]}
                ]
            )
            logger.info(f"✅ Gemini chat session initialized for {session_id} (Raw Text Mode).")
            
        except Exception as e:
            if session_id in self._session_token_counts: del self._session_token_counts[session_id]
            if session_id in self._session_tts_char_counts: del self._session_tts_char_counts[session_id]
            with self._transcript_lock:
                 if session_id in self._latest_transcript_for_gemini: 
                     del self._latest_transcript_for_gemini[session_id]
            logger.error(f"❌ Failed init Gemini chat: {e}", exc_info=True)
            raise RuntimeError(f"Failed start Gemini: {e}")
            
        return session_id

    def generate_initial_greeting(self) -> str:
        """Provides the opening greeting with instructions."""
        return ("Hello and welcome. I am your Interviewer for today. "
                "For our session today, please ensure your camera is turned on and remains on throughout the interview. "
                "Also, please make sure your microphone is enabled when you are speaking. "
                "Let's get started. "
                "Please introduce yourself.")

    def _save_audio_thread(self, audio_data: np.ndarray, sample_rate: int, audio_path: str, session_id: str, text: str, is_follow_up: bool):
        try: 
            sf.write(str(audio_path), audio_data, sample_rate)
            self.db.add_message_to_session(session_id, "assistant", text, str(audio_path), is_follow_up=is_follow_up)
            logger.info(f"💾 BG save ok: {audio_path}")
        except Exception as e: 
            logger.error(f"❌ BG audio save fail {audio_path}: {e}")

    def speech_to_text(self, audio_path: str) -> str:
        if not Path(audio_path).exists(): 
            logger.error(f"STT file missing: {audio_path}")
            return "[Error: Audio file missing]"
        with sr.AudioFile(audio_path) as source:
            try: 
                logger.debug(f"STT record: {audio_path}...")
                audio_data = self.recognizer.record(source)
                logger.debug("STT recognize...")
            except Exception as e: 
                logger.error(f"STT record fail: {e}")
                return "[Error recording segment]"
            try: 
                text = self.recognizer.recognize_google(audio_data)
                logger.info(f"STT ok: '{text}'")
                return text
            except sr.UnknownValueError: 
                logger.warning("STT unclear.")
                return "[Unintelligible]"
            except sr.RequestError as e: 
                logger.error(f"STT service err: {e}")
                return "[Speech service error]"
            except Exception as e: 
                logger.error(f"STT unexpected err: {e}", exc_info=True)
                return "[STT Error]"

    def _stream_gemini_sentences(self, session_id: str) -> iter:
        """
        Calls Gemini in streaming mode and yields full sentences.
        This is a generator function.
        """
        chat = self.active_chat_sessions.get(session_id)
        if not chat:
            logger.error(f"No active chat for {session_id}.")
            yield "Error: Session state lost."
            return

        with self._transcript_lock:
            last_user_answer = self._latest_transcript_for_gemini.get(session_id, "[Error retrieving last answer]")

        logger.debug(f"Sending to Gemini (streaming): '{last_user_answer[:100]}...'")
        
        try:
            response_stream = chat.send_message(last_user_answer, stream=True)
            
            sentence_buffer = ""
            full_response_text = ""
            
            for chunk in response_stream:
                text = chunk.text
                sentence_buffer += text
                full_response_text += text
                
                # Check for sentence-ending punctuation
                while True:
                    match = re.search(r'([^.!?]+[.!?])(\s+|$)', sentence_buffer)
                    if match:
                        sentence = match.group(1).strip()
                        if sentence:
                            logger.debug(f"Yielding sentence: '{sentence}'")
                            yield sentence
                        sentence_buffer = sentence_buffer[len(match.group(0)):].lstrip()
                    else:
                        break
            
            # Yield any remaining text
            if sentence_buffer.strip():
                logger.debug(f"Yielding final buffer: '{sentence_buffer.strip()}'")
                yield sentence_buffer.strip()
            
            logger.info(f"Generated Q (full): '{full_response_text[:100]}...'")
            
            # Log token usage
            try:
                metadata = getattr(response_stream, 'usage_metadata', None)
                if not metadata and 'chunk' in locals():
                     metadata = getattr(chunk, 'usage_metadata', None)
                
                if metadata:
                    prompt_tokens = getattr(metadata, 'prompt_token_count', 0)
                    response_tokens = getattr(metadata, 'candidates_token_count', 0)
                    total_tokens = getattr(metadata, 'total_token_count', 0)
                    self._session_token_counts[session_id]['prompt'] += prompt_tokens
                    self._session_token_counts[session_id]['response'] += response_tokens
                    self._session_token_counts[session_id]['total'] += total_tokens
                    logger.debug(f"[Turn Tokens] P:{prompt_tokens}, R:{response_tokens}, T:{total_tokens}")
                else: 
                    logger.warning("No usage_metadata in Gemini streaming response.")
            except Exception as e: 
                logger.error(f"Error logging Gemini tokens: {e}")
                
        except Exception as e:
            logger.error(f"Gemini streaming generate fail: {e}", exc_info=True)
            yield "Apologies, an error occurred. What is a key skill you possess?"

    def stream_interview_turn(self, session_id: str, turn_count: int, candidate_id: str) -> iter:
        """
        Chains Gemini sentence streaming to TTS streaming.
        Yields raw audio_bytes (MULAW encoded).
        """
        if not self.tts_client:
            logger.error("TTS client not available.")
            return

        full_question_text = []

        def request_generator(sentence_gen):
            try:
                # FIRST REQUEST: Config with voice and streaming audio settings
                # Use MULAW encoding instead of LINEAR16 for streaming
                yield texttospeech.StreamingSynthesizeRequest(
                    streaming_config={
                        'voice': {
                            'language_code': 'en-IN',
                            'name': 'en-IN-Chirp3-HD-Alnilam'
                        },
                        'streaming_audio_config': {
                            'audio_encoding': texttospeech.AudioEncoding.MULAW,
                            'sample_rate_hertz': 24000
                        }
                    }
                )
                
                # SUBSEQUENT REQUESTS: Send text inputs
                for sentence in sentence_gen:
                    full_question_text.append(sentence)
                    logger.debug(f"Streaming sentence to TTS: '{sentence}'")
                    yield texttospeech.StreamingSynthesizeRequest(
                        input={'text': sentence}
                    )
            
            except Exception as e:
                logger.error(f"Error in TTS request_generator: {e}", exc_info=True)

        try:
            sentence_generator = self._stream_gemini_sentences(session_id)
            tts_stream = self.tts_client.streaming_synthesize(
                requests=request_generator(sentence_generator)
            )

            # Yield audio chunks as they arrive
            for tts_response in tts_stream:
                if tts_response.audio_content:
                    yield tts_response.audio_content

            # Log the full text to DB
            combined_text = " ".join(full_question_text)
            if combined_text:
                audio_dir = Path("data") / candidate_id / session_id / "audio"
                audio_dir.mkdir(parents=True, exist_ok=True)
                audio_path = audio_dir / f"bot_{session_id}_{turn_count}_streamed.txt"
                
                with open(audio_path, "w", encoding="utf-8") as f:
                    f.write(combined_text)
                
                self.db.add_message_to_session(session_id, "assistant", combined_text, str(audio_path), is_follow_up=False)
                logger.info(f"💾 BG save ok (streamed text): {audio_path}")
            
        except Exception as e:
            logger.error(f"Error in stream_interview_turn: {e}", exc_info=True)
            fallback_gen = self.stream_plain_text("Apologies, an error occurred.", session_id, turn_count, candidate_id)
            for chunk in fallback_gen:
                yield chunk

    def stream_plain_text(self, plain_text: str, session_id: str, turn_count: int, candidate_id: str, is_follow_up: bool = False) -> iter:
        """
        Takes a full plain_text string and streams the TTS audio output.
        Yields raw audio_bytes (MULAW encoded).
        """
        if not self.tts_client:
            logger.error("TTS client not available.")
            return

        try:
            char_count = len(plain_text)
            self._session_tts_char_counts[session_id] += char_count
            logger.info(f"Generating TTS audio (streaming TEXT, {char_count} chars): '{plain_text[:50]}...'")

            def request_generator():
                try:
                    # FIRST REQUEST: Config with voice and streaming audio settings
                    # Use MULAW encoding instead of LINEAR16
                    yield texttospeech.StreamingSynthesizeRequest(
                        streaming_config={
                            'voice': {
                                'language_code': 'en-IN',
                                'name': 'en-IN-Chirp3-HD-Alnilam'
                            },
                            'streaming_audio_config': {
                                'audio_encoding': texttospeech.AudioEncoding.MULAW,
                                'sample_rate_hertz': 24000
                            }
                        }
                    )
                    
                    # SECOND REQUEST: The text to synthesize
                    yield texttospeech.StreamingSynthesizeRequest(
                        input={'text': plain_text}
                    )
                except Exception as e:
                    logger.error(f"Error in plain_text request_generator: {e}", exc_info=True)

            tts_stream = self.tts_client.streaming_synthesize(requests=request_generator())

            full_audio_data = []

            for tts_response in tts_stream:
                if tts_response.audio_content:
                    full_audio_data.append(tts_response.audio_content)
                    yield tts_response.audio_content
            
            # Background save - decode MULAW to LINEAR16 for saving as WAV
            audio_bytes = b''.join(full_audio_data)
            # Convert MULAW to LINEAR16 for saving
            linear_audio = audioop.ulaw2lin(audio_bytes, 2)
            audio_data_np = np.frombuffer(linear_audio, dtype=np.int16)
            sample_rate = 24000

            audio_dir = Path("data") / candidate_id / session_id / "audio"
            audio_dir.mkdir(parents=True, exist_ok=True)
            suffix = "_followup" if is_follow_up else ""
            audio_path = audio_dir / f"bot_{session_id}_{turn_count}{suffix}_24k.wav"

            save_thread = threading.Thread(
                target=self._save_audio_thread,
                args=(audio_data_np, sample_rate, audio_path, session_id, plain_text, is_follow_up),
                daemon=True
            )
            save_thread.start()

        except Exception as e:
            logger.error(f"Plain text streaming TTS fail: {e}", exc_info=True)
            return

    def process_and_log_transcript(
        self, session_id: str, audio_path: str, turn_count: int,
        candidate_id: str, start_time: Optional[datetime], end_time: Optional[datetime],
        is_follow_up_response: bool = False
    ) -> None:
        """Transcribes, logs to DB, AND updates latest transcript state."""
        transcript = "[Error during transcription process]"
        try:
            logger.info(f"(BG Thread) Transcribing user audio: {audio_path}")
            transcript = self.speech_to_text(audio_path)
            self.db.add_message_to_session(session_id, "user", transcript, audio_path, start_time, end_time, is_follow_up=is_follow_up_response)
            
            with self._transcript_lock:
                if not transcript.startswith("[Error") and transcript != "[Unintelligible]":
                    self._latest_transcript_for_gemini[session_id] = transcript
                    logger.info(f"(BG Thread) Updated latest transcript for session {session_id}")
                else:
                    logger.warning(f"(BG Thread) Transcription failed for {audio_path}, Gemini will use previous transcript.")
        except Exception as e:
            logger.error(f"(BG Thread) Error in process_and_log_transcript: {e}", exc_info=True)
            try: 
                self.db.add_message_to_session(session_id, "user", transcript, audio_path, start_time, end_time, is_follow_up=is_follow_up_response)
            except Exception as db_e: 
                logger.error(f"(BG Thread) Failed log audio error to DB: {db_e}")

    def generate_final_transcript_file(self, session_id: str) -> None:
        """Generates text transcript file."""
        session_data = self.db.get_session(session_id)
        if not session_data: 
            logger.error(f"No session data for transcript: {session_id}")
            return
            
        candidate_id = session_data.get("candidate_id", "unknown")
        transcript_dir = Path("data") / candidate_id / "transcripts"
        transcript_dir.mkdir(parents=True, exist_ok=True)
        transcript_path = transcript_dir / f"transcript_{session_id}.txt"
        
        try:
            with open(transcript_path, "w", encoding="utf-8") as f:
                f.write(f"--- Interview Transcript ---\nSession ID: {session_id}\nCandidate ID: {candidate_id}\n")
                created_at = session_data.get('created_at')
                f.write(f"Date: {created_at.strftime('%Y-%m-%d %H:%M:%S')}\n" if created_at else "Date: N/A\n")
                f.write(f"Questionnaire Provided: {'Yes' if session_data.get('questionnaire') else 'No'} ({len(session_data.get('questionnaire', []))} questions)\n")
                f.write("--------------------------------\n\n")
                
                for msg in session_data.get("conversation", []):
                    role = msg.get("role", "unknown").capitalize()
                    text = msg.get("text", "")
                    ts = msg.get("timestamp")
                    ts_str = ts.strftime('%Y-%m-%d %H:%M:%S') if ts else "N/A"
                    is_follow_up = msg.get("is_follow_up", False)
                    
                    audio_path = msg.get("audio_path", "")
                    if audio_path.endswith("_streamed.txt"):
                        prefix = "(Streamed Text) "
                    else:
                        prefix = "(Follow-up Context) " if is_follow_up else ""
                    
                    f.write(f"[{ts_str}] {prefix}{role}:\n{text}\n\n---\n\n")
                    
            logger.info(f"Transcript saved: {transcript_path}")
        except Exception as e: 
            logger.error(f"Failed write transcript: {e}", exc_info=True)

    def end_interview_session(self, session_id: str):
        """Logs totals and clears session state from memory."""
        if session_id in self._session_token_counts:
            totals = self._session_token_counts[session_id]
            logger.info(f"--- Total Interview Chat Tokens (Session: {session_id}) ---")
            logger.info(f"Prompt: {totals['prompt']}, Response: {totals['response']}, Total: {totals['total']}")
            del self._session_token_counts[session_id]
        else: 
            logger.warning(f"No token count data for session {session_id} on end.")
            
        if session_id in self._session_tts_char_counts:
            total_chars = self._session_tts_char_counts[session_id]
            logger.info(f"--- Total TTS Characters Synthesized (Session: {session_id}) ---")
            logger.info(f"Total Characters: {total_chars}")
            logger.info("-------------------------------------------------")
            del self._session_tts_char_counts[session_id]
        else: 
            logger.warning(f"No TTS char count data for session {session_id} on end.")
            
        if session_id in self.active_chat_sessions:
            del self.active_chat_sessions[session_id]
            logger.info(f"Cleared Gemini chat session {session_id}")
        else: 
            logger.warning(f"End session {session_id}, but no active chat.")
            
        with self._transcript_lock:
            if session_id in self._latest_transcript_for_gemini:
                del self._latest_transcript_for_gemini[session_id]
                logger.info(f"Cleared transcript state for {session_id}")
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
import re  # For cleaning markdown

from app.core.services.database_service import DBHandler

load_dotenv()
logger = logging.getLogger(__name__)

# Initialize Gemini model
chat_model = None
try:
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key: raise ValueError("GEMINI_API_KEY not found.")
    genai.configure(api_key=gemini_api_key)
    chat_model = genai.GenerativeModel('gemini-2.5-flash')
    logger.info("✅ Gemini API configured.")
except Exception as e:
    logger.error(f"❌ Failed to configure Gemini API: {e}")

class InterviewService:
    def __init__(self):
        self.db = DBHandler()
        self.active_chat_sessions: Dict[str, Any] = {}
        self._session_token_counts = defaultdict(lambda: {'prompt': 0, 'response': 0, 'total': 0})
        self._session_tts_char_counts = defaultdict(int)
        self.recognizer = sr.Recognizer()
        self._latest_transcript_for_gemini: Dict[str, Optional[str]] = {}
        self._transcript_lock = threading.Lock()

        logger.info("Initializing Google Cloud TTS client...")
        try:
            custom_endpoint = "texttospeech.googleapis.com:443"
            self.tts_client = texttospeech.TextToSpeechClient(client_options=ClientOptions(api_endpoint=custom_endpoint))
            logger.info(f"✅ Google Cloud TTS client loaded (Endpoint: {custom_endpoint}).")
            self.tts_voice = texttospeech.VoiceSelectionParams(language_code="en-IN", name="en-IN-Chirp3-HD-Alnilam")
            self.tts_audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.LINEAR16,
                sample_rate_hertz=24000,
                speaking_rate=0.9,
            )
            logger.info(f"Set TTS speaking rate to {self.tts_audio_config.speaking_rate}")
        except Exception as e:
            logger.error(f"❌ Failed load TTS client: {e}")
            self.tts_client = None

    def start_new_interview(self, resume_text: str, candidate_id: str, questionnaire: List[str]) -> str:
        """Initializes Gemini chat and all tracking."""
        session_id = self.db.create_session(resume_text, candidate_id, questionnaire or [])
        if not chat_model: raise RuntimeError("Gemini chat model failed.")
        
        logger.info(f"Initializing Gemini chat for session {session_id}")
        
        questionnaire_context = ""
        if questionnaire: 
            questionnaire_str = "\n".join([f"- {q}" for q in questionnaire])
            questionnaire_context = f"\n**Reference Questionnaire:**\n---\n{questionnaire_str}\n---"
        else: 
            questionnaire_context = "\n**Reference Questionnaire:** Not provided."

        # --- MODIFIED SYSTEM PROMPT ---
        # This prompt is now shorter and explicitly asks for brief, general questions.
        system_prompt = f"""
You are an AI Interviewer. Your role is to conduct a professional, conversational interview.

**Your Primary Goal:**
Assess the candidate's skills and experience based on their resume and their responses.

**Critical Instructions:**
1.  **Keep Questions Short:** Ask clear, concise, and general questions. Aim for 1-2 sentences maximum.
2.  **Be Conversational:** Do not ask long, complex, or multi-part questions. Maintain a natural, flowing conversation.
3.  **Use the Context:** Refer to the candidate's resume and the provided questionnaire (if any) to guide your questions, but frame them generally.
4.  **Flow:** Start with an introductory question and adapt based on the candidate's last answer.

**Candidate Resume:**
---
{resume_text}
---

{questionnaire_context}

**Task:**
You will receive the candidate's responses one by one. After each response, generate the *next logical question*. Remember to keep it short and general. Do not greet the candidate or add filler text; just ask the question.
"""
        # --- END MODIFIED SYSTEM PROMPT ---
        
        try:
            self._session_token_counts[session_id] = {'prompt': 0, 'response': 0, 'total': 0}
            self._session_tts_char_counts[session_id] = 0
            with self._transcript_lock: self._latest_transcript_for_gemini[session_id] = "[Candidate introduction pending]"
            
            # Initialize chat history
            self.active_chat_sessions[session_id] = chat_model.start_chat(
                history=[
                    {'role': 'user', 'parts': [system_prompt]},
                    {'role': 'model', 'parts': ["Understood. I will ask short, general questions based on the context. Ready for the candidate's introduction."]}
                ]
            )
            logger.info(f"✅ Gemini chat session initialized for {session_id} with new brief prompt.")
            
        except Exception as e:
            # Clean up on failure
            if session_id in self._session_token_counts: del self._session_token_counts[session_id]
            if session_id in self._session_tts_char_counts: del self._session_tts_char_counts[session_id]
            with self._transcript_lock:
                 if session_id in self._latest_transcript_for_gemini: del self._latest_transcript_for_gemini[session_id]
            logger.error(f"❌ Failed init Gemini chat: {e}", exc_info=True); raise RuntimeError(f"Failed start Gemini: {e}")
            
        return session_id

    def generate_initial_greeting(self) -> str:
        """Provides the opening greeting with instructions."""
        return ("Hello and welcome. I am your AI interviewer. "
                "For our session today, please ensure your camera is turned on and remains on throughout the interview. "
                "Also, please make sure your microphone is enabled when you are speaking. "
                "To start, please introduce yourself.")

    def _save_audio_thread( self, audio_data: np.ndarray, sample_rate: int, audio_path: str, session_id: str, text: str, is_follow_up: bool ):
        try: 
            sf.write(str(audio_path), audio_data, sample_rate)
            self.db.add_message_to_session(session_id, "assistant", text, str(audio_path), is_follow_up=is_follow_up)
            logger.info(f"💾 BG save ok: {audio_path}")
        except Exception as e: 
            logger.error(f"❌ BG audio save fail {audio_path}: {e}")

    def text_to_speech( self, text: str, session_id: str, turn_count: int, candidate_id: str, is_follow_up: bool = False ) -> Tuple[Optional[np.ndarray], Optional[int]]:
        if not self.tts_client: 
            logger.error("TTS client N/A."); 
            return None, None
        try:
            audio_dir = Path("data") / candidate_id / "audio"; audio_dir.mkdir(parents=True, exist_ok=True)
            suffix = "_followup" if is_follow_up else ""; audio_path = audio_dir / f"bot_{session_id}_{turn_count}{suffix}_24k.wav"
            
            # Clean text of Markdown symbols (*, _, #) before sending to TTS
            cleaned_text_for_tts = re.sub(r'[#\*_]', '', text)
            
            synthesis_input = texttospeech.SynthesisInput(text=cleaned_text_for_tts) # (Use cleaned text)
            char_count = len(cleaned_text_for_tts) # (Count cleaned text)
            
            self._session_tts_char_counts[session_id] += char_count
            
            # Log the original text for context, but note the cleaned char count
            logger.info(f"Generating TTS audio ({char_count} cleaned chars){' (followup)' if is_follow_up else ''}: '{text[:50]}...'")
            
            response = self.tts_client.synthesize_speech( input=synthesis_input, voice=self.tts_voice, audio_config=self.tts_audio_config )
            logger.info("✅ TTS received.")
            audio_data = np.frombuffer(response.audio_content, dtype=np.int16); sample_rate = self.tts_audio_config.sample_rate_hertz
            
            # Save the *original* text (with markdown) to the database log
            save_thread = threading.Thread( target=self._save_audio_thread, args=(audio_data, sample_rate, audio_path, session_id, text, is_follow_up), daemon=True )
            save_thread.start()
            
            return audio_data, sample_rate
        except Exception as e: 
            logger.error(f"TTS gen fail: {e}", exc_info=True)
            return None, None

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

    def generate_next_question(self, session_id: str) -> str:
        """Gets the next question from Gemini using the last completed transcript."""
        chat = self.active_chat_sessions.get(session_id)
        if not chat: 
            logger.error(f"No active chat for {session_id}.")
            return "Error: Session state lost."
        
        with self._transcript_lock: 
            last_user_answer = self._latest_transcript_for_gemini.get(session_id, "[Error retrieving last answer]")
        
        try:
            logger.debug(f"Sending to Gemini: '{last_user_answer[:100]}...'")
            response = chat.send_message(last_user_answer)
            next_question = response.text.strip()
            
            try: # Log tokens
                metadata = getattr(response, 'usage_metadata', None)
                if metadata:
                    prompt_tokens = getattr(metadata, 'prompt_token_count', 0)
                    response_tokens = getattr(metadata, 'candidates_token_count', 0)
                    total_tokens = getattr(metadata, 'total_token_count', 0)
                    self._session_token_counts[session_id]['prompt'] += prompt_tokens
                    self._session_token_counts[session_id]['response'] += response_tokens
                    self._session_token_counts[session_id]['total'] += total_tokens
                    logger.debug(f"[Turn Tokens] P:{prompt_tokens}, R:{response_tokens}, T:{total_tokens}")
                else: 
                    logger.warning("No usage_metadata in Gemini response.")
            except Exception as e: 
                logger.error(f"Error logging Gemini tokens: {e}")

            logger.info(f"Generated Q: '{next_question[:100]}...'")
            
            if not next_question or len(next_question) < 5: 
                logger.warning("Gemini Q short/empty, fallback.")
                return "Tell me more about a project you're proud of."
                
            return next_question
        except Exception as e: 
            logger.error(f"Gemini generate fail: {e}", exc_info=True)
            return "Apologies, an error occurred. What is a key skill you possess?"

    def process_and_log_transcript(
        self, session_id: str, audio_path: str, turn_count: int,
        candidate_id: str, start_time: Optional[datetime], end_time: Optional[datetime],
        is_follow_up_response: bool = False
    ) -> None:
        """Transcribes, logs to DB, AND updates latest transcript state (for background thread)."""
        transcript = "[Error during transcription process]"
        try:
            logger.info(f"(BG Thread) Transcribing user audio: {audio_path}")
            transcript = self.speech_to_text(audio_path) # Blocking STT call
            self.db.add_message_to_session(session_id, "user", transcript, audio_path, start_time, end_time, is_follow_up=is_follow_up_response)
            
            with self._transcript_lock:
                # Update only if transcription was reasonably successful
                if not transcript.startswith("[Error") and transcript != "[Unintelligible]":
                    self._latest_transcript_for_gemini[session_id] = transcript
                    logger.info(f"(BG Thread) Updated latest transcript for session {session_id}")
                else:
                    logger.warning(f"(BG Thread) Transcription failed for {audio_path}, Gemini will use previous transcript.")
        except Exception as e:
            logger.error(f"(BG Thread) Error in process_and_log_transcript: {e}", exc_info=True)
            try: 
                self.db.add_message_to_session(session_id, "user", transcript, audio_path, start_time, end_time, is_follow_up=is_follow_up_response) # Log even on error
            except Exception as db_e: 
                logger.error(f"(BG Thread) Failed log audio error to DB: {db_e}")

    def generate_final_transcript_file(self, session_id: str) -> None:
        """Generates text transcript file."""
        session_data = self.db.get_session(session_id)
        if not session_data: 
            logger.error(f"No session data for transcript: {session_id}"); 
            return
            
        candidate_id = session_data.get("candidate_id", "unknown")
        transcript_dir = Path("data") / candidate_id / "transcripts"
        transcript_dir.mkdir(parents=True, exist_ok=True)
        transcript_path = transcript_dir / f"transcript_{session_id}.txt"
        
        try:
            # --- THIS IS THE FIX ---
            with open(transcript_path, "w", encoding="utf-8") as f:
            # --- END OF FIX ---
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
                    prefix = "(Follow-up Context)" if is_follow_up else ""
                    
                    f.write(f"[{ts_str}] {prefix}{role}:\n{text}\n\n---\n\n")
                    
            logger.info(f"Transcript saved: {transcript_path}")
        except Exception as e: 
            logger.error(f"Failed write transcript: {e}", exc_info=True)

    def end_interview_session(self, session_id: str):
        """Logs totals and clears session state from memory."""
        # Log Gemini Tokens
        if session_id in self._session_token_counts:
            totals = self._session_token_counts[session_id]
            logger.info(f"--- Total Interview Chat Tokens (Session: {session_id}) ---")
            logger.info(f"Prompt: {totals['prompt']}, Response: {totals['response']}, Total: {totals['total']}")
            del self._session_token_counts[session_id]
        else: 
            logger.warning(f"No token count data for session {session_id} on end.")
            
        # Log TTS Characters
        if session_id in self._session_tts_char_counts:
            total_chars = self._session_tts_char_counts[session_id]
            logger.info(f"--- Total TTS Characters Synthesized (Session: {session_id}) ---")
            logger.info(f"Total Characters: {total_chars}")
            logger.info("-------------------------------------------------")
            del self._session_tts_char_counts[session_id]
        else: 
            logger.warning(f"No TTS char count data for session {session_id} on end.")
            
        # Clear Chat Session
        if session_id in self.active_chat_sessions:
            del self.active_chat_sessions[session_id]
            logger.info(f"Cleared Gemini chat session {session_id}")
        else: 
            logger.warning(f"End session {session_id}, but no active chat.")
            
        # Clear latest transcript state
        with self._transcript_lock:
            if session_id in self._latest_transcript_for_gemini:
                del self._latest_transcript_for_gemini[session_id]
                logger.info(f"Cleared transcript state for {session_id}")
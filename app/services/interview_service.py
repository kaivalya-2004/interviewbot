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

# --- NO MORE GLOBAL VARIABLE HERE ---

class InterviewService:
    def __init__(self):
        self.db = DBHandler()
        self.active_chat_sessions: Dict[str, Any] = {}
        self._session_token_counts = defaultdict(lambda: {'prompt': 0, 'response': 0, 'total': 0})
        self._session_tts_char_counts = defaultdict(int)
        self.recognizer = sr.Recognizer()
        self._latest_transcript_for_gemini: Dict[str, Optional[str]] = {}
        self._transcript_lock = threading.Lock()

        # --- FIX: Initialize model as an INSTANCE variable ---
        try:
            gemini_api_key = os.getenv("GEMINI_API_KEY")
            if not gemini_api_key: raise ValueError("GEMINI_API_KEY not found.")
            genai.configure(api_key=gemini_api_key)
            # This is now 'self.model', not 'chat_model'
            self.model = genai.GenerativeModel('gemini-2.5-flash') 
            logger.info("✅ Gemini API configured.")
        except Exception as e:
            logger.error(f"❌ Failed to configure Gemini API: {e}")
            self.model = None # Set to None on failure
        # --- END FIX ---

        logger.info("Initializing Google Cloud TTS client...")
        try:
            custom_endpoint = "texttospeech.googleapis.com:443"
            self.tts_client = texttospeech.TextToSpeechClient(client_options=ClientOptions(api_endpoint=custom_endpoint))
            logger.info(f"✅ Google Cloud TTS client loaded (Endpoint: {custom_endpoint}).")
            self.tts_voice = texttospeech.VoiceSelectionParams(language_code="en-IN", name="en-IN-Chirp3-HD-Alnilam")
            
            # --- MODIFICATION: Removed all incompatible params for Chirp voice ---
            self.tts_audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.LINEAR16
                # Chirp models do not support sample_rate_hertz, pitch, or speaking_rate overrides.
            )
            logger.info(f"Set TTS audio config for Chirp model (native 24kHz).")
            # --- END MODIFICATION ---

        except Exception as e:
            logger.error(f"❌ Failed load TTS client: {e}")
            self.tts_client = None

    def start_new_interview(self, resume_text: str, candidate_id: str, questionnaire: List[str]) -> str:
        """Initializes Gemini chat and all tracking."""
        session_id = self.db.create_session(resume_text, candidate_id, questionnaire or [])
        
        # --- FIX: Use self.model ---
        if not self.model: 
            raise RuntimeError("Gemini chat model failed.")
        # --- END FIX ---
        
        logger.info(f"Initializing Gemini chat for session {session_id}")
        
        questionnaire_context = ""
        if questionnaire: 
            questionnaire_str = "\n".join([f"- {q}" for q in questionnaire])
            questionnaire_context = f"\n**Reference Questionnaire:**\n---\n{questionnaire_str}\n---"
        else: 
            questionnaire_context = "\n**Reference Questionnaire:** Not provided."

        # --- MODIFICATION: More specific SSML instructions for pauses ---
        system_prompt = f"""
You are an AI Interviewer. Your role is to conduct a professional, conversational interview.
Your responses MUST be wrapped in <speak> tags.

**IMPORTANT:** To sound natural, you MUST add a pause after punctuation.
- Use <break time='700ms'/> after commas.
- Use <break time='1000ms'/> (1 second) after full stops (periods), question marks, or exclamation marks.
- Use <emphasis> tags to stress key words.
- Your questions must be **very short and direct, ideally under 15 words.** They must be concise, clear, and conversational. Ask only one question at a time.

Your primary goal is to assess the candidate's skills, experience, and suitability for a role. You must be professional, polite, and curious. You will be provided with the candidate's resume and an optional list of topics to guide the conversation.

**Candidate Resume:**
---
{resume_text}
---
{questionnaire_context}

**Task:**
1.  You will be given the transcript of the candidate's last answer.
2.  Your job is to generate the *next* logical question.
3.  This question **must** be relevant to their previous answer AND the context from their **Resume**.
4.  Use the **Reference Questionnaire** as a high-level guide for key topics to cover, but do not just ask the questions verbatim.
5.  Ask open-ended questions (e.g., "Why did you choose that approach?", "What was the main challenge?", "How did that project turn out?").
6.  Actively listen and ask relevant follow-up questions. For example, if they mention a project from their resume, ask about a specific challenge they faced in that project.
7.  Do not repeat questions.
8.  When the interview is concluding, your final response should be a polite closing remark (e.g., "<speak>Thank you for your time today. <break time='1000ms'/> That concludes our interview.</speak>").
"""
        # --- END MODIFICATION ---
        
        try:
            self._session_token_counts[session_id] = {'prompt': 0, 'response': 0, 'total': 0}
            self._session_tts_char_counts[session_id] = 0
            with self._transcript_lock: self._latest_transcript_for_gemini[session_id] = "[Candidate introduction pending]"
            
            # Initialize chat history
            # --- FIX: Use self.model ---
            self.active_chat_sessions[session_id] = self.model.start_chat(
                history=[
                    {'role': 'user', 'parts': [system_prompt]},
                    # --- MODIFICATION: Update model's example response to use SSML ---
                    {'role': 'model', 'parts': ["<speak>Understood. <break time='700ms'/> I will ask short, clear, and conversational questions using SSML with pauses after punctuation. <break time='1000ms'/> Ready for the candidate's introduction.</speak>"]}
                ]
            )
            # --- END FIX ---
            logger.info(f"✅ Gemini chat session initialized for {session_id} with SSML prompt.")
            
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
        # --- MODIFICATION: "A. I." to fix pronunciation ---
        return ("<speak>Hello and welcome. <break time='300ms'/> I am your Interviewer for today. "
                "For our session today, <break time='200ms'/> please ensure your camera is turned on and remains on throughout the interview. "
                "Also, please make sure your microphone is enabled when you are speaking. "
                "<break time='500ms'/> Let's get started. "
                "<break time='500ms'/> Please introduce yourself.</speak>")

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
            # --- MODIFICATION: Standardized audio path ---
            audio_dir = Path("data") / candidate_id / session_id / "audio"
            audio_dir.mkdir(parents=True, exist_ok=True)
            # --- END MODIFICATION ---

            suffix = "_followup" if is_follow_up else ""; audio_path = audio_dir / f"bot_{session_id}_{turn_count}{suffix}_24k.wav"
            
            # --- This line is not needed when sending SSML ---
            ##cleaned_text_for_tts = re.sub(r'[#\*_]', '', text)#
            
            # --- This is correct: it assumes 'text' is valid SSML ---
            synthesis_input = texttospeech.SynthesisInput(ssml=text)
            char_count = len(text) 
            
            self._session_tts_char_counts[session_id] += char_count
            
            # --- MODIFICATION: Fixed misleading log message ---
            logger.info(f"Generating TTS audio ({char_count} SSML chars){' (followup)' if is_follow_up else ''}: '{text[:50]}...'")
            
            response = self.tts_client.synthesize_speech( input=synthesis_input, voice=self.tts_voice, audio_config=self.tts_audio_config )
            logger.info("✅ TTS received.")
            
            audio_data = np.frombuffer(response.audio_content, dtype=np.int16); 
            # --- This is correct: hard-coded sample rate for Chirp model ---
            sample_rate = 24000
            
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
                # --- MODIFICATION: Make sure fallback is also SSML ---
                return "<speak>Tell me more about a project you're proud of.</speak>"
            
            # --- MODIFICATION: Ensure response is valid SSML ---
            if not next_question.startswith("<speak>") or not next_question.endswith("</speak>"):
                logger.warning("Gemini response was not valid SSML. Wrapping it...")
                next_question = f"<speak>{next_question}</speak>"
                
            return next_question
        except Exception as e: 
            logger.error(f"Gemini generate fail: {e}", exc_info=True)
            return "<speak>Apologies, <break time='500ms'/> an error occurred. What is a key skill you possess?</speak>"

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
            logger.error(f"No session data for transcript: {session_id}"); 
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
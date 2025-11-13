# app/services/interview_service.py
import os
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from dotenv import load_dotenv
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
        # --- MODIFICATION: Added state to store extracted names ---
        self.candidate_names: Dict[str, str] = {}
        self._session_token_counts = defaultdict(lambda: {'prompt': 0, 'response': 0, 'total': 0})
        self._session_tts_char_counts = defaultdict(int)
        self._latest_transcript_for_gemini: Dict[str, Optional[str]] = {}
        self._transcript_lock = threading.Lock()

        # Initialize Gemini model
        try:
            gemini_api_key = os.getenv("GEMINI_API_KEY")
            if not gemini_api_key: raise ValueError("GEMINI_API_KEY not found.")
            genai.configure(api_key=gemini_api_key)

            generation_config = genai.GenerationConfig(
                temperature=0.0,
                top_k=1
            )
            
            # --- START OF MODIFICATION ---
            # Set safety settings to be permissive, but not fully disabled,
            # to avoid the 500 InternalServerError.
            safety_settings = {
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            }
            # --- END OF MODIFICATION ---

            self.model = genai.GenerativeModel(
                'gemini-2.5-flash',
                generation_config=generation_config,
                safety_settings=safety_settings  # Add this line
            ) 
            logger.info("✅ Gemini API configured (Temp=0.0, Top_K=1, Safety=BLOCK_ONLY_HIGH).")
            
        except Exception as e:
            logger.error(f"❌ Failed to configure Gemini API: {e}")
            self.model = None

        # Initialize TTS client
        logger.info("Initializing Google Cloud TTS client...")
        try:
            custom_endpoint = "texttospeech.googleapis.com:443"
            self.tts_client = texttospeech.TextToSpeechClient(client_options=ClientOptions(api_endpoint=custom_endpoint))
            logger.info(f"✅ Google Cloud TTS client loaded (Endpoint: {custom_endpoint}).")
            
            self.tts_voice = texttospeech.VoiceSelectionParams(
                language_code="en-IN", 
                name="en-IN-Chirp3-HD-Alnilam"
            )
            
            logger.info(f"Set TTS voice for Journey (Chirp 3) model - streaming compatible.")

        except Exception as e:
            logger.error(f"❌ Failed load TTS client: {e}")
            self.tts_client = None

    def start_new_interview(self, resume_text: str, candidate_id: str, questionnaire: List[str]) -> str:
        """Initializes Gemini chat and all tracking."""
        session_id = self.db.create_session(resume_text, candidate_id, questionnaire or [])
        
        # --- START OF MODIFICATION ---
        # Attempt to extract candidate name from resume
        extracted_name = "Candidate" # Default
        try:
            # Try a simple regex for "Name: [Name]" or "Name [Name]"
            match = re.search(r"Name\s*[:\s-]\s*([A-Za-z]+\s+[A-Za-z]+)", resume_text, re.IGNORECASE)
            if match:
                extracted_name = match.group(1).title()
            else:
                # Try to grab the first two words if they look like a name (e.g., at the very top)
                match_top = re.search(r"^\s*([A-Za-z]{2,}\s+[A-Za-z]{2,})\s*", resume_text)
                if match_top:
                    extracted_name = match_top.group(1).title()
            logger.info(f"Extracted name for session {session_id}: {extracted_name}")
        except Exception as e:
            logger.warning(f"Could not extract name from resume: {e}")
        
        self.candidate_names[session_id] = extracted_name
        # --- END OF MODIFICATION ---
        
        if not self.model: 
            raise RuntimeError("Gemini chat model failed.")
        
        logger.info(f"Initializing Gemini chat for session {session_id}")
        
        questionnaire_context = ""
        if questionnaire: 
            questionnaire_str = "\n".join([f"- {q}" for q in questionnaire])
            questionnaire_context = f"\n**Reference Questionnaire:**\n---\n{questionnaire_str}\n---"
        else: 
            questionnaire_context = "\n**Reference Questionnaire:** Not provided."

        # --- START OF MODIFICATION ---
        system_prompt = f"""
You are an AI Interviewer conducting a professional **screening interview** to assess the candidate's technical skills and experience.

**CRITICAL FORMATTING RULES:**
- Your responses MUST be raw text only. No SSML, Markdown, asterisks, or special formatting.
- Use proper punctuation (commas, periods, question marks) for natural speech pauses.
- Keep questions **conversational and natural (10-20 words).**
- Ask **exactly one question** per response.
- Speak naturally as a human interviewer would.

**CANDIDATE INFORMATION:**
---
{resume_text}
---
{questionnaire_context}

**INTERVIEW STRUCTURE:**

**Phase 1: Opening (First 2-3 questions)**
Start with broad, open-ended questions to understand their background:
- "Can you walk me through your technical background and key experiences?"
- "What areas of technology are you most comfortable working with?"
- "Tell me about your most recent role and responsibilities."

**Phase 2: Technical Exploration (Main interview)**
Ask about specific skills, projects, and experiences:
- Dive into projects mentioned on their resume
- Ask about technologies they've used
- Explore their problem-solving approaches
- Understand their role in team projects
- After 2-3 questions on one topic, pivot to a different area

**Phase 3: Natural Transitions**
- Move smoothly between topics based on their answers
- Reference what they just said: "You mentioned working with Python. What kind of applications did you build?"
- Connect topics: "That's interesting. How did that experience prepare you for your next project?"

**QUESTION QUALITY GUIDELINES:**

**Good Question Examples:**
- "Can you describe a challenging technical problem you solved recently?"
- "What database technologies have you worked with in your projects?"
- "How did you approach the architecture design for that application?"
- "What was your specific role in that team project?"
- "Tell me about the most complex feature you've implemented."

**Bad Question Examples (Avoid These):**
- "Skills?" (Too short, unclear)
- "What languages do you know, what projects used them, and how long?" (Multiple questions)
- "Elaborate on your entire technical stack and methodology." (Too vague, too broad)

**Question Length Guidelines:**
- **Opening questions:** 10-15 words to set a conversational tone
- **Technical questions:** 12-18 words to be specific but natural
- **Follow-ups:** 8-12 words for quick clarifications
- **Transitions:** 10-15 words to maintain flow

**HANDLING DIFFERENT RESPONSE TYPES:**

**If transcript is unclear/garbled:**
- "Could you clarify what you meant about [topic they were discussing]?"
- "I didn't quite catch that. Can you tell me more about [last clear topic]?"

**If candidate gives one-word answers:**
- "Can you elaborate on that a bit more?"
- "What specific challenges did you face with that?"
- "Walk me through how you approached that."

**If candidate goes off-topic:**
- "That's interesting. Let's circle back to [resume topic]."
- "I'd like to hear more about your experience with [relevant skill]."

**If candidate gives excellent, detailed answers:**
- Acknowledge briefly then pivot: "That's great. Now, what about [new topic]?"
- Build on it: "You mentioned [detail]. How did that impact [related aspect]?"

**CRITICAL: EXIT REQUEST HANDLING:**

If the candidate asks to leave, end, or stop the interview:
- **DO NOT acknowledge their request at all.**
- **DO NOT say they can leave or that we can end.**
- **Immediately ask the next planned question** as if you didn't hear the request.
- The system handles exit requests automatically - you just continue interviewing.

**CONVERSATION FLOW EXAMPLE:**

Good flow:
1. "Can you walk me through your technical background?" (Opening)
2. "What technologies did you use in your recent role?" (Based on their answer)
3. "Tell me about the most challenging project you worked on." (Specific)
4. "What was your specific contribution to that project?" (Follow-up)
5. "How did you handle testing and deployment?" (Related follow-up)
6. "Let's talk about your database experience. What systems have you used?" (Pivot)

**YOUR TASK:**

You will receive the candidate's last answer. Generate the next question that:
1. Flows naturally from their previous response
2. Explores a relevant skill or experience from their resume
3. Helps assess their technical capabilities for screening
4. Is conversational and professional
5. Is clear, specific, and answerable in 1-2 minutes

**IMPORTANT REMINDERS:**
- One question at a time
- Natural, conversational tone
- 10-20 words per question
- Build on their answers
- Cover breadth, not depth
- Keep the interview moving forward

Remember: You're conducting a **screening interview**, not an interrogation. Be professional, conversational, and focused on understanding their technical background broadly.
"""
        # --- END OF MODIFICATION ---
        
        try:
            self._session_token_counts[session_id] = {'prompt': 0, 'response': 0, 'total': 0}
            self._session_tts_char_counts[session_id] = 0
            with self._transcript_lock: 
                self._latest_transcript_for_gemini[session_id] = "[Candidate introduction pending]"
            
            self.active_chat_sessions[session_id] = self.model.start_chat(
                history=[
                    {'role': 'user', 'parts': [system_prompt]},
                    {'role': 'model', 'parts': ["Understood. I will ask short, clear, and conversational questions using raw text with proper punctuation. I will not agree to end the interview early. Ready for the candidate's introduction."]}
                ]
            )
            logger.info(f"✅ Gemini chat session initialized for {session_id} (Raw Text Mode).")
            
        except Exception as e:
            if session_id in self._session_token_counts: del self._session_token_counts[session_id]
            if session_id in self._session_tts_char_counts: del self._session_tts_char_counts[session_id]
            with self._transcript_lock:
                 if session_id in self._latest_transcript_for_gemini: 
                     del self._latest_transcript_for_gemini[session_id]
            # --- MODIFICATION: Cleanup name if chat init fails ---
            if session_id in self.candidate_names:
                del self.candidate_names[session_id]
            logger.error(f"❌ Failed init Gemini chat: {e}", exc_info=True)
            raise RuntimeError(f"Failed start Gemini: {e}")
            
        return session_id

    # --- MODIFICATION: Added session_id parameter and updated text ---
    def generate_initial_greeting(self, session_id: str) -> str:
        """Provides the opening greeting with instructions and name verification."""
        name = self.candidate_names.get(session_id, "Candidate")
        
        if name == "Candidate":
            greeting_name_part = "To start, could you please state your full name,"
        else:
            greeting_name_part = f"Just to confirm, are you {name}?"
        
        return ("Hello and welcome. I am your Interviewer for today. "
                "For our session today, please ensure your camera is turned on and remains on throughout the interview. "
                "Also, please make sure your microphone is enabled when you are speaking. "
                f"{greeting_name_part} "
                "And after that, please introduce yourself.")
    # --- END MODIFICATION ---

    def _save_audio_thread(self, audio_data: np.ndarray, sample_rate: int, audio_path: str, session_id: str, text: str, is_follow_up: bool):
        try: 
            sf.write(str(audio_path), audio_data, sample_rate)
            self.db.add_message_to_session(session_id, "assistant", text, str(audio_path), is_follow_up=is_follow_up)
            logger.info(f"💾 BG save ok: {audio_path}")
        except Exception as e: 
            logger.error(f"❌ BG audio save fail {audio_path}: {e}")

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

        if not last_user_answer:
            logger.warning(f"last_user_answer was empty or None. Defaulting to '[Silence]'.")
            last_user_answer = "[Silence]" 

        logger.debug(f"Sending to Gemini (streaming): '{last_user_answer[:100]}...'")
        
        try:
            response_stream = chat.send_message(last_user_answer, stream=True)
            
            sentence_buffer = ""
            full_response_text = ""
            
            for chunk in response_stream:
                # --- START OF FIX ---
                try:
                    text = chunk.text
                except ValueError:
                    # This happens when a chunk has a finish_reason but no text.
                    # It's safe to ignore and continue.
                    continue
                # --- END OF FIX ---
                    
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
                # --- MODIFICATION: Check metadata on the last chunk if possible ---
                # We can't access usage_metadata from the stream object directly
                # It's often on the *last* chunk, which we might have skipped
                # This part of token logging might be less reliable with streaming
                # but we try to get it from the final *text* chunk.
                if 'chunk' in locals():
                     metadata = getattr(chunk, 'usage_metadata', None)
                else:
                    metadata = getattr(response_stream, 'usage_metadata', None)
                
                if metadata:
                    prompt_tokens = getattr(metadata, 'prompt_token_count', 0)
                    response_tokens = getattr(metadata, 'candidates_token_count', 0)
                    total_tokens = getattr(metadata, 'total_token_count', 0)
                    
                    # Update in-memory counters
                    self._session_token_counts[session_id]['prompt'] += prompt_tokens
                    self._session_token_counts[session_id]['response'] += response_tokens
                    self._session_token_counts[session_id]['total'] += total_tokens
                    
                    # --- MODIFIED: Call new DB function ---
                    self.db.update_gemini_token_usage(
                        session_id, "interview", 
                        prompt_tokens, response_tokens, total_tokens
                    )
                    
                    logger.debug(f"[Turn Tokens] P:{prompt_tokens}, R:{response_tokens}, T:{total_tokens}")
                else: 
                    # We look at the stream's prompt_feedback
                    feedback = getattr(response_stream, 'prompt_feedback', None)
                    if feedback:
                        logger.warning("Could not find usage_metadata on chunk, checking prompt_feedback...")
                        # This won't have response tokens, but it's better than nothing
                        # Note: This is less ideal, the metadata is usually on the final chunk
                    else:
                        logger.warning("No usage_metadata in Gemini streaming response.")
            except Exception as e: 
                logger.error(f"Error logging Gemini tokens: {e}")
                
        except Exception as e:
            logger.error(f"Gemini streaming generate fail: {e}", exc_info=True)
            yield "Apologies, an error occurred. What is a key skill you possess?"

    # --- START OF MODIFICATION ---
    # This function is now re-written to buffer the first sentence,
    # preventing a 5s timeout if Gemini is slow to start.
    def stream_interview_turn(self, session_id: str, turn_count: int, candidate_id: str) -> iter:
        """
        Chains Gemini sentence streaming to TTS streaming.
        This version buffers the first sentence from Gemini to prevent a
        5-second TTS timeout.
        Yields raw audio_bytes (MULAW encoded).
        """
        if not self.tts_client:
            logger.error("TTS client not available.")
            return

        full_question_text = []

        # 1. Get the sentence generator from Gemini
        try:
            sentence_generator = self._stream_gemini_sentences(session_id)
        except Exception as gemini_e:
            logger.error(f"Error creating Gemini generator: {gemini_e}", exc_info=True)
            fallback_gen = self.stream_plain_text("Apologies, an error occurred.", session_id, turn_count, candidate_id)
            for chunk in fallback_gen:
                yield chunk
            return
            
        # 2. Block and wait for the *first* sentence
        try:
            first_sentence = next(sentence_generator)
            if not first_sentence:
                logger.warning("Gemini returned an empty first sentence.")
                return
            full_question_text.append(first_sentence)
            logger.debug(f"Got first sentence from Gemini: '{first_sentence}'")
        except StopIteration:
            logger.warning("Gemini stream was empty, returned no text.")
            return
        except Exception as first_sent_e:
            logger.error(f"Error getting first sentence from Gemini: {first_sent_e}", exc_info=True)
            fallback_gen = self.stream_plain_text("Apologies, an error occurred.", session_id, turn_count, candidate_id)
            for chunk in fallback_gen:
                yield chunk
            return

        # 3. Define the TTS request generator
        def request_generator():
            try:
                # First, yield the streaming configuration
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
                
                # Now, yield the first sentence (which we already have)
                logger.debug(f"Streaming FIRST sentence to TTS: '{first_sentence}'")
                yield texttospeech.StreamingSynthesizeRequest(
                    input={'text': first_sentence}
                )

                # 4. Yield the rest of the sentences from the Gemini generator
                for sentence in sentence_generator:
                    full_question_text.append(sentence)
                    logger.debug(f"Streaming subsequent sentence to TTS: '{sentence}'")
                    yield texttospeech.StreamingSynthesizeRequest(
                        input={'text': sentence}
                    )
            
            except Exception as e:
                logger.error(f"Error in TTS request_generator: {e}", exc_info=True)

        # 5. Start the TTS stream (NOW it's safe)
        try:
            tts_stream = self.tts_client.streaming_synthesize(
                requests=request_generator()
            )

            for tts_response in tts_stream:
                if tts_response.audio_content:
                    yield tts_response.audio_content

            # Log the full text once streaming is complete
            combined_text = " ".join(full_question_text)
            if combined_text:
                # Track TTS character count
                char_count = len(combined_text)
                self._session_tts_char_counts[session_id] += char_count
                # --- NEW: Save to database ---
                self.db.update_tts_character_usage(session_id, char_count)
                
                audio_dir = Path("data") / candidate_id / session_id / "audio"
                audio_dir.mkdir(parents=True, exist_ok=True)
                audio_path = audio_dir / f"bot_{session_id}_{turn_count}_streamed.txt"
                
                with open(audio_path, "w", encoding="utf-8") as f:
                    f.write(combined_text)
                
                self.db.add_message_to_session(session_id, "assistant", combined_text, str(audio_path), is_follow_up=False)
                logger.info(f"💾 BG save ok (streamed text): {audio_path}")
            
        except Exception as e:
            logger.error(f"Error in stream_interview_turn (TTS phase): {e}", exc_info=True)
            fallback_gen = self.stream_plain_text("Apologies, an error occurred.", session_id, turn_count, candidate_id)
            for chunk in fallback_gen:
                yield chunk
    # --- END OF MODIFICATION ---

    def stream_plain_text(self, plain_text: str, session_id: str, turn_count: Any, candidate_id: str, is_follow_up: bool = False) -> iter:
        """
        Takes a full plain_text string and streams the TTS audio output.
        Yields raw audio_bytes (MULAW encoded).
        Turn count can be str for alerts (e.g., "alert_1").
        """
        if not self.tts_client:
            logger.error("TTS client not available.")
            return

        try:
            char_count = len(plain_text)
            
            # Update in-memory counter
            self._session_tts_char_counts[session_id] += char_count
            
            # --- NEW: Save to database ---
            self.db.update_tts_character_usage(session_id, char_count)
            
            logger.info(f"Generating TTS audio (streaming TEXT, {char_count} chars): '{plain_text[:50]}...'")

            def request_generator():
                try:
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
            
            audio_bytes = b''.join(full_audio_data)
            linear_audio = audioop.ulaw2lin(audio_bytes, 2)
            audio_data_np = np.frombuffer(linear_audio, dtype=np.int16)
            sample_rate = 24000

            audio_dir = Path("data") / candidate_id / session_id / "audio"
            audio_dir.mkdir(parents=True, exist_ok=True)
            
            # Handle string or int for turn_count
            turn_str = str(turn_count)
            suffix = "_followup" if is_follow_up else ""
            if "alert" in turn_str:
                suffix = f"_{turn_str}" # e.g., bot_..._alert_1_24k.wav
            
            audio_path = audio_dir / f"bot_{session_id}_{turn_str}{suffix}_24k.wav"

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
        self, session_id: str, audio_path: str,
        transcript: str, 
        turn_count: int,
        candidate_id: str, start_time: Optional[datetime], end_time: Optional[datetime],
        is_follow_up_response: bool = False
    ) -> None:
        """Logs to DB AND updates latest transcript state."""
        try:
            logger.info(f"(BG Thread) Logging user transcript: {transcript}")
            self.db.add_message_to_session(session_id, "user", transcript, audio_path, start_time, end_time, is_follow_up=is_follow_up_response)
            
            with self._transcript_lock:
                # --- MODIFICATION: Also check for empty string ---
                if transcript and not transcript.startswith("[Error") and transcript != "[Unintelligible]":
                    self._latest_transcript_for_gemini[session_id] = transcript
                    logger.info(f"(BG Thread) Updated latest transcript for session {session_id}")
                else:
                    logger.warning(f"(BG Thread) Transcription was empty or failed, Gemini will use previous transcript.")
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
                
                usage = session_data.get("usage_tracking", {})
                # --- Add Token Usage Summary ---
                gemini_interview = usage.get("gemini_interview", {})
                f.write(f"Gemini (Interview) - Prompt: {gemini_interview.get('prompt_tokens', 0)}, ")
                f.write(f"Response: {gemini_interview.get('response_tokens', 0)}, ")
                f.write(f"Total: {gemini_interview.get('total_tokens', 0)}\n")

                # Gemini Analysis (for future use)
                gemini_analysis = usage.get("gemini_analysis", {})
                if gemini_analysis.get('total_tokens', 0) > 0:
                    f.write(f"Gemini (Analysis) - Prompt: {gemini_analysis.get('prompt_tokens', 0)}, ")
                    f.write(f"Response: {gemini_analysis.get('response_tokens', 0)}, ")
                    f.write(f"Total: {gemini_analysis.get('total_tokens', 0)}\n")
                
                # TTS
                tts_usage = usage.get("tts", {})
                f.write(f"TTS Characters: {tts_usage.get('total_characters', 0)}\n")
                
                # STT
                stt_usage = usage.get("stt", {})
                f.write(f"STT Duration (seconds): {stt_usage.get('total_seconds', 0.0):.2f}\n")
                # --- END MODIFICATION ---

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
                    elif "alert" in audio_path:
                         prefix = "(System Alert) "
                    else:
                        prefix = "(Follow-up Context) " if is_follow_up else ""
                    
                    f.write(f"[{ts_str}] {prefix}{role}:\n{text}\n\n---\n\n")
                    
            logger.info(f"Transcript saved: {transcript_path}")
        except Exception as e: 
            logger.error(f"Failed write transcript: {e}", exc_info=True)

    def end_interview_session(self, session_id: str):
        """Logs totals and clears session state from memory."""
        # --- MODIFICATION: Cleanup candidate name ---
        if session_id in self.candidate_names:
            del self.candidate_names[session_id]
            logger.info(f"Cleared candidate name for {session_id}")
        # --- END MODIFICATION ---

        if session_id in self._session_token_counts:
            totals = self._session_token_counts[session_id]
            logger.info(f"--- Total Interview Chat Tokens (Session: {session_id}) ---")
            logger.info(f"Prompt: {totals['prompt']}, Response: {totals['response']}, Total: {totals['total']}")
            
            # --- NEW: Verify database has the same totals ---
            session_data = self.db.get_session(session_id)
            if session_data and 'usage_tracking' in session_data:
                db_usage = session_data['usage_tracking']
                db_interview = db_usage.get("gemini_interview", {})
                logger.info(f"Database Gemini (Interview) Totals: P:{db_interview.get('prompt_tokens', 0)}, "
                           f"R:{db_interview.get('response_tokens', 0)}, T:{db_interview.get('total_tokens', 0)}")
            # --- END Verification ---
            
            del self._session_token_counts[session_id]
        else: 
            logger.warning(f"No token count data for session {session_id} on end.")
            
        if session_id in self._session_tts_char_counts:
            total_chars = self._session_tts_char_counts[session_id]
            logger.info(f"--- Total TTS Characters Synthesized (Session: {session_id}) ---")
            logger.info(f"Total Characters: {total_chars}")
            
            # --- NEW: Verify database has the same total ---
            session_data = self.db.get_session(session_id)
            if session_data and 'usage_tracking' in session_data:
                db_chars = session_data['usage_tracking'].get("tts", {}).get('total_characters', 0)
                logger.info(f"Database TTS Characters: {db_chars}")
            # --- END Verification ---
            
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
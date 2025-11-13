# app/core/services/database_service.py
import os
import pymongo
from bson.objectid import ObjectId
from dotenv import load_dotenv
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List

load_dotenv()
logger = logging.getLogger(__name__)

class DBHandler:
    def __init__(self):
        mongo_uri = os.getenv("MONGO_URI")
        if not mongo_uri: raise ValueError("MONGO_URI not found.")
        try:
            self.client = pymongo.MongoClient(mongo_uri)
            self.db = self.client['ai_interviewer_db']
            self.sessions = self.db['interview_sessions']
            logger.info("Successfully connected to MongoDB.")
        except pymongo.errors.ConnectionFailure as e:
            logger.error(f"Could not connect to MongoDB: {e}"); raise

    def create_session(self, resume_text: str, candidate_id: str, questionnaire: List[str]) -> str:
        """Creates a new interview session record with bifurcated token tracking."""
        session_data = {
            "candidate_id": candidate_id, 
            "resume_text": resume_text,
            "questionnaire": questionnaire, 
            "conversation": [],
            "created_at": datetime.now(),
            "status": "pending",
            
            # --- NEW BIFURCATED USAGE TRACKING ---
            "usage_tracking": {
                "gemini_interview": {
                    "prompt_tokens": 0,
                    "response_tokens": 0,
                    "total_tokens": 0
                },
                "gemini_analysis": {
                    "prompt_tokens": 0,
                    "response_tokens": 0,
                    "total_tokens": 0
                },
                "stt": {
                    "total_seconds": 0.0
                },
                "tts": {
                    "total_characters": 0
                }
            }
            # --- Old token_usage and tts_usage removed ---
        }
        result = self.sessions.insert_one(session_data)
        logger.info(f"Created session {result.inserted_id} with {len(questionnaire)} questions.")
        return str(result.inserted_id)

    def add_message_to_session(
        self, session_id: str, role: str, text: str,
        audio_path: Optional[str] = None, start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None, is_follow_up: bool = False
    ) -> None:
        """Adds a message to the conversation history of a session."""
        if not session_id: logger.error("Attempted to add message with empty session_id."); return
        try:
            message: Dict[str, Any] = {
                "role": role, "text": text, "timestamp": datetime.now(),
                "is_follow_up": is_follow_up
            }
            if audio_path: message["audio_path"] = audio_path
            if start_time and end_time and role == 'user':
                message["start_timestamp"] = start_time
                message["end_timestamp"] = end_time

            update_result = self.sessions.update_one(
                 {"_id": ObjectId(session_id)},
                 {"$push": {"conversation": message}}
            )
            if update_result.matched_count == 0:
                 logger.error(f"Session {session_id} not found when trying to add message.")
            elif update_result.modified_count == 1:
                 log_prefix = "[Easy Follow-up] " if is_follow_up and role == 'assistant' else "[Response to Follow-up] " if is_follow_up and role == 'user' else ""
                 logger.debug(f"{log_prefix}Added {role} message to session {session_id}")
            else:
                  logger.warning(f"Update operation didn't modify session {session_id} when adding message (modified_count={update_result.modified_count}).")

        except pymongo.errors.PyMongoError as e:
            logger.error(f"MongoDB error adding message to session {session_id}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error adding message to session {session_id}: {e}", exc_info=True)

    # --- REPLACED update_token_usage ---
    def update_gemini_token_usage(
        self, session_id: str, usage_type: str,
        prompt_tokens: int, response_tokens: int, total_tokens: int
    ) -> None:
        """Updates a specific Gemini token usage counter for a session."""
        if not session_id:
            logger.error("Attempted to update tokens with empty session_id.")
            return
        
        if usage_type not in ["interview", "analysis"]:
            logger.error(f"Invalid Gemini usage_type '{usage_type}' for session {session_id}.")
            return
            
        update_key_prefix = f"usage_tracking.gemini_{usage_type}"
        
        try:
            update_result = self.sessions.update_one(
                {"_id": ObjectId(session_id)},
                {
                    "$inc": {
                        f"{update_key_prefix}.prompt_tokens": prompt_tokens,
                        f"{update_key_prefix}.response_tokens": response_tokens,
                        f"{update_key_prefix}.total_tokens": total_tokens
                    }
                }
            )
            if update_result.matched_count == 0:
                logger.error(f"Session {session_id} not found when updating {usage_type} token usage.")
            elif update_result.modified_count == 1:
                logger.debug(f"Updated {usage_type} token usage for session {session_id}: +{prompt_tokens}P, +{response_tokens}R, +{total_tokens}T")
            else:
                logger.warning(f"{usage_type} token update didn't modify session {session_id}")
        except pymongo.errors.PyMongoError as e:
            logger.error(f"MongoDB error updating {usage_type} tokens for session {session_id}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error updating {usage_type} tokens for {session_id}: {e}", exc_info=True)

    # --- UPDATED PATH ---
    def update_tts_character_usage(self, session_id: str, character_count: int) -> None:
        """Updates the TTS character usage counter for a session."""
        if not session_id:
            logger.error("Attempted to update TTS chars with empty session_id.")
            return
        try:
            update_result = self.sessions.update_one(
                {"_id": ObjectId(session_id)},
                {"$inc": {"usage_tracking.tts.total_characters": character_count}}
            )
            if update_result.matched_count == 0:
                logger.error(f"Session {session_id} not found when updating TTS usage.")
            elif update_result.modified_count == 1:
                logger.debug(f"Updated TTS usage for session {session_id}: +{character_count} chars")
            else:
                logger.warning(f"TTS update didn't modify session {session_id}")
        except pymongo.errors.PyMongoError as e:
            logger.error(f"MongoDB error updating TTS chars for session {session_id}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error updating TTS chars for session {session_id}: {e}", exc_info=True)

    # --- NEW METHOD FOR STT ---
    def update_stt_usage(self, session_id: str, duration_seconds: float) -> None:
        """Updates the STT duration (in seconds) counter for a session."""
        if not session_id:
            logger.error("Attempted to update STT duration with empty session_id.")
            return
        if duration_seconds <= 0:
            logger.warning(f"Attempted to log zero or negative STT duration for {session_id}")
            return
        
        try:
            update_result = self.sessions.update_one(
                {"_id": ObjectId(session_id)},
                {"$inc": {"usage_tracking.stt.total_seconds": duration_seconds}}
            )
            if update_result.matched_count == 0:
                logger.error(f"Session {session_id} not found when updating STT usage.")
            elif update_result.modified_count == 1:
                logger.debug(f"Updated STT usage for session {session_id}: +{duration_seconds:.2f} seconds")
            else:
                logger.warning(f"STT usage update didn't modify session {session_id}")
        except pymongo.errors.PyMongoError as e:
            logger.error(f"MongoDB error updating STT usage for session {session_id}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error updating STT usage for {session_id}: {e}", exc_info=True)

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a session document by its ID."""
        if not session_id: logger.error("Attempted to get session with empty session_id."); return None
        try:
            if not ObjectId.is_valid(session_id):
                 logger.error(f"Invalid ObjectId format for session_id: {session_id}")
                 return None
            return self.sessions.find_one({"_id": ObjectId(session_id)})
        except pymongo.errors.PyMongoError as e:
             logger.error(f"MongoDB error fetching session {session_id}: {e}")
             return None
        except Exception as e:
            logger.error(f"Unexpected error fetching session {session_id}: {e}", exc_info=True)
            return None

    def update_session_status(self, session_id: str, status: str):
        """Updates the 'status' field of a session."""
        if self.db is None:
            logger.error("Database not initialized, cannot update status.")
            return
        try:
            self.sessions.update_one(
                {"_id": ObjectId(session_id)},
                {"$set": {"status": status}}
            )
        except Exception as e:
            logger.error(f"DB Error updating status for {session_id}: {e}", exc_info=True)
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
        """Creates a new interview session record."""
        session_data = {
            "candidate_id": candidate_id, 
            "resume_text": resume_text,
            "questionnaire": questionnaire, 
            "conversation": [],
            # --- tab_switch_events REMOVED ---
            "created_at": datetime.now(),
            "status": "pending",
            # --- device_type and user_agent REMOVED ---
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

    # --- add_tab_switch_events method REMOVED ---

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

    # --- update_session_device_info method REMOVED ---
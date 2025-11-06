import pytest
import pymongo
from unittest.mock import patch, MagicMock, call
from bson.objectid import ObjectId
from datetime import datetime
import os
import sys

# --- Add project root to path ---
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, project_root)
# --- End Path Setup ---

from app.core.services.database_service import DBHandler

# --- Fixtures ---

@pytest.fixture
def mock_mongo_client():
    """Mocks pymongo.MongoClient and yields the mock 'sessions' collection."""
    with patch('app.core.services.database_service.pymongo.MongoClient') as mock_client_class:
        mock_client_instance = MagicMock(name="MongoClientInstance")
        mock_db = MagicMock(name="MockDB")
        mock_sessions_collection = MagicMock(name="MockSessionsCollection")
        
        # Setup the chained access: client['db']['collection']
        mock_client_instance.__getitem__.return_value = mock_db
        mock_db.__getitem__.return_value = mock_sessions_collection
        
        mock_client_class.return_value = mock_client_instance
        
        yield mock_sessions_collection

@pytest.fixture
def mock_objectid():
    """
    Mocks bson.objectid.ObjectId to return controllable test IDs.
    """
    with patch('app.core.services.database_service.ObjectId') as mock_obj_id_class:
        # When ObjectId(id_str) is called, just return id_str for simplicity
        mock_obj_id_class.side_effect = lambda x: x
        mock_obj_id_class.is_valid = MagicMock(return_value=True)
        yield mock_obj_id_class

@pytest.fixture
def db_handler(mock_mongo_client, mock_objectid):
    """
    Provides a DBHandler instance with mocked dependencies.
    """
    with patch('app.core.services.database_service.os.getenv', return_value="mongodb://mock_uri:27017/"):
        with patch('app.core.services.database_service.load_dotenv'):
            handler = DBHandler()
            yield handler

# --- Test Cases ---

class TestDBHandlerInit:
    """Tests for DBHandler.__init__"""

    def test_init_success(self, mock_mongo_client):
        """Tests successful initialization and connection."""
        with patch('app.core.services.database_service.os.getenv', return_value="mongodb://mock_uri:27017/"):
            with patch('app.core.services.database_service.load_dotenv'):
                with patch('app.core.services.database_service.pymongo.MongoClient') as mock_client_class:
                    mock_client_instance = MagicMock()
                    mock_db = MagicMock()
                    mock_sessions = MagicMock()
                    
                    mock_client_instance.__getitem__.return_value = mock_db
                    mock_db.__getitem__.return_value = mock_sessions
                    mock_client_class.return_value = mock_client_instance
                    
                    handler = DBHandler()
                    
                    mock_client_class.assert_called_once_with("mongodb://mock_uri:27017/")
                    assert handler.client is not None
                    assert handler.db is not None
                    assert handler.sessions is not None

    def test_init_no_mongo_uri(self):
        """Tests that a ValueError is raised if MONGO_URI is not set."""
        with patch('app.core.services.database_service.os.getenv', return_value=None):
            with patch('app.core.services.database_service.load_dotenv'):
                with pytest.raises(ValueError, match="MONGO_URI not found"):
                    DBHandler()

    def test_init_empty_mongo_uri(self):
        """Tests that a ValueError is raised if MONGO_URI is empty string."""
        with patch('app.core.services.database_service.os.getenv', return_value=""):
            with patch('app.core.services.database_service.load_dotenv'):
                with pytest.raises(ValueError, match="MONGO_URI not found"):
                    DBHandler()

    def test_init_connection_failure(self):
        """Tests that a ConnectionFailure is propagated."""
        with patch('app.core.services.database_service.os.getenv', return_value="mongodb://mock_uri:27017/"):
            with patch('app.core.services.database_service.load_dotenv'):
                with patch('app.core.services.database_service.pymongo.MongoClient', 
                          side_effect=pymongo.errors.ConnectionFailure("Test connection fail")):
                    with pytest.raises(pymongo.errors.ConnectionFailure, match="Test connection fail"):
                        DBHandler()


class TestCreateSession:
    """Tests for DBHandler.create_session"""

    def test_create_session_success(self, db_handler, mock_mongo_client):
        """Tests successful creation of a new session."""
        test_resume = "Test resume text"
        test_candidate_id = "test_candidate_123"
        test_questionnaire = ["Q1", "Q2", "Q3"]
        
        test_object_id_str = "60c72b2f9b1d8b001f8e4c6d"
        mock_result = MagicMock()
        mock_result.inserted_id = test_object_id_str
        mock_mongo_client.insert_one.return_value = mock_result
        
        session_id = db_handler.create_session(test_resume, test_candidate_id, test_questionnaire)
        
        assert session_id == test_object_id_str
        mock_mongo_client.insert_one.assert_called_once()
        
        # Verify the inserted data structure
        inserted_data = mock_mongo_client.insert_one.call_args[0][0]
        assert inserted_data['candidate_id'] == test_candidate_id
        assert inserted_data['resume_text'] == test_resume
        assert inserted_data['questionnaire'] == test_questionnaire
        assert inserted_data['conversation'] == []
        assert 'created_at' in inserted_data
        assert isinstance(inserted_data['created_at'], datetime)

    def test_create_session_empty_questionnaire(self, db_handler, mock_mongo_client):
        """Tests session creation with empty questionnaire list."""
        test_resume = "Test resume"
        test_candidate_id = "test_candidate_456"
        test_questionnaire = []
        
        test_object_id_str = "60c72b2f9b1d8b001f8e4c6e"
        mock_result = MagicMock()
        mock_result.inserted_id = test_object_id_str
        mock_mongo_client.insert_one.return_value = mock_result
        
        session_id = db_handler.create_session(test_resume, test_candidate_id, test_questionnaire)
        
        assert session_id == test_object_id_str
        inserted_data = mock_mongo_client.insert_one.call_args[0][0]
        assert inserted_data['questionnaire'] == []

    def test_create_session_special_characters(self, db_handler, mock_mongo_client):
        """Tests session creation with special characters in text fields."""
        test_resume = "Resume with special chars: @#$%^&*()"
        test_candidate_id = "candidate_with_underscore_123"
        test_questionnaire = ["Question with 'quotes'?", "Question with \"double quotes\""]
        
        test_object_id_str = "60c72b2f9b1d8b001f8e4c6f"
        mock_result = MagicMock()
        mock_result.inserted_id = test_object_id_str
        mock_mongo_client.insert_one.return_value = mock_result
        
        session_id = db_handler.create_session(test_resume, test_candidate_id, test_questionnaire)
        
        assert session_id == test_object_id_str
        inserted_data = mock_mongo_client.insert_one.call_args[0][0]
        assert inserted_data['resume_text'] == test_resume
        assert inserted_data['questionnaire'] == test_questionnaire


class TestAddMessageToSession:
    """Tests for DBHandler.add_message_to_session"""

    def test_add_simple_user_message(self, db_handler, mock_mongo_client, mock_objectid):
        """Tests adding a simple user message."""
        test_session_id = "60c72b2f9b1d8b001f8e4c6a"
        
        mock_result = MagicMock()
        mock_result.matched_count = 1
        mock_result.modified_count = 1
        mock_mongo_client.update_one.return_value = mock_result
        
        db_handler.add_message_to_session(test_session_id, "user", "Hello, this is my answer")
        
        mock_mongo_client.update_one.assert_called_once()
        call_filter, call_update = mock_mongo_client.update_one.call_args[0]
        
        assert call_filter == {"_id": test_session_id}
        assert "$push" in call_update
        
        message = call_update["$push"]["conversation"]
        assert message['role'] == 'user'
        assert message['text'] == "Hello, this is my answer"
        assert 'timestamp' in message
        assert isinstance(message['timestamp'], datetime)
        assert message['is_follow_up'] is False

    def test_add_assistant_message(self, db_handler, mock_mongo_client, mock_objectid):
        """Tests adding an assistant message."""
        test_session_id = "60c72b2f9b1d8b001f8e4c6b"
        
        mock_result = MagicMock()
        mock_result.matched_count = 1
        mock_result.modified_count = 1
        mock_mongo_client.update_one.return_value = mock_result
        
        db_handler.add_message_to_session(test_session_id, "assistant", "What is your experience?")
        
        message = mock_mongo_client.update_one.call_args[0][1]["$push"]["conversation"]
        assert message['role'] == 'assistant'
        assert message['text'] == "What is your experience?"

    def test_add_message_with_audio_path(self, db_handler, mock_mongo_client, mock_objectid):
        """Tests adding a message with audio path."""
        test_session_id = "60c72b2f9b1d8b001f8e4c6c"
        audio_path = "/data/candidate_123/session_456/audio/turn_1.wav"
        
        mock_result = MagicMock()
        mock_result.matched_count = 1
        mock_result.modified_count = 1
        mock_mongo_client.update_one.return_value = mock_result
        
        db_handler.add_message_to_session(
            test_session_id, "user", "Test answer", audio_path=audio_path
        )
        
        message = mock_mongo_client.update_one.call_args[0][1]["$push"]["conversation"]
        assert message['audio_path'] == audio_path

    def test_add_message_with_timestamps(self, db_handler, mock_mongo_client, mock_objectid):
        """Tests adding a user message with start and end timestamps."""
        test_session_id = "60c72b2f9b1d8b001f8e4c6d"
        start_time = datetime(2025, 11, 5, 10, 0, 0)
        end_time = datetime(2025, 11, 5, 10, 0, 30)
        
        mock_result = MagicMock()
        mock_result.matched_count = 1
        mock_result.modified_count = 1
        mock_mongo_client.update_one.return_value = mock_result
        
        db_handler.add_message_to_session(
            test_session_id, "user", "Answer with timing",
            start_time=start_time, end_time=end_time
        )
        
        message = mock_mongo_client.update_one.call_args[0][1]["$push"]["conversation"]
        assert message['start_timestamp'] == start_time
        assert message['end_timestamp'] == end_time

    def test_add_message_with_all_fields(self, db_handler, mock_mongo_client, mock_objectid):
        """Tests adding a message with all optional fields."""
        test_session_id = "60c72b2f9b1d8b001f8e4c6e"
        start_time = datetime(2025, 11, 5, 10, 0, 0)
        end_time = datetime(2025, 11, 5, 10, 0, 30)
        audio_path = "/data/audio.wav"
        
        mock_result = MagicMock()
        mock_result.matched_count = 1
        mock_result.modified_count = 1
        mock_mongo_client.update_one.return_value = mock_result
        
        db_handler.add_message_to_session(
            session_id=test_session_id,
            role="user",
            text="Complete answer",
            audio_path=audio_path,
            start_time=start_time,
            end_time=end_time,
            is_follow_up=True
        )
        
        message = mock_mongo_client.update_one.call_args[0][1]["$push"]["conversation"]
        assert message['role'] == "user"
        assert message['text'] == "Complete answer"
        assert message['audio_path'] == audio_path
        assert message['start_timestamp'] == start_time
        assert message['end_timestamp'] == end_time
        assert message['is_follow_up'] is True

    def test_add_follow_up_message(self, db_handler, mock_mongo_client, mock_objectid):
        """Tests adding a follow-up message."""
        test_session_id = "60c72b2f9b1d8b001f8e4c6f"
        
        mock_result = MagicMock()
        mock_result.matched_count = 1
        mock_result.modified_count = 1
        mock_mongo_client.update_one.return_value = mock_result
        
        db_handler.add_message_to_session(
            test_session_id, "assistant", "Can you elaborate?", is_follow_up=True
        )
        
        message = mock_mongo_client.update_one.call_args[0][1]["$push"]["conversation"]
        assert message['is_follow_up'] is True

    def test_add_message_session_not_found(self, db_handler, mock_mongo_client, mock_objectid):
        """Tests behavior when session_id does not match any document."""
        test_session_id = "60c72b2f9b1d8b001f8e4c70"
        
        mock_result = MagicMock()
        mock_result.matched_count = 0
        mock_result.modified_count = 0
        mock_mongo_client.update_one.return_value = mock_result
        
        # Should not raise an error, but should log
        db_handler.add_message_to_session(test_session_id, "user", "Hello")
        
        mock_mongo_client.update_one.assert_called_once()

    def test_add_message_matched_not_modified(self, db_handler, mock_mongo_client, mock_objectid):
        """Tests behavior when session is matched but not modified."""
        test_session_id = "60c72b2f9b1d8b001f8e4c71"
        
        mock_result = MagicMock()
        mock_result.matched_count = 1
        mock_result.modified_count = 0
        mock_mongo_client.update_one.return_value = mock_result
        
        db_handler.add_message_to_session(test_session_id, "user", "Hello")
        
        mock_mongo_client.update_one.assert_called_once()

    @pytest.mark.parametrize("session_id", [None, ""])
    def test_add_message_empty_session_id(self, db_handler, mock_mongo_client, session_id):
        """Tests that no DB call is made if session_id is empty or None."""
        db_handler.add_message_to_session(session_id, "user", "Hello")
        mock_mongo_client.update_one.assert_not_called()

    def test_add_message_mongo_error(self, db_handler, mock_mongo_client, mock_objectid):
        """Tests that PyMongoErrors are caught and logged."""
        test_session_id = "60c72b2f9b1d8b001f8e4c72"
        
        mock_mongo_client.update_one.side_effect = pymongo.errors.PyMongoError("DB Test Error")
        
        # Should not raise, just log
        db_handler.add_message_to_session(test_session_id, "user", "Hello")
        
        mock_mongo_client.update_one.assert_called_once()

    def test_add_message_unexpected_error(self, db_handler, mock_mongo_client, mock_objectid):
        """Tests that general exceptions are caught."""
        test_session_id = "60c72b2f9b1d8b001f8e4c73"
        
        mock_mongo_client.update_one.side_effect = Exception("Unexpected error")
        
        # Should not raise, just log
        db_handler.add_message_to_session(test_session_id, "user", "Hello")
        
        mock_mongo_client.update_one.assert_called_once()

    def test_timestamps_only_added_for_user(self, db_handler, mock_mongo_client, mock_objectid):
        """Tests that start/end timestamps are only added for user role."""
        test_session_id = "60c72b2f9b1d8b001f8e4c74"
        start_time = datetime.now()
        end_time = datetime.now()
        
        mock_result = MagicMock()
        mock_result.matched_count = 1
        mock_result.modified_count = 1
        mock_mongo_client.update_one.return_value = mock_result
        
        # Test with assistant role
        db_handler.add_message_to_session(
            test_session_id, "assistant", "Question?",
            start_time=start_time, end_time=end_time
        )
        
        message = mock_mongo_client.update_one.call_args[0][1]["$push"]["conversation"]
        assert 'start_timestamp' not in message
        assert 'end_timestamp' not in message


class TestGetSession:
    """Tests for DBHandler.get_session"""

    def test_get_session_success(self, db_handler, mock_mongo_client, mock_objectid):
        """Tests successfully retrieving a session."""
        test_session_id = "60c72b2f9b1d8b001f8e4c75"
        mock_doc = {
            "_id": test_session_id,
            "candidate_id": "test_candidate_123",
            "resume_text": "Test resume",
            "conversation": []
        }
        
        mock_mongo_client.find_one.return_value = mock_doc
        
        result = db_handler.get_session(test_session_id)
        
        assert result == mock_doc
        mock_mongo_client.find_one.assert_called_once_with({"_id": test_session_id})

    def test_get_session_not_found(self, db_handler, mock_mongo_client, mock_objectid):
        """Tests retrieving a session that doesn't exist."""
        test_session_id = "60c72b2f9b1d8b001f8e4c76"
        
        mock_mongo_client.find_one.return_value = None
        
        result = db_handler.get_session(test_session_id)
        
        assert result is None
        mock_mongo_client.find_one.assert_called_once_with({"_id": test_session_id})

    @pytest.mark.parametrize("session_id", [None, ""])
    def test_get_session_empty_id(self, db_handler, mock_mongo_client, session_id):
        """Tests that no DB call is made if session_id is empty or None."""
        result = db_handler.get_session(session_id)
        
        assert result is None
        mock_mongo_client.find_one.assert_not_called()

    def test_get_session_invalid_objectid_format(self, db_handler, mock_mongo_client):
        """Tests behavior when session_id is not a valid ObjectId format."""
        with patch('app.core.services.database_service.ObjectId') as mock_oid:
            mock_oid.is_valid.return_value = False
            
            result = db_handler.get_session("not_a_valid_object_id")
            
            assert result is None
            mock_oid.is_valid.assert_called_once_with("not_a_valid_object_id")
            mock_mongo_client.find_one.assert_not_called()

    def test_get_session_mongo_error(self, db_handler, mock_mongo_client, mock_objectid):
        """Tests that PyMongoErrors are caught and return None."""
        test_session_id = "60c72b2f9b1d8b001f8e4c77"
        
        mock_mongo_client.find_one.side_effect = pymongo.errors.PyMongoError("DB Test Error")
        
        result = db_handler.get_session(test_session_id)
        
        assert result is None
        mock_mongo_client.find_one.assert_called_once()

    def test_get_session_unexpected_error(self, db_handler, mock_mongo_client, mock_objectid):
        """Tests that general exceptions are caught and return None."""
        test_session_id = "60c72b2f9b1d8b001f8e4c78"
        
        mock_mongo_client.find_one.side_effect = Exception("Unexpected error")
        
        result = db_handler.get_session(test_session_id)
        
        assert result is None
        mock_mongo_client.find_one.assert_called_once()

    def test_get_session_with_conversation_history(self, db_handler, mock_mongo_client, mock_objectid):
        """Tests retrieving a session with conversation history."""
        test_session_id = "60c72b2f9b1d8b001f8e4c79"
        mock_doc = {
            "_id": test_session_id,
            "candidate_id": "test_candidate_123",
            "resume_text": "Test resume",
            "conversation": [
                {"role": "assistant", "text": "Question 1", "timestamp": datetime.now()},
                {"role": "user", "text": "Answer 1", "timestamp": datetime.now()}
            ]
        }
        
        mock_mongo_client.find_one.return_value = mock_doc
        
        result = db_handler.get_session(test_session_id)
        
        assert result == mock_doc
        assert len(result['conversation']) == 2


class TestIntegrationScenarios:
    """Integration-style tests for common workflows"""

    def test_complete_interview_workflow(self, db_handler, mock_mongo_client, mock_objectid):
        """Tests a complete interview workflow from creation to retrieval."""
        # Create session
        test_resume = "Software Engineer Resume"
        test_candidate_id = "candidate_001"
        test_questionnaire = ["Q1", "Q2"]
        test_session_id = "60c72b2f9b1d8b001f8e4c7a"
        
        mock_insert_result = MagicMock()
        mock_insert_result.inserted_id = test_session_id
        mock_mongo_client.insert_one.return_value = mock_insert_result
        
        session_id = db_handler.create_session(test_resume, test_candidate_id, test_questionnaire)
        assert session_id == test_session_id
        
        # Add messages
        mock_update_result = MagicMock()
        mock_update_result.matched_count = 1
        mock_update_result.modified_count = 1
        mock_mongo_client.update_one.return_value = mock_update_result
        
        db_handler.add_message_to_session(session_id, "assistant", "First question")
        db_handler.add_message_to_session(session_id, "user", "First answer")
        
        # Retrieve session
        mock_doc = {
            "_id": test_session_id,
            "candidate_id": test_candidate_id,
            "resume_text": test_resume,
            "questionnaire": test_questionnaire,
            "conversation": [
                {"role": "assistant", "text": "First question"},
                {"role": "user", "text": "First answer"}
            ]
        }
        mock_mongo_client.find_one.return_value = mock_doc
        
        retrieved_session = db_handler.get_session(session_id)
        assert retrieved_session is not None
        assert retrieved_session['candidate_id'] == test_candidate_id

    def test_multiple_message_additions(self, db_handler, mock_mongo_client, mock_objectid):
        """Tests adding multiple messages in sequence."""
        test_session_id = "60c72b2f9b1d8b001f8e4c7b"
        
        mock_result = MagicMock()
        mock_result.matched_count = 1
        mock_result.modified_count = 1
        mock_mongo_client.update_one.return_value = mock_result
        
        messages = [
            ("assistant", "Question 1"),
            ("user", "Answer 1"),
            ("assistant", "Question 2"),
            ("user", "Answer 2"),
            ("assistant", "Question 3")
        ]
        
        for role, text in messages:
            db_handler.add_message_to_session(test_session_id, role, text)
        
        assert mock_mongo_client.update_one.call_count == len(messages)
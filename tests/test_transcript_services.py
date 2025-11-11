# tests/test_transcript_services.py
import pytest
from unittest.mock import Mock, patch, MagicMock
import json
import os # Import os to patch.dict
import sys # Import sys to fix path

# --- Setup Path ---
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, project_root)
# --- End Path Setup ---

from app.services.transcript_parser import TranscriptParser
from app.services.transcript_analyzer import TranscriptAnalyzer
from app.models.analysis_schemas import OverallAnalysis


@pytest.fixture
def sample_transcript():
    """Sample transcript content"""
    return """--- Interview Transcript ---
Session ID: test_session_123
Candidate ID: candidate_001
Date: 2024-01-15 10:00:00
Questionnaire Provided: Yes (3 questions)
--------------------------------

[2024-01-15 10:00:15] Assistant:
Hello and welcome. Please introduce yourself.

---

[2024-01-15 10:00:45] User:
I am John Doe, a software engineer with 5 years of experience in Python and Django.

---

[2024-01-15 10:01:30] Assistant:
Tell me about your experience with Python.

---

[2024-01-15 10:02:00] User:
I have been working with Python for over 5 years, primarily in web development using Django and Flask frameworks.

---

[2024-01-15 10:03:00] [Easy Follow-up] Assistant:
Can you give me a specific example?

---

[2024-01-15 10:03:30] [Response to Follow-up] User:
I built an e-commerce platform that handles 10,000 transactions per day.

---
"""


@pytest.fixture
def sample_resume():
    """Sample resume content"""
    return """John Doe
Software Engineer

Experience:
- Senior Python Developer at Tech Corp (2019-2024)
- Developed scalable web applications using Django
- Led team of 5 developers

Skills:
- Python, Django, Flask
- PostgreSQL, Redis
- Docker, Kubernetes

Education:
- BS Computer Science, University of Tech (2015-2019)
"""


class TestTranscriptParser:
    """Test suite for TranscriptParser"""

    def test_parse_valid_transcript(self, sample_transcript):
        """Test parsing a valid transcript"""
        result = TranscriptParser.parse(sample_transcript)
        
        assert 'metadata' in result
        assert 'qa_pairs' in result
        assert result['metadata']['session_id'] == 'test_session_123'
        assert result['metadata']['user_id'] == 'candidate_001'
        assert len(result['qa_pairs']) > 0

    @pytest.mark.parametrize("content", ["", "   \n   "])
    def test_parse_empty_or_whitespace_transcript(self, content):
        """Test parsing empty or whitespace-only transcript"""
        with pytest.raises(ValueError, match="Transcript content is empty."):
            TranscriptParser.parse(content)

    def test_extract_metadata(self, sample_transcript):
        """Test metadata extraction"""
        lines = sample_transcript.split('\n')
        metadata = TranscriptParser._extract_metadata(lines)
        
        assert metadata['session_id'] == 'test_session_123'
        assert metadata['user_id'] == 'candidate_001'
        assert '2024-01-15' in metadata['date']

    def test_extract_qa_entries(self, sample_transcript):
        """Test Q&A entry extraction"""
        lines = sample_transcript.split('\n')
        qa_entries = TranscriptParser._extract_qa_entries(lines)
        
        assert len(qa_entries) > 0
        assert qa_entries[0]['speaker'] == 'Bot'
        assert 'introduce yourself' in qa_entries[0]['text'].lower()

    def test_extract_qa_entries_with_followups(self, sample_transcript):
        """Test extraction handles follow-up tags"""
        lines = sample_transcript.split('\n')
        qa_entries = TranscriptParser._extract_qa_entries(lines)
        
        followup_questions = [e for e in qa_entries if e['is_follow_up'] and e['speaker'] == 'Bot']
        followup_answers = [e for e in qa_entries if e['is_follow_up'] and e['speaker'] == 'User']
        
        assert len(followup_questions) > 0
        assert len(followup_answers) > 0

    def test_pair_questions_answers(self, sample_transcript):
        """Test Q&A pairing logic"""
        lines = sample_transcript.split('\n')
        qa_entries = TranscriptParser._extract_qa_entries(lines)
        paired = TranscriptParser._pair_questions_answers(qa_entries)
        
        assert len(paired) > 0
        for pair in paired:
            assert 'question' in pair
            assert 'answer' in pair
            assert 'timestamp' in pair
            assert 'is_follow_up_question' in pair
            assert 'is_follow_up_answer' in pair

    def test_pair_unanswered_questions(self):
        """Test pairing when questions don't have answers"""
        transcript = """--- Interview Transcript ---
Session ID: test_session
Candidate ID: test_candidate
Date: 2024-01-15 10:00:00
--------------------------------

[2024-01-15 10:00:00] Assistant:
What is your name?

---
"""
        result = TranscriptParser.parse(transcript)
        
        assert len(result['qa_pairs']) == 1
        assert result['qa_pairs'][0]['answer'] == "No answer provided or recorded"

    def test_multiline_responses(self):
        """Test handling of multiline responses"""
        transcript = """--- Interview Transcript ---
Session ID: test_session
Candidate ID: test_candidate
Date: 2024-01-15 10:00:00
--------------------------------

[2024-01-15 10:00:00] Assistant:
Tell me about your experience.

---

[2024-01-15 10:00:30] User:
I have worked on multiple projects.
First, I developed a web application.
Then, I led a team of developers.

---
"""
        result = TranscriptParser.parse(transcript)
        
        answer = result['qa_pairs'][0]['answer']
        assert 'multiple projects' in answer
        assert 'web application' in answer
        assert 'team of developers' in answer


class TestTranscriptAnalyzer:
    """Test suite for TranscriptAnalyzer"""

    @pytest.fixture
    def mock_gemini(self):
        """Mock Gemini API for transcript analysis"""
        with patch('app.services.transcript_analyzer.genai') as mock_genai:
            mock_model = Mock()
            mock_response = Mock()
            
            analysis_data = {
                "session_id": "test_session_123",
                "user_id": "candidate_001",
                "interview_date": "2024-01-15",
                "questions_analyzed": [
                    {
                        "timestamp": "2024-01-15 10:00:00",
                        "question": "Tell me about yourself",
                        "answer": "I am a software engineer",
                        "relevance_to_resume": "Directly related",
                        "answer_quality": "Good",
                        "technical_accuracy": "Not Applicable",
                        "analysis": "Clear and concise response",
                        "score": 8.0
                    }
                ],
                "overall_summary": "Strong candidate",
                "strengths": ["Clear communication", "Technical knowledge"],
                "weaknesses": ["Could be more specific"],
                "resume_alignment_score": 8.5,
                "communication_score": 8.0,
                "technical_knowledge_score": 7.5,
                "overall_score": 8.0,
                "recommendations": ["Ask about specific projects"]
            }
            
            mock_response.text = json.dumps(analysis_data)
            mock_response.usage_metadata = Mock(
                prompt_token_count=1000,
                candidates_token_count=500,
                total_token_count=1500
            )
            
            mock_model.generate_content.return_value = mock_response
            mock_genai.GenerativeModel.return_value = mock_model
            mock_genai.configure = Mock()
            
            yield mock_genai

    @pytest.fixture
    def transcript_analyzer(self, mock_gemini):
        """Create TranscriptAnalyzer with mocked Gemini"""
        # --- FIX: Patch the module-level variables directly ---
        with patch('app.services.transcript_analyzer.GEMINI_API_KEY', 'fake-api-key'), \
             patch('app.services.transcript_analyzer.TRANSCRIPT_ANALYSIS_MODEL_NAME', 'gemini-test-pro'):
            analyzer = TranscriptAnalyzer()
            return analyzer
        # --- END FIX ---

    def test_initialization(self, transcript_analyzer, mock_gemini):
        """Test analyzer initializes correctly"""
        assert transcript_analyzer.model is not None
        # Check that it used the (mocked) env var model name
        mock_gemini.GenerativeModel.assert_called_with('gemini-test-pro')


    def test_analyze_success(self, transcript_analyzer, sample_resume, sample_transcript):
        """Test successful transcript analysis"""
        parsed = TranscriptParser.parse(sample_transcript)
        
        result = transcript_analyzer.analyze(sample_resume, parsed)
        
        assert isinstance(result, OverallAnalysis)
        assert result.session_id == "test_session_123"
        assert result.overall_score > 0
        assert len(result.strengths) > 0
        assert len(result.weaknesses) > 0

    def test_analyze_empty_resume(self, transcript_analyzer, sample_transcript):
        """Test analysis with empty resume"""
        parsed = TranscriptParser.parse(sample_transcript)
        
        result = transcript_analyzer.analyze("", parsed)
        
        assert isinstance(result, OverallAnalysis)

    def test_analyze_no_qa_pairs(self, transcript_analyzer, sample_resume):
        """Test analysis with no Q&A pairs"""
        invalid_transcript = {
            'metadata': {'session_id': 'test', 'user_id': 'test', 'date': '2024-01-15'},
            'qa_pairs': []
        }
        
        with pytest.raises(ValueError, match="Cannot analyze transcript without Q&A pairs"):
            transcript_analyzer.analyze(sample_resume, invalid_transcript)

    def test_analyze_invalid_response_format(self, transcript_analyzer, sample_resume, sample_transcript, mock_gemini):
        """Test handling of invalid API response format (not JSON)"""
        mock_gemini.GenerativeModel().generate_content.return_value.text = "Not valid JSON"
        
        parsed = TranscriptParser.parse(sample_transcript)
        
        # Should fail in _clean_response
        with pytest.raises(ValueError, match="AI response did not contain a valid JSON object"):
            transcript_analyzer.analyze(sample_resume, parsed)

    def test_analyze_missing_required_keys(self, transcript_analyzer, sample_resume, sample_transcript, mock_gemini):
        """Test handling of response with missing required keys"""
        mock_gemini.GenerativeModel().generate_content.return_value.text = json.dumps({
            "session_id": "test",
            "overall_summary": "Test"
            # Missing "strengths", "weaknesses", "overall_score", etc.
        })
        
        parsed = TranscriptParser.parse(sample_transcript)
        
        with pytest.raises(ValueError, match="missing required keys"):
            transcript_analyzer.analyze(sample_resume, parsed)

    def test_create_analysis_prompt(self, transcript_analyzer, sample_resume, sample_transcript):
        """Test prompt creation"""
        parsed = TranscriptParser.parse(sample_transcript)
        
        prompt = transcript_analyzer._create_analysis_prompt(sample_resume, parsed)
        
        assert isinstance(prompt, str)
        assert 'John Doe' in prompt
        assert 'test_session_123' in prompt
        assert 'JSON' in prompt
        assert len(parsed['qa_pairs']) > 0

    def test_clean_response_with_markdown(self, transcript_analyzer):
        """Test cleaning markdown code blocks from response"""
        response_with_markdown = """```json
    {
        "test": "value"
    }
    ```"""
        
        cleaned = transcript_analyzer._clean_response(response_with_markdown)
        
        assert '```' not in cleaned
        assert 'json' not in cleaned.lower()
        
        # --- FIX: Parse the JSON and check its content ---
        # This is more robust than checking for an exact string
        try:
            data = json.loads(cleaned)
            assert data['test'] == 'value'
        except json.JSONDecodeError:
            pytest.fail(f"The cleaned response is not valid JSON: {cleaned}")
        # --- END FIX ---

    def test_clean_response_no_markdown(self, transcript_analyzer):
        """Test cleaning response without markdown"""
        response_no_markdown = '{"test": "value"}'
        
        cleaned = transcript_analyzer._clean_response(response_no_markdown)
        
        assert cleaned == '{"test": "value"}'

    def test_clean_response_no_json(self, transcript_analyzer):
        """Test cleaning response with no valid JSON"""
        response_no_json = "This is just text without JSON"
        
        with pytest.raises(ValueError, match="did not contain a valid JSON"):
            transcript_analyzer._clean_response(response_no_json)

    def test_score_validation(self, transcript_analyzer, sample_resume, sample_transcript, mock_gemini):
        """Test that scores are validated to be in range (Pydantic)"""
        invalid_scores = {
            "session_id": "test_session_123",
            "user_id": "candidate_001",
            "interview_date": "2024-01-15",
            "questions_analyzed": [],
            "overall_summary": "Test",
            "strengths": [],
            "weaknesses": [],
            "resume_alignment_score": 15.0,  # Invalid: > 10
            "communication_score": -2.0,     # Invalid: < 0
            "technical_knowledge_score": 7.5,
            "overall_score": 8.0,
            "recommendations": []
        }
        
        mock_gemini.GenerativeModel().generate_content.return_value.text = json.dumps(invalid_scores)
        
        parsed = TranscriptParser.parse(sample_transcript)
        
        with pytest.raises(Exception): # Catch Pydantic's validation error
            transcript_analyzer.analyze(sample_resume, parsed)

    def test_token_usage_logging(self, transcript_analyzer, sample_resume, sample_transcript, mock_gemini):
        """Test that token usage is logged"""
        parsed = TranscriptParser.parse(sample_transcript)
        
        with patch('app.services.transcript_analyzer.logger') as mock_logger:
            result = transcript_analyzer.analyze(sample_resume, parsed)
            
            info_calls = [call for call in mock_logger.info.call_args_list]
            token_logged = any('Token' in str(call) for call in info_calls)
            assert token_logged
            # Check for the specific log
            assert any('Gemini Transcript Analysis Tokens' in str(call) for call in info_calls)

    @pytest.mark.parametrize("answer_quality", ["Excellent", "Good", "Fair", "Poor"])
    def test_answer_quality_levels(self, transcript_analyzer, sample_resume, mock_gemini, answer_quality):
        """Test different answer quality levels"""
        transcript_data = {
            'metadata': {
                'session_id': 'test_session',
                'user_id': 'test_user',
                'date': '2024-01-15'
            },
            'qa_pairs': [
                {
                    'timestamp': '2024-01-15 10:00:00',
                    'question': 'Test question',
                    'answer': 'Test answer',
                    'is_follow_up_question': False,
                    'is_follow_up_answer': False
                }
            ]
        }
        
        response_data = {
            "session_id": "test_session",
            "user_id": "test_user",
            "interview_date": "2024-01-15",
            "questions_analyzed": [{
                "timestamp": "2024-01-15 10:00:00",
                "question": "Test question",
                "answer": "Test answer",
                "relevance_to_resume": "Related",
                "answer_quality": answer_quality,
                "technical_accuracy": "Accurate",
                "analysis": "Test analysis",
                "score": 8.0
            }],
            "overall_summary": "Test",
            "strengths": [],
            "weaknesses": [],
            "resume_alignment_score": 8.0,
            "communication_score": 8.0,
            "technical_knowledge_score": 8.0,
            "overall_score": 8.0,
            "recommendations": []
        }
        
        mock_gemini.GenerativeModel().generate_content.return_value.text = json.dumps(response_data)
        
        result = transcript_analyzer.analyze(sample_resume, transcript_data)
        
        assert result.questions_analyzed[0].answer_quality == answer_quality
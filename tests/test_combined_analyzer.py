# tests/test_combined_analyzer.py
import pytest
import unittest # <-- Import unittest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import json
from PIL import Image
import io
import sys, os # Import sys and os

# --- Setup Path ---
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..')) 
sys.path.insert(0, project_root)
# --- End Path Setup ---

from app.services.combined_analyzer import CombinedAnalyzer
from app.models.analysis_schemas import OverallAnalysis, CombinedAnalysisReport


@pytest.fixture
def mock_gemini():
    """Mock Gemini API for behavioral analysis"""
    with patch('app.services.combined_analyzer.genai') as mock_genai:
        mock_model = Mock()
        mock_response = Mock()
        
        # Mock behavioral analysis response
        mock_response.text = """```json
{
    "identity_consistency": {
        "is_consistent": true,
        "confidence_score": 9,
        "observations": "Same person visible in all frames"
    },
    "overall_confidence_score": 8,
    "emotion_analysis": {
        "dominant_emotions": ["confident", "engaged"],
        "emotional_stability_score": 8,
        "emotional_appropriateness_score": 9,
        "emotion_timeline": {
            "beginning": "nervous",
            "middle": "confident",
            "end": "relaxed"
        },
        "anxiety_level": 3,
        "enthusiasm_level": 8
    },
    "eye_contact": {
        "quality_score": 8,
        "consistency_score": 7,
        "percentage_maintained": 75,
        "appropriateness_rating": "good"
    },
    "gaze_behavior": {
        "gaze_shift_frequency": "moderate",
        "average_shifts_per_minute": 5,
        "looking_away_percentage": 25,
        "shift_patterns": ["thinking", "natural"],
        "dishonesty_indicators": 2
    },
    "body_movement": {
        "movement_level": "moderate",
        "fidgeting_score": 3,
        "hand_gesture_naturalness": 8,
        "self_soothing_frequency": "rare",
        "distracting_movements_score": 2
    },
    "posture_composure": {
        "posture_score": 8,
        "engagement_level": 9,
        "professional_presentation": 8,
        "posture_consistency": 7,
        "physical_comfort_score": 8
    },
    "communication_quality": {
        "body_language_clarity": 8,
        "non_verbal_confidence": 8,
        "attentiveness_score": 9
    },
    "red_flags": {
        "detected": false,
        "count": 0,
        "descriptions": []
    },
    "positive_indicators": {
        "count": 5,
        "descriptions": ["Good eye contact", "Professional posture", "Engaged throughout", "Clear body language", "Appropriate enthusiasm"]
    },
    "key_strengths": ["Strong non-verbal communication", "Maintained composure", "Demonstrated engagement"],
    "areas_for_improvement": ["Could maintain eye contact more consistently", "Slight fidgeting when answering technical questions"],
    "interview_readiness_score": 8,
    "hiring_recommendation": "yes",
    "hiring_confidence": 8,
    "summary": "Candidate demonstrated strong interview skills with good engagement and professional presentation.",
    "detailed_observations": "Throughout the interview, the candidate maintained professional posture and demonstrated strong engagement through appropriate body language and eye contact."
}
```"""
        
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model
        mock_genai.configure = Mock()
        yield mock_genai


@pytest.fixture
def combined_analyzer(mock_gemini):
    """Create CombinedAnalyzer instance with mocked dependencies"""
    with patch.dict('os.environ', {'GEMINI_API_KEY': 'test_key'}):
        analyzer = CombinedAnalyzer()
        return analyzer


@pytest.fixture
def mock_snapshots(tmp_path):
    """Create mock snapshot images for testing"""
    snapshot_dir = tmp_path / "data" / "candidate_001" / "session_123" / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    
    # Create 5 mock images
    for i in range(5):
        img = Image.new('RGB', (640, 480), color=(73, 109, 137))
        img.save(snapshot_dir / f"snapshot_{i:04d}.jpg")
    
    return snapshot_dir


@pytest.fixture
def sample_transcript_analysis():
    """Sample transcript analysis result"""
    return OverallAnalysis(
        session_id="session_123",
        user_id="candidate_001",
        interview_date="2024-01-15",
        questions_analyzed=[
            {
                "timestamp": "2024-01-15 10:00:00",
                "question": "Tell me about yourself",
                "answer": "I am a software engineer with 5 years of experience",
                "relevance_to_resume": "Directly related to resume experience",
                "answer_quality": "Good",
                "technical_accuracy": "Not Applicable",
                "analysis": "Clear and concise introduction covering key points",
                "score": 8.0
            }
        ],
        overall_summary="Strong candidate with good technical background",
        strengths=["Clear communication", "Strong technical knowledge"],
        weaknesses=["Could provide more specific examples"],
        resume_alignment_score=8.5,
        communication_score=8.0,
        technical_knowledge_score=7.5,
        overall_score=8.0,
        recommendations=["Probe deeper into project details", "Ask about team leadership experience"]
    )


class TestCombinedAnalyzer:
    """Test suite for CombinedAnalyzer"""

    def test_initialization(self, combined_analyzer):
        """Test analyzer initializes correctly"""
        assert combined_analyzer.model is not None

    def test_behavioral_analysis_success(self, combined_analyzer, mock_snapshots, tmp_path):
        """Test successful behavioral analysis"""
        user_id = "candidate_001"
        session_id = "session_123"
        
        with patch('app.services.combined_analyzer.Path') as mock_path:
            mock_path.return_value = tmp_path / "data" 
            
            result = combined_analyzer.perform_behavioral_analysis(user_id, session_id)
            
            assert result is not None
            assert result['status'] == 'success'
            assert 'metrics' in result
            assert result['metrics']['overall_confidence_score'] == 8

    def test_behavioral_analysis_no_snapshot_dir(self, combined_analyzer, tmp_path):
        """Test behavioral analysis with no snapshot *directory*"""
        user_id = "candidate_002"
        session_id = "session_456"
        
        # Create session dir but *not* the 'snapshots' subdirectory
        session_dir = tmp_path / "data" / user_id / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        
        with patch('app.services.combined_analyzer.Path') as mock_path:
            mock_path.return_value = tmp_path / "data"
            
            result = combined_analyzer.perform_behavioral_analysis(user_id, session_id)
            
            assert result['status'] == 'error'
            assert 'No snapshot directory' in result['error_message']

    def test_behavioral_analysis_no_snapshots(self, combined_analyzer, tmp_path):
        """Test behavioral analysis with an empty snapshot directory"""
        user_id = "candidate_002"
        session_id = "session_456"
        
        # Create an *empty* snapshots directory
        snapshot_dir = tmp_path / "data" / user_id / session_id / "snapshots"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        
        with patch('app.services.combined_analyzer.Path') as mock_path:
            mock_path.return_value = tmp_path / "data"
            
            result = combined_analyzer.perform_behavioral_analysis(user_id, session_id)
            
            assert result['status'] == 'error'
            assert 'No snapshots found' in result['error_message']

    def test_behavioral_analysis_identity_inconsistency(self, combined_analyzer, mock_snapshots, mock_gemini, tmp_path):
        """Test behavioral analysis detects identity inconsistency"""
        mock_gemini.GenerativeModel().generate_content.return_value.text = """```json
{
    "identity_consistency": {
        "is_consistent": false,
        "confidence_score": 8,
        "observations": "Different person appears in later frames"
    },
    "overall_confidence_score": 3,
    "hiring_recommendation": "no",
    "summary": "Test"
}
```"""
        
        user_id = "candidate_001"
        session_id = "session_123"
        
        with patch('app.services.combined_analyzer.Path') as mock_path:
            mock_path.return_value = tmp_path / "data"
            
            result = combined_analyzer.perform_behavioral_analysis(user_id, session_id)
            
            assert result['status'] == 'success'
            metrics = result['metrics']
            assert metrics['identity_consistency']['is_consistent'] is False
            assert 'Different person' in metrics['identity_consistency']['observations']

    def test_combine_analyses_both_present(self, combined_analyzer, sample_transcript_analysis):
        """Test combining both behavioral and transcript analyses"""
        behavioral_result = {
            'status': 'success',
            'metrics': {
                'identity_consistency': {'is_consistent': True, 'observations': 'N/A'},
                'overall_confidence_score': 8.0,
                'key_strengths': ['Good body language'],
                'areas_for_improvement': ['Eye contact'],
                'summary': 'Behavioral analysis shows good engagement'
            }
        }
        
        report = combined_analyzer.combine_analyses(
            behavioral_result=behavioral_result,
            transcript_result=sample_transcript_analysis,
            session_id='session_123',
            candidate_id='candidate_001'
        )
        
        assert isinstance(report, CombinedAnalysisReport)
        assert report.behavioral_score == 8.0
        assert report.transcript_overall_score == 8.0
        assert report.final_weighted_score == 8.0  # (8.0 + 8.0) / 2

    def test_combine_analyses_only_behavioral(self, combined_analyzer):
        """Test combining with only behavioral analysis"""
        behavioral_result = {
            'status': 'success',
            'metrics': {
                'identity_consistency': {'is_consistent': True, 'observations': 'N/A'},
                'overall_confidence_score': 7.0,
                'key_strengths': ['Professional demeanor'],
                'areas_for_improvement': ['Nervous initially'],
                'summary': 'Good overall performance'
            }
        }
        
        report = combined_analyzer.combine_analyses(
            behavioral_result=behavioral_result,
            transcript_result=None,
            session_id='session_123',
            candidate_id='candidate_001'
        )
        
        assert isinstance(report, CombinedAnalysisReport)
        assert report.behavioral_score == 7.0
        assert report.transcript_overall_score == 0.0
        assert report.final_weighted_score == 7.0  # (7.0 + 0) / 1

    def test_combine_analyses_only_transcript(self, combined_analyzer, sample_transcript_analysis):
        """Test combining with only transcript analysis"""
        report = combined_analyzer.combine_analyses(
            behavioral_result=None,
            transcript_result=sample_transcript_analysis,
            session_id='session_123',
            candidate_id='candidate_001'
        )
        
        assert isinstance(report, CombinedAnalysisReport)
        assert report.behavioral_score is None
        assert report.transcript_overall_score == 8.0
        assert report.final_weighted_score == 8.0  # (8.0 + 0) / 1

    def test_combine_analyses_identity_warning(self, combined_analyzer, sample_transcript_analysis):
        """Test combining analyses with identity inconsistency warning"""
        behavioral_result = {
            'status': 'success',
            'metrics': {
                'identity_consistency': {
                    'is_consistent': False,
                    'observations': 'Different person detected'
                },
                'overall_confidence_score': 5,
                'summary': 'Normal summary'
            }
        }
        
        report = combined_analyzer.combine_analyses(
            behavioral_result=behavioral_result,
            transcript_result=sample_transcript_analysis,
            session_id='session_123',
            candidate_id='candidate_001'
        )
        
        assert 'CRITICAL IDENTITY WARNING' in report.behavioral_summary

    def test_combine_analyses_no_data(self, combined_analyzer):
        """Test combining with no valid analysis data"""
        with pytest.raises(ValueError) as e:
            combined_analyzer.combine_analyses(
                behavioral_result={'status': 'error'},
                transcript_result=None,
                session_id='session_123',
                candidate_id='candidate_001'
            )
        assert "No valid analysis data" in str(e.value)

    def test_parse_metrics_response_valid_json(self, combined_analyzer):
        """Test parsing valid JSON metrics response"""
        response_text = """```json
{
    "overall_confidence_score": 8,
    "emotion_analysis": {"dominant_emotions": ["confident"]}
}
```"""
        
        result = combined_analyzer._parse_metrics_response(
            response_text, "c1", "s1", 10
        )
        
        assert result['parsing_status'] == 'success'
        assert result['metrics']['overall_confidence_score'] == 8

    def test_parse_metrics_response_invalid_json(self, combined_analyzer):
        """Test parsing invalid JSON response"""
        response_text = "This is not valid JSON"
        
        result = combined_analyzer._parse_metrics_response(
            response_text, "c1", "s1", 10
        )
        
        assert result['parsing_status'] == 'failed'
        assert 'error' in result

    def test_save_all_behavioral_reports(self, combined_analyzer, tmp_path):
        """Test saving all behavioral reports"""
        
        # --- FIX: Provide a complete metrics dict to prevent helper errors ---
        metrics_data = {
            "identity_consistency": {"is_consistent": True, "confidence_score": 9, "observations": "N/A"},
            "overall_confidence_score": 8,
            "emotion_analysis": {"dominant_emotions": ["confident"], "emotional_stability_score": 8, "emotion_timeline": {}},
            "eye_contact": {"quality_score": 8, "appropriateness_rating": "good"},
            "gaze_behavior": {},
            "body_movement": {},
            "posture_composure": {},
            "communication_quality": {},
            "red_flags": {"detected": False, "count": 0, "descriptions": []},
            "positive_indicators": {"count": 1, "descriptions": ["Good"]},
            "key_strengths": ["Test strength"],
            "areas_for_improvement": ["Test improvement"],
            "interview_readiness_score": 8,
            "hiring_recommendation": "yes",
            "hiring_confidence": 8,
            "summary": "Test summary",
            "detailed_observations": "Test details"
        }
        # --- END FIX ---

        analysis_result = {
            'user_id': 'candidate_001',
            'session_id': 'session_123',
            'analysis_timestamp': '2024-01-15T10:00:00',
            'total_snapshots_analyzed': 10,
            'metadata': {
                'capture_interval_seconds': 5,
                'total_duration_seconds': 50,
                'model_used': 'gemini-2.5-flash',
                'analysis_version': '2.1'
            },
            'metrics': metrics_data, # Use the complete metrics
            'raw_response': 'Test raw response'
        }
        
        # --- FIX: Mock Path and builtins.open ---
        with patch('app.services.combined_analyzer.Path') as mock_path:
            # We mock Path() to return a mock directory
            mock_report_dir = MagicMock(name="report_dir_mock")
            mock_path.return_value.__truediv__().__truediv__().__truediv__.return_value = mock_report_dir
            
            # We also need to mock open()
            with patch("builtins.open", new_callable=unittest.mock.mock_open) as mock_open:
                file_paths = combined_analyzer._save_all_behavioral_reports(
                    'candidate_001',
                    'session_123',
                    analysis_result
                )
            
                # Check that file paths were returned (and not an empty dict)
                assert 'json_complete' in file_paths
                assert 'text_report' in file_paths
                assert 'executive_summary' in file_paths
                assert 'csv_metrics' in file_paths
                
                # Check that directory was created
                mock_report_dir.mkdir.assert_called_with(parents=True, exist_ok=True)
                
                # Check that files were opened
                assert mock_open.call_count >= 7 
        # --- END FIX ---


    def test_metrics_prompt_creation(self, combined_analyzer):
        """Test metrics prompt is created correctly"""
        prompt = combined_analyzer._create_metrics_prompt(10)
        
        assert isinstance(prompt, str)
        assert '10' in prompt
        assert 'identity_consistency' in prompt.lower()
        assert 'json' in prompt.lower()

    @pytest.mark.parametrize("score,expected_clamped,expected_final", [
        (5.5, 5.5, 6.8),  # (5.5 + 8.0) / 2 = 6.75 -> 6.8
        (8.2, 8.2, 8.1),  # (8.2 + 8.0) / 2 = 8.1
        (10.0, 10.0, 9.0), # (10.0 + 8.0) / 2 = 9.0
        (0.0, 0.0, 4.0),   # (0.0 + 8.0) / 2 = 4.0
        (11.0, 10.0, 9.0), # (10.0 + 8.0) / 2 = 9.0
        (-1.0, 0.0, 4.0)   # (0.0 + 8.0) / 2 = 4.0
    ])
    def test_score_clamping(self, combined_analyzer, sample_transcript_analysis, score, expected_clamped, expected_final):
        """Test that scores are properly clamped to valid ranges (0-10)"""
        behavioral_result = {
            'status': 'success',
            'metrics': {
                'identity_consistency': {'is_consistent': True},
                'overall_confidence_score': score
            }
        }
        
        report = combined_analyzer.combine_analyses(
            behavioral_result=behavioral_result,
            transcript_result=sample_transcript_analysis,
            session_id='session_123',
            candidate_id='candidate_001'
        )
        
        assert report.behavioral_score == expected_clamped
        assert report.final_weighted_score == expected_final

    def test_thread_safety(self, combined_analyzer, mock_snapshots, tmp_path):
        """Test analyzer handles concurrent requests safely"""
        import threading
        
        results = []
        
        with patch('app.services.combined_analyzer.Path') as mock_path:
            mock_path.return_value = tmp_path / "data"
        
            def run_analysis(user_id, session_id):
                result = combined_analyzer.perform_behavioral_analysis(user_id, session_id)
                results.append(result)
            
            snapshot_dir_2 = tmp_path / "data" / "candidate_002" / "session_456" / "snapshots"
            snapshot_dir_2.mkdir(parents=True, exist_ok=True)
            img = Image.new('RGB', (10, 10), color='red')
            img.save(snapshot_dir_2 / "snapshot_001.jpg")

            threads = [
                threading.Thread(target=run_analysis, args=("candidate_001", "session_123")),
                threading.Thread(target=run_analysis, args=("candidate_002", "session_456"))
            ]
            
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            
            assert len(results) == 2
            assert results[0]['status'] == 'success'
            assert results[1]['status'] == 'success'
            assert results[0]['user_id'] != results[1]['user_id']
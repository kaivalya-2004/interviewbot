# tests/conftest.py
"""
Pytest configuration and shared fixtures for AI Interviewer Bot tests
"""
import pytest
import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch
import tempfile
import shutil

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="session")
def project_root():
    """Get project root directory"""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def test_data_dir(project_root):
    """Get test data directory"""
    test_dir = project_root / "tests" / "test_data"
    test_dir.mkdir(parents=True, exist_ok=True)
    return test_dir


@pytest.fixture
def temp_dir():
    """Create temporary directory for test files"""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    # Cleanup after test
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def mock_env_vars():
    """Mock environment variables"""
    env_vars = {
        'GEMINI_API_KEY': 'test_gemini_key_12345',
        'GOOGLE_APPLICATION_CREDENTIALS': 'test_credentials.json'
    }
    with patch.dict('os.environ', env_vars, clear=False):
        yield env_vars


@pytest.fixture
def sample_candidate_data():
    """Sample candidate data for testing"""
    return {
        'candidate_id': 'candidate_test_001',
        'name': 'John Doe',
        'email': 'john.doe@example.com',
        'position': 'Software Engineer',
        'experience_years': 5
    }


@pytest.fixture
def sample_session_data():
    """Sample session data for testing"""
    return {
        'session_id': 'session_test_123',
        'candidate_id': 'candidate_test_001',
        'created_at': '2024-01-15T10:00:00',
        'status': 'active',
        'meet_link': 'https://meet.google.com/abc-defg-hij'
    }


@pytest.fixture
def sample_resume_text():
    """Sample resume text for testing"""
    return """John Doe
Software Engineer

PROFESSIONAL SUMMARY
Experienced software engineer with 5+ years in Python development

WORK EXPERIENCE
Senior Software Engineer | Tech Corp | 2020-2024
- Developed scalable web applications using Django and Flask
- Led team of 5 developers on major projects
- Improved system performance by 40%

Software Developer | StartupXYZ | 2018-2020
- Built RESTful APIs using Python and PostgreSQL
- Implemented CI/CD pipelines with Docker and Kubernetes

EDUCATION
Bachelor of Science in Computer Science
University of Technology | 2014-2018
GPA: 3.8/4.0

SKILLS
Programming Languages: Python, JavaScript, SQL
Frameworks: Django, Flask, React
Tools: Docker, Kubernetes, Git, Jenkins
Databases: PostgreSQL, MongoDB, Redis
"""


@pytest.fixture
def sample_questionnaire():
    """Sample interview questionnaire"""
    return [
        "Tell me about your experience with Python",
        "Describe a challenging project you worked on",
        "How do you handle tight deadlines?",
        "What is your experience with cloud platforms?",
        "Tell me about a time you had to debug a complex issue"
    ]


@pytest.fixture
def mock_audio_data():
    """Generate mock audio data for testing"""
    import numpy as np
    # Create 1 second of audio at 24000 Hz
    sample_rate = 24000
    duration = 1.0
    samples = int(sample_rate * duration)
    audio_data = np.random.randint(-32768, 32767, samples, dtype=np.int16)
    return audio_data, sample_rate


@pytest.fixture
def mock_image_data():
    """Generate mock image data for testing"""
    from PIL import Image
    import io
    
    # Create a simple test image
    img = Image.new('RGB', (640, 480), color=(100, 150, 200))
    
    # Convert to bytes
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    
    return img_bytes.read()


@pytest.fixture
def create_mock_snapshot_dir(temp_dir):
    """Factory to create mock snapshot directories with images"""
    def _create_dir(candidate_id, session_id, num_snapshots=5):
        snapshot_dir = temp_dir / "data" / candidate_id / session_id / "snapshots"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        
        from PIL import Image
        for i in range(num_snapshots):
            img = Image.new('RGB', (640, 480), color=(i*50, 100, 150))
            img.save(snapshot_dir / f"snapshot_{i:04d}.jpg")
        
        return snapshot_dir
    
    return _create_dir


@pytest.fixture
def create_mock_transcript(temp_dir):
    """Factory to create mock transcript files"""
    def _create_transcript(candidate_id, session_id, qa_pairs=None):
        transcript_dir = temp_dir / "data" / candidate_id / "transcripts"
        transcript_dir.mkdir(parents=True, exist_ok=True)
        
        transcript_path = transcript_dir / f"transcript_{session_id}.txt"
        
        content = f"""--- Interview Transcript ---
Session ID: {session_id}
Candidate ID: {candidate_id}
Date: 2024-01-15 10:00:00
Questionnaire Provided: Yes (3 questions)
--------------------------------

"""
        
        if qa_pairs:
            for i, (question, answer) in enumerate(qa_pairs, start=1):
                timestamp = f"2024-01-15 10:{i:02d}:00"
                content += f"[{timestamp}] Assistant:\n{question}\n\n---\n\n"
                content += f"[{timestamp}] User:\n{answer}\n\n---\n\n"
        else:
            # Default Q&A pairs
            content += """[2024-01-15 10:00:15] Assistant:
Tell me about yourself.

---

[2024-01-15 10:00:45] User:
I am a software engineer with 5 years of experience.

---
"""
        
        transcript_path.write_text(content, encoding='utf-8')
        return transcript_path
    
    return _create_transcript


@pytest.fixture
def mock_db_session():
    """Mock database session"""
    session = Mock()
    session.add = Mock()
    session.commit = Mock()
    session.query = Mock()
    session.flush = Mock()
    session.refresh = Mock()
    return session


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset singleton instances between tests"""
    # Add any singleton reset logic here if needed
    yield
    # Cleanup after test


@pytest.fixture
def disable_logging(caplog):
    """Disable logging for cleaner test output"""
    import logging
    caplog.set_level(logging.CRITICAL)


@pytest.fixture
def mock_time():
    """Mock time.time() for consistent timing tests"""
    with patch('time.time') as mock:
        mock.return_value = 1705315200.0  # 2024-01-15 10:00:00
        yield mock


@pytest.fixture
def mock_datetime():
    """Mock datetime.now() for consistent datetime tests"""
    from datetime import datetime
    with patch('app.services.interview_service.datetime') as mock:
        mock.now.return_value = datetime(2024, 1, 15, 10, 0, 0)
        mock.side_effect = lambda *args, **kw: datetime(*args, **kw)
        yield mock


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "selenium: marks tests that require selenium/chrome"
    )


def pytest_collection_modifyitems(config, items):
    """Auto-mark tests based on their location"""
    for item in items:
        # Auto-mark integration tests
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        
        # Auto-mark selenium tests
        if "meet_controller" in str(item.fspath) or "meet_session" in str(item.fspath):
            item.add_marker(pytest.mark.selenium)


# Helper functions for tests
def assert_valid_session_id(session_id):
    """Assert session ID is valid format"""
    assert isinstance(session_id, str)
    assert len(session_id) > 0
    assert '_' in session_id or '-' in session_id


def assert_valid_score(score, min_val=0, max_val=10):
    """Assert score is in valid range"""
    assert isinstance(score, (int, float))
    assert min_val <= score <= max_val


def assert_file_exists(file_path):
    """Assert file exists at path"""
    path = Path(file_path)
    assert path.exists(), f"File not found: {file_path}"
    assert path.is_file(), f"Not a file: {file_path}"


def create_test_audio_file(file_path, duration=1.0, sample_rate=16000):
    """Create a test audio file"""
    import numpy as np
    import soundfile as sf
    
    samples = int(sample_rate * duration)
    audio_data = np.random.randn(samples).astype(np.float32) * 0.1
    
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(file_path), audio_data, sample_rate)
    
    return file_path
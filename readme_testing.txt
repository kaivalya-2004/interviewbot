# AI Interviewer Bot - Test Suite

Comprehensive test suite for the AI Interviewer Bot project.

## 📋 Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Running Tests](#running-tests)
- [Test Structure](#test-structure)
- [Writing Tests](#writing-tests)
- [Coverage](#coverage)
- [CI/CD Integration](#cicd-integration)

## 🎯 Overview

This test suite provides comprehensive coverage for:

- **Interview Service**: Core interview logic, Gemini integration, TTS/STT
- **Combined Analyzer**: Behavioral and transcript analysis
- **Transcript Services**: Parsing and analysis of interview transcripts
- **Meet Controller**: Selenium automation for Google Meet
- **API Endpoints**: FastAPI route testing

### Test Types

- **Unit Tests**: Test individual functions and classes in isolation
- **Integration Tests**: Test interactions between components
- **Selenium Tests**: Test browser automation (requires Chrome)

## 🔧 Installation

### 1. Install Test Dependencies

```bash
pip install -r requirements-test.txt
```

### 2. Set Environment Variables

Create a `.env.test` file:

```bash
GEMINI_API_KEY=your_test_api_key
GOOGLE_APPLICATION_CREDENTIALS=path/to/test/credentials.json
```

### 3. Install Chrome/ChromeDriver (for Selenium tests)

```bash
# macOS
brew install chromedriver

# Ubuntu
sudo apt-get install chromium-chromedriver

# Or use webdriver-manager
pip install webdriver-manager
```

## 🚀 Running Tests

### Quick Start

```bash
# Run all tests
pytest

# Or use the test runner script
chmod +x run_tests.sh
./run_tests.sh
```

### Specific Test Categories

```bash
# Unit tests only (fast)
pytest -m unit
# or
./run_tests.sh --unit

# Integration tests
pytest -m integration
./run_tests.sh --integration

# Skip slow tests
pytest -m "not slow"
./run_tests.sh --fast

# Selenium tests (requires Chrome)
pytest -m selenium
./run_tests.sh --selenium
```

### Parallel Execution

```bash
# Run tests in parallel (faster)
pytest -n auto
./run_tests.sh --parallel
```

### Specific Test Files

```bash
# Test specific file
pytest tests/test_interview_service.py

# Test specific class
pytest tests/test_interview_service.py::TestInterviewService

# Test specific function
pytest tests/test_interview_service.py::TestInterviewService::test_initialization
```

### Verbose Output

```bash
# More detailed output
pytest -v

# Even more detailed
pytest -vv
./run_tests.sh --verbose
```

## 📁 Test Structure

```
tests/
├── conftest.py                    # Shared fixtures and configuration
├── test_interview_service.py       # Interview service tests
├── test_combined_analyzer.py       # Analysis service tests
├── test_transcript_services.py     # Transcript parsing/analysis tests
├── test_meet_controller.py         # Selenium/Meet automation tests
├── test_api_endpoints.py           # FastAPI route tests (to be added)
├── integration/                    # Integration tests
│   └── test_full_interview_flow.py
├── test_data/                      # Test fixtures and data files
│   ├── sample_resumes/
│   ├── sample_transcripts/
│   └── sample_snapshots/
└── README.md                       # This file
```

## ✍️ Writing Tests

### Basic Test Structure

```python
import pytest
from unittest.mock import Mock, patch

def test_example():
    """Test description"""
    # Arrange
    service = MyService()
    
    # Act
    result = service.do_something()
    
    # Assert
    assert result == expected_value
```

### Using Fixtures

```python
@pytest.fixture
def sample_data():
    """Provide test data"""
    return {"key": "value"}

def test_with_fixture(sample_data):
    """Test using fixture"""
    assert sample_data["key"] == "value"
```

### Mocking External APIs

```python
@patch('app.services.interview_service.genai')
def test_with_mock(mock_gemini):
    """Test with mocked Gemini API"""
    mock_gemini.GenerativeModel().start_chat().send_message.return_value.text = "Test response"
    
    service = InterviewService()
    result = service.generate_next_question("session_123")
    
    assert result == "Test response"
```

### Parametrized Tests

```python
@pytest.mark.parametrize("input,expected", [
    (1, 2),
    (2, 4),
    (3, 6)
])
def test_multiply(input, expected):
    """Test with multiple inputs"""
    assert input * 2 == expected
```

### Async Tests

```python
@pytest.mark.asyncio
async def test_async_function():
    """Test async function"""
    result = await some_async_function()
    assert result is not None
```

## 📊 Coverage

### Generate Coverage Report

```bash
# Run tests with coverage
pytest --cov=app --cov-report=html

# Open HTML report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

### Coverage Goals

- **Overall Coverage**: > 70%
- **Critical Services**: > 85%
  - interview_service.py
  - combined_analyzer.py
  - transcript_analyzer.py

### View Coverage in Terminal

```bash
pytest --cov=app --cov-report=term-missing
```

## 🔄 CI/CD Integration

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r requirements-test.txt
    
    - name: Run tests
      run: pytest --cov=app --cov-report=xml
      env:
        GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}
    
    - name: Upload coverage
      uses: codecov/codecov-action@v2
```

## 🐛 Debugging Tests

### Run Specific Test with Print Statements

```bash
pytest -s tests/test_interview_service.py::test_specific_function
```

### Drop into Debugger on Failure

```bash
pytest --pdb
```

### Show Local Variables on Failure

```bash
pytest -l
```

## 📝 Best Practices

1. **Keep Tests Independent**: Each test should be able to run in isolation
2. **Use Fixtures**: Share setup code across tests
3. **Mock External Services**: Don't call real APIs in tests
4. **Test Edge Cases**: Include tests for error conditions
5. **Clear Test Names**: Use descriptive test function names
6. **Keep Tests Fast**: Use markers to separate slow tests
7. **Clean Up**: Use fixtures with cleanup or context managers

## 🔍 Common Issues

### Issue: Import Errors

```bash
# Make sure you're in the project root
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Issue: Selenium Tests Fail

```bash
# Install/update ChromeDriver
pip install --upgrade webdriver-manager

# Run without selenium tests
pytest -m "not selenium"
```

### Issue: API Key Not Found

```bash
# Set environment variable
export GEMINI_API_KEY="your_key_here"

# Or create .env file
echo "GEMINI_API_KEY=your_key" > .env
```

## 📞 Support

For issues or questions:
- Open an issue on GitHub
- Check existing test examples
- Review fixture documentation in `conftest.py`

## 📚 Additional Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Unittest Mock Guide](https://docs.python.org/3/library/unittest.mock.html)
- [Coverage.py Documentation](https://coverage.readthedocs.io/)

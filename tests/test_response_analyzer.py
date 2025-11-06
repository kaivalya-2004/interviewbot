# tests/utils/test_response_analyzer.py

import pytest
from app.utils.response_analyzer import (
    detect_non_committal_response,
    detect_repeat_request
)

# --- Tests for detect_non_committal_response ---

@pytest.mark.parametrize("transcript, expected", [
    # Positive Cases (should be True)
    ("I don't know", True),
    ("Well, i do not know about that.", True),
    ("I have no idea.", True),
    ("I'm not sure", True),
    ("I AM NOT SURE!", True),
    ("Honestly, I am confused.", True),
    ("I cant answer that question.", True),
    ("I really have no clue.", True),
    
    # Negative Cases (should be False)
    ("", False),
    ("   ", False),
    ("I am absolutely sure.", False),
    ("I know the answer.", False),
    ("This idea is great.", False),
    ("I can answer that.", False),
    ("The clues are all there.", False),
    ("My confusion is gone.", False)
])
def test_detect_non_committal_response(transcript, expected):
    """
    Tests the detect_non_committal_response function with various inputs.
    """
    assert detect_non_committal_response(transcript) == expected

# --- Tests for detect_repeat_request ---

@pytest.mark.parametrize("transcript, expected", [
    # Positive Cases (should be True)
    ("Could you repeat the question?", True),
    ("Please repeat that.", True),
    ("SAY THAT AGAIN.", True),
    ("Could you repeat yourself?", True),
    ("Pardon?", True),
    ("Sorry, I missed that.", True),
    ("I didn't hear you.", True),
    
    # Negative Cases (should be False)
    ("", False),
    ("   ", False),
    ("I heard you perfectly.", False),
    ("I am not sorry for my answer.", False),
    ("Don't repeat yourself.", False),
    ("I can hear you just fine.", False)
])
def test_detect_repeat_request(transcript, expected):
    """
    Tests the detect_repeat_request function with various inputs.
    """
    assert detect_repeat_request(transcript) == expected
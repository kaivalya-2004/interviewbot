# app/utils/response_analyzer.py
import logging
import re  # Import the regular expression module

logger = logging.getLogger(__name__)

# --- Non-committal Detection ---

# This list is fine as-is
_non_committal_phrases = [
    "i don't know", "i do not know", "no idea", "not sure",
    "i'm not sure", "i am confused", "cant answer", "no clue"
]
# We still use word boundaries \b and re.IGNORECASE
_non_committal_pattern = re.compile(
    r"\b(" + "|".join(re.escape(p) for p in _non_committal_phrases) + r")\b",
    re.IGNORECASE
)

def detect_non_committal_response(transcript: str) -> bool:
    """Detects non-committal phrases using whole-word matching."""
    if not transcript:
        return False
    return _non_committal_pattern.search(transcript) is not None

# --- Repeat Request Detection (FIXED) ---

# We must treat "sorry" as a special case.
# First, list all the phrases that are NOT "sorry":
_repeat_phrases_normal = [
    "repeat the question", "repeat that", "say that again",
    "could you repeat", "pardon", "didn't hear", "did not hear"
]

# 1. Create a pattern for all the normal phrases
pattern_normal = r"\b(" + "|".join(re.escape(p) for p in _repeat_phrases_normal) + r")\b"

# 2. Create a special pattern for "sorry"
# (?<!...) is a "Negative Lookbehind".
# It must be FIXED-WIDTH. We use \s (exactly one space) instead of \s*
# This matches \bsorry\b ONLY IF it is NOT preceded by "not " or "n't ".
# Both 'not\s' and 'n\'t\s' are exactly 4 characters, so this is valid.
pattern_sorry = r"(?<!not\s|n't\s)\bsorry\b"

# 3. Combine both patterns with an OR (|)
_repeat_pattern = re.compile(
    f"({pattern_normal}|{pattern_sorry})",  # Combine the two patterns
    re.IGNORECASE
)

def detect_repeat_request(transcript: str) -> bool:
    """Detects repeat request phrases using whole-word matching."""
    if not transcript:
        return False
    
    # This will now match "Sorry, I missed that"
    # but will NOT match "I am not sorry"
    return _repeat_pattern.search(transcript) is not None
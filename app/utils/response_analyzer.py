# app/utils/response_analyzer.py
import logging

logger = logging.getLogger(__name__)

def detect_non_committal_response(transcript: str) -> bool:
    phrases = ["i don't know", "i do not know", "no idea", "not sure", "i'm not sure", "i am confused", "cant answer", "no clue"]
    return any(p in transcript.lower() for p in phrases)

def detect_repeat_request(transcript: str) -> bool:
    phrases = ["repeat the question", "repeat that", "say that again", "could you repeat", "pardon", "sorry", "didn't hear"]
    return any(p in transcript.lower() for p in phrases)
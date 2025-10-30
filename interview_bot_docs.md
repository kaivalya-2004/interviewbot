# AI Interview Bot - Complete Documentation

## Table of Contents
1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Features](#features)
4. [Installation & Setup](#installation--setup)
5. [User Guide](#user-guide)
6. [Technical Documentation](#technical-documentation)
7. [API Reference](#api-reference)
8. [Testing Guide](#testing-guide)
9. [Troubleshooting](#troubleshooting)
10. [FAQ](#faq)

---

## Overview

### What is the AI Interview Bot?

The AI Interview Bot is an intelligent, real-time interviewing system that conducts automated technical interviews. It uses advanced speech recognition, natural language processing, and text-to-speech technology to create a seamless interview experience.

### Key Capabilities

- **Automated Questioning**: Generates intelligent follow-up questions based on candidate responses and resume content
- **Voice Interaction**: Real-time speech-to-text and text-to-speech capabilities
- **Smart Response Detection**: Identifies when candidates can't answer questions or need clarification
- **Video Analysis** (Optional): Records and analyzes body language during interviews
- **Transcript Generation**: Creates detailed interview transcripts with timestamps
- **Early Termination Logic**: Intelligently ends interviews when candidates consistently can't answer

---

## System Architecture

### Technology Stack

```
┌─────────────────────────────────────────────┐
│          User Interface (Streamlit)          │
├─────────────────────────────────────────────┤
│                                              │
│  ┌──────────────┐      ┌─────────────────┐ │
│  │   Camera     │      │  Audio Recorder │ │
│  │   Handler    │      │                 │ │
│  └──────────────┘      └─────────────────┘ │
│                                              │
├─────────────────────────────────────────────┤
│                                              │
│  ┌──────────────────────────────────────┐  │
│  │    Response Analyzer (NEW)           │  │
│  │  • Non-committal detection           │  │
│  │  • Repeat request detection          │  │
│  └──────────────────────────────────────┘  │
│                                              │
├─────────────────────────────────────────────┤
│                                              │
│  ┌──────────────────────────────────────┐  │
│  │    Interview Service                 │  │
│  │  • Question generation (Gemini AI)   │  │
│  │  • Speech-to-Text (Google)           │  │
│  │  • Text-to-Speech (Kokoro)           │  │
│  └──────────────────────────────────────┘  │
│                                              │
├─────────────────────────────────────────────┤
│                                              │
│  ┌──────────────────────────────────────┐  │
│  │    Database Handler (MongoDB)        │  │
│  │  • Session management                │  │
│  │  • Conversation logging              │  │
│  │  • Video tracking                    │  │
│  └──────────────────────────────────────┘  │
│                                              │
└─────────────────────────────────────────────┘
```

### File Structure

```
interview-bot/
├── app.py                      # Main Streamlit application
├── interview_service.py        # Core interview logic
├── db_handler.py              # MongoDB operations
├── camera_handler.py          # Video recording (not shown)
├── pdf_parser.py              # Resume parsing
├── response_analyzer.py       # NEW: Response analysis module
├── requirements.txt           # Python dependencies
├── .env                       # Environment variables
└── data/                      # Stored interviews
    └── [candidate_id]/
        ├── audio/             # Audio recordings
        ├── video/             # Video recordings
        └── transcripts/       # Interview transcripts
```

---

## Features

### 1. Intelligent Question Generation

The bot generates contextually relevant questions based on:
- Candidate's resume content
- Previous responses in the conversation
- Interview progress (ensures variety across 10 questions)

**Example Flow:**
```
Resume mentions: "Python developer with ML experience"
↓
Bot asks: "Tell me about your experience with Python?"
↓
Candidate mentions: "I worked on a sentiment analysis project"
↓
Bot follows up: "Can you explain the ML algorithms you used?"
```

### 2. Enhanced "I Don't Know" Detection ⭐ NEW

**Problem Solved:**
Traditional systems only detect exact phrases. Our system uses multi-layered detection to identify when candidates genuinely cannot answer.

**Detection Layers:**

#### Layer 1: Explicit Phrase Matching
Detects 40+ variations including:
- "I don't know"
- "I'm not sure"
- "No idea"
- "I can't answer that"
- "I'm confused"
- "I don't remember"

#### Layer 2: Pattern-Based Detection
Combines negative words with uncertainty/action keywords:

| Negative Words | Uncertainty Keywords | Action Keywords |
|---------------|---------------------|-----------------|
| not           | sure                | know            |
| n't           | certain             | remember        |
| no            | confident           | recall          |
| never         | idea                | understand      |
| neither       | clue                | answer          |

**Examples Detected:**
- "I'm **not sure** about that" ✓
- "I have **no idea** how that works" ✓
- "I **can't recall** the details" ✓
- "I'm **not familiar** with that" ✓

#### Layer 3: Confusion Indicators
Standalone words that indicate confusion:
- confused
- unclear
- unsure
- uncertain
- confusing

**Termination Logic:**
- Threshold: **2 non-committal responses**
- Action: Interview ends with message: "Thank you, we are done with the interview."
- Counter resets: Every new interview starts fresh

**Code Example:**
```python
from response_analyzer import detect_non_committal_response

transcript = "I'm not really sure about that topic"
result = detect_non_committal_response(transcript)
# Returns: {"detected": True, "reason": "Pattern match: not sure"}
```

### 3. Repeat Question Functionality ⭐ NEW

**Problem Solved:**
Candidates sometimes miss or don't fully hear questions. Previously, this would count as no answer.

**How It Works:**

1. **User requests repeat** using phrases like:
   - "Can you repeat the question?"
   - "Sorry, I didn't hear you"
   - "Pardon?"
   - "What was that?"

2. **Bot responds:**
   - "Sure, let me repeat that. [ORIGINAL QUESTION]"

3. **System behavior:**
   - ✅ Question repeats with audio
   - ✅ Recording restarts for new answer
   - ❌ Turn counter does NOT increment
   - ❌ "I don't know" counter does NOT increment
   - ❌ No-answer counter does NOT increment

**Supported Phrases (30+ variations):**

| Category | Examples |
|----------|----------|
| Direct requests | "repeat the question", "say that again" |
| Polite requests | "could you repeat", "please repeat" |
| Clarification | "what was the question", "what did you say" |
| Hearing issues | "didn't hear", "couldn't catch" |
| Short forms | "pardon", "sorry", "what", "huh" |

**Code Example:**
```python
from response_analyzer import detect_repeat_request

transcript = "Sorry, can you repeat that?"
result = detect_repeat_request(transcript)
# Returns: {"detected": True, "phrase": "can you repeat"}
```

### 4. Automatic Recording & Silence Detection

**Recording Behavior:**
- Starts automatically after bot finishes speaking
- Stops when:
  - **7 seconds of silence** detected, OR
  - **60 seconds maximum** recording time reached

**Visual Feedback:**
```
🎙️ Recording... (Stops after 7s silence or 54.3s remaining)
```

### 5. Early Termination Conditions

The interview can end early under these conditions:

| Condition | Threshold | Message |
|-----------|-----------|---------|
| Non-committal responses | 2 occurrences | "Thank you, we are done with the interview." |
| No audio responses | 2 occurrences | "Thank you, we are done with the interview." |
| Normal completion | 10 questions asked | "Great, that's all the questions I have for you." |
| Audio issues at 10Q | 10 questions + no answer | "It seems we're having trouble with the audio..." |

**Termination Flow:**
```
Question 1 → User: "I don't know"        [Counter: 1]
Question 2 → User: "I'm not sure"        [Counter: 2]
         ↓
    TERMINATE
         ↓
Bot: "Thank you, we are done with the interview."
```

### 6. Video Recording (Optional)

When enabled:
- Records candidate during their responses
- Logs video files to database
- Syncs with audio recordings
- Prepares data for future body language analysis

### 7. Transcript Generation

**Generated After Each Interview:**
- Full conversation history
- Timestamps for each message
- Speech timing information for user responses
- Interview completion status (COMPLETED or EARLY TERMINATION)

**Example Transcript:**
```
--- Interview Transcript ---
Session ID: 507f1f77bcf86cd799439011
Candidate ID: CAND001
Date: 2025-10-16 14:30:00
Status: EARLY TERMINATION (Due to termination conditions)
--------------------------------

[2025-10-16 14:30:15] Bot:
Hello and welcome to the interview. My name is Avnish...

[2025-10-16 14:30:45] User:
My name is John Doe, I'm a software engineer...
(Speech Timestamps: 2025-10-16T14:30:42 to 2025-10-16T14:30:44)

---
```

---

## Installation & Setup

### Prerequisites

- Python 3.8 or higher
- MongoDB database
- Google Cloud account (for Speech-to-Text)
- Gemini API key
- Microphone and speakers
- (Optional) Webcam

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/interview-bot.git
cd interview-bot
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

**Required packages:**
```
streamlit
pymongo
google-generativeai
speechrecognition
sounddevice
soundfile
pydub
numpy
kokoro
PyPDF2
pdfplumber
python-dotenv
opencv-python  # If using camera
```

### Step 3: Download TTS Model

```bash
# Create models directory
mkdir -p models/kokoro

# Download Kokoro TTS model
# Follow instructions at: https://github.com/hexgrad/kokoro
```

### Step 4: Configure Environment Variables

Create a `.env` file in the root directory:

```env
# MongoDB Configuration
MONGO_URI=mongodb://localhost:27017/

# Gemini AI Configuration
GEMINI_API_KEY=your_gemini_api_key_here

# Optional: Other configurations
TTS_VOICE=af_alloy
MAX_QUESTIONS=10
```

### Step 5: Initialize Database

MongoDB collections are created automatically:
- `ai_interviewer_db.interview_sessions`

### Step 6: Run the Application

```bash
streamlit run app.py
```

Access the application at: `http://localhost:8501`

---

## User Guide

### Starting an Interview

#### Step 1: Setup
1. Open the application in your browser
2. Select your **microphone** from the dropdown
3. (Optional) Enable **camera** for body language analysis
4. Enter a **Candidate ID** (e.g., "CAND001")

#### Step 2: Upload Resume
- Click "Upload Resume"
- Supported formats: **PDF** or **TXT**
- The system will extract and analyze the resume text

#### Step 3: Begin Interview
- Click **"Start Interview"**
- The bot will greet you and ask the first question
- Wait for the bot to finish speaking before answering

### During the Interview

#### Answering Questions
1. **Listen** to the bot's question
2. **Wait** for the recording indicator: 🎙️ Recording...
3. **Speak** your answer clearly
4. **Pause** for 7 seconds to auto-submit your answer

#### If You Need the Question Repeated
Simply say any of:
- "Can you repeat the question?"
- "Sorry, I didn't hear you"
- "Pardon?"
- "Say that again"

The bot will repeat the question and wait for your answer.

#### What to Avoid
❌ Saying "I don't know" repeatedly (causes early termination)
❌ Staying completely silent (causes early termination)
❌ Speaking immediately after the bot (wait for recording indicator)

### Ending the Interview

The interview ends automatically when:
- ✅ 10 questions have been completed
- ⚠️ You say "I don't know" (or similar) twice
- ⚠️ You provide no response twice
- 🔴 You click "End Interview" button

After ending:
- Full transcript is generated
- All recordings are saved
- Session is stored in database

---

## Technical Documentation

### Response Analysis System

#### Architecture

```python
ResponseAnalyzer
├── detect_non_committal_response()
│   ├── Explicit phrase matching
│   ├── Pattern-based detection
│   └── Confusion indicator detection
│
└── detect_repeat_request()
    ├── Direct repeat phrases
    ├── Polite requests
    └── Short form indicators
```

#### Detection Algorithms

##### Non-Committal Detection Algorithm

```
INPUT: transcript (string)
OUTPUT: {detected: bool, reason: string}

STEP 1: Normalize transcript to lowercase

STEP 2: Check explicit phrases
  FOR EACH phrase IN explicit_phrases:
    IF phrase IN transcript:
      RETURN {detected: True, reason: phrase}

STEP 3: Check pattern combinations
  FOR EACH negative_word IN ["not", "n't", "no", "never"]:
    FOR EACH keyword IN [uncertainty_keywords + action_keywords]:
      pattern = negative_word + " " + keyword
      IF pattern IN transcript:
        RETURN {detected: True, reason: pattern}

STEP 4: Check confusion indicators
  FOR EACH indicator IN confusion_indicators:
    IF indicator is standalone word IN transcript:
      RETURN {detected: True, reason: indicator}

STEP 5: No detection
  RETURN {detected: False, reason: None}
```

##### Repeat Request Detection Algorithm

```
INPUT: transcript (string)
OUTPUT: {detected: bool, phrase: string}

STEP 1: Normalize transcript to lowercase

STEP 2: Check repeat phrases
  FOR EACH phrase IN repeat_phrases:
    IF phrase IN transcript:
      RETURN {detected: True, phrase: phrase}

STEP 3: Check short indicators
  IF transcript IN ["what", "pardon", "sorry", "huh", "eh"]:
    RETURN {detected: True, phrase: transcript}

STEP 4: No detection
  RETURN {detected: False, phrase: None}
```

### State Management

#### Session State Variables

| Variable | Type | Purpose | Reset On |
|----------|------|---------|----------|
| `session_id` | str | MongoDB session identifier | New interview |
| `candidate_id` | str | Unique candidate identifier | New interview |
| `messages` | list | Conversation history | New interview |
| `turn_count` | int | Current question number (1-10) | New interview |
| `dont_know_count` | int | Non-committal response counter | New interview |
| `no_answer_count` | int | Silent response counter | New interview |
| `current_question` | str | Last asked question (for repeats) | New interview |
| `interview_active` | bool | Interview in progress flag | End interview |
| `waiting_for_response` | bool | Bot waiting for user | After response |
| `processing` | bool | Processing user response | After processing |

#### State Transitions

```
START
  ↓
[Interview Inactive]
  ↓ (User clicks "Start Interview")
[Interview Active]
  ↓
[Bot asks question] → current_question = question
  ↓
[Bot speaks] → waiting_for_response = True
  ↓
[Recording user]
  ↓
[Processing response] → processing = True
  ↓
  ├─→ [Repeat request] → Repeat question, stay same turn
  ├─→ [Non-committal] → Increment dont_know_count
  ├─→ [No answer] → Increment no_answer_count
  ├─→ [Valid answer] → Continue to next question
  ↓
[Check termination conditions]
  ├─→ [Terminate] → End interview
  └─→ [Continue] → turn_count++, next question
```

### Database Schema

#### Session Document

```javascript
{
  _id: ObjectId("..."),
  candidate_id: "CAND001",
  resume_text: "Full resume text...",
  conversation: [
    {
      role: "bot" | "user",
      text: "Message text",
      timestamp: ISODate("2025-10-16T14:30:00Z"),
      audio_path: "/path/to/audio.wav",
      // For user messages only:
      start_timestamp: ISODate("2025-10-16T14:30:42Z"),
      end_timestamp: ISODate("2025-10-16T14:30:44Z")
    }
  ],
  videos: [
    {
      path: "/path/to/video.mp4",
      turn_count: 1,
      analysis: null | {...}
    }
  ],
  total_tokens: 2500,
  analysis: null | "Interview analysis text",
  analysis_generated_at: null | ISODate("..."),
  created_at: ISODate("2025-10-16T14:00:00Z")
}
```

### Audio Processing Pipeline

```
User speaks
    ↓
[Microphone Input]
    ↓
[AudioRecorder class] → Detects silence/max duration
    ↓
[WAV file saved] → Temporary file
    ↓
[Google Speech-to-Text] → Transcript
    ↓
[Response Analyzer] → Intent detection
    ↓
[Interview Service] → Decision logic
    ↓
[Gemini AI] → Next question generation
    ↓
[Kokoro TTS] → Audio output
    ↓
[Audio playback] → User hears next question
```

---

## API Reference

### ResponseAnalyzer Class

#### `detect_non_committal_response(transcript: str) -> Dict`

Detects if a response indicates inability to answer.

**Parameters:**
- `transcript` (str): The transcribed user response

**Returns:**
- `dict`: 
  - `detected` (bool): True if non-committal response detected
  - `reason` (str): Description of what triggered detection

**Example:**
```python
analyzer = ResponseAnalyzer()
result = analyzer.detect_non_committal_response("I'm not sure")
# Returns: {"detected": True, "reason": "Pattern match: not sure"}
```

#### `detect_repeat_request(transcript: str) -> Dict`

Detects if user wants question repeated.

**Parameters:**
- `transcript` (str): The transcribed user response

**Returns:**
- `dict`:
  - `detected` (bool): True if repeat request detected
  - `phrase` (str): The phrase that triggered detection

**Example:**
```python
analyzer = ResponseAnalyzer()
result = analyzer.detect_repeat_request("Can you repeat that?")
# Returns: {"detected": True, "phrase": "can you repeat"}
```

#### `get_analysis_summary(transcript: str) -> Dict`

Comprehensive analysis of a transcript.

**Returns:**
- `dict`:
  - `transcript` (str): Original transcript
  - `non_committal` (dict): Non-committal detection results
  - `repeat_request` (dict): Repeat request detection results
  - `length` (int): Word count
  - `is_very_short` (bool): True if fewer than 3 words

### InterviewService Class

#### `start_new_interview(resume_text: str, candidate_id: str) -> str`

Creates a new interview session.

**Parameters:**
- `resume_text` (str): Extracted resume content
- `candidate_id` (str): Unique candidate identifier

**Returns:**
- `str`: MongoDB session ID

#### `generate_next_question(session_id: str) -> str`

Generates the next interview question using Gemini AI.

**Parameters:**
- `session_id` (str): Current session identifier

**Returns:**
- `str`: The next question to ask

#### `text_to_speech(text: str, session_id: str, turn_count: int, candidate_id: str) -> str`

Converts text to speech audio file.

**Parameters:**
- `text` (str): Text to convert
- `session_id` (str): Current session
- `turn_count` (int): Question number
- `candidate_id` (str): Candidate identifier

**Returns:**
- `str`: Path to generated audio file

#### `speech_to_text(audio_path: str) -> str`

Transcribes audio file to text.

**Parameters:**
- `audio_path` (str): Path to audio file

**Returns:**
- `str`: Transcribed text

### DBHandler Class

#### `create_session(resume_text: str, candidate_id: str) -> str`

Creates new session document in MongoDB.

#### `add_message_to_session(session_id, role, text, audio_path, start_time, end_time)`

Adds a message to the conversation history.

#### `add_video_to_session(session_id, turn_count, video_path)`

Logs a video recording reference.

#### `get_session(session_id) -> dict`

Retrieves complete session document.

---

## Testing Guide

### Unit Testing

#### Test Non-Committal Detection

```python
def test_non_committal_detection():
    from response_analyzer import detect_non_committal_response
    
    # Test cases
    test_cases = [
        ("I don't know", True),
        ("I'm not sure about that", True),
        ("I have no idea", True),
        ("I'm confused", True),
        ("Let me explain my approach", False),
        ("I worked on Python projects", False)
    ]
    
    for transcript, expected in test_cases:
        result = detect_non_committal_response(transcript)
        assert result == expected, f"Failed for: {transcript}"
    
    print("✓ All non-committal detection tests passed")
```

#### Test Repeat Request Detection

```python
def test_repeat_request_detection():
    from response_analyzer import detect_repeat_request
    
    test_cases = [
        ("Can you repeat the question?", True),
        ("Pardon?", True),
        ("Sorry, I didn't hear you", True),
        ("Yes, I can answer that", False),
        ("Let me think about it", False)
    ]
    
    for transcript, expected in test_cases:
        result = detect_repeat_request(transcript)
        assert result == expected, f"Failed for: {transcript}"
    
    print("✓ All repeat request tests passed")
```

### Integration Testing

#### Test Interview Flow

```python
def test_complete_interview_flow():
    """
    Tests a complete interview from start to finish.
    """
    # 1. Start interview
    # 2. Answer first question normally
    # 3. Request repeat on second question
    # 4. Say "I don't know" on third question
    # 5. Say "Not sure" on fourth question
    # 6. Verify early termination
    
    # Implementation details...
    pass
```

### Manual Testing Checklist

#### Basic Functionality
- [ ] Start new interview with PDF resume
- [ ] Start new interview with TXT resume
- [ ] Answer 10 questions normally (full completion)
- [ ] Click "End Interview" manually

#### Non-Committal Detection
- [ ] Say "I don't know" once → continues
- [ ] Say "I don't know" twice → terminates
- [ ] Say "I'm not sure" twice → terminates
- [ ] Say "I'm confused" → increments counter
- [ ] Mix different non-committal phrases

#### Repeat Functionality
- [ ] Say "Repeat the question" → question repeats
- [ ] Say "Pardon?" → question repeats
- [ ] Say "Didn't hear you" → question repeats
- [ ] Verify turn count doesn't increment
- [ ] Verify counters don't increment
- [ ] Request multiple repeats in a row

#### Edge Cases
- [ ] Stay completely silent → no answer count increments
- [ ] Stay silent twice → early termination
- [ ] Very short answers (1-2 words)
- [ ] Very long answers (>1 minute)
- [ ] Background noise during silence detection
- [ ] Rapid repeated requests

#### Audio/Video
- [ ] Test with different microphones
- [ ] Test audio playback quality
- [ ] Enable camera and verify recording
- [ ] Disable camera and verify audio-only

---

## Troubleshooting

### Common Issues

#### Issue: Microphone Not Detected

**Symptoms:** No microphone appears in dropdown

**Solutions:**
1. Check system microphone permissions
2. Verify microphone is properly connected
3. Restart the application
4. Check `sounddevice` installation:
   ```bash
   python -c "import sounddevice; print(sounddevice.query_devices())"
   ```

#### Issue: Speech Not Transcribed

**Symptoms:** All responses show as "[Unintelligible]"

**Solutions:**
1. Check internet connection (Google Speech-to-Text requires internet)
2. Verify microphone input level
3. Reduce background noise
4. Speak more clearly and louder
5. Check Google Cloud API quota

#### Issue: TTS Not Working

**Symptoms:** No audio playback, error messages about TTS

**Solutions:**
1. Verify Kokoro model is downloaded:
   ```bash
   ls models/kokoro/kokoro-v1.0.onnx
   ```
2. Check model directory structure
3. Review error logs for specific issues
4. Reinstall kokoro package:
   ```bash
   pip install --upgrade kokoro
   ```

#### Issue: Questions Don't Repeat

**Symptoms:** Saying "repeat" doesn't trigger repeat functionality

**Solutions:**
1. Check if phrase is in detection list
2. Verify `current_question` is being stored
3. Check logs for detection messages
4. Ensure clear pronunciation of "repeat"

#### Issue: False Non-Committal Detections

**Symptoms:** Valid answers trigger "I don't know" counter

**Solutions:**
1. Review detection patterns in `response_analyzer.py`
2. Add exceptions for specific phrases
3. Adjust standalone word detection
4. Check transcript accuracy (STT might be wrong)

#### Issue: Interview Doesn't Terminate

**Symptoms:** Saying "I don't know" multiple times doesn't end interview

**Solutions:**
1. Verify counter incrementing in logs
2. Check threshold (should be 2)
3. Ensure detection function is being called
4. Review response processing logic

#### Issue: MongoDB Connection Failed

**Symptoms:** Error: "MONGO_URI not found" or connection timeout

**Solutions:**
1. Verify MongoDB is running:
   ```bash
   # On Linux/Mac
   sudo systemctl status mongod
   
   # On Windows
   net start MongoDB
   ```
2. Check `.env` file has correct MONGO_URI
3. Test connection:
   ```python
   import pymongo
   client = pymongo.MongoClient("mongodb://localhost:27017/")
   client.server_info()  # Should not raise exception
   ```

### Debug Mode

Enable detailed logging:

```python
# Add to top of app.py
import logging
logging.basicConfig(
    level=logging.DEBUG,  # Changed from INFO to DEBUG
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

### Log Analysis

Key log messages to look for:

```
✓ Non-committal response detected: pattern 'not sure'
✓ Repeat request detected: phrase 'can you repeat'
Non-committal response count: 1
Early termination: dont_know=2, no_answer=0
User requested to repeat the question
Generated next follow-up question. Tokens used: 350
```

---

## FAQ

### General Questions

**Q: How many questions does the interview include?**
A: Up to 10 questions, unless early termination occurs.

**Q: Can I pause the interview?**
A: No, interviews run continuously. You can end it manually with the "End Interview" button.

**Q: Where are recordings stored?**
A: In the `data/[candidate_id]/` directory, organized by audio, video, and transcripts.

**Q: Can I customize the termination threshold?**
A: Yes, modify the threshold in `app.py`:
```python
if st.session_state.dont_know_count >= 2:  # Change to desired number
```

### Feature Questions

**Q: What happens if I say "I don't know" and then "repeat the question"?**
A: The "I don't know" counter increments first, then the question repeats. You get a chance to provide a better answer.

**Q: Can I request multiple repeats?**
A: Yes, you can request repeats as many times as needed without penalty.

**Q: Does the bot remember previous answers?**
A: Yes, the bot uses conversation history to generate contextually relevant follow-up questions.

**Q: Can I review my transcript after the interview?**
A: Yes, transcripts are saved in `data/[candidate_id]/transcripts/transcript_[session_id].txt`

**Q: What languages are supported?**
A: Currently English only. The speech recognition and TTS are configured for English.

### Technical Questions

**Q: Can I use a different TTS engine?**
A: Yes, modify the `InterviewService.__init__()` method to use a different TTS provider.

**Q: Can I use a different AI model instead of Gemini?**
A: Yes, replace the Gemini calls in `generate_next_question()` with your preferred LLM API.

**Q: How do I add new detection phrases?**
A: Edit the lists in `response_analyzer.py`:
```python
self.non_committal_explicit_phrases.append("your new phrase")
```

**Q: Can I deploy this to a cloud server?**
A: Yes, but note:
- Requires persistent audio/video permissions
- Streamlit works best with WebRTC for audio
- Consider using cloud-based STT/TTS for better performance

**Q: How do I backup interview data?**
A: MongoDB backup:
```bash
mongodump --db ai_interviewer_db --out backup_folder/
```
File backup: Copy entire `data/` directory

### Customization Questions

**Q: Can I change the bot's voice?**
A: Yes, modify the voice selection in `interview_service.py`:
```python
self.tts_voice = 'af_alloy'  # Change to different Kokoro voice
```

**Q: Can I modify the termination message?**
A: Yes, search for "Thank you, we are done with the interview." in `app.py` and change it.

**Q: Can I add more termination conditions?**
A: Yes, add new logic in the termination check section:
```python
should_terminate = (
    st.session_state.dont_know_count >= 2 or 
    st.session_state.no_answer_count >= 2 or
    st.session_state.your_new_condition  # Add here
)
```

---

## Appendix

### A. Complete Detection Phrase Lists

#### Non-Committal Phrases (40+)
```
i don't know, i do not know, i dont know
don't know, dont know, do not know
no idea, have no idea, i have no idea
not sure, i'm not sure, im not sure
i am not sure, not really sure
i am confused, i'm confused, im confused
don't have an answer, dont have an answer
i don't have an answer, i dont have an answer
i do not have an answer
can't answer, cannot answer, cant answer
i can't answer, i cannot answer, i cant answer
no clue, don't remember, dont remember
do not remember, can't recall, cannot recall
cant recall, i can't recall, i cannot recall
not familiar, unfamiliar, never heard
don't understand, dont understand
i don't understand, i dont understand
```

#### Pattern Combinations
```
Negative Words: not, n't, no, never, neither
Uncertainty: sure, certain, confident, idea, clue
Action: know, remember, recall, understand, answer, think
Confusion: confused, confusing, unclear, unsure, uncertain

Examples: "not sure", "no idea", "can't answer", "don't remember"
```

#### Repeat Request Phrases (30+)
```
Direct:
  repeat the question, repeat that, repeat it
  say that again, say it again, say again

Polite:
  could you repeat, can you repeat, please repeat
  would you repeat, will you repeat
  could you say that again, can you say that again

Clarification:
  what was the question, what did you say
  what did you ask, what was that

Apologetic:
  pardon, sorry, excuse me, come again
  beg your pardon, i beg your pardon

Hearing Issues:
  didn't hear, didnt hear, didn't catch, didnt catch
  couldn't hear, couldnt hear, can't hear, cant hear
  i didn't hear, i didnt hear
  i couldn't hear, i couldnt hear

Repetition:
  one more time, again please, once more
  say that one more time

Short Forms:
  what, huh, eh
```

### B. Configuration Options

#### Environment Variables

Create a `.env` file with these options:

```env
# Required
MONGO_URI=mongodb://localhost:27017/
GEMINI_API_KEY=your_api_key_here

# Optional - Interview Settings
MAX_QUESTIONS=10
DONT_KNOW_THRESHOLD=2
NO_ANSWER_THRESHOLD=2
SILENCE_DURATION=7.0
MAX_RECORDING_DURATION=60.0

# Optional - TTS Settings
TTS_VOICE=af_alloy
TTS_SAMPLE_RATE=24000

# Optional - Model Settings
GEMINI_MODEL=models/gemini-2.5-pro
STT_LANGUAGE=en-US

# Optional - Storage Settings
DATA_DIR=./data
AUDIO_FORMAT=wav
VIDEO_FORMAT=mp4
```

#### Streamlit Configuration

Create `.streamlit/config.toml`:

```toml
[server]
port = 8501
enableCORS = false
enableXsrfProtection = true
maxUploadSize = 200

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#FF4B4B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"
```

### C. Performance Metrics

#### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 2 cores | 4+ cores |
| RAM | 4 GB | 8+ GB |
| Storage | 10 GB | 50+ GB (for recordings) |
| Internet | 1 Mbps | 10+ Mbps |
| Microphone | Any USB/built-in | High-quality USB |
| Camera (optional) | 480p | 720p+ |

#### Performance Benchmarks

| Operation | Average Time | Notes |
|-----------|-------------|-------|
| Start Interview | 2-3 seconds | Includes model loading |
| Question Generation | 1-2 seconds | Depends on Gemini API |
| Speech-to-Text | 1-3 seconds | Depends on audio length |
| Text-to-Speech | 0.5-1 second | Kokoro local processing |
| Response Analysis | <0.1 seconds | Pure Python logic |
| Database Write | <0.1 seconds | Local MongoDB |
| Transcript Generation | 0.5-1 second | Depends on length |

#### Token Usage (Gemini AI)

Approximate tokens per interview:

| Item | Tokens | Cost (at $0.001/1K) |
|------|--------|---------------------|
| Resume analysis | 500-1000 | $0.0005-$0.001 |
| Question generation (×10) | 300-500 each | $0.003-$0.005 |
| Total per interview | 3500-6000 | $0.0035-$0.006 |

### D. Security Considerations

#### Data Privacy

**Personal Information:**
- Resumes contain sensitive personal data
- Audio recordings contain biometric voice data
- Video recordings contain facial data

**Recommendations:**
1. Encrypt data at rest:
   ```python
   # Use cryptography library
   from cryptography.fernet import Fernet
   ```

2. Implement access controls:
   ```python
   # Add authentication to Streamlit
   # Use streamlit-authenticator library
   ```

3. Secure MongoDB:
   ```javascript
   // Enable authentication
   use admin
   db.createUser({
     user: "interview_admin",
     pwd: "secure_password",
     roles: ["readWrite"]
   })
   ```

4. Use HTTPS in production:
   ```bash
   streamlit run app.py --server.sslCertFile=cert.pem --server.sslKeyFile=key.pem
   ```

#### API Key Security

**Never commit `.env` file to version control:**

```bash
# Add to .gitignore
.env
*.pem
*.key
data/
```

**Use environment variables in production:**

```bash
export GEMINI_API_KEY="your_key_here"
export MONGO_URI="mongodb://user:pass@host:port/db"
```

### E. Advanced Customization Examples

#### Custom Detection Logic

```python
# Add to response_analyzer.py

def detect_deflection(self, transcript: str) -> Dict:
    """
    Detects if candidate is deflecting or avoiding the question.
    """
    deflection_phrases = [
        "that's a good question",
        "interesting question",
        "let me think about",
        "it depends",
        "that's complicated"
    ]
    
    transcript_lower = transcript.lower()
    for phrase in deflection_phrases:
        if phrase in transcript_lower:
            return {"detected": True, "phrase": phrase}
    
    return {"detected": False, "phrase": None}
```

#### Custom Question Types

```python
# Add to interview_service.py

def generate_behavioral_question(self, session_id: str) -> str:
    """
    Generates a behavioral (STAR method) question.
    """
    prompt = f"""
    Generate a behavioral interview question using the STAR format
    (Situation, Task, Action, Result). Base it on the candidate's
    background but make it open-ended.
    
    Resume: {resume_text}
    
    Question:
    """
    # Use Gemini to generate...
```

#### Integration with External Systems

```python
# Example: Send results to HR system

def send_to_hr_system(self, session_id: str):
    """
    Sends interview results to external HR system.
    """
    import requests
    
    session = self.db.get_session(session_id)
    
    payload = {
        "candidate_id": session["candidate_id"],
        "interview_date": session["created_at"],
        "status": "COMPLETED",
        "transcript_url": f"/transcripts/{session_id}.txt"
    }
    
    response = requests.post(
        "https://hr-system.company.com/api/interviews",
        json=payload,
        headers={"Authorization": "Bearer TOKEN"}
    )
    
    return response.status_code == 200
```

### F. Troubleshooting Decision Tree

```
Problem: Interview not starting
├─ Resume upload fails?
│  ├─ YES → Check file format (PDF/TXT only)
│  └─ NO → Continue
├─ Candidate ID missing?
│  ├─ YES → Enter valid ID
│  └─ NO → Continue
├─ Error message shown?
│  ├─ YES → Check MongoDB connection
│  └─ NO → Check browser console logs

Problem: Audio not working
├─ Can't hear bot?
│  ├─ YES → Check speakers/volume
│  │        Check TTS model installation
│  └─ NO → Continue
├─ Bot can't hear you?
│  ├─ YES → Check microphone selection
│  │        Check microphone permissions
│  │        Test with: sounddevice.query_devices()
│  └─ NO → Continue
├─ Transcript shows [Unintelligible]?
│  └─ YES → Check internet connection
│           Speak more clearly
│           Reduce background noise

Problem: Detection not working
├─ Repeat not detected?
│  ├─ YES → Check if phrase in list
│  │        Check pronunciation
│  │        Enable debug logging
│  └─ NO → Continue
├─ "I don't know" not detected?
│  ├─ YES → Check transcription accuracy
│  │        Add phrase to list
│  │        Check counter in logs
│  └─ NO → Works correctly!
```

### G. Migration Guide

#### Upgrading from Previous Version

If you're upgrading from a version without the new detection features:

**Step 1: Backup Data**
```bash
# Backup MongoDB
mongodump --db ai_interviewer_db --out ./backup

# Backup files
cp -r data/ data_backup/
```

**Step 2: Update Code**
```bash
git pull origin main
pip install -r requirements.txt --upgrade
```

**Step 3: Update Database Schema**

No schema changes required - the new features use existing fields.

**Step 4: Test New Features**
```bash
streamlit run app.py
# Test repeat functionality
# Test enhanced detection
```

**Step 5: Update Configuration**
```bash
# Add to .env if needed
echo "DONT_KNOW_THRESHOLD=2" >> .env
echo "NO_ANSWER_THRESHOLD=2" >> .env
```

### H. Contribution Guidelines

#### Adding New Detection Phrases

1. **Edit response_analyzer.py**:
```python
self.non_committal_explicit_phrases.append("your phrase")
```

2. **Test the phrase**:
```python
def test_new_phrase():
    result = detect_non_committal_response("your phrase")
    assert result == True
```

3. **Update documentation** in this file

4. **Submit pull request** with:
   - Description of new phrases
   - Test cases
   - Real-world examples

#### Reporting Bugs

**Include:**
- Python version
- Streamlit version
- Operating system
- Error messages/logs
- Steps to reproduce
- Expected vs actual behavior

**Template:**
```markdown
**Bug Description:**
Brief description

**Environment:**
- Python: 3.9.5
- Streamlit: 1.28.0
- OS: Ubuntu 20.04

**Steps to Reproduce:**
1. Start interview
2. Say "phrase that causes bug"
3. Observe error

**Expected:** Should work normally
**Actual:** Error occurs

**Logs:**
```
[paste relevant logs here]
```
```

### I. Roadmap

#### Planned Features

**Version 2.0 (Q1 2026)**
- [ ] Multi-language support
- [ ] Real-time body language analysis
- [ ] Sentiment analysis during responses
- [ ] Custom question templates
- [ ] Interview scheduling system

**Version 2.1 (Q2 2026)**
- [ ] AI-powered resume screening
- [ ] Automated interview scoring
- [ ] Comparison with job requirements
- [ ] Video analysis feedback
- [ ] Integration with ATS systems

**Version 3.0 (Q3 2026)**
- [ ] Multi-interviewer support
- [ ] Live collaboration features
- [ ] Advanced analytics dashboard
- [ ] Mobile app support
- [ ] Cloud deployment templates

### J. Resources

#### Official Documentation
- [Streamlit Docs](https://docs.streamlit.io/)
- [Google Gemini AI](https://ai.google.dev/docs)
- [MongoDB Manual](https://docs.mongodb.com/manual/)
- [Kokoro TTS](https://github.com/hexgrad/kokoro)

#### Community
- GitHub Issues: [Report bugs and request features]
- Discussions: [Ask questions and share ideas]
- Discord: [Join our community]

#### Training Materials
- Video Tutorial: [Getting Started with Interview Bot]
- Blog Posts: [Best Practices for AI Interviews]
- Case Studies: [Success Stories]

---

## Changelog

### Version 1.1.0 (October 2025)
**New Features:**
- ✨ Enhanced "I don't know" detection with 40+ phrase variations
- ✨ Pattern-based detection for non-committal responses
- ✨ "Repeat the question" functionality
- ✨ Smart request detection with 30+ variations
- ✨ New ResponseAnalyzer module for better code organization

**Improvements:**
- 📈 Increased question limit from 5 to 10
- 🔍 More accurate intent detection
- 📝 Better transcript status tracking
- 🪵 Enhanced logging for debugging

**Bug Fixes:**
- 🐛 Fixed turn counter incrementing on repeats
- 🐛 Fixed false positives in detection
- 🐛 Improved silence detection accuracy

### Version 1.0.0 (Initial Release)
- Basic interview functionality
- Audio recording and transcription
- Question generation with Gemini AI
- MongoDB session storage
- Simple "I don't know" detection
- Video recording support

---

## License

This project is licensed under the MIT License.

```
MIT License

Copyright (c) 2025 [Your Organization]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## Support

### Getting Help

**For technical issues:**
- Check the [Troubleshooting](#troubleshooting) section
- Search existing GitHub issues
- Create a new issue with details

**For questions:**
- Review the [FAQ](#faq) section
- Ask in community discussions
- Contact support: support@interview-bot.com

**For security issues:**
- Email: security@interview-bot.com
- Do NOT create public issues for security vulnerabilities

### Contact Information

- **Website**: https://interview-bot.com
- **Email**: contact@interview-bot.com
- **GitHub**: https://github.com/yourorg/interview-bot
- **Twitter**: @InterviewBot

---

## Acknowledgments

**Contributors:**
- Original concept and development
- Community feedback and testing
- Open-source libraries used

**Special Thanks:**
- Streamlit team for the excellent framework
- Google for Gemini AI and Speech-to-Text APIs
- Kokoro team for the TTS model
- MongoDB for database solutions
- All beta testers and early adopters

---

**Last Updated:** October 16, 2025  
**Version:** 1.1.0  
**Document Version:** 1.0

---

*For the latest documentation, visit: https://docs.interview-bot.com*
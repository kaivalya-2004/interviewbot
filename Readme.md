# AI Interviewer - Automated Interview System

A comprehensive AI-powered interview platform that conducts live interviews via Google Meet, captures behavioral analysis through video snapshots, transcribes conversations, and generates detailed combined analysis reports.

## 🌟 Features

### Core Capabilities
- **Automated Google Meet Integration**: Bot joins meetings, conducts interviews, and manages audio/video
- **Real-time Interview Orchestration**: AI-driven question generation with dynamic follow-ups
- **Dual Analysis System**:
  - **Behavioral Analysis**: Computer vision analysis of candidate body language, emotions, eye contact, and posture
  - **Transcript Analysis**: NLP-based evaluation of responses, technical knowledge, and communication skills
- **Combined Reporting**: Unified analysis merging behavioral and conversational insights
- **Resume-Driven Questions**: Questions tailored to candidate's resume and optional questionnaire
- **Identity Verification**: Ensures consistent participant throughout interview
- **Multi-Participant Detection**: Automatically terminates if extra participants join

### Technical Highlights
- FastAPI backend with async support
- MongoDB for session management
- Google Gemini AI (2.5-Flash & 2.5-Pro models)
- VB-Audio Cable integration for real audio capture
- Undetected ChromeDriver for Meet automation
- Thread-safe concurrent analysis processing

## 📋 Prerequisites

### System Requirements
- **Python**: 3.9 or higher
- **Operating System**: Windows/Linux/macOS
- **RAM**: 8GB minimum (16GB recommended)
- **Storage**: 5GB free space

### Required Accounts & Keys
1. **Google Cloud Platform**
   - Gemini API Key (for AI models)
   - Text-to-Speech API credentials
2. **MongoDB** 
   - Atlas cluster or local instance
3. **VB-Audio Cable** (for real audio capture)
   - Download: [VB-Audio Virtual Cable](https://vb-audio.com/Cable/)

## 🚀 Installation

### 1. Clone Repository
```bash
git clone <repository-url>
cd ai-interviewer
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

**Key Dependencies:**
```
fastapi>=0.104.1
uvicorn[standard]>=0.24.0
google-generativeai>=0.3.0
google-cloud-texttospeech>=2.14.0
pymongo>=4.6.0
undetected-chromedriver>=3.5.4
selenium>=4.15.0
sounddevice>=0.4.6
soundfile>=0.12.1
librosa>=0.10.1
speech-recognition>=3.10.0
Pillow>=10.1.0
pydantic>=2.5.0
python-dotenv>=1.0.0
PyPDF2>=3.0.0
pdfplumber>=0.10.0
```

### 4. Environment Configuration

Create `.env` file in project root:

```env
# Google AI
GEMINI_API_KEY=your_gemini_api_key_here
GOOGLE_APPLICATION_CREDENTIALS=path/to/google-cloud-credentials.json

# MongoDB
MONGO_URI=your mongodb uri.

# Model Configuration
TRANSCRIPT_ANALYSIS_MODEL=gemini-2.5-pro

# Optional: Audio Device Indices (check with sounddevice.query_devices())
VIRTUAL_OUTPUT_DEVICE=<device_index>
VIRTUAL_INPUT_DEVICE=<device_index>
```

### 5. Google Cloud Setup

#### A. Enable APIs
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Enable:
   - Gemini API (AI Studio)
   - Text-to-Speech API

#### B. Create Service Account (for TTS)
```bash
# Download JSON credentials and save to project root
# Set path in .env: GOOGLE_APPLICATION_CREDENTIALS=./your-credentials.json
```

### 6. VB-Audio Cable Setup

**Windows:**
1. Install VB-Audio Cable
2. Set as default recording device in Windows Sound Settings
3. Configure your TTS output to route to "VB-Audio Input"

**Find Device Indices:**
```python
import sounddevice as sd
print(sd.query_devices())
# Note the index numbers for VB-Audio Input/Output
```

### 7. MongoDB Setup

**Option A: MongoDB Atlas (Cloud)**
1. Create free cluster at [MongoDB Atlas](https://www.mongodb.com/cloud/atlas)
2. Whitelist your IP
3. Create database user
4. Get connection string → add to `.env`

**Option B: Local MongoDB**
```bash
# Install MongoDB Community Edition
# Start service
mongod --dbpath /path/to/data

# Connection string
MONGO_URI=mongodb://localhost:27017/ai_interviewer_db
```

## 🎮 Usage

### Starting the API Server

```bash
python main_api.py
```

Server runs at: `http://127.0.0.1:8000`  
API Docs: `http://127.0.0.1:8000/docs`

### API Endpoints

#### 1. Start Interview
```bash
POST /interview/start-google-meet

Form Data:
- candidate_id: string (required)
- meet_link: string (required) 
- resume: file (required, .pdf or .txt)
- questionnaire_json: string (optional, JSON array)
- audio_device: int (optional)
- enable_video: bool (default: true)
- video_capture_method: string (default: "javascript")

Response:
{
  "status": "pending",
  "session_id": "65f1a2b3c4d5e6f7a8b9c0d1",
  "message": "Interview session scheduled",
  "questions_in_session": 5
}
```

**Example with curl:**
```bash
curl -X POST "http://127.0.0.1:8000/interview/start-google-meet" \
  -F "candidate_id=john_doe_001" \
  -F "meet_link=https://meet.google.com/abc-defg-hij" \
  -F "resume=@resume.pdf" \
  -F "questionnaire_json=[\"Tell me about your experience\", \"Describe a challenging project\"]" \
  -F "enable_video=true"
```

#### 2. Check Status
```bash
GET /interview/{session_id}/status

Response:
{
  "session_id": "...",
  "candidate_id": "john_doe_001",
  "bot_status": "active",
  "questionnaire_length": 5,
  "video_enabled": true,
  "snapshots_captured": 42,
  "conversation_length": 12
}
```

#### 3. Get Snapshot Info
```bash
GET /interview/{session_id}/snapshots
```

#### 4. End Interview
```bash
POST /interview/{session_id}/end
```

#### 5. Health Check
```bash
GET /health
```

### Interview Flow

1. **Bot joins Google Meet** using provided link
2. **Waits for candidate** (2-minute timeout with rejoin support)
3. **Conducts interview**:
   - Asks initial greeting
   - Records candidate introduction
   - Generates questions based on resume/questionnaire
   - Records responses (30s max per answer)
   - Monitors for drops (<2 participants) or extra participants (>2)
4. **Captures snapshots** every 5 seconds (if video enabled)
5. **Ends gracefully** after duration/question limit
6. **Generates transcript** with timestamps
7. **Runs dual analysis** (behavioral + transcript) in parallel
8. **Combines results** into unified report

### Interview Termination Conditions

- **Time limit reached** (default: 10 minutes)
- **Max questions asked** (default: 10)
- **Manual termination** via API
- **Multiple participants detected** (>2 in call)
- **Candidate left and didn't rejoin** within 2 minutes

## 📁 Project Structure

```
ai-interviewer/
├── app/
│   ├── core/
│   │   ├── services/
│   │   │   └── database_service.py      # MongoDB handler
│   │   ├── log_config.py                # Logging setup
│   │   └── safe_chrome_driver.py        # Chrome wrapper
│   ├── models/
│   │   └── analysis_schemas.py          # Pydantic models
│   ├── services/
│   │   ├── combined_analyzer.py         # Unified analysis service
│   │   ├── interview_service.py         # Core interview logic
│   │   ├── meet_controller.py           # Browser automation
│   │   ├── meet_session_manager.py      # Session orchestration
│   │   ├── meet_interview_orchestrator.py # Interview flow
│   │   ├── transcript_parser.py         # Parse transcripts
│   │   └── transcript_analyzer.py       # NLP analysis
│   ├── utils/
│   │   ├── pdf_parser.py                # Resume parsing
│   │   ├── response_analyzer.py         # Response detection
│   │   └── metrics_visualizer.py        # Optional: Charts
│   └── config/
│       └── audio_config.py              # Audio device config
├── data/                                # Generated data
│   └── {candidate_id}/
│       └── {session_id}/
│           ├── snapshots/               # Video frames
│           ├── audio/                   # Audio files
│           ├── transcripts/             # Text transcripts
│           ├── behavioral_analysis/     # Behavioral reports
│           └── final_report/            # Combined analysis
├── logs/                                # Application logs
├── chrome_profile/                      # Persistent Chrome data
├── main_api.py                          # FastAPI application
├── requirements.txt                     # Python dependencies
├── .env                                 # Environment variables
└── README.md                            # This file
```

## 📊 Output Files

After each interview, the system generates:

### Transcripts (`data/{candidate_id}/transcripts/`)
- `transcript_{session_id}.txt` - Full conversation log with timestamps

### Behavioral Analysis (`data/{candidate_id}/{session_id}/behavioral_analysis/`)
- `executive_summary.txt` - Quick overview
- `report_formatted.txt` - Detailed human-readable report
- `analysis_complete.json` - Full structured data
- `metrics_only.json` - Simplified metrics
- `metrics_data.csv` - Tabular format
- `raw_response.txt` - Gemini raw output

### Snapshots (`data/{candidate_id}/{session_id}/snapshots/`)
- `snapshot_{timestamp}.jpg` - Captured every 5 seconds

### Final Report (`data/{candidate_id}/{session_id}/final_report/`)
- `combined_analysis_{session_id}.json` - Unified analysis with:
  - Behavioral scores & summary
  - Transcript scores (overall, communication, technical)
  - Combined strengths/weaknesses/recommendations
  - Final weighted score

## 🔧 Configuration

### Interview Parameters (in API call)
```python
interview_duration_minutes = 10  # Total interview time
max_questions = 10               # Question limit
enable_video = True              # Capture behavioral data
video_capture_method = "javascript"  # or "screenshot"
```

### Audio Settings (`app/config/audio_config.py`)
```python
VirtualAudioConfig.get_virtual_output_device()  # TTS output
VirtualAudioConfig.get_virtual_input_device()   # Candidate input
```

### Gemini Models
- **Chat**: `gemini-2.5-flash` (fast, cost-effective)
- **Analysis**: `gemini-2.5-pro` (high-quality insights)
- **TTS**: Google Cloud TTS (en-IN-Chirp3-HD-Alnilam)

### Chrome Options (headless mode)
Edit `meet_controller.py`:
```python
MeetController(headless=True)  # Set False for debugging
```

## 🐛 Troubleshooting

### Common Issues

**1. "GEMINI_API_KEY not found"**
```bash
# Check .env file exists and contains:
GEMINI_API_KEY=your_key_here
```

**2. Chrome driver errors**
```bash
pip install --upgrade undetected-chromedriver
# Or manually download ChromeDriver matching your Chrome version
```

**3. "Audio device not found"**
```python
# Run to list devices:
import sounddevice as sd
print(sd.query_devices())

# Update .env with correct indices:
VIRTUAL_OUTPUT_DEVICE=5
VIRTUAL_INPUT_DEVICE=6
```

**4. MongoDB connection failed**
```bash
# Test connection:
mongosh "mongodb+srv://cluster.mongodb.net/" --username <user>

# Check firewall/IP whitelist in Atlas
```

**5. "Failed to join meeting"**
- Verify meet link is valid and accessible
- Check Chrome can access camera/microphone (even in headless)
- Ensure VB-Audio Cable is installed and set as default

**6. No snapshots captured**
```bash
# Check video element detection:
# Set headless=False temporarily to debug visually
# Ensure candidate video is visible in Meet
```

**7. Transcription errors**
- Verify Google Speech Recognition API is accessible
- Check internet connection stability
- Ensure audio quality is sufficient

### Debug Mode

**Enable verbose logging:**
```python
# In log_config.py, change:
'level': 'DEBUG'  # Instead of 'INFO'
```

**Run without headless:**
```python
# In API call:
headless = False
# Watch browser automation live
```

## 📈 Performance Metrics

### Typical Interview Statistics
- **Duration**: 8-12 minutes
- **Questions**: 8-12 questions
- **Snapshots**: 100-150 frames (5s interval)
- **Transcript**: 2,000-5,000 words
- **Analysis Time**: 30-60 seconds (concurrent)

### Token Usage (Gemini API)
- **Chat (per interview)**: ~5,000-10,000 tokens
- **Transcript Analysis**: ~8,000-15,000 tokens
- **Behavioral Analysis**: ~3,000-6,000 tokens

### Storage Requirements
- **Per Interview**: 50-150 MB
  - Snapshots: 30-100 MB
  - Audio: 10-20 MB
  - Reports: 1-5 MB

## 🔒 Security Considerations

1. **API Keys**: Never commit `.env` to version control
2. **MongoDB**: Use authentication and IP whitelisting
3. **Chrome Profiles**: Persistent profiles may store credentials
4. **Data Privacy**: Snapshots contain PII - implement retention policies
5. **Meet Links**: Validate format to prevent injection attacks

## 🚦 Rate Limits

### Google APIs
- **Gemini**: 60 requests/minute (free tier)
- **TTS**: 5,000 characters/minute
- **Speech-to-Text**: 60 seconds audio/minute

### Recommendations
- Use Gemini paid tier for production
- Implement request queuing for high volume
- Cache TTS audio for repeated phrases

## 🤝 Contributing

### Development Setup
```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests (if available)
pytest tests/

# Format code
black app/
```

### Code Standards
- Follow PEP 8
- Add type hints
- Document functions with docstrings
- Log important events with appropriate levels

## 📜 License

[Specify your license here - e.g., MIT, Apache 2.0]

## 👥 Support

- **Issues**: [GitHub Issues](your-repo-url/issues)
- **Docs**: [API Documentation](http://127.0.0.1:8000/docs)
- **Email**: [your-email@domain.com]

## 🗺️ Roadmap

- [ ] Streamlit UI for non-technical users
- [ ] Multi-language support (i18n)
- [ ] Real-time analysis during interview
- [ ] Custom scoring models
- [ ] Interview replay functionality
- [ ] Integration with ATS systems
- [ ] Sentiment analysis timeline
- [ ] Automated highlight extraction
- [ ] Compliance & GDPR features

## 📝 Changelog

### Version 2.1 (Current)
- ✅ Combined analyzer with identity verification
- ✅ Concurrent transcript + behavioral analysis
- ✅ Improved participant counting logic
- ✅ Robust drop/rejoin handling
- ✅ Better error handling and logging

### Version 2.0
- ✅ Behavioral analysis with metrics
- ✅ Transcript NLP analysis
- ✅ Meet automation with video capture

### Version 1.0
- ✅ Basic interview flow
- ✅ Speech-to-text transcription
- ✅ MongoDB session management

---

**Built with ❤️ using Google Gemini, FastAPI, and Python**

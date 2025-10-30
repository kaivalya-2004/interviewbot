# main_api.py
import logging
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from typing import Optional, Any, Dict, List
import uvicorn
import io
import json
import threading
from pathlib import Path

# --- Import Services ---
from app.services.interview_service import InterviewService
from app.services.behavioral_analyzer import BehavioralAnalyzer
from app.services.transcript_parser import TranscriptParser
from app.services.transcript_analyzer import TranscriptAnalyzer
from app.services.combined_analyzer import combine_analyses
from app.core.services.database_service import DBHandler
from app.services.meet_session_manager import MeetSessionManager
from app.services.meet_interview_orchestrator import MeetInterviewOrchestrator
from app.utils.pdf_parser import parse_resume_pdf # Assuming this exists
from app.models.analysis_schemas import CombinedAnalysisReport, OverallAnalysis

# --- Setup Logging ---
from app.core.log_config import setup_logging
setup_logging()
logger = logging.getLogger(__name__)

# --- Initialize FastAPI App ---
app = FastAPI( title="AI Interviewer API", description="Conducts interviews and provides combined analysis." )

# --- Service Initialization ---
# (Keep initialization try-except block as before)
db_handler: Optional[DBHandler] = None
interview_service: Optional[InterviewService] = None
behavioral_analyzer: Optional[BehavioralAnalyzer] = None
transcript_parser: Optional[TranscriptParser] = None
transcript_analyzer: Optional[TranscriptAnalyzer] = None
meet_session_mgr: Optional[MeetSessionManager] = None
meet_orchestrator: Optional[MeetInterviewOrchestrator] = None

try:
    db_handler = DBHandler()
    interview_service = InterviewService()
    behavioral_analyzer = BehavioralAnalyzer()
    transcript_parser = TranscriptParser()
    transcript_analyzer = TranscriptAnalyzer()
    meet_session_mgr = MeetSessionManager(db_handler)
    meet_orchestrator = MeetInterviewOrchestrator( meet_session_mgr, interview_service )
    logger.info("✅ All services initialized successfully.")
except Exception as e:
    logger.critical(f"❌ CRITICAL: Failed to initialize services: {e}", exc_info=True)


# --- Helper Function to run analysis in thread ---
def run_analysis_in_thread(target_func, args_tuple, results_dict, key_name):
    """Runs a target function in a thread and stores result/error in dict."""
    try:
        logger.info(f"Starting analysis thread for '{key_name}'...")
        result = target_func(*args_tuple)
        results_dict[key_name] = result
        logger.info(f"Analysis thread '{key_name}' completed successfully.")
    except Exception as e:
        logger.error(f"Error in analysis thread '{key_name}': {e}", exc_info=True)
        results_dict[key_name] = {"status": "error", "detail": str(e)}

# --- Background Task ---
def start_and_conduct_interview_task(
    session_id: str, candidate_id: str, meet_link: str, audio_device: Optional[int],
    enable_video: bool, video_capture_method: str, resume_content: str
):
    """ Background task for the full interview and analysis pipeline. """
    logger.info(f"[Task: {session_id}] Background task started")
    interview_result = None; transcript_analysis_result = None; behavioral_analysis_result = None; final_report_path = None
    snapshot_count = 0 # Initialize snapshot count

    try:
        # --- Service Check ---
        if not all([meet_session_mgr, meet_orchestrator, interview_service, behavioral_analyzer, transcript_analyzer, transcript_parser]):
            logger.error(f"[Task: {session_id}] Essential services missing. Aborting."); return

        # --- 1. Start Bot & Join Meet ---
        logger.info(f"[Task: {session_id}] Starting bot...")
        success = meet_session_mgr.start_bot_session( session_id, meet_link, candidate_id, audio_device, enable_video, True, video_capture_method )
        if not success: logger.error(f"[Task: {session_id}] Bot failed to join Meet."); return
        logger.info(f"[Task: {session_id}] ✅ Bot joined Meet.")

        # --- 2. Wait for Candidate ---
        logger.info(f"[Task: {session_id}] ⏳ Waiting for candidate...");
        candidate_joined = meet_session_mgr.wait_for_candidate(session_id, timeout=300)
        if not candidate_joined: logger.warning(f"[Task: {session_id}] Candidate didn't join."); return # Cleanup in finally

        logger.info(f"[Task: {session_id}] ✅ Candidate joined.")

        # --- 3. Start Video Capture ---
        if enable_video: logger.info(f"[Task: {session_id}] 📸 Starting video capture..."); meet_session_mgr._start_candidate_video_capture(session_id)

        # --- 4. Conduct Interview ---
        logger.info(f"[Task: {session_id}] Starting interview orchestration...")
        interview_result = meet_orchestrator.conduct_interview(session_id=session_id)
        logger.info(f"[Task: {session_id}] Interview finished: {interview_result.get('status', 'unknown')}")

        # --- 5. Get Final Snapshot Count BEFORE Ending Session ---
        # Get the count while the session might still be technically active to retrieve stats
        snapshot_count = meet_session_mgr.get_snapshot_count(session_id)
        capture_stats = meet_session_mgr.get_capture_stats(session_id) # Attempt to get stats too
        logger.info(f"[Task: {session_id}] Final snapshot count (before session end): {snapshot_count}")


        # --- MODIFICATION: End Meet Session (Leave Call & Cleanup Browser) BEFORE Analysis ---
        logger.info(f"[Task: {session_id}] Ending Google Meet session...")
        try:
            if meet_session_mgr: meet_session_mgr.end_session(session_id) # This stops video capture, leaves call, cleans up driver
            logger.info(f"[Task: {session_id}] ✅ Google Meet session ended.")
        except Exception as end_e:
            logger.error(f"[Task: {session_id}] Error during explicit meet session end: {end_e}", exc_info=True)
            # Continue to analysis anyway
        # --- END MODIFICATION ---

        # --- 6. Generate & Read Transcript ---
        # Transcript generation should have happened in conduct_interview
        transcript_path = Path("data") / candidate_id / "transcripts" / f"transcript_{session_id}.txt"
        transcript_content = None
        if transcript_path.exists():
             try: transcript_content = transcript_path.read_text(encoding="utf-8"); logger.info(f"[Task: {session_id}] Read transcript: {transcript_path}")
             except Exception as e: logger.error(f"[Task: {session_id}] Failed read transcript {transcript_path}: {e}")
        else: logger.error(f"[Task: {session_id}] Transcript not found: {transcript_path}")

        # --- 7. Run Analyses Concurrently ---
        analysis_results = {"behavioral": None, "transcript": None}; analysis_threads = []

        # Behavioral Analysis Thread (uses snapshot_count determined before session end)
        if enable_video and snapshot_count > 0:
            logger.info(f"[Task: {session_id}] Preparing behavioral analysis thread ({snapshot_count} snapshots)...")
            b_thread = threading.Thread( target=run_analysis_in_thread, args=(behavioral_analyzer.analyze_interview_snapshots, (candidate_id, session_id), analysis_results, "behavioral"), daemon=True )
            analysis_threads.append(b_thread)
        elif enable_video: logger.warning(f"[Task: {session_id}] No snapshots for behavioral analysis.")

        # Transcript Analysis Thread
        if transcript_content:
            logger.info(f"[Task: {session_id}] Preparing transcript analysis thread...")
            try:
                parsed_transcript = transcript_parser.parse(transcript_content)
                t_thread = threading.Thread( target=run_analysis_in_thread, args=(transcript_analyzer.analyze, (resume_content, parsed_transcript), analysis_results, "transcript"), daemon=True )
                analysis_threads.append(t_thread)
            except Exception as parse_e: logger.error(f"[Task: {session_id}] Failed parse transcript: {parse_e}")
        else: logger.error(f"[Task: {session_id}] No transcript content for analysis.")

        # Start and wait
        if analysis_threads:
            logger.info(f"[Task: {session_id}] Starting {len(analysis_threads)} analysis threads..."); [t.start() for t in analysis_threads]; logger.info(f"Waiting for analyses..."); [t.join() for t in analysis_threads]; logger.info(f"Analysis threads finished.")
        else: logger.warning(f"[Task: {session_id}] No analysis tasks run.")

        # --- 8. Combine Analyses ---
        behavioral_analysis_result = analysis_results.get("behavioral")
        transcript_analysis_result = analysis_results.get("transcript")

        # Log completion
        if isinstance(behavioral_analysis_result, dict) and behavioral_analysis_result.get('status') == 'success': logger.info("[Task: {session_id}] Behavioral analysis OK.")
        elif behavioral_analysis_result: logger.error(f"[Task: {session_id}] Behavioral analysis FAIL: {behavioral_analysis_result.get('detail')}")
        if isinstance(transcript_analysis_result, OverallAnalysis): logger.info("[Task: {session_id}] Transcript analysis OK.")
        elif transcript_analysis_result: logger.error(f"[Task: {session_id}] Transcript analysis FAIL: {transcript_analysis_result.get('detail')}")

        if behavioral_analysis_result or transcript_analysis_result:
            logger.info(f"[Task: {session_id}] Combining analysis results...")
            try:
                combined_report: CombinedAnalysisReport = combine_analyses(
                    behavioral_result=behavioral_analysis_result,
                    transcript_result=transcript_analysis_result if isinstance(transcript_analysis_result, OverallAnalysis) else None,
                    session_id=session_id, candidate_id=candidate_id )
                report_dir = Path("data") / candidate_id / session_id / "final_report"; report_dir.mkdir(parents=True, exist_ok=True)
                final_report_path = report_dir / f"combined_analysis_{session_id}.json"
                with open(final_report_path, "w", encoding="utf-8") as f: f.write(combined_report.model_dump_json(indent=4))
                logger.info(f"[Task: {session_id}] ✅ Combined report saved: {final_report_path}")
            except Exception as combine_e: logger.error(f"[Task: {session_id}] ❌ Combine/save failed: {combine_e}", exc_info=True)
        else: logger.warning(f"[Task: {session_id}] No successful analysis results to combine.")

    except Exception as e: logger.error(f"❌ FATAL ERROR in task {session_id}: {e}", exc_info=True)
    finally:
        # --- 9. Final Cleanup (Mainly Interview Service state now) ---
        logger.info(f"[Task: {session_id}] Final cleanup...")
        # Meet session should already be ended unless an error occurred before step 5
        # Call end_session again just in case, it handles already ended sessions
        try:
            if meet_session_mgr: meet_session_mgr.end_session(session_id)
        except Exception as cleanup_e: logger.error(f"Error during final Meet cleanup check: {cleanup_e}")
        try: # Clear Gemini chat state
            if interview_service: interview_service.end_interview_session(session_id)
        except Exception as cleanup_e: logger.error(f"Error during final Interview state cleanup: {cleanup_e}")

        logger.info(f"[Task: {session_id}] ✅ Task finished.")
        # Optionally update DB status here to 'completed' or 'failed'


# --- API Endpoints ---
@app.on_event("startup")
async def startup_event():
    if not all([db_handler, interview_service, behavioral_analyzer, transcript_parser, transcript_analyzer, meet_session_mgr, meet_orchestrator]):
        logger.critical("❌ One or more essential services failed to initialize.")

@app.get("/")
def read_root(): return { "status": "AI Interviewer API is running" }

@app.post("/interview/start-google-meet", status_code=202)
async def start_google_meet_interview(
    background_tasks: BackgroundTasks,
    candidate_id: str = Form(...),
    meet_link: str = Form(...),
    questionnaire_json: Optional[str] = Form(None),
    audio_device: Optional[int] = Form(None),
    enable_video: bool = Form(True),
    video_capture_method: str = Form("javascript"),
    resume: UploadFile = File(...)
):
    """ Starts a new interview session. Questionnaire is optional. """
    if not all([db_handler, interview_service, meet_session_mgr, meet_orchestrator]):
        raise HTTPException(status_code=503, detail="Services not initialized.")

    if video_capture_method not in ["javascript", "screenshot"]:
        raise HTTPException(status_code=400, detail="Invalid video_capture_method.")

    logger.info(f"📥 Start interview request for: {candidate_id}")

    # 1. Parse Resume
    resume_content: Optional[str] = None
    try:
        resume_bytes = await resume.read(); fname = resume.filename.lower() if resume.filename else ""
        if fname.endswith('.pdf'): resume_content = parse_resume_pdf(io.BytesIO(resume_bytes)); logger.info("Parsed PDF resume.")
        elif fname.endswith('.txt'): resume_content = resume_bytes.decode("utf-8"); logger.info("Parsed TXT resume.")
        else: raise HTTPException(status_code=400, detail="Invalid resume file type (.txt or .pdf).")
        if not resume_content: raise ValueError("Failed to extract text from resume.")
    except Exception as e: logger.error(f"Failed parse resume: {e}", exc_info=True); raise HTTPException(status_code=400, detail=f"Failed parse resume: {e}")

    # 2. Parse Optional Questionnaire
    questionnaire: List[str] = []
    if questionnaire_json:
        try:
            parsed_q = json.loads(questionnaire_json)
            if not isinstance(parsed_q, list) or not all(isinstance(q, str) for q in parsed_q): raise ValueError("Questionnaire must be JSON list of strings.")
            questionnaire = parsed_q; logger.info(f"Received questionnaire ({len(questionnaire)} questions).")
        except Exception as e: logger.error(f"Error parsing questionnaire: {e}", exc_info=True); raise HTTPException(status_code=400, detail=f"Invalid questionnaire_json: {e}")
    else: logger.info("No questionnaire provided.")

    # 3. Create database session
    try:
        if resume_content is None: raise ValueError("Resume content missing.")
        session_id = interview_service.start_new_interview(resume_content, candidate_id, questionnaire)
        logger.info(f"✅ DB session created: {session_id}")
    except Exception as e: logger.error(f"Failed create DB session: {e}", exc_info=True); raise HTTPException(status_code=500, detail="Failed initialize session.")

    # 4. Schedule background task
    background_tasks.add_task(
        start_and_conduct_interview_task,
        session_id=session_id, candidate_id=candidate_id, meet_link=meet_link,
        audio_device=audio_device, enable_video=enable_video, video_capture_method=video_capture_method,
        resume_content=resume_content # Pass resume content
    )

    # 5. Return response
    return { "status": "pending", "message": "Interview session scheduled.", "session_id": session_id, "questions_in_session": len(questionnaire) }

# --- Other Endpoints ---
@app.get("/interview/{session_id}/status")
async def get_interview_status(session_id: str) -> Dict[str, Any]:
    if db_handler is None or meet_session_mgr is None: raise HTTPException(status_code=503, detail="Services offline.")
    session_data = db_handler.get_session(session_id)
    if not session_data: raise HTTPException(status_code=404, detail="Session not found.")
    active_session = meet_session_mgr.get_session(session_id)
    bot_status = "ended_or_unknown"; video_enabled = False; video_capture_method = "unknown"; snapshot_count = 0; capture_stats = None
    if active_session:
        bot_status = active_session.get('status', 'active'); video_enabled = active_session.get('video_enabled', False)
        video_capture_method = active_session.get('video_capture_method', 'js'); snapshot_count = meet_session_mgr.get_snapshot_count(session_id)
        capture_stats = meet_session_mgr.get_capture_stats(session_id)
    else: snapshot_count = meet_session_mgr.get_snapshot_count(session_id)
    return { "session_id": session_id, "candidate_id": session_data.get("candidate_id"), "bot_status": bot_status, "questionnaire_length": len(session_data.get("questionnaire", [])),
        "video_enabled": video_enabled, "video_capture_method": video_capture_method, "snapshots_captured": snapshot_count, "capture_stats": capture_stats,
        "created_at": session_data.get("created_at"), "conversation_length": len(session_data.get("conversation", [])) }

@app.post("/interview/{session_id}/end")
async def end_interview(session_id: str, background_tasks: BackgroundTasks):
     if db_handler is None or meet_session_mgr is None or interview_service is None: raise HTTPException(status_code=503, detail="Services offline.")
     logger.info(f"Received request to end interview: {session_id}")
     snapshot_count = meet_session_mgr.get_snapshot_count(session_id); capture_stats = meet_session_mgr.get_capture_stats(session_id)
     try: meet_session_mgr.end_session(session_id); interview_service.end_interview_session(session_id); logger.info(f"✅ Session {session_id} signaled end.")
     except Exception as e: logger.error(f"Error ending session {session_id}: {e}", exc_info=True)
     return { "status": "ending_triggered", "message": f"Termination signal sent for {session_id}.", "snapshots": snapshot_count, "capture_stats": capture_stats }

@app.get("/interview/{session_id}/snapshots")
async def get_snapshot_info(session_id: str):
     if meet_session_mgr is None or db_handler is None: raise HTTPException(status_code=503, detail="Services offline.")
     active_session = meet_session_mgr.get_session(session_id)
     if active_session:
        snapshot_count = meet_session_mgr.get_snapshot_count(session_id); capture_stats = meet_session_mgr.get_capture_stats(session_id)
        video_enabled = active_session.get('video_enabled', False); video_method = active_session.get('video_capture_method', 'js')
        return { "session_id": session_id, "status": "active" if video_enabled else "active_vid_off", "video_enabled": video_enabled, "video_method": video_method, "snapshots": snapshot_count, "capture_stats": capture_stats }
     else:
        session_data = db_handler.get_session(session_id);
        if not session_data: raise HTTPException(status_code=404, detail="Session not found.")
        candidate_id = session_data.get("candidate_id"); snapshot_dir = None; snapshot_count = 0
        if candidate_id: snapshot_dir = Path("data") / candidate_id / session_id / "snapshots";
        if snapshot_dir and snapshot_dir.exists(): snapshot_count = len(list(snapshot_dir.glob("*.jpg")))
        return { "session_id": session_id, "status": "completed_or_unknown", "snapshots": snapshot_count, "dir": str(snapshot_dir) if snapshot_dir else None }

@app.get("/health")
async def health_check():
    services = { "db": db_handler is not None, "interview": interview_service is not None, "behavioral": behavioral_analyzer is not None, "transcript_parser": transcript_parser is not None, "transcript_analyzer": transcript_analyzer is not None, "meet_mgr": meet_session_mgr is not None, "orchestrator": meet_orchestrator is not None }
    healthy = all(services.values())
    return { "status": "healthy" if healthy else "degraded", "services": services }

if __name__ == "__main__":
    print("="*70 + "\n🚀 Starting FastAPI server - AI Interviewer\n" + "="*70)
    print("Server: http://127.0.0.1:8000 | Docs: http://127.0.0.1:8000/docs")
    print("\n`/start` expects: `candidate_id`, `meet_link`, `resume` (file), OPTIONAL `questionnaire_json` (string list)")
    print("="*70 + "\n")
    uvicorn.run("main_api:app", host="127.0.0.1", port=8000, reload=True)
# app/services/meet_session_manager.py
"""
Google Meet Session Manager with Candidate Video Capture
UPDATED: Added check for exactly one candidate after stable join.
"""
import logging
import time
import threading
from pathlib import Path
from typing import Dict, Optional, Any
from datetime import datetime

from app.services.meet_controller import MeetController
from app.core.services.database_service import DBHandler

logger = logging.getLogger(__name__)


class MeetSessionManager:
    """Manages Google Meet bot sessions with candidate video capture."""

    CHROME_PROFILE_PATH = Path("chrome_profile").resolve()

    def __init__(self, db_handler: DBHandler):
        self.db = db_handler
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        logger.info("✅ Session manager initialized (using persistent Chrome profiles)")

    # ... (create_interview_session, _generate_meet_link, start_bot_session, _start_candidate_video_capture, get_snapshot_count, get_capture_stats remain the same) ...
    def create_interview_session( self, candidate_id: str, session_id: str, job_position: str = "Software Engineer", meet_link: Optional[str] = None ) -> Dict[str, Any]:
        meet_link = meet_link or self._generate_meet_link()
        existing_session = self.db.get_session(session_id)
        if existing_session: logger.info(f"✅ Using existing session: {session_id}")
        else: logger.warning(f"⚠️ Session {session_id} not found in DB")
        if meet_link: logger.info(f"🔎 Using Meet link: {meet_link}")
        return { "session_id": session_id, "candidate_id": candidate_id, "job_position": job_position, "meet_link": meet_link, "status": "pending", "created_at": datetime.utcnow().isoformat() }

    def _generate_meet_link(self) -> str:
        import random; import string
        code = ''.join(random.choices(string.ascii_lowercase, k=3)) + '-' + ''.join(random.choices(string.ascii_lowercase, k=4)) + '-' + ''.join(random.choices(string.ascii_lowercase, k=3))
        return f"https://meet.google.com/{code}"

    def start_bot_session( self, session_id: str, meet_link: str, candidate_id: str, audio_device: Optional[int] = None, enable_video: bool = True, headless: bool = True, video_capture_method: str = "javascript" ) -> bool:
        try:
            logger.info(f"🤖 Starting bot session: {session_id} | Headless: {headless}")
            logger.info(f"📹 Video: {'Enabled' if enable_video else 'Disabled'} | Method: {video_capture_method}")
            controller = MeetController( headless=headless, audio_device_index=audio_device, user_data_dir=str(self.CHROME_PROFILE_PATH), use_vb_audio=True )
            if not controller.setup_driver(): logger.error(f"❌ Failed setup driver {session_id}"); return False
            bot_name = f"AI Interviewer Bot"
            if not controller.join_meeting(meet_link, bot_name): logger.error(f"❌ Failed join meeting {session_id}"); controller.cleanup(); return False
            logger.info(f"✅ Bot joined meeting: {session_id}")
            snapshot_dir = Path("data") / candidate_id / session_id / "snapshots"; snapshot_dir.mkdir(parents=True, exist_ok=True)
            self.active_sessions[session_id] = { 'controller': controller, 'meet_link': meet_link, 'candidate_id': candidate_id, 'status': 'active', 'video_enabled': enable_video,
                'video_capture_method': video_capture_method, 'snapshot_dir': snapshot_dir, 'snapshot_count': 0, 'snapshot_thread': None,
                'stop_capture': threading.Event(), 'stop_interview': threading.Event(),
                'capture_stats': { 'total_attempts': 0, 'successful_captures': 0, 'failed_captures': 0, 'js_captures': 0, 'screenshot_captures': 0 } }
            return True
        except Exception as e: logger.error(f"❌ Error starting bot session: {e}", exc_info=True); return False

    def _start_candidate_video_capture(self, session_id: str):
        session = self.active_sessions.get(session_id)
        if not session or not session.get('controller'): logger.error(f"Session/Ctrl missing for vidcap {session_id}"); return
        controller = session['controller']
        def capture_loop():
            # ... (capture loop logic remains the same as previous correct version) ...
             logger.info(f"📸 Starting capture thread {session_id} | Method: {session['video_capture_method']}")
             snapshot_dir = session['snapshot_dir']; stop_event = session['stop_capture']; capture_method = session['video_capture_method']; stats = session['capture_stats']
             capture_interval = 5; last_capture_time = 0; consecutive_failures = 0; max_consecutive_failures = 10
             logger.info("Waiting 5s for video stream..."); time.sleep(5) # Use time.sleep here
             while not stop_event.is_set():
                 try:
                     current_controller = self.active_sessions.get(session_id, {}).get('controller') # Re-check controller
                     if not current_controller: logger.warning("Ctrl missing in capture loop."); break
                     current_time = time.time()
                     if current_time - last_capture_time >= capture_interval:
                         stats['total_attempts'] += 1; screenshot_data = None; width = 640; height = 480
                         if capture_method == "javascript":
                             result = current_controller.capture_candidate_video_js()
                             if result: screenshot_data, width, height = result; stats['js_captures'] += 1; consecutive_failures = 0; logger.debug(f"JS capture ok ({width}x{height})")
                             else: logger.debug("JS capture None")
                         if screenshot_data is None:
                             if capture_method == "javascript": logger.debug("JS fail, fallback screenshot.")
                             screenshot_data = current_controller.capture_candidate_video_screenshot()
                             if screenshot_data: stats['screenshot_captures'] += 1; consecutive_failures = 0; logger.debug("Screenshot ok")
                             else: logger.debug("Screenshot None")
                         if screenshot_data:
                             timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f"); snapshot_path = snapshot_dir / f"snapshot_{timestamp}.jpg"
                             try:
                                 from PIL import Image; import io
                                 img = Image.open(io.BytesIO(screenshot_data));
                                 if img.mode in ('RGBA', 'P'): img = img.convert('RGB')
                                 img.save(snapshot_path, "JPEG", quality=90)
                             except Exception as img_e: logger.error(f"Save snapshot fail: {img_e}"); stats['failed_captures'] += 1; consecutive_failures += 1; continue
                             session['snapshot_count'] += 1; stats['successful_captures'] += 1
                             if session['snapshot_count'] % 5 == 0: method_used = "JS" if stats['js_captures'] >= stats['screenshot_captures'] else "Scr"; logger.info(f"📸 Snaps: {session['snapshot_count']} ({method_used}) | Rate: {stats['successful_captures']}/{stats['total_attempts']}")
                             last_capture_time = current_time
                         else: # Handle failure
                             stats['failed_captures'] += 1; consecutive_failures += 1; logger.warning(f"Capture fail #{stats['total_attempts']} (Consec: {consecutive_failures})")
                             if consecutive_failures >= max_consecutive_failures: logger.error(f"❌ Max capture fails ({max_consecutive_failures}). Stopping."); break
                     if stop_event.wait(0.5): break
                 except Exception as e: logger.error(f"Capture loop error: {e}", exc_info=True);
                 if stop_event.wait(1): break
             # Final logging
             final_session_data = self.active_sessions.get(session_id); final_snapshot_count = final_session_data.get('snapshot_count', session.get('snapshot_count', 0)) if final_session_data else session.get('snapshot_count', 0)
             final_stats = final_session_data.get('capture_stats', session.get('capture_stats', {})) if final_session_data else session.get('capture_stats', {})
             logger.info(f"📸 Video capture thread stopped {session_id}"); logger.info(f"📊 Final Stats: Snaps={final_snapshot_count}, Attempts={final_stats.get('total_attempts', 0)}, Success={final_stats.get('successful_captures', 0)}, JS={final_stats.get('js_captures', 0)}, Scr={final_stats.get('screenshot_captures', 0)}")
        capture_thread = threading.Thread( target=capture_loop, daemon=True, name=f"VideoCapture-{session_id}" ); capture_thread.start(); session['snapshot_thread'] = capture_thread; logger.info(f"✅ Vidcap thread init ok {session_id}")

    def get_snapshot_count(self, session_id: str) -> int:
        session = self.active_sessions.get(session_id);
        if session: return session.get('snapshot_count', 0)
        try:
            session_data = self.db.get_session(session_id);
            if session_data: candidate_id = session_data.get('candidate_id');
            if candidate_id: snapshot_dir = Path("data") / candidate_id / session_id / "snapshots";
            if snapshot_dir and snapshot_dir.exists(): return len(list(snapshot_dir.glob("*.jpg")))
        except Exception as e: logger.error(f"Err count disk snaps: {e}")
        return 0

    def get_capture_stats(self, session_id: str) -> Optional[Dict[str, Any]]:
        session = self.active_sessions.get(session_id);
        if session:
            stats = session.get('capture_stats', {}); success_rate_str = "N/A"; total_attempts = stats.get('total_attempts', 0);
            if total_attempts > 0: success_rate_str = f"{stats.get('successful_captures', 0)}/{total_attempts}"
            return { 'snapshot_count': session.get('snapshot_count', 0), 'total_attempts': total_attempts, 'successful_captures': stats.get('successful_captures', 0), 'failed_captures': stats.get('failed_captures', 0),
                'js_captures': stats.get('js_captures', 0), 'screenshot_captures': stats.get('screenshot_captures', 0), 'success_rate': success_rate_str }
        return None

    def wait_for_candidate(self, session_id: str, timeout: int = 300) -> bool:
        """Wait for candidate, requiring stable count == 2."""
        session = self.active_sessions.get(session_id)
        if not session or not session.get('controller'): logger.error(f"Session/Ctrl missing wait {session_id}"); return False
        controller = session['controller']
        start_time = time.time()
        logger.info(f"⏳ Waiting for candidate (timeout: {timeout}s, requires 3 stable checks at count 2)...")
        consecutive_joins = 0; required_stable_checks = 3; check_interval = 3
        last_checked_count = 1 # Assume initially only bot is present

        while time.time() - start_time < timeout:
            stop_event = session.get('stop_interview')
            if stop_event and stop_event.is_set(): logger.info("Stop signal during wait."); return False
            try:
                participant_count = controller.get_participant_count() # Returns TOTAL count
                logger.debug(f"Participant count check: {participant_count}")
                last_checked_count = participant_count # Store the latest count

                # --- MODIFIED: Check for EXACTLY 2 participants ---
                if participant_count == 2:
                    consecutive_joins += 1
                    logger.info(f"Potential stable join (Count=2). Stable checks: {consecutive_joins}/{required_stable_checks}")
                    if consecutive_joins >= required_stable_checks:
                        logger.info(f"✅ Candidate joined stably! Total participants: 2")
                        session['status'] = 'candidate_joined'
                        session['candidate_joined_at'] = datetime.utcnow().isoformat()
                        return True # Successfully joined with exactly one candidate
                # --- END MODIFICATION ---
                elif participant_count > 2:
                     # --- ADDED: Handle too many participants immediately ---
                     logger.error(f"❌ Too many participants ({participant_count}) detected during initial wait. Aborting.")
                     # Optionally try to send a message or play audio here if critical
                     # meet.send_chat_message("Error: More than one candidate detected. Interview cannot proceed.") # Requires meet controller access
                     session['status'] = 'aborted_multiple_participants'
                     return False # Fail the join process
                     # --- END ADDED ---
                else: # Count is 1 (or 0/error)
                    if consecutive_joins > 0: logger.info("Count dropped below 2. Resetting stable checks.")
                    consecutive_joins = 0

                if stop_event and stop_event.wait(check_interval): logger.info("Stop signal during wait interval."); return False
            except Exception as e:
                logger.error(f"Error checking participants: {e}", exc_info=True); consecutive_joins = 0
                if stop_event and stop_event.wait(check_interval): logger.info("Stop signal during error wait."); return False

        # --- Check final count on timeout ---
        if last_checked_count > 2:
             logger.error(f"❌ Timeout waiting for candidate, and participant count ({last_checked_count}) > 2. Aborting.")
             session['status'] = 'aborted_multiple_participants_timeout'
             return False
        # --- END Check ---

        logger.warning(f"⏰ Timeout waiting for candidate {session_id} (Final count: {last_checked_count})"); return False


    def end_session(self, session_id: str):
        """End a bot session, stop threads, cleanup resources."""
        session = self.active_sessions.pop(session_id, None)
        if not session: logger.warning(f"Session {session_id} not found/ended."); return
        try:
            logger.info(f"🛑 Ending session: {session_id}")
            stop_interview_event = session.get('stop_interview');
            if stop_interview_event and not stop_interview_event.is_set(): logger.info("Signaling orchestrator stop."); stop_interview_event.set()
            if session.get('video_enabled'):
                stop_event = session.get('stop_capture');
                if stop_event and not stop_event.is_set():
                    stop_event.set(); capture_thread = session.get('snapshot_thread')
                    if capture_thread and capture_thread.is_alive():
                        logger.info("Waiting video capture..."); capture_thread.join(timeout=5)
                        if capture_thread.is_alive(): logger.warning("Video thread didn't finish.")
                        else: logger.info(f"✅ Video capture stopped")
                    else: logger.info("Video capture not running.")
            stats = session.get('capture_stats', {}); snapshot_count = session.get('snapshot_count', 0)
            controller = session.get('controller')
            if controller: controller.leave_meeting(); controller.cleanup()
            logger.info(f"✅ Session {session_id} ended. Snaps: {snapshot_count}")
            total_attempts = stats.get('total_attempts', 0); successful_captures = stats.get('successful_captures', 0)
            if total_attempts > 0: logger.info(f"📊 Final capture rate: {successful_captures}/{total_attempts} ({(successful_captures/total_attempts)*100:.1f}%)")
            else: logger.info("📊 No capture attempts.")
        except Exception as e:
            logger.error(f"Error ending session {session_id}: {e}", exc_info=True)
            controller = session.get('controller');
            if controller:
                 try: controller.cleanup()
                 except Exception as cleanup_e: logger.error(f"Fallback cleanup error: {cleanup_e}")

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        return self.active_sessions.get(session_id)

    def get_all_active_sessions(self) -> Dict[str, Dict[str, Any]]:
        active_sessions_copy = dict(self.active_sessions); summary = {}
        # ... (summary generation logic remains the same) ...
        for sid, session in active_sessions_copy.items():
             stats = session.get('capture_stats', {}); success_rate_str = "N/A"; total_attempts = stats.get('total_attempts', 0)
             if total_attempts > 0: success_rate_str = f"{stats.get('successful_captures', 0)}/{total_attempts}"
             summary[sid] = { 'status': session.get('status', '?'), 'candidate_id': session.get('candidate_id'), 'meet_link': session.get('meet_link'),
                'video': session.get('video_enabled'), 'method': session.get('video_capture_method'), 'snaps': session.get('snapshot_count', 0),
                'capture_stats': { 'attempts': total_attempts, 'success': stats.get('successful_captures', 0), 'rate': success_rate_str } }
        return summary
# app/services/meet_session_manager.py
"""
Google Meet Session Manager with Candidate Video Capture
UPDATED: Made wait_for_candidate more robust with stability check.
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

    # Chrome profile path for persistent login
    CHROME_PROFILE_PATH = Path("chrome_profile").resolve()

    def __init__(self, db_handler: DBHandler):
        self.db = db_handler
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        logger.info("✅ Session manager initialized (using persistent Chrome profiles)")

    def create_interview_session(
        self,
        candidate_id: str,
        session_id: str,
        job_position: str = "Software Engineer",
        meet_link: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create or update interview session in database.
        """
        meet_link = meet_link or self._generate_meet_link()

        existing_session = self.db.get_session(session_id)

        if existing_session:
            logger.info(f"✅ Using existing session: {session_id}")
        else:
            logger.warning(f"⚠️ Session {session_id} not found in database")

        if meet_link:
            logger.info(f"🔎 Using Meet link: {meet_link}")

        session_data = {
            "session_id": session_id,
            "candidate_id": candidate_id,
            "job_position": job_position,
            "meet_link": meet_link,
            "status": "pending",
            "created_at": datetime.utcnow().isoformat()
        }

        return session_data

    def _generate_meet_link(self) -> str:
        """Generate a random Google Meet link."""
        import random
        import string
        code = ''.join(random.choices(string.ascii_lowercase, k=3))
        code += '-' + ''.join(random.choices(string.ascii_lowercase, k=4))
        code += '-' + ''.join(random.choices(string.ascii_lowercase, k=3))
        return f"https://meet.google.com/{code}"

    def start_bot_session(
        self,
        session_id: str,
        meet_link: str,
        candidate_id: str,
        audio_device: Optional[int] = None,
        enable_video: bool = True,
        headless: bool = True, # Set back to True
        video_capture_method: str = "javascript"
    ) -> bool:
        """
        Start bot session and join Google Meet.
        """
        try:
            logger.info(f"🤖 Starting bot session: {session_id}")
            logger.info(f"📹 Candidate video capture: {'Enabled' if enable_video else 'Disabled'}")
            logger.info(f"🎯 Video capture method: {video_capture_method}")

            controller = MeetController(
                headless=headless,
                audio_device_index=audio_device,
                user_data_dir=str(self.CHROME_PROFILE_PATH),
                use_vb_audio=True
            )

            if not controller.setup_driver():
                logger.error(f"❌ Failed to setup Chrome driver for {session_id}")
                return False

            bot_name = f"AI Interviewer Bot"
            if not controller.join_meeting(meet_link, bot_name):
                logger.error(f"❌ Failed to join meeting for {session_id}")
                controller.cleanup()
                return False

            logger.info(f"✅ Bot joined meeting: {session_id}")

            snapshot_dir = Path("data") / candidate_id / session_id / "snapshots"
            snapshot_dir.mkdir(parents=True, exist_ok=True)

            self.active_sessions[session_id] = {
                'controller': controller,
                'meet_link': meet_link,
                'candidate_id': candidate_id,
                'status': 'active',
                'video_enabled': enable_video,
                'video_capture_method': video_capture_method,
                'snapshot_dir': snapshot_dir,
                'snapshot_count': 0,
                'snapshot_thread': None,
                'stop_capture': threading.Event(),
                'stop_interview': threading.Event(),
                'capture_stats': {
                    'total_attempts': 0,
                    'successful_captures': 0,
                    'failed_captures': 0,
                    'js_captures': 0,
                    'screenshot_captures': 0
                }
            }
            return True

        except Exception as e:
            logger.error(f"❌ Error starting bot session: {e}", exc_info=True)
            return False

    def _start_candidate_video_capture(self, session_id: str):
        """Start capturing candidate video snapshots in background thread."""
        session = self.active_sessions.get(session_id)
        if not session:
            logger.error(f"Session {session_id} not found")
            return

        def capture_loop():
            logger.info(f"📸 Starting candidate video capture for {session_id}")
            logger.info(f"🎯 Using {session['video_capture_method']} capture method")

            controller = session['controller']
            snapshot_dir = session['snapshot_dir']
            stop_event = session['stop_capture']
            capture_method = session['video_capture_method']
            stats = session['capture_stats']

            capture_interval = 5
            last_capture_time = 0
            consecutive_failures = 0
            max_consecutive_failures = 10

            logger.info("Waiting 5s for video stream to potentially stabilize...")
            time.sleep(5)

            while not stop_event.is_set():
                try:
                    current_time = time.time()
                    if current_time - last_capture_time >= capture_interval:
                        stats['total_attempts'] += 1
                        screenshot_data = None
                        width, height = 640, 480

                        if capture_method == "javascript":
                            result = controller.capture_candidate_video_js()
                            if result:
                                screenshot_data, width, height = result
                                stats['js_captures'] += 1
                                consecutive_failures = 0
                                logger.debug(f"JS capture successful ({width}x{height})")
                            else:
                                logger.debug("JS capture returned None")

                        if screenshot_data is None:
                            if capture_method == "javascript":
                                logger.debug("JS capture failed, falling back to screenshot method.")
                            screenshot_data = controller.capture_candidate_video_screenshot()
                            if screenshot_data:
                                stats['screenshot_captures'] += 1
                                consecutive_failures = 0
                                logger.debug("Screenshot capture successful")
                            else:
                                logger.debug("Screenshot capture returned None")

                        if screenshot_data:
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            snapshot_path = snapshot_dir / f"snapshot_{timestamp}.jpg"
                            try:
                                from PIL import Image
                                import io
                                img = Image.open(io.BytesIO(screenshot_data))
                                if img.mode in ('RGBA', 'P'):
                                    img = img.convert('RGB')
                                img.save(snapshot_path, "JPEG", quality=90)
                                logger.debug(f"Saved snapshot to {snapshot_path}")
                            except Exception as img_e:
                                logger.error(f"Failed to save snapshot image: {img_e}")
                                stats['failed_captures'] += 1
                                consecutive_failures += 1
                                continue

                            session['snapshot_count'] += 1
                            stats['successful_captures'] += 1
                            if session['snapshot_count'] % 5 == 0:
                                method = "JS" if stats['js_captures'] > stats['screenshot_captures'] else "Screenshot"
                                logger.info(
                                    f"📸 Captured {session['snapshot_count']} candidate snapshots ({method}) "
                                    f"(Success rate: {stats['successful_captures']}/{stats['total_attempts']})"
                                )
                            last_capture_time = current_time
                        else:
                            stats['failed_captures'] += 1
                            consecutive_failures += 1
                            logger.warning(f"Capture attempt {stats['total_attempts']} failed (Consecutive: {consecutive_failures})")
                            if consecutive_failures == 5:
                                logger.warning(
                                    f"⚠️ {consecutive_failures} consecutive capture failures. "
                                    f"Candidate may not have camera on or Meet layout changed."
                                )
                            elif consecutive_failures >= max_consecutive_failures:
                                logger.error(
                                    f"❌ Too many consecutive failures ({consecutive_failures}). "
                                    f"Stopping capture attempts. Check selectors or if candidate camera is on."
                                )
                                break

                    if stop_event.wait(0.5):
                        break
                except Exception as e:
                    logger.error(f"Error in capture loop: {e}", exc_info=True)
                    if stop_event.wait(1):
                         break

            logger.info(f"📸 Video capture stopped for {session_id}")
            logger.info(f"📊 Capture Statistics:")
            logger.info(f"   Total snapshots: {session['snapshot_count']}")
            if stats['total_attempts'] > 0:
                success_percent = (stats['successful_captures'] / stats['total_attempts']) * 100
                logger.info(f"   Success rate: {stats['successful_captures']}/{stats['total_attempts']} ({success_percent:.1f}%)")
            else:
                 logger.info(f"   Success rate: 0/0")
            logger.info(f"   JS captures: {stats['js_captures']}")
            logger.info(f"   Screenshot captures: {stats['screenshot_captures']}")

        capture_thread = threading.Thread(
            target=capture_loop, daemon=True, name=f"VideoCapture-{session_id}"
        )
        capture_thread.start()
        session['snapshot_thread'] = capture_thread
        logger.info(f"✅ Candidate video capture thread started for {session_id}")

    def get_snapshot_count(self, session_id: str) -> int:
        """Get number of snapshots captured for a session."""
        session = self.active_sessions.get(session_id)
        if session:
            return session.get('snapshot_count', 0)
        try:
            session_data = self.db.get_session(session_id)
            if session_data:
                candidate_id = session_data.get('candidate_id')
                if candidate_id:
                    snapshot_dir = Path("data") / candidate_id / session_id / "snapshots"
                    if snapshot_dir.exists():
                        return len(list(snapshot_dir.glob("snapshot_*.jpg")))
        except Exception as e:
            logger.error(f"Error counting snapshots from disk: {e}")
        return 0

    def get_capture_stats(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get capture statistics for a session."""
        session = self.active_sessions.get(session_id)
        if session:
            stats = session.get('capture_stats', {})
            success_rate_str = "0/0"
            if stats.get('total_attempts', 0) > 0:
                 success_rate_str = f"{stats.get('successful_captures', 0)}/{stats.get('total_attempts', 0)}"
            return {
                'snapshot_count': session.get('snapshot_count', 0),
                'total_attempts': stats.get('total_attempts', 0),
                'successful_captures': stats.get('successful_captures', 0),
                'failed_captures': stats.get('failed_captures', 0),
                'js_captures': stats.get('js_captures', 0),
                'screenshot_captures': stats.get('screenshot_captures', 0),
                'success_rate': success_rate_str
            }
        return None

    def wait_for_candidate(self, session_id: str, timeout: int = 300) -> bool:
        """
        Wait for candidate to join the meeting, requiring stable participant count.
        """
        session = self.active_sessions.get(session_id)
        if not session:
            logger.error(f"Session {session_id} not found for candidate wait")
            return False

        controller = session.get('controller')
        if not controller:
            logger.error(f"Controller not found in session {session_id} for candidate wait")
            return False

        start_time = time.time()
        logger.info(f"⏳ Waiting for candidate to join (timeout: {timeout}s, requires 3 stable checks)...")

        consecutive_joins = 0
        required_stable_checks = 3
        check_interval = 3

        while time.time() - start_time < timeout:
            stop_event = session.get('stop_interview')
            if stop_event and stop_event.is_set():
                logger.info("Stop signal received while waiting for candidate.")
                return False

            try:
                # get_participant_count() now returns the TOTAL count including bot
                participant_count = controller.get_participant_count()
                logger.debug(f"Participant count check: {participant_count}")

                # Check if MORE than 1 participant is present
                if participant_count > 1:
                    consecutive_joins += 1
                    logger.info(f"Candidate potentially joined (Count={participant_count}). Stable checks: {consecutive_joins}/{required_stable_checks}")
                    if consecutive_joins >= required_stable_checks:
                        logger.info(f"✅ Candidate joined stably! Total participants: {participant_count}")
                        session['status'] = 'candidate_joined'
                        session['candidate_joined_at'] = datetime.utcnow().isoformat()
                        return True
                else: # participant_count is 1 (only bot) or 0 (error)
                    if consecutive_joins > 0:
                        logger.info("Participant count dropped back to 1. Resetting stable checks.")
                    consecutive_joins = 0

                if stop_event and stop_event.wait(check_interval):
                    logger.info("Stop signal received during wait interval.")
                    return False

            except Exception as e:
                logger.error(f"Error checking participants: {e}", exc_info=True)
                consecutive_joins = 0
                if stop_event and stop_event.wait(check_interval):
                     logger.info("Stop signal received during participant check error wait.")
                     return False

        logger.warning(f"⏰ Timeout waiting for candidate to join stably {session_id}")
        return False


    def end_session(self, session_id: str):
        """End a bot session and cleanup resources."""
        session = self.active_sessions.pop(session_id, None)
        if not session:
            logger.warning(f"Session {session_id} not found or already ended.")
            return

        try:
            logger.info(f"🛑 Ending session: {session_id}")
            stop_interview_event = session.get('stop_interview')
            if stop_interview_event and not stop_interview_event.is_set():
                logger.info("Signaling interview orchestrator to stop.")
                stop_interview_event.set()

            if session.get('video_enabled'):
                stop_event = session.get('stop_capture')
                if stop_event and not stop_event.is_set():
                    stop_event.set()
                    capture_thread = session.get('snapshot_thread')
                    if capture_thread and capture_thread.is_alive():
                        logger.info("Waiting for video capture thread to finish...")
                        capture_thread.join(timeout=5)
                        if capture_thread.is_alive(): logger.warning("Video capture thread did not finish cleanly.")
                        else: logger.info(f"✅ Video capture thread stopped")
                    else: logger.info("Video capture thread was not running or already finished.")

            stats = session.get('capture_stats', {})
            snapshot_count = session.get('snapshot_count', 0)
            controller = session.get('controller')
            if controller:
                controller.leave_meeting()
                controller.cleanup()

            logger.info(f"✅ Session {session_id} ended successfully")
            logger.info(f"📊 Total snapshots captured: {snapshot_count}")
            if stats and stats.get('total_attempts', 0) > 0:
                success_percent = (stats.get('successful_captures', 0) / stats['total_attempts']) * 100
                logger.info(f"📊 Capture success rate: {stats.get('successful_captures', 0)}/{stats['total_attempts']} ({success_percent:.1f}%)")
            else: logger.info("📊 No capture attempts were made or stats unavailable.")

        except Exception as e:
            logger.error(f"Error ending session {session_id}: {e}", exc_info=True)
            controller = session.get('controller')
            if controller:
                 try: controller.cleanup()
                 except Exception as cleanup_e: logger.error(f"Error during fallback cleanup: {cleanup_e}")

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get active session data."""
        return self.active_sessions.get(session_id)

    def get_all_active_sessions(self) -> Dict[str, Dict[str, Any]]:
        """Get all active sessions with summary info."""
        active_sessions_copy = dict(self.active_sessions)
        summary = {}
        for sid, session in active_sessions_copy.items():
             stats = session.get('capture_stats', {})
             success_rate_str = "0/0"
             if stats.get('total_attempts', 0) > 0:
                 success_rate_str = f"{stats.get('successful_captures', 0)}/{stats.get('total_attempts', 0)}"
             summary[sid] = {
                'status': session.get('status', 'unknown'),
                'candidate_id': session.get('candidate_id'),
                'meet_link': session.get('meet_link'),
                'video_enabled': session.get('video_enabled'),
                'video_capture_method': session.get('video_capture_method'),
                'snapshot_count': session.get('snapshot_count', 0),
                'capture_stats': {
                     'total_attempts': stats.get('total_attempts', 0),
                     'successful_captures': stats.get('successful_captures', 0),
                     'success_rate': success_rate_str
                }
             }
        return summary
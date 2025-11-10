# app/services/meet_controller.py
"""
Google Meet Controller - Handles browser automation for Google Meet
UPDATED: Corrected participant counting logic to find unique IDs.
MODIFIED: Removed automatic caption enabling to speed up bot join time.
"""
import logging
import time
import os
import base64
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, ElementNotInteractableException, JavascriptException
from typing import Optional, Tuple, List, Dict 

# --- DeviceDetector IMPORT REMOVED ---

logger = logging.getLogger(__name__)


class MeetController:
    """Controls Chrome browser for Google Meet interactions."""

    def __init__(
        self,
        headless: bool = True, # Set to True for production
        audio_device_index: Optional[int] = None,
        user_data_dir: Optional[str] = None,
        use_vb_audio: bool = True
    ):
        self.headless = headless
        self.audio_device_index = audio_device_index
        self.user_data_dir = user_data_dir
        self.use_vb_audio = use_vb_audio
        self.driver: Optional[uc.Chrome] = None
        self.current_meet_link: Optional[str] = None
        # --- device_detector and device_type attributes REMOVED ---

    def setup_driver(self) -> bool:
        """Setup Chrome driver with appropriate options."""
        try:
            options = uc.ChromeOptions()

            if self.headless:
                options.add_argument('--headless=new')

            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--disable-blink-features=AutomationControlled')

            if self.use_vb_audio:
                logger.info("🎵 Configuring Chrome for VB-Audio Cable (REAL audio)")
                options.add_argument('--use-fake-ui-for-media-stream')
                options.add_argument('--enable-usermedia-screen-capturing')
                options.add_argument('--allow-file-access-from-files')

                prefs = {
                    "profile.default_content_setting_values.media_stream_mic": 1,
                    "profile.default_content_setting_values.media_stream_camera": 1,
                    "profile.default_content_setting_values.notifications": 2,
                }

                if self.audio_device_index is not None:
                    logger.info(f"🎤 Using VB-Audio device index: {self.audio_device_index}")
                    prefs["media.audio_capture_device"] = str(self.audio_device_index)

            else:
                logger.info("🔇 Configuring Chrome with fake audio devices")
                options.add_argument('--use-fake-ui-for-media-stream')
                options.add_argument('--use-fake-device-for-media-stream')

                prefs = {
                    "profile.default_content_setting_values.media_stream_mic": 1,
                    "profile.default_content_setting_values.media_stream_camera": 1,
                    "profile.default_content_setting_values.notifications": 2,
                }

            options.add_experimental_option("prefs", prefs)

            if self.user_data_dir:
                profile_path = os.path.abspath(self.user_data_dir)
                os.makedirs(profile_path, exist_ok=True)
                logger.info(f"💾 Using persistent Chrome profile: {profile_path}")
                options.add_argument(f'--user-data-dir={profile_path}')
            else:
                logger.info("Using temporary Chrome profile")

            options.add_argument('--disable-extensions')
            options.add_argument('--disable-plugins-discovery')

            logger.info("🔧 Creating Chrome driver...")
            self.driver = uc.Chrome(options=options)

            self.driver.set_page_load_timeout(60)
            self.driver.implicitly_wait(10) # Keep implicit wait low

            logger.info("✅ Chrome driver created successfully")
            
            # --- check_device_type() call REMOVED ---

            if self.use_vb_audio:
                logger.info("🎵 Chrome is configured to use REAL audio devices (VB-Audio)")
                logger.info("💡 Make sure:")
                logger.info("   1. VB-Audio Cable is installed")
                logger.info("   2. Your TTS output is routed to VB-Audio Input")
                logger.info("   3. Chrome will capture from VB-Audio Output")

            return True

        except Exception as e:
            logger.error(f"❌ Failed to setup Chrome driver: {e}", exc_info=True)
            return False

    # --- check_device_type method REMOVED ---

    def join_meeting(self, meet_link: str, display_name: str = "AI Bot") -> bool:
        """Join a Google Meet meeting."""
        try:
            if not self.driver:
                logger.error("Driver not initialized")
                return False

            logger.info(f"🔗 Navigating to Meet: {meet_link}")
            self.current_meet_link = meet_link

            self.driver.get(meet_link)
            time.sleep(5) # Allow page to load initially

            # Handle name input if needed
            try:
                name_input = WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.XPATH, "//input[@placeholder='Enter your name'] | //input[@aria-label='Your name']"))
                )
                name_input.clear()
                name_input.send_keys(display_name)
                logger.info(f"✅ Entered display name: {display_name}")
            except TimeoutException:
                logger.info("Name input not required or already set.")
            except Exception as e:
                 logger.warning(f"Could not set display name: {e}")

            # Turn off camera using JS
            try:
                js_turn_off_camera = """
                const buttons = document.querySelectorAll("div[role='button'][aria-label*='camera' i]");
                const offButton = Array.from(buttons).find(b => b.getAttribute('aria-label').toLowerCase().includes('turn off'));
                if (offButton) {
                    offButton.click();
                    return true;
                }
                return false;
                """
                if self.driver.execute_script(js_turn_off_camera):
                    logger.info("📷 Camera turned off (JS)")
                    time.sleep(0.5)
                else:
                    logger.info("📷 Camera already off or button not found (JS).")
            except Exception as e:
                logger.warning(f"Could not toggle camera (JS): {e}")

            # Turn off mic if not using VB Audio, using JS
            if not self.use_vb_audio:
                try:
                    js_turn_off_mic = """
                    const buttons = document.querySelectorAll("div[role='button'][aria-label*='microphone' i]");
                    const offButton = Array.from(buttons).find(b => b.getAttribute('aria-label').toLowerCase().includes('turn off'));
                    if (offButton) {
                        offButton.click();
                        return true;
                    }
                    return false;
                    """
                    if self.driver.execute_script(js_turn_off_mic):
                        logger.info("🎤 Microphone turned off (JS)")
                        time.sleep(0.5)
                    else:
                        logger.info("🎤 Microphone already off or button not found (JS).")
                except Exception as e:
                    logger.warning(f"Could not toggle microphone (JS): {e}")
            else:
                 logger.info("🎤 Microphone configured for VB-Audio (keeping ON)")


            # Click join button
            try:
                logger.info("Finding 'Join' button (will try for 60 seconds)...")
                join_locators = [
                    (By.XPATH, "//button[.//span[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'join now')]]"),
                    (By.XPATH, "//button[.//span[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'ask to join')]]")
                ]
                
                join_button = None
                start_time = time.time()
                while time.time() - start_time < 60:
                    for locator_type, selector_string in join_locators:
                        try:
                            element = WebDriverWait(self.driver, 2).until(
                                EC.element_to_be_clickable((locator_type, selector_string))
                            )
                            if element:
                                join_button = element
                                logger.info(f"✅ Found clickable join button via {selector_string}")
                                break
                        except TimeoutException:
                            continue
                    if join_button:
                        break
                    time.sleep(1)

                if join_button:
                    self.driver.execute_script("arguments[0].click();", join_button)
                    logger.info("✅ Clicked join button (JS)")
                    time.sleep(8) # Wait for meeting connection
                else:
                    logger.warning(f"Could not find join button after 60 seconds")
                    return False
                    
            except Exception as e:
                logger.error(f"❌ Failed to click join button: {e}", exc_info=True)
                return False
            
            # Verify join
            try:
                WebDriverWait(self.driver, 15).until(
                    lambda d: "meet.google.com/" in d.current_url and d.execute_script("return document.readyState") == "complete"
                )
                logger.info("✅ Successfully joined Google Meet")

                if self.use_vb_audio:
                    logger.info("🎵 Bot is now listening via VB-Audio Cable")

                logger.info("Join verified. Proceeding without enabling captions.")

                return True
            except TimeoutException:
                logger.error("❌ Failed to verify meeting join or page didn't complete loading")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error joining meeting: {e}", exc_info=True)
            return False
    
    def enable_microphone(self):
        """Enable microphone during call (for VB-Audio output)."""
        try:
            if not self.driver: return
            js_script = """
            const mic_button = Array.from(document.querySelectorAll("div[data-is-muted='true'][aria-label*='microphone' i]"))
                                  .find(e => e.offsetParent !== null);
            if (mic_button) { mic_button.click(); return true; } return false;
            """
            if self.driver.execute_script(js_script): logger.info("🎤 Microphone enabled (JS)")
            else: logger.debug("Microphone already enabled or button not found (JS).")
            time.sleep(0.5)
        except Exception as e: logger.debug(f"Error enabling microphone (JS): {e}")
    
    def disable_microphone(self):
        """Disable microphone during call."""
        try:
            if not self.driver: return
            js_script = """
            const mic_button = Array.from(document.querySelectorAll("div[data-is-muted='false'][aria-label*='microphone' i]"))
                                  .find(e => e.offsetParent !== null);
            if (mic_button) { mic_button.click(); return true; } return false;
            """
            if self.driver.execute_script(js_script): logger.info("🔇 Microphone disabled (JS)")
            else: logger.debug("Microphone already disabled or button not found (JS).")
            time.sleep(0.5)
        except Exception as e: logger.debug(f"Error disabling microphone (JS): {e}")

    def capture_candidate_video_js(self) -> Optional[Tuple[bytes, int, int]]:
        """Capture candidate's video using JavaScript method."""
        try:
            if not self.driver: return None
            js_script = """
            function isVisible(elem) { /* ... visibility check ... */ }
            const videos = Array.from(document.querySelectorAll('video'));
            let candidateVideo = null; let maxArea = 0;
            for (let video of videos) {
                if (video.paused || video.ended || !video.videoWidth || video.videoWidth < 100 || video.videoHeight < 100) continue;
                const rect = video.getBoundingClientRect();
                if (rect.width <= 0 || rect.height <= 0 || rect.top < -5 || rect.left < -5 || rect.bottom > (window.innerHeight + 5) || rect.right > (window.innerWidth + 5)) continue;
                let parent = video.closest('[data-self-view="true"], [jsname="Ne3sF"]');
                if (parent) continue;
                const area = rect.width * rect.height;
                if (area > maxArea) { maxArea = area; candidateVideo = video; }
            }
            if (!candidateVideo) return null;
            const canvas = document.createElement('canvas'); const videoWidth = candidateVideo.videoWidth; const videoHeight = candidateVideo.videoHeight;
            canvas.width = videoWidth; canvas.height = videoHeight; const ctx = canvas.getContext('2d');
            ctx.drawImage(candidateVideo, 0, 0, videoWidth, videoHeight);
            const dataUrl = canvas.toDataURL('image/jpeg', 0.85);
            return { data: dataUrl, width: videoWidth, height: videoHeight };
            """
            result = self.driver.execute_script(js_script.replace("/* ... visibility check ... */", """
            if (!elem) return false; return !!( elem.offsetWidth || elem.offsetHeight || elem.getClientRects().length );
            """))
            if result and result.get('data'):
                data_url = result['data']
                if data_url.startswith('data:image/jpeg;base64,'):
                    base64_data = data_url.split(',', 1)[1]
                    try:
                         image_bytes = base64.b64decode(base64_data)
                         width = result.get('width', 0); height = result.get('height', 0)
                         if width > 0 and height > 0: return (image_bytes, width, height)
                         else: logger.debug("JS capture invalid dimensions."); return None
                    except base64.binascii.Error as b64_error: logger.error(f"JS capture invalid base64: {b64_error}"); return None
                else: logger.debug(f"JS capture unexpected format: {data_url[:50]}..."); return None
            return None
        except JavascriptException as e: logger.error(f"JS error video capture: {e}"); return None
        except Exception as e: logger.error(f"Unexpected error JS video capture: {e}", exc_info=True); return None

    def capture_candidate_video_screenshot(self) -> Optional[bytes]:
        """Fallback: Capture candidate video using screenshot method."""
        try:
            if not self.driver: return None
            logger.debug("Attempting screenshot capture (fallback)...")
            screenshot_bytes = self.driver.get_screenshot_as_png()
            if screenshot_bytes: logger.debug("Screenshot ok."); return screenshot_bytes
            else: logger.warning("Screenshot empty."); return None
        except Exception as e: logger.error(f"Screenshot fail: {e}", exc_info=True); return None

    def get_participant_count(self) -> int:
        """Get the number of *unique* participants in the meeting, including the bot."""
        try:
            if not self.driver:
                return 0

            # Primary Method: Use JS to count unique data-participant-id attributes
            try:
                js_script = """
                const elements = document.querySelectorAll('[data-participant-id]');
                const uniqueIds = new Set();
                elements.forEach(el => {
                    const id = el.getAttribute('data-participant-id');
                    if (id) {
                        uniqueIds.add(id);
                    }
                });
                return uniqueIds.size;
                """
                count = self.driver.execute_script(js_script)
                if count is not None and count >= 1:
                    logger.debug(f"Participant count via unique JS IDs: {count}")
                    return count
                else:
                     logger.debug("JS querySelector for unique IDs returned 0 or None. Trying fallbacks...")
            except Exception as e:
                 logger.warning(f"JS unique participant count failed: {e}. Trying fallbacks.")

            # Fallback 1: Check the participant button text/label
            try:
                participant_button = WebDriverWait(self.driver, 3).until(
                     EC.presence_of_element_located((By.XPATH, "//button[contains(@aria-label, 'participants (')] | //button[contains(@aria-label, 'Show everyone (')]" ))
                )
                button_text = participant_button.text or participant_button.get_attribute('aria-label')
                import re
                match = re.search(r'\((\d+)\)', button_text)
                if match:
                    num = int(match.group(1))
                    if num >= 1: logger.debug(f"Participant count via UI button: {num}"); return num
            except Exception as e: logger.debug(f"UI participant button count fail: {e}. Trying video count.")

            # Fallback 2: Count visible video elements (less reliable)
            try:
                js_script_videos = """
                return Array.from(document.querySelectorAll('video'))
                            .filter(el => {
                                const rect = el.getBoundingClientRect();
                                return !!(el.offsetWidth || el.offsetHeight || el.getClientRects().length) && rect.width > 50 && rect.height > 50;
                            }).length;
                """
                count_videos = self.driver.execute_script(js_script_videos)
                if count_videos is not None and count_videos >= 1:
                     logger.debug(f"Participant count via visible videos: {count_videos}")
                     return count_videos
            except Exception as e: logger.warning(f"JS video count fail: {e}.")

            logger.error("Could not determine participant count reliably. Returning 1 (assuming only bot).")
            return 1
        except Exception as e:
            logger.error(f"Critical error getting participant count: {e}", exc_info=True)
            return 1

    # --- tab monitoring methods REMOVED ---

    def leave_meeting(self):
        """Leave the current Google Meet."""
        try:
            if not self.driver: return
            logger.info("👋 Leaving meeting...")
            js_leave = """ /* ... js leave script ... */ """
            try:
                if self.driver.execute_script(js_leave.replace("/* ... js leave script ... */", """
                const leaveButton = document.querySelector("button[aria-label*='Leave call' i], button[aria-label*='Hang up' i]");
                if (leaveButton) { leaveButton.click(); return true; }
                const iconButton = document.querySelector("button i.google-material-icons:contains('call_end')");
                if(iconButton) { const clickableParent = iconButton.closest('button'); if(clickableParent) { clickableParent.click(); return true; } }
                return false;
                """)):
                    logger.info("✅ Left meeting (JS)"); time.sleep(2); return
            except Exception as e: logger.warning(f"JS leave failed: {e}. Trying Selenium.")
            try:
                leave_selectors = [ "button[aria-label*='Leave call'][data-mdc-value='end_call']", "button[aria-label*='Leave call' i]", "button[aria-label*='Hang up' i]" ]
                for selector in leave_selectors:
                    try:
                        leave_button = WebDriverWait(self.driver, 3).until(EC.element_to_be_clickable((By.CSS_SELECTOR, selector)))
                        leave_button.click(); logger.info("✅ Left meeting (Selenium)"); time.sleep(2); return
                    except TimeoutException: continue
                logger.warning("Could not find leave button via Selenium, navigating away"); self.driver.get("about:blank")
            except Exception as e: logger.warning(f"Error leaving via Selenium: {e}"); self.driver.get("about:blank")
        except Exception as e: logger.error(f"Error in leave_meeting: {e}")

    def cleanup(self):
        """Cleanup and close browser."""
        try:
            if self.driver:
                logger.info("🧹 Cleaning up Chrome driver...")
                try: self.driver.close(); self.driver.quit()
                except Exception as e:
                    logger.warning(f"Error driver close/quit: {e}")
                    try: self.driver.quit()
                    except Exception as e_quit: logger.error(f"Force quit failed: {e_quit}")
                self.driver = None; logger.info("✅ Chrome driver cleaned up")
        except Exception as e: logger.error(f"Error during cleanup: {e}")

    def __enter__(self): return self
    def __exit__(self, exc_type, exc_val, exc_tb): self.cleanup(); return False
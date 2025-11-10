"""
Minimal test script to verify VideoProcessor is being called
Run this standalone to test: streamlit run tests/test_video_processor.py

NOTE: This file is NOT a pytest test. It is a manual diagnostic script.
The Streamlit code is wrapped in a main() function to prevent
pytest from executing it during test collection.
"""

import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase, RTCConfiguration
import av
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestVideoProcessor(VideoProcessorBase):
    def __init__(self) -> None:
        self.frame_count = 0
        logger.info("🎬 TestVideoProcessor __init__ called!")
        print("🎬 TestVideoProcessor __init__ called!")
    
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            logger.info(f"📹 Frame {self.frame_count} processed")
            print(f"📹 Frame {self.frame_count} processed")
        return frame

# --- FIX: Move all Streamlit code into a function ---
def run_streamlit_app():
    st.title("Video Processor Test")

    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    st.write("Click START to begin video stream")

    ctx = webrtc_streamer(
        key="test",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=TestVideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    if ctx.state.playing:
        st.success("✅ Video is playing - check logs for frame processing messages")
    else:
        st.info("⚪ Click START button above to test")

    st.write(f"State: {ctx.state.playing}")

# --- FIX: Use __name__ == "__main__" guard ---
# This ensures the code only runs when you execute
# `streamlit run tests/test_video_processor.py`
# and NOT when pytest imports it.
if __name__ == "__main__":
    run_streamlit_app()
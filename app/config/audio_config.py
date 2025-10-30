# app/config/audio_config.py
"""
Audio Configuration for Google Meet Integration
Handles virtual audio cable setup for bot audio routing
"""
import logging
import sounddevice as sd
# --- FIX: Import cast ---
from typing import Optional, Tuple, Any, cast

logger = logging.getLogger(__name__)


class VirtualAudioConfig:
    """Configuration for virtual audio cable devices."""
    
    # Common virtual audio cable names (add your specific names here)
    VIRTUAL_OUTPUT_NAMES = [
        "CABLE Input",  # VB-Audio Virtual Cable
        "VoiceMeeter Input",  # VoiceMeeter
        "Stereo Mix",  # Windows Stereo Mix
        "BlackHole 2ch",  # MacOS BlackHole
        "Monitor of",  # Linux PulseAudio monitor
    ]
    
    VIRTUAL_INPUT_NAMES = [
        "CABLE Output",  # VB-Audio Virtual Cable
        "VoiceMeeter Output",  # VoiceMeeter
        "BlackHole 2ch",  # MacOS BlackHole
        "Monitor of",  # Linux PulseAudio monitor
    ]
    
    @staticmethod
    def list_all_audio_devices():
        """List all available audio devices with details."""
        try:
            # --- FIX: Cast query_devices() to DeviceList ---
            devices: sd.DeviceList = cast(sd.DeviceList, sd.query_devices())
            
            logger.info("\n" + "="*60)
            logger.info("AVAILABLE AUDIO DEVICES:")
            logger.info("="*60)
            
            for idx, device_untyped in enumerate(devices):
                device: dict[str, Any] = device_untyped # Cast to dict
                
                device_type = []
                if device['max_input_channels'] > 0:
                    device_type.append("INPUT")
                if device['max_output_channels'] > 0:
                    device_type.append("OUTPUT")
                
                logger.info(f"\n[{idx}] {device['name']}")
                logger.info(f"    Type: {' & '.join(device_type)}")
                logger.info(f"    Sample Rate: {device['default_samplerate']} Hz")
                logger.info(f"    Channels: In={device['max_input_channels']}, Out={device['max_output_channels']}")
            
            logger.info("\n" + "="*60)
            
            return devices
        except Exception as e:
            logger.error(f"Error listing audio devices: {e}")
            return []
    
    @staticmethod
    def find_device_by_name(name_patterns: list, device_type: str = "both") -> Optional[int]:
        """
        Find audio device by name pattern.
        
        Args:
            name_patterns: List of name patterns to search for
            device_type: "input", "output", or "both"
            
        Returns:
            Device index if found, None otherwise
        """
        try:
            # --- FIX: Cast query_devices() to DeviceList ---
            devices: sd.DeviceList = cast(sd.DeviceList, sd.query_devices())
            
            for idx, device_untyped in enumerate(devices):
                device: dict[str, Any] = device_untyped # Cast to dict
                device_name = device['name'].lower()
                
                # Check device type
                is_input = device['max_input_channels'] > 0
                is_output = device['max_output_channels'] > 0
                
                if device_type == "input" and not is_input:
                    continue
                if device_type == "output" and not is_output:
                    continue
                
                # Check name patterns
                for pattern in name_patterns:
                    if pattern.lower() in device_name:
                        logger.info(f"Found {device_type} device: [{idx}] {device['name']}")
                        return idx
            
            logger.warning(f"No {device_type} device found matching patterns: {name_patterns}")
            return None
            
        except Exception as e:
            logger.error(f"Error finding device: {e}")
            return None
    
    @classmethod
    def get_virtual_output_device(cls) -> Optional[int]:
        """
        Get virtual audio output device (where bot plays audio).
        This should be set as the microphone input in Google Meet.
        
        Returns:
            Device index or None
        """
        return cls.find_device_by_name(cls.VIRTUAL_OUTPUT_NAMES, device_type="output")
    
    @classmethod
    def get_virtual_input_device(cls) -> Optional[int]:
        """
        Get virtual audio input device (where bot receives audio from Meet).
        This captures audio coming from Google Meet.
        
        Returns:
            Device index or None
        """
        return cls.find_device_by_name(cls.VIRTUAL_INPUT_NAMES, device_type="input")
    
    @staticmethod
    def test_virtual_audio_setup() -> Tuple[bool, str]:
        """
        Test virtual audio cable setup.
        
        Returns:
            Tuple of (success: bool, message: str)
        """
        logger.info("\n" + "="*60)
        logger.info("TESTING VIRTUAL AUDIO SETUP")
        logger.info("="*60)
        
        # List all devices
        VirtualAudioConfig.list_all_audio_devices()
        
        # Check for virtual output (bot speaks)
        output_device = VirtualAudioConfig.get_virtual_output_device()
        if output_device is None:
            msg = (
                "❌ Virtual OUTPUT device not found!\n"
                "Bot audio will not work in Google Meet.\n\n"
                "SETUP INSTRUCTIONS:\n"
                "1. Install VB-Audio Virtual Cable: https://vb-audio.com/Cable/\n"
                "2. After installation, restart your computer\n"
                "3. In Google Meet settings, set Microphone to 'CABLE Input'\n"
                "4. Bot will play audio through this virtual cable"
            )
            logger.error(msg)
            return False, msg
        
        # Check for virtual input (bot listens)
        input_device = VirtualAudioConfig.get_virtual_input_device()
        if input_device is None:
            msg = (
                "⚠️ Virtual INPUT device not found!\n"
                "Bot may not hear candidate responses.\n\n"
                "SETUP INSTRUCTIONS:\n"
                "1. Make sure VB-Audio Virtual Cable is installed\n"
                "2. The 'CABLE Output' device should appear\n"
                "3. Configure audio routing to capture Meet audio to this device"
            )
            logger.warning(msg)
            return False, msg
        
        # Success
        msg = (
            "✅ Virtual audio setup looks good!\n\n"
            f"Output device: {output_device} (Bot will play audio here)\n"
            f"Input device: {input_device} (Bot will record from here)\n\n"
            "IMPORTANT: In Google Meet settings:\n"
            "- Set Microphone to 'CABLE Input' (or similar virtual device)\n"
            "- Set Speakers to your normal speakers/headphones"
        )
        logger.info(msg)
        return True, msg
    
    @staticmethod
    def verify_sample_rate_support(device_idx: int, target_rate: int = 48000) -> bool:
        """
        Verify if a device supports a specific sample rate.
        
        Args:
            device_idx: Device index
            target_rate: Target sample rate (default 48000 for Google Meet)
            
        Returns:
            True if supported
        """
        try:
            # --- FIX: Check instance type for single device query ---
            device_info = sd.query_devices(device_idx)
            if not isinstance(device_info, dict):
                logger.error(f"Could not query device {device_idx}, got {type(device_info)}")
                return False
            
            device: dict[str, Any] = device_info
            # --- END FIX ---
            
            # Try to check if the device supports the target rate
            try:
                sd.check_output_settings(
                    device=device_idx,
                    samplerate=target_rate
                )
                logger.info(f"✅ Device {device_idx} supports {target_rate}Hz")
                return True
            except Exception as e:
                logger.warning(f"⚠️ Device {device_idx} may not support {target_rate}Hz: {e}")
                logger.info(f"Default rate: {device['default_samplerate']}Hz")
                return False
                
        except Exception as e:
            logger.error(f"Error checking sample rate: {e}")
            return False


# Run self-test when module is imported
if __name__ == "__main__":
    # Self-test
    success, message = VirtualAudioConfig.test_virtual_audio_setup()
    print("\n" + message)
    
    # Test 48kHz support
    output_device = VirtualAudioConfig.get_virtual_output_device()
    if output_device is not None:
        VirtualAudioConfig.verify_sample_rate_support(output_device, 48000)
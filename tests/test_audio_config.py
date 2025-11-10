import pytest
from unittest.mock import patch, MagicMock, call
import sounddevice as sd
import os
import sys

# --- Add project root to path ---
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, project_root)
# --- End Path Setup ---

from app.config.audio_config import VirtualAudioConfig

# --- Mock Device Data ---

MOCK_DEVICES = [
    {
        'name': 'Microphone (Realtek High Definition Audio)',
        'index': 0,
        'max_input_channels': 2,
        'max_output_channels': 0,
        'default_samplerate': 48000.0
    },
    {
        'name': 'Speakers (Realtek High Definition Audio)',
        'index': 1,
        'max_input_channels': 0,
        'max_output_channels': 2,
        'default_samplerate': 48000.0
    },
    {
        'name': 'CABLE Input (VB-Audio Virtual Cable)',
        'index': 2,
        'max_input_channels': 0,
        'max_output_channels': 2,
        'default_samplerate': 48000.0
    },
    {
        'name': 'CABLE Output (VB-Audio Virtual Cable)',
        'index': 3,
        'max_input_channels': 2,
        'max_output_channels': 0,
        'default_samplerate': 48000.0
    },
    {
        'name': 'VoiceMeeter Input (VB-Audio VoiceMeeter)',
        'index': 4,
        'max_input_channels': 0,
        'max_output_channels': 8,
        'default_samplerate': 48000.0
    },
    {
        'name': 'VoiceMeeter Output (VB-Audio VoiceMeeter)',
        'index': 5,
        'max_input_channels': 8,
        'max_output_channels': 0,
        'default_samplerate': 48000.0
    },
    {
        'name': 'Stereo Mix (Realtek)',
        'index': 6,
        'max_input_channels': 2,
        'max_output_channels': 2,
        'default_samplerate': 44100.0
    },
    {
        'name': 'BlackHole 2ch',
        'index': 7,
        'max_input_channels': 2,
        'max_output_channels': 2,
        'default_samplerate': 48000.0
    }
]

# Mock DeviceList that behaves like a list
class MockDeviceList(list):
    """Mock DeviceList that can be enumerated."""
    pass

# --- Fixtures ---

@pytest.fixture
def mock_query_devices():
    """Mock sounddevice.query_devices to return mock devices."""
    with patch('sounddevice.query_devices') as mock_query:
        mock_device_list = MockDeviceList(MOCK_DEVICES)
        mock_query.return_value = mock_device_list
        yield mock_query

@pytest.fixture
def mock_query_devices_no_virtual():
    """Mock sounddevice.query_devices with no virtual devices."""
    devices_no_virtual = [d for d in MOCK_DEVICES if 'CABLE' not in d['name'] and 'VoiceMeeter' not in d['name'] and 'BlackHole' not in d['name'] and 'Stereo Mix' not in d['name']]
    with patch('sounddevice.query_devices') as mock_query:
        mock_device_list = MockDeviceList(devices_no_virtual)
        mock_query.return_value = mock_device_list
        yield mock_query

@pytest.fixture
def mock_query_devices_error():
    """Mock sounddevice.query_devices to raise an error."""
    with patch('sounddevice.query_devices', side_effect=Exception("Device query failed")):
        yield

@pytest.fixture
def mock_check_output_settings():
    """Mock sounddevice.check_output_settings."""
    with patch('sounddevice.check_output_settings') as mock_check:
        yield mock_check


# --- Test Cases ---

class TestListAllAudioDevices:
    """Tests for VirtualAudioConfig.list_all_audio_devices"""

    def test_list_devices_success(self, mock_query_devices):
        """Tests successful listing of all audio devices."""
        devices = VirtualAudioConfig.list_all_audio_devices()
        
        assert len(devices) == len(MOCK_DEVICES)
        mock_query_devices.assert_called_once()

    def test_list_devices_formats_output(self, mock_query_devices, caplog):
        """Tests that device listing produces formatted output."""
        with caplog.at_level('INFO'):
            VirtualAudioConfig.list_all_audio_devices()
        
        # Check that device names appear in logs
        assert 'CABLE Input' in caplog.text
        assert 'CABLE Output' in caplog.text
        assert 'AVAILABLE AUDIO DEVICES' in caplog.text

    def test_list_devices_shows_device_types(self, mock_query_devices, caplog):
        """Tests that device types (INPUT/OUTPUT) are displayed."""
        with caplog.at_level('INFO'):
            VirtualAudioConfig.list_all_audio_devices()
        
        assert 'INPUT' in caplog.text
        assert 'OUTPUT' in caplog.text

    def test_list_devices_shows_sample_rates(self, mock_query_devices, caplog):
        """Tests that sample rates are displayed."""
        with caplog.at_level('INFO'):
            VirtualAudioConfig.list_all_audio_devices()
        
        assert '48000' in caplog.text or '44100' in caplog.text

    def test_list_devices_error_handling(self, mock_query_devices_error, caplog):
        """Tests error handling when device query fails."""
        with caplog.at_level('ERROR'):
            devices = VirtualAudioConfig.list_all_audio_devices()
        
        assert devices == []
        assert 'Error listing audio devices' in caplog.text

    def test_list_devices_empty(self):
        """Tests listing when no devices are available."""
        with patch('sounddevice.query_devices', return_value=MockDeviceList([])):
            devices = VirtualAudioConfig.list_all_audio_devices()
            assert len(devices) == 0


class TestFindDeviceByName:
    """Tests for VirtualAudioConfig.find_device_by_name"""

    def test_find_output_device_success(self, mock_query_devices):
        """Tests finding an output device by name."""
        result = VirtualAudioConfig.find_device_by_name(['CABLE Input'], device_type="output")
        
        assert result == 2  # CABLE Input is at index 2

    def test_find_input_device_success(self, mock_query_devices):
        """Tests finding an input device by name."""
        result = VirtualAudioConfig.find_device_by_name(['CABLE Output'], device_type="input")
        
        assert result == 3  # CABLE Output is at index 3

    def test_find_device_both_types(self, mock_query_devices):
        """Tests finding device with 'both' type (input and output)."""
        result = VirtualAudioConfig.find_device_by_name(['Stereo Mix'], device_type="both")
        
        assert result == 6  # Stereo Mix supports both

    def test_find_device_case_insensitive(self, mock_query_devices):
        """Tests that device name search is case-insensitive."""
        result = VirtualAudioConfig.find_device_by_name(['cable input'], device_type="output")
        
        assert result == 2

    def test_find_device_partial_match(self, mock_query_devices):
        """Tests that partial name matches work."""
        result = VirtualAudioConfig.find_device_by_name(['CABLE'], device_type="output")
        
        assert result == 2  # First CABLE device found

    def test_find_device_multiple_patterns(self, mock_query_devices):
        """Tests finding device with multiple name patterns."""
        result = VirtualAudioConfig.find_device_by_name(
            ['NonExistent', 'CABLE Input', 'VoiceMeeter'], 
            device_type="output"
        )
        
        assert result == 2  # Should find CABLE Input

    def test_find_device_not_found(self, mock_query_devices, caplog):
        """Tests when device is not found."""
        with caplog.at_level('WARNING'):
            result = VirtualAudioConfig.find_device_by_name(
                ['NonExistentDevice'], 
                device_type="output"
            )
        
        assert result is None
        assert 'No output device found' in caplog.text

    def test_find_device_wrong_type_input(self, mock_query_devices):
        """Tests that input-only devices are not returned when searching for output."""
        result = VirtualAudioConfig.find_device_by_name(['CABLE Output'], device_type="output")
        
        assert result is None  # CABLE Output is input-only

    def test_find_device_wrong_type_output(self, mock_query_devices):
        """Tests that output-only devices are not returned when searching for input."""
        result = VirtualAudioConfig.find_device_by_name(['CABLE Input'], device_type="input")
        
        assert result is None  # CABLE Input is output-only

    def test_find_device_error_handling(self, mock_query_devices_error, caplog):
        """Tests error handling during device search."""
        with caplog.at_level('ERROR'):
            result = VirtualAudioConfig.find_device_by_name(['CABLE'], device_type="output")
        
        assert result is None
        assert 'Error finding device' in caplog.text

    def test_find_voicemeeter_device(self, mock_query_devices):
        """Tests finding VoiceMeeter devices."""
        result = VirtualAudioConfig.find_device_by_name(['VoiceMeeter Input'], device_type="output")
        
        assert result == 4

    def test_find_blackhole_device(self, mock_query_devices):
        """Tests finding BlackHole device."""
        result = VirtualAudioConfig.find_device_by_name(['BlackHole'], device_type="both")
        
        assert result == 7


class TestGetVirtualOutputDevice:
    """Tests for VirtualAudioConfig.get_virtual_output_device"""

    def test_get_virtual_output_device_cable(self, mock_query_devices):
        """Tests getting virtual output device (CABLE Input)."""
        result = VirtualAudioConfig.get_virtual_output_device()
        
        assert result == 2  # CABLE Input

    def test_get_virtual_output_device_voicemeeter(self):
        """Tests getting VoiceMeeter as virtual output."""
        devices_only_voicemeeter = [d for d in MOCK_DEVICES if 'VoiceMeeter' in d['name'] or 'Realtek' in d['name']]
        with patch('sounddevice.query_devices', return_value=MockDeviceList(devices_only_voicemeeter)):
            result = VirtualAudioConfig.get_virtual_output_device()
            
            assert result is not None
            assert result == 2  # VoiceMeeter Input in filtered list

    def test_get_virtual_output_device_not_found(self, mock_query_devices_no_virtual, caplog):
        """Tests when no virtual output device is found."""
        with caplog.at_level('WARNING'):
            result = VirtualAudioConfig.get_virtual_output_device()
        
        assert result is None
        assert 'No output device found' in caplog.text

    def test_get_virtual_output_device_error(self, mock_query_devices_error, caplog):
        """Tests error handling when getting virtual output."""
        with caplog.at_level('ERROR'):
            result = VirtualAudioConfig.get_virtual_output_device()
        
        assert result is None


class TestGetVirtualInputDevice:
    """Tests for VirtualAudioConfig.get_virtual_input_device"""

    def test_get_virtual_input_device_cable(self, mock_query_devices):
        """Tests getting virtual input device (CABLE Output)."""
        result = VirtualAudioConfig.get_virtual_input_device()
        
        assert result == 3  # CABLE Output

    def test_get_virtual_input_device_voicemeeter(self):
        """Tests getting VoiceMeeter as virtual input."""
        devices_only_voicemeeter = [d for d in MOCK_DEVICES if 'VoiceMeeter' in d['name'] or 'Realtek' in d['name']]
        with patch('sounddevice.query_devices', return_value=MockDeviceList(devices_only_voicemeeter)):
            result = VirtualAudioConfig.get_virtual_input_device()
            
            assert result is not None
            assert result == 3  # VoiceMeeter Output in filtered list

    def test_get_virtual_input_device_not_found(self, mock_query_devices_no_virtual, caplog):
        """Tests when no virtual input device is found."""
        with caplog.at_level('WARNING'):
            result = VirtualAudioConfig.get_virtual_input_device()
        
        assert result is None
        assert 'No input device found' in caplog.text

    def test_get_virtual_input_device_error(self, mock_query_devices_error, caplog):
        """Tests error handling when getting virtual input."""
        with caplog.at_level('ERROR'):
            result = VirtualAudioConfig.get_virtual_input_device()
        
        assert result is None


class TestTestVirtualAudioSetup:
    """Tests for VirtualAudioConfig.test_virtual_audio_setup"""

    def test_setup_success_both_devices(self, mock_query_devices, caplog):
        """Tests successful setup with both input and output devices."""
        with caplog.at_level('INFO'):
            success, message = VirtualAudioConfig.test_virtual_audio_setup()
        
        assert success is True
        assert 'Virtual audio setup looks good' in message
        assert 'Output device: 2' in message
        assert 'Input device: 3' in message
        assert 'TESTING VIRTUAL AUDIO SETUP' in caplog.text

    def test_setup_no_output_device(self, mock_query_devices_no_virtual, caplog):
        """Tests setup failure when output device is missing."""
        with caplog.at_level('ERROR'):
            success, message = VirtualAudioConfig.test_virtual_audio_setup()
        
        assert success is False
        assert 'Virtual OUTPUT device not found' in message
        assert 'Install VB-Audio Virtual Cable' in message

    def test_setup_no_input_device(self, caplog):
        """Tests setup warning when input device is missing."""
        # Create devices with only output (no input) - exclude ALL virtual input devices
        devices_no_input = [
            d for d in MOCK_DEVICES 
            if d['name'] != 'CABLE Output (VB-Audio Virtual Cable)' 
            and 'VoiceMeeter Output' not in d['name']
            and 'Stereo Mix' not in d['name']
            and 'BlackHole' not in d['name']
            and 'Monitor of' not in d['name']
        ]
        
        with patch('sounddevice.query_devices', return_value=MockDeviceList(devices_no_input)):
            with caplog.at_level('WARNING'):
                success, message = VirtualAudioConfig.test_virtual_audio_setup()
        
        assert success is False
        assert 'Virtual INPUT device not found' in message
        assert 'CABLE Output' in message

    def test_setup_calls_list_devices(self, mock_query_devices):
        """Tests that setup calls list_all_audio_devices."""
        with patch.object(VirtualAudioConfig, 'list_all_audio_devices') as mock_list:
            VirtualAudioConfig.test_virtual_audio_setup()
            mock_list.assert_called_once()

    def test_setup_provides_instructions(self, mock_query_devices):
        """Tests that setup provides usage instructions."""
        success, message = VirtualAudioConfig.test_virtual_audio_setup()
        
        assert 'Google Meet settings' in message or 'IMPORTANT' in message
        assert 'Microphone' in message


class TestVerifySampleRateSupport:
    """Tests for VirtualAudioConfig.verify_sample_rate_support"""

    def test_verify_sample_rate_success(self, mock_query_devices, mock_check_output_settings):
        """Tests successful sample rate verification."""
        # Mock single device query
        with patch('sounddevice.query_devices', return_value=MOCK_DEVICES[2]):
            result = VirtualAudioConfig.verify_sample_rate_support(2, 48000)
        
        assert result is True
        mock_check_output_settings.assert_called_once_with(device=2, samplerate=48000)

    def test_verify_sample_rate_not_supported(self, mock_check_output_settings, caplog):
        """Tests when sample rate is not supported."""
        mock_check_output_settings.side_effect = Exception("Sample rate not supported")
        
        with patch('sounddevice.query_devices', return_value=MOCK_DEVICES[2]):
            with caplog.at_level('WARNING'):
                result = VirtualAudioConfig.verify_sample_rate_support(2, 48000)
        
        assert result is False
        assert 'may not support' in caplog.text

    def test_verify_sample_rate_custom_rate(self, mock_query_devices, mock_check_output_settings):
        """Tests verification with custom sample rate."""
        with patch('sounddevice.query_devices', return_value=MOCK_DEVICES[2]):
            result = VirtualAudioConfig.verify_sample_rate_support(2, 44100)
        
        mock_check_output_settings.assert_called_once_with(device=2, samplerate=44100)

    def test_verify_sample_rate_device_query_error(self, caplog):
        """Tests error handling when device query fails."""
        with patch('sounddevice.query_devices', side_effect=Exception("Device not found")):
            with caplog.at_level('ERROR'):
                result = VirtualAudioConfig.verify_sample_rate_support(99, 48000)
        
        assert result is False
        assert 'Error checking sample rate' in caplog.text

    def test_verify_sample_rate_invalid_device_type(self, caplog):
        """Tests when query_devices returns non-dict type."""
        with patch('sounddevice.query_devices', return_value="invalid_type"):
            with caplog.at_level('ERROR'):
                result = VirtualAudioConfig.verify_sample_rate_support(2, 48000)
        
        assert result is False
        assert 'Could not query device' in caplog.text

    def test_verify_sample_rate_shows_default_rate(self, mock_check_output_settings, caplog):
        """Tests that default sample rate is shown when target is not supported."""
        mock_check_output_settings.side_effect = Exception("Not supported")
        
        with patch('sounddevice.query_devices', return_value=MOCK_DEVICES[2]):
            with caplog.at_level('INFO'):
                VirtualAudioConfig.verify_sample_rate_support(2, 96000)
        
        assert 'Default rate: 48000' in caplog.text


class TestConstantDefinitions:
    """Tests for class constant definitions"""

    def test_virtual_output_names_defined(self):
        """Tests that VIRTUAL_OUTPUT_NAMES is properly defined."""
        assert hasattr(VirtualAudioConfig, 'VIRTUAL_OUTPUT_NAMES')
        assert isinstance(VirtualAudioConfig.VIRTUAL_OUTPUT_NAMES, list)
        assert len(VirtualAudioConfig.VIRTUAL_OUTPUT_NAMES) > 0
        assert 'CABLE Input' in VirtualAudioConfig.VIRTUAL_OUTPUT_NAMES

    def test_virtual_input_names_defined(self):
        """Tests that VIRTUAL_INPUT_NAMES is properly defined."""
        assert hasattr(VirtualAudioConfig, 'VIRTUAL_INPUT_NAMES')
        assert isinstance(VirtualAudioConfig.VIRTUAL_INPUT_NAMES, list)
        assert len(VirtualAudioConfig.VIRTUAL_INPUT_NAMES) > 0
        assert 'CABLE Output' in VirtualAudioConfig.VIRTUAL_INPUT_NAMES

    def test_output_names_include_common_devices(self):
        """Tests that output names include common virtual audio devices."""
        output_names = VirtualAudioConfig.VIRTUAL_OUTPUT_NAMES
        assert any('CABLE' in name for name in output_names)
        assert any('VoiceMeeter' in name for name in output_names)
        assert any('BlackHole' in name for name in output_names)

    def test_input_names_include_common_devices(self):
        """Tests that input names include common virtual audio devices."""
        input_names = VirtualAudioConfig.VIRTUAL_INPUT_NAMES
        assert any('CABLE' in name for name in input_names)
        assert any('VoiceMeeter' in name for name in input_names)
        assert any('BlackHole' in name for name in input_names)


class TestIntegrationScenarios:
    """Integration-style tests for complete workflows"""

    def test_complete_audio_detection_workflow(self, mock_query_devices):
        """Tests complete audio device detection workflow."""
        # List devices
        devices = VirtualAudioConfig.list_all_audio_devices()
        assert len(devices) > 0
        
        # Find output device
        output_idx = VirtualAudioConfig.get_virtual_output_device()
        assert output_idx is not None
        
        # Find input device
        input_idx = VirtualAudioConfig.get_virtual_input_device()
        assert input_idx is not None
        
        # Test setup
        success, message = VirtualAudioConfig.test_virtual_audio_setup()
        assert success is True

    def test_no_virtual_devices_workflow(self, mock_query_devices_no_virtual):
        """Tests workflow when no virtual devices are available."""
        devices = VirtualAudioConfig.list_all_audio_devices()
        assert len(devices) > 0  # Still has regular devices
        
        output_idx = VirtualAudioConfig.get_virtual_output_device()
        assert output_idx is None
        
        input_idx = VirtualAudioConfig.get_virtual_input_device()
        assert input_idx is None
        
        success, message = VirtualAudioConfig.test_virtual_audio_setup()
        assert success is False
        assert 'not found' in message

    def test_device_verification_workflow(self, mock_query_devices, mock_check_output_settings):
        """Tests device verification workflow."""
        # Get device
        output_idx = VirtualAudioConfig.get_virtual_output_device()
        assert output_idx is not None
        
        # Verify sample rate
        with patch('sounddevice.query_devices', return_value=MOCK_DEVICES[output_idx]):
            supported = VirtualAudioConfig.verify_sample_rate_support(output_idx, 48000)
            assert supported is True
# app/core/log_config.py
import logging.config
import os
from pathlib import Path

# Create a 'logs' directory at the project root if it doesn't exist
log_dir = Path(__file__).resolve().parents[2] / 'logs'
log_dir.mkdir(exist_ok=True)

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'default': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S',
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'default',
            'level': 'INFO',
            'stream': 'ext://sys.stdout',
        },
        'info_file_handler': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': os.path.join(log_dir, 'info.log'),
            'maxBytes': 10485760,  # 10 MB
            'backupCount': 5,
            'formatter': 'default',
            'level': 'INFO',
            'encoding': 'utf-8',
        },
        'error_file_handler': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': os.path.join(log_dir, 'error.log'),
            'maxBytes': 10485760,  # 10 MB
            'backupCount': 5,
            'formatter': 'default',
            'level': 'ERROR',
            'encoding': 'utf-8',
        },
    },
    'loggers': {
        # Suppress noisy libav errors (video decoding issues are common and non-critical)
        'libav.libvpx': {
            'level': 'CRITICAL',
            'handlers': ['console'],
            'propagate': False,
        },
        'libav': {
            'level': 'CRITICAL',
            'handlers': ['console'],
            'propagate': False,
        },
        # Suppress streamlit_webrtc shutdown errors (harmless event loop warnings)
        'streamlit_webrtc.shutdown': {
            'level': 'CRITICAL',
            'handlers': ['console'],
            'propagate': False,
        },
    },
    'root': {
        'level': 'INFO',
        'handlers': ['console', 'info_file_handler', 'error_file_handler'],
    },
}

def setup_logging():
    """Applies the logging configuration."""
    logging.config.dictConfig(LOGGING_CONFIG)
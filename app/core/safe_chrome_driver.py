# app/core/safe_chrome_driver.py
"""
Safe Chrome Driver Context Manager
Ensures proper cleanup without handle errors
"""
import logging
import time
import atexit
import undetected_chromedriver as uc
from typing import Optional, List

logger = logging.getLogger(__name__)

# --- START FIX: Register cleanup on program exit ---
_active_drivers: List['SafeChromeDriver'] = []

def _cleanup_all_drivers():
    """Emergency cleanup of all active drivers on program exit"""
    logger.info(f"ATEIXT: Cleaning up {len(_active_drivers)} dangling drivers...")
    for driver_wrapper in list(_active_drivers): # Iterate over a copy
        try:
            driver_wrapper.close()
        except:
            pass

atexit.register(_cleanup_all_drivers)
# --- END FIX ---


class SafeChromeDriver:
    """
    Context manager for undetected_chromedriver that handles cleanup safely.
    
    Usage:
        with SafeChromeDriver(headless=True) as driver:
            driver.get("https://google.com")
            # ... use driver ...
        # Automatically cleaned up here
    """
    
    def __init__(self, options: Optional[uc.ChromeOptions] = None, **kwargs):
        """
        Initialize safe Chrome driver.
        
        Args:
            options: Chrome options (will be created if None)
            **kwargs: Additional arguments for uc.Chrome()
        """
        self.options = options or uc.ChromeOptions()
        self.kwargs = kwargs
        self.driver: Optional[uc.Chrome] = None
        self._closed = False
        
        # --- START FIX: Register driver for atexit cleanup ---
        _active_drivers.append(self)
        # --- END FIX ---
    
    def __enter__(self):
        """Start the Chrome driver"""
        try:
            self.driver = uc.Chrome(options=self.options, **self.kwargs)
            logger.info("✅ Chrome driver started")
            return self.driver
        except Exception as e:
            logger.error(f"❌ Failed to start Chrome driver: {e}")
            raise
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Safely cleanup Chrome driver"""
        self.close()
        return False  # Don't suppress exceptions
    
    def close(self):
        """Safely close the Chrome driver"""
        if self._closed or not self.driver:
            return
        
        try:
            # Step 1: Close all windows
            try:
                handles = self.driver.window_handles
                for handle in handles:
                    try:
                        self.driver.switch_to.window(handle)
                        self.driver.close()
                    except:
                        pass
            except:
                pass
            
            # Step 2: Quit driver with error suppression
            try:
                self.driver.quit()
            except OSError as e:
                # Ignore "invalid handle" errors on Windows
                if "invalid" not in str(e).lower():
                    logger.warning(f"Driver quit warning: {e}")
            except Exception as e:
                logger.warning(f"Driver quit error: {e}")
            
            # Step 3: Wait for cleanup
            time.sleep(0.2)
            
            logger.info("🧹 Chrome driver closed safely")
            
        except Exception as e:
            logger.error(f"Error during Chrome cleanup: {e}")
        finally:
            # --- START FIX: Unregister driver from atexit ---
            if self in _active_drivers:
                _active_drivers.remove(self)
            # --- END FIX ---
            self._closed = True
            self.driver = None


def create_safe_chrome_driver(
    headless: bool = False,
    no_sandbox: bool = True,
    disable_dev_shm: bool = True,
    **kwargs
) -> SafeChromeDriver:
    """
    Factory function to create a safe Chrome driver with common options.
    
    Args:
        headless: Run in headless mode
        no_sandbox: Disable sandbox (needed for some environments)
        disable_dev_shm: Disable /dev/shm usage
        **kwargs: Additional Chrome options
        
    Returns:
        SafeChromeDriver context manager
        
    Usage:
        with create_safe_chrome_driver(headless=True) as driver:
            driver.get("https.example.com")
    """
    options = uc.ChromeOptions()
    
    if headless:
        options.add_argument('--headless=new')
    
    if no_sandbox:
        options.add_argument('--no-sandbox')
    
    if disable_dev_shm:
        options.add_argument('--disable-dev-shm-usage')
    
    # Additional common arguments
    options.add_argument('--disable-gpu')
    options.add_argument('--disable-blink-features=AutomationControlled')
    
    return SafeChromeDriver(options=options, **kwargs)
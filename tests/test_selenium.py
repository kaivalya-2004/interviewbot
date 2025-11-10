from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import sys # Import sys for the guard

def main():
    """Main function to run the Selenium diagnostic test."""
    options = Options()
    options.add_argument("--start-maximized")

    # Use a raw string (r"...") to avoid escape sequence errors
    service = Service(r"C:\Tools\chromedriver\chromedriver.exe")  # path optional if added to PATH

    driver = None
    try:
        driver = webdriver.Chrome(service=service, options=options)
        
        # --- FIX: Corrected the URL ---
        driver.get("https://www.google.com")
        # --- END FIX ---

        print("Title:", driver.title)
        
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if driver:
            driver.quit()

# --- FIX: Add __name__ == "__main__" guard ---
# This prevents pytest from running this file during test collection.
# You can still run this file directly to test Selenium:
# python tests/test_selenium.py
if __name__ == "__main__":
    main()
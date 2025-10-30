from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options

options = Options()
options.add_argument("--start-maximized")

# Create a Service object for ChromeDriver
service = Service("C:\Tools\chromedriver\chromedriver.exe")  # path optional if added to PATH

driver = webdriver.Chrome(service=service, options=options)
driver.get("https://www.google.com")

print("Title:", driver.title)
driver.quit()

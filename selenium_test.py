"""
Selenium Test Script for DemandAI Application
Tests: Login, Navigation, CSV Upload, and Prediction Display
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time
import os

class DemandAITester:
    def __init__(self, base_url="http://localhost:5173"):
        self.base_url = base_url
        print("🚀 Initializing Selenium WebDriver...")
        
        options = webdriver.ChromeOptions()
        options.add_argument("--start-maximized")
        # options.add_argument("--headless")  # Uncomment for headless mode
        
        self.driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=options
        )
        self.wait = WebDriverWait(self.driver, 10)
        print("✅ Browser initialized!")
    
    def test_login_page(self):
        """Test 1: Verify login page loads correctly"""
        print("\n📝 TEST 1: Login Page Load")
        self.driver.get(f"{self.base_url}/login")
        time.sleep(2)
        
        try:
            # Check for email input
            email_input = self.wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='email'], input[name='email']"))
            )
            print("  ✅ Email input found")
            
            # Check for password input
            password_input = self.driver.find_element(By.CSS_SELECTOR, "input[type='password']")
            print("  ✅ Password input found")
            
            # Check for submit button
            submit_btn = self.driver.find_element(By.CSS_SELECTOR, "button[type='submit']")
            print("  ✅ Submit button found")
            
            return True
        except Exception as e:
            print(f"  ❌ FAILED: {e}")
            return False
    
    def test_login_flow(self, email="test@test.com", password="password123"):
        """Test 2: Attempt login with credentials"""
        print(f"\n📝 TEST 2: Login Flow (email={email})")
        
        try:
            self.driver.get(f"{self.base_url}/login")
            time.sleep(2)
            
            # Fill email
            email_input = self.wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='email'], input[name='email']"))
            )
            email_input.clear()
            email_input.send_keys(email)
            print(f"  ✅ Entered email: {email}")
            
            # Fill password
            password_input = self.driver.find_element(By.CSS_SELECTOR, "input[type='password']")
            password_input.clear()
            password_input.send_keys(password)
            print("  ✅ Entered password")
            
            # Click submit
            submit_btn = self.driver.find_element(By.CSS_SELECTOR, "button[type='submit']")
            submit_btn.click()
            print("  ✅ Clicked login button")
            
            time.sleep(3)
            
            # Check if redirected (successful login)
            current_url = self.driver.current_url
            if "/login" not in current_url:
                print(f"  ✅ Login successful! Redirected to: {current_url}")
                return True
            else:
                # Check for error message
                try:
                    error_msg = self.driver.find_element(By.CSS_SELECTOR, ".error, .alert, [role='alert']")
                    print(f"  ⚠️ Login failed: {error_msg.text}")
                except:
                    print("  ⚠️ Login failed (still on login page)")
                return False
                
        except Exception as e:
            print(f"  ❌ FAILED: {e}")
            return False
    
    def test_signup_and_login(self, email="selenium_test@test.com", password="Test123!"):
        """Test 3: Create account then login"""
        print(f"\n📝 TEST 3: Signup then Login (email={email})")
        
        try:
            # Go to signup
            self.driver.get(f"{self.base_url}/signup")
            time.sleep(2)
            
            # Fill signup form
            email_input = self.wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='email'], input[name='email']"))
            )
            email_input.clear()
            email_input.send_keys(email)
            
            password_input = self.driver.find_element(By.CSS_SELECTOR, "input[type='password']")
            password_input.clear()
            password_input.send_keys(password)
            
            # Look for confirm password field
            confirm_inputs = self.driver.find_elements(By.CSS_SELECTOR, "input[type='password']")
            if len(confirm_inputs) > 1:
                confirm_inputs[1].clear()
                confirm_inputs[1].send_keys(password)
            
            # Submit signup
            submit_btn = self.driver.find_element(By.CSS_SELECTOR, "button[type='submit']")
            submit_btn.click()
            print("  ✅ Submitted signup form")
            
            time.sleep(3)
            
            # Now try login
            return self.test_login_flow(email, password)
            
        except Exception as e:
            print(f"  ❌ FAILED: {e}")
            return False
    
    def test_upload_csv(self, csv_path):
        """Test 4: Upload CSV and check predictions"""
        print(f"\n📝 TEST 4: CSV Upload ({os.path.basename(csv_path)})")
        
        try:
            # Navigate to upload page
            self.driver.get(f"{self.base_url}/upload")
            time.sleep(2)
            
            # Find file input
            file_input = self.wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='file']"))
            )
            
            # Upload file
            file_input.send_keys(os.path.abspath(csv_path))
            print(f"  ✅ Selected file: {csv_path}")
            
            time.sleep(2)
            
            # Click upload/submit button
            upload_btn = self.driver.find_element(By.CSS_SELECTOR, "button[type='submit'], .upload-btn, button:contains('Upload')")
            upload_btn.click()
            print("  ✅ Clicked upload button")
            
            time.sleep(5)
            
            # Check for predictions
            try:
                prediction_element = self.wait.until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, ".prediction, .forecast, [class*='prediction']"))
                )
                print(f"  ✅ Predictions displayed!")
                return True
            except:
                print("  ⚠️ Predictions element not found")
                return False
                
        except Exception as e:
            print(f"  ❌ FAILED: {e}")
            return False
    
    def take_screenshot(self, name="screenshot"):
        """Save a screenshot"""
        path = f"{name}_{int(time.time())}.png"
        self.driver.save_screenshot(path)
        print(f"  📸 Screenshot saved: {path}")
        return path
    
    def cleanup(self):
        """Close browser"""
        print("\n🧹 Closing browser...")
        self.driver.quit()
        print("✅ Test complete!")

def run_tests():
    """Run all tests"""
    print("=" * 60)
    print("  DEMANDAI SELENIUM TEST SUITE")
    print("=" * 60)
    
    tester = DemandAITester()
    
    try:
        # Test 1: Login page loads
        tester.test_login_page()
        tester.take_screenshot("01_login_page")
        
        # Test 2: Try login with test account
        login_success = tester.test_login_flow("test@example.com", "password123")
        
        if not login_success:
            # Test 3: Try signup then login
            tester.test_signup_and_login()
        
        tester.take_screenshot("02_after_login")
        
        # Test 4: Upload CSV (if logged in)
        csv_path = os.path.join(os.path.dirname(__file__), "akshay.csv")
        if os.path.exists(csv_path):
            tester.test_upload_csv(csv_path)
            tester.take_screenshot("03_predictions")
        
    finally:
        tester.cleanup()

if __name__ == "__main__":
    run_tests()

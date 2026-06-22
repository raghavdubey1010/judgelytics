#!/usr/bin/env python3
"""
Judgelytics End-to-End API Test
Tests all major API endpoints
"""

import httpx
import json
import time
from typing import Optional

BASE_URL = "http://localhost:8000"
TEST_USER_USERNAME = "testuser_e2e"
TEST_USER_PASSWORD = "TestPassword123!"
TEST_USER_EMAIL = f"{TEST_USER_USERNAME}@test.com"

class JudgelyticsAPITester:
    """Test Judgelytics API endpoints."""
    
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.access_token: Optional[str] = None
        self.case_id: Optional[str] = None
        
    def print_header(self, title: str):
        """Print a formatted header."""
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}\n")
    
    def print_success(self, message: str):
        """Print success message."""
        print(f"✅ {message}")
    
    def print_error(self, message: str):
        """Print error message."""
        print(f"❌ {message}")
    
    def print_info(self, message: str):
        """Print info message."""
        print(f"ℹ️  {message}")
    
    def test_health_check(self) -> bool:
        """Test health check endpoint."""
        self.print_header("Test 1: Health Check")
        
        try:
            response = httpx.get(f"{self.base_url}/health")
            response.raise_for_status()
            data = response.json()
            
            self.print_info(f"Status: {data.get('status')}")
            self.print_info(f"Models Loaded: {data.get('models_loaded')}")
            self.print_info(f"Version: {data.get('version')}")
            
            self.print_success("Health check passed")
            return True
            
        except Exception as e:
            self.print_error(f"Health check failed: {str(e)}")
            return False
    
    def test_root_endpoint(self) -> bool:
        """Test root endpoint."""
        self.print_header("Test 2: Root Endpoint")
        
        try:
            response = httpx.get(f"{self.base_url}/")
            response.raise_for_status()
            data = response.json()
            
            self.print_info(f"App: {data.get('app')}")
            self.print_info(f"Version: {data.get('version')}")
            self.print_info(f"Docs: {data.get('docs')}")
            
            self.print_success("Root endpoint accessible")
            return True
            
        except Exception as e:
            self.print_error(f"Root endpoint failed: {str(e)}")
            return False
    
    def test_register(self) -> bool:
        """Test user registration."""
        self.print_header("Test 3: User Registration")
        
        try:
            payload = {
                "username": TEST_USER_USERNAME,
                "email": TEST_USER_EMAIL,
                "password": TEST_USER_PASSWORD,
                "full_name": "E2E Test User"
            }
            
            response = httpx.post(
                f"{self.base_url}/api/v1/auth/register",
                json=payload
            )
            
            if response.status_code == 409:
                self.print_info("User already exists (from previous test run)")
                self.print_success("Registration endpoint working")
                return True
            else:
                response.raise_for_status()
                data = response.json()
                self.print_info(f"User ID: {data.get('uid')}")
                self.print_success("Registration successful")
                return True
                
        except Exception as e:
            self.print_error(f"Registration failed: {str(e)}")
            return False
    
    def test_login(self) -> bool:
        """Test user login."""
        self.print_header("Test 4: User Login")
        
        try:
            payload = {
                "username": TEST_USER_USERNAME,
                "password": TEST_USER_PASSWORD
            }
            
            response = httpx.post(
                f"{self.base_url}/api/v1/auth/login",
                data=payload
            )
            response.raise_for_status()
            data = response.json()
            
            self.access_token = data.get("access_token")
            token_type = data.get("token_type")
            
            self.print_info(f"Token Type: {token_type}")
            self.print_info(f"Token: {self.access_token[:20]}...")
            self.print_success("Login successful")
            return True
            
        except Exception as e:
            self.print_error(f"Login failed: {str(e)}")
            return False
    
    def test_case_analysis(self) -> bool:
        """Test case analysis endpoint."""
        self.print_header("Test 5: Case Analysis")
        
        if not self.access_token:
            self.print_error("No access token available")
            return False
        
        try:
            payload = {
                "description": "We purchased a defective electronic appliance from an online retailer. "
                              "The product stopped working after 3 days despite being brand new. "
                              "The retailer refuses to provide a refund or replacement.",
                "category": "Product Defect",
                "sector": "E-commerce",
                "claim_amount": 50000.0,
                "evidence_count": 4,
                "has_legal_notice": "Yes",
                "opponent_type": "Business"
            }
            
            headers = {"Authorization": f"Bearer {self.access_token}"}
            
            response = httpx.post(
                f"{self.base_url}/api/v1/case/analyze",
                json=payload,
                headers=headers
            )
            response.raise_for_status()
            data = response.json()
            
            self.case_id = data.get("case_id")
            prediction = data.get("prediction", {})
            
            self.print_info(f"Case ID: {self.case_id}")
            self.print_info(f"Outcome: {prediction.get('outcome')}")
            self.print_info(f"Win Probability: {prediction.get('win_probability_pct')}%")
            self.print_info(f"Confidence: {prediction.get('confidence')}")
            self.print_info(f"Recommended Forum: {prediction.get('recommended_forum')}")
            self.print_info(f"Filing Fee: {prediction.get('filing_fee')}")
            self.print_info(f"Evidence Strength: {prediction.get('evidence_strength')}")
            self.print_info(f"Models Used: {', '.join(prediction.get('models_used', []))}")
            
            self.print_success("Case analysis successful")
            return True
            
        except Exception as e:
            self.print_error(f"Case analysis failed: {str(e)}")
            return False
    
    def test_case_history(self) -> bool:
        """Test case history endpoint."""
        self.print_header("Test 6: Case History")
        
        if not self.access_token:
            self.print_error("No access token available")
            return False
        
        try:
            headers = {"Authorization": f"Bearer {self.access_token}"}
            
            response = httpx.get(
                f"{self.base_url}/api/v1/case/history",
                headers=headers,
                params={"limit": 10, "offset": 0}
            )
            response.raise_for_status()
            data = response.json()
            
            self.print_info(f"Cases found: {len(data)}")
            for i, case in enumerate(data[:3], 1):
                self.print_info(f"  Case {i}: {case.get('case_id')} - {case.get('predicted_outcome')}")
            
            self.print_success("Case history retrieved")
            return True
            
        except Exception as e:
            self.print_error(f"Case history failed: {str(e)}")
            return False
    
    def test_case_detail(self) -> bool:
        """Test case detail endpoint."""
        self.print_header("Test 7: Case Detail")
        
        if not self.access_token or not self.case_id:
            self.print_error("No access token or case ID available")
            return False
        
        try:
            headers = {"Authorization": f"Bearer {self.access_token}"}
            
            response = httpx.get(
                f"{self.base_url}/api/v1/case/{self.case_id}",
                headers=headers
            )
            response.raise_for_status()
            data = response.json()
            
            self.print_info(f"Case ID: {data.get('case_id')}")
            prediction = data.get("prediction", {})
            self.print_info(f"Outcome: {prediction.get('outcome')}")
            self.print_info(f"Applicable Sections: {', '.join(prediction.get('applicable_sections', []))}")
            
            self.print_success("Case detail retrieved")
            return True
            
        except Exception as e:
            self.print_error(f"Case detail failed: {str(e)}")
            return False
    
    def run_all_tests(self) -> bool:
        """Run all tests."""
        print("\n" + "="*60)
        print("  🧪 Judgelytics End-to-End API Test")
        print("="*60)
        
        tests = [
            ("Health Check", self.test_health_check),
            ("Root Endpoint", self.test_root_endpoint),
            ("User Registration", self.test_register),
            ("User Login", self.test_login),
            ("Case Analysis", self.test_case_analysis),
            ("Case History", self.test_case_history),
            ("Case Detail", self.test_case_detail),
        ]
        
        results = []
        for name, test_func in tests:
            try:
                result = test_func()
                results.append((name, result))
            except Exception as e:
                self.print_error(f"Test '{name}' crashed: {str(e)}")
                results.append((name, False))
            time.sleep(0.2)
        
        # Summary
        self.print_header("Test Summary")
        passed = sum(1 for _, result in results if result)
        total = len(results)
        
        for name, result in results:
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"  {status}: {name}")
        
        print(f"\n  Result: {passed}/{total} tests passed")
        
        if passed == total:
            print("\n✅ All tests passed! API is ready for deployment.\n")
            return True
        else:
            print(f"\n⚠️  {total - passed} test(s) failed. Please review the errors above.\n")
            return False


def main():
    """Main entry point."""
    print("\n⏳ Waiting for API to be ready...")
    print(f"   Target: {BASE_URL}")
    
    # Wait for API to be ready
    max_retries = 30
    for i in range(max_retries):
        try:
            response = httpx.get(f"{BASE_URL}/health", timeout=2)
            if response.status_code == 200:
                print("✅ API is ready!\n")
                break
        except:
            pass
        
        if i < max_retries - 1:
            print(f"   Waiting... ({i+1}/{max_retries})")
            time.sleep(1)
    else:
        print(f"❌ API did not become ready after {max_retries} seconds")
        return False
    
    # Run tests
    tester = JudgelyticsAPITester(BASE_URL)
    success = tester.run_all_tests()
    
    return success


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)

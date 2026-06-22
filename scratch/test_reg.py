import requests

url = "http://localhost:8000/api/v1/auth/register"
data = {
    "name": "Test User",
    "email": "test2@test.com",
    "phone": "123",
    "password": "password123"
}

try:
    response = requests.post(url, json=data)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Error: {e}")

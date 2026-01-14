import requests
import json
import os

url = "http://127.0.0.1:5000/predict_csv"
file_path = os.path.abspath("test_upload.csv")

print(f"Testing URL: {url}")
print(f"Sending File: {file_path}")

try:
    response = requests.post(url, json={"file_path": file_path})
    print(f"Status Code: {response.status_code}")
    print("Response Body:")
    print(response.text)
except Exception as e:
    print(f"Request failed: {e}")

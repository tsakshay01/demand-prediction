
import requests
import json
import time

URL = "http://127.0.0.1:5000/predict"

# Test Case 1: Full Multimodal Data (Image + Text + Time Series)
print("\n--- TEST CASE 1: Full Multimodal Data ---")
payload_full = {
    "description": "High-performance gaming laptop with RTX 4090, 32GB RAM, and 240Hz OLED display.",
    "sales_history": [120, 135, 142, 128, 156, 189, 234, 198, 210, 225, 245, 267, 289, 310, 295],
    "image_url": "https://img.freepik.com/free-photo/laptop-wooden-table_53876-101452.jpg" # Sample generic laptop image
}

try:
    start = time.time()
    res = requests.post(URL, json=payload_full)
    duration = time.time() - start
    
    if res.status_code == 200:
        data = res.json()
        print("✅ Success!")
        print(f"Prediction: {data['predicted_demand']:.2f}")
        print(f"Confidence Interval: {data['confidence_interval']}")
        print(f"Modalities Used: {data['modalities_used']}")
        print(f"Model Version: {data['model_version']}")
        print(f"M odel Latency: {duration:.2f}s")
    else:
        print(f"❌ Failed: {res.text}")
except Exception as e:
    print(f"❌ Error: {e}")

# Test Case 2: Cold Start (No History, Description + Image Only)
print("\n--- TEST CASE 2: Cold Start (New Product) ---")
payload_cold = {
    "description": "Revolutionary new AI smart glasses with holographic display.",
    "sales_history": [], # Empty history
    "image_url": "https://img.freepik.com/free-photo/smart-glasses-white-background_23-2147895432.jpg"
}

try:
    start = time.time()
    res = requests.post(URL, json=payload_cold)
    
    if res.status_code == 200:
        data = res.json()
        print("✅ Success!")
        print(f"Prediction: {data['predicted_demand']:.2f}")
        print(f"Modalities Used: {data['modalities_used']}")
        # Verify that it handled cold start (likely non-zero prediction based on semantic/visual similarity)
    else:
        print(f"❌ Failed: {res.text}")
except Exception as e:
    print(f"❌ Error: {e}")

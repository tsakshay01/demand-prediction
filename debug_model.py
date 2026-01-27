import pandas as pd
import requests
import json
import ast

def test_debug():
    # Load CSV
    try:
        df = pd.read_csv('akshay.csv')
        print("CSV Loaded. Columns:", df.columns)
        first_row = df.iloc[0]
        
        # Parse history
        hist_str = first_row['sales_history']
        try:
            hist = ast.literal_eval(hist_str)
            print(f"Parsed History (First 5): {hist[:5]}")
            print(f"History Length: {len(hist)}")
        except:
            print("Failed to parse history string")
            return

        # Payload
        payload = {
            "description": first_row['description'],
            "sales_history": hist,
            "image_url": first_row['image_url']
        }
        
        # Send Request
        print("\nSending request to http://127.0.0.1:5000/predict ...")
        res = requests.post("http://127.0.0.1:5000/predict", json=payload)
        
        print(f"Status Code: {res.status_code}")
        data = res.json()
        print("Response Body:")
        print(json.dumps(data, indent=2))
        
        if 'daily_forecast' in data:
            print("\nVERIFICATION: Daily Forecast Vector:")
            print(data['daily_forecast'])
        else:
            print("\nWARNING: No daily_forecast found in response!")
        
    except Exception as e:
        print(f"Debug Script Error: {e}")

if __name__ == "__main__":
    test_debug()

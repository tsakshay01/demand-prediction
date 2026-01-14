from flask import Flask, request, jsonify
import numpy as np
import time

app = Flask(__name__)

print("="*60)
print("SIMPLIFIED ML SERVICE - FAST START MODE")
print("="*60)
print("✓ Bypassing TensorFlow/Transformers for instant startup")
print("✓ Using mock predictions with realistic logic")
print("="*60)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "service": "demand-prediction-mock", "mode": "fast-start"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        print(f"Prediction request received: {data}")
        
        # Extract request context
        sentiment = data.get('sentiment_score', 0)
        category = data.get('event_category', 'NONE')
        severity = data.get('event_severity', 'MODERATE')
        
        # Base demand with realistic variance
        base_demand = np.random.randint(1000, 1500)
        
        # Apply risk multiplier based on event
        multiplier = 1.0
        
        if category == 'LEGAL' and severity == 'CRITICAL':
            multiplier = 0.3  # 70% drop
        elif category == 'WEATHER' and severity in ['CRITICAL', 'HIGH']:
            multiplier = 0.5  # 50% drop
        elif severity == 'POSITIVE':
            multiplier = 1.5  # 50% surge
        elif sentiment:
            # Standard sentiment impact
            multiplier = 1 + (sentiment * 0.3)
        
        predicted_demand = int(base_demand * multiplier)
        confidence_lower = int(predicted_demand * 0.85)
        confidence_upper = int(predicted_demand * 1.15)
        
        response = {
            "predicted_demand": predicted_demand,
            "confidence_interval": [confidence_lower, confidence_upper],
            "model_version": "simplified-mock-v1",
            "risk_factors": {
                "category": category,
                "severity": severity,
                "multiplier_applied": round(multiplier, 2)
            },
            "note": "Fast-start mode active. Upgrade to TensorFlow for full ML capabilities."
        }
        
        print(f"→ Prediction: {predicted_demand} (multiplier: {multiplier})")
        return jsonify(response)
        
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/train', methods=['POST'])
def train():
    print("Training request received (simulated)")
    time.sleep(2)  # Simulate training time
    return jsonify({
        "message": "Mock training completed successfully", 
        "mode": "Fast-Start-Simulation",
        "note": "Install TensorFlow stack for real training"
    })

if __name__ == '__main__':
    print("\n🚀 ML Service ready on http://127.0.0.1:5000")
    print("📊 Serving intelligent mock predictions")
    print("-"*60 + "\n")
    app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False)

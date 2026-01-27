"""
Flask ML Service - Lightweight Multimodal Demand Prediction
Uses TF-IDF + Time-Series Features + RandomForest (No deep learning)
"""

from flask import Flask, request, jsonify
import numpy as np
import threading
import os
import ast

# Import the Advanced Multimodal Model (MobileNetV2 + DistilBERT + LSTM + Attention)
from advanced_model import AdvancedMultimodalModel

app = Flask(__name__)

# Global model instance
print("Loading Advanced Multimodal Model...")
demand_model = AdvancedMultimodalModel(model_dir='ml_service')
print("Model loaded!")


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "service": "multimodal-demand-prediction",
        "model_type": "lightweight-tfidf-randomforest",
        "is_trained": demand_model.is_trained
    })


@app.route('/predict', methods=['POST'])
def predict():
    """Single product prediction."""
    try:
        data = request.json
        
        description = data.get('description', '')
        image_url = data.get('image_url', None)
        sales_history = data.get('sales_history', [])
        
        # Parse sales_history if string
        if isinstance(sales_history, str):
            try:
                sales_history = ast.literal_eval(sales_history)
            except:
                sales_history = []
        
        result = demand_model.predict_single(description, sales_history, image_url)
        
        # Handle error case (insufficient data)
        if 'error' in result:
            return jsonify(result), 400
        
        # Format response for API compatibility
        predicted_value = result['prediction']
        
        return jsonify({
            "predicted_demand": predicted_value,
            "confidence_interval": result['confidence_interval'],
            "model_version": "v3.0-advanced-attention-fusion",
            "modalities_used": result['modalities_used'],
            "text_features_active": result['text_features_active'],
            "ts_features_active": result['ts_features_active']
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Batch prediction for multiple products."""
    try:
        items = request.json.get('items', [])
        if not items:
            return jsonify({"predictions": []})
        
        results = demand_model.predict_batch(items)
        
        # Extract just the prediction values for API compatibility
        predictions = []
        for r in results:
            if 'error' in r:
                predictions.append(0)  # Default for insufficient data
            else:
                predictions.append(r['prediction'])
        
        return jsonify({"predictions": predictions})

    except Exception as e:
        print(f"Batch Error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/predict_csv', methods=['POST'])
def predict_csv():
    """Process uploaded CSV file and return predictions."""
    import pandas as pd
    
    try:
        file_path = request.json.get('file_path')
        if not file_path or not os.path.exists(file_path):
            return jsonify({"error": "File not found"}), 400
            
        print(f"Processing CSV: {file_path}")
        df = pd.read_csv(file_path)
        
        # Validate Columns
        if 'sales_history' not in df.columns:
            return jsonify({"error": "CSV must have 'sales_history' column"}), 400

        # Prepare items for batch prediction
        items = []
        for _, row in df.iterrows():
            desc = row['description'] if 'description' in df.columns else "Generic Product"
            img_url = row['image_url'] if 'image_url' in df.columns else None
            items.append({
                "description": str(desc),
                "sales_history": row['sales_history'],
                "image_url": img_url
            })
        
        # Get predictions
        results = demand_model.predict_batch(items)
        
        # Structure detailed response for frontend
        detailed = []
        for i, result in enumerate(results):
            # Parse history for response
            hist = items[i]['sales_history']
            if isinstance(hist, str):
                try:
                    hist = ast.literal_eval(hist)
                except:
                    hist = []
            elif isinstance(hist, np.ndarray):
                hist = hist.tolist()
            
            pred_value = result.get('prediction', 0) if 'error' not in result else 0
            
            detailed.append({
                "product_id": df.iloc[i].get('product_id', f"P{i+1}") if 'product_id' in df.columns else f"P{i+1}",
                "description": items[i]['description'],
                "history": hist,
                "prediction": pred_value,
                "modalities_used": result.get('modalities_used', [])
            })

        return jsonify({
            "success": True, 
            "count": len(results),
            "detailed_predictions": detailed
        })

    except Exception as e:
        print(f"CSV Predict Error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/train', methods=['POST'])
def train():
    """Train/retrain the model on dataset."""
    
    def training_task():
        print("Starting training task...")
        try:
            history = demand_model.train(csv_path='ml_service/dataset.csv')
            print(f"Training finished. MSE: {history['loss'][-1]:.2f}")
        except Exception as e:
            print(f"Training failed: {e}")
    
    thread = threading.Thread(target=training_task)
    thread.start()
    
    return jsonify({
        "message": "Training started (TF-IDF + RandomForest)", 
        "mode": "Lightweight-Multimodal",
        "estimated_duration": "5-15 seconds"
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)

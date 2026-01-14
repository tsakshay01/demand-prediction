from flask import Flask, request, jsonify
from model import MultimodalDemandModel
import numpy as np
import threading
import time
import os

app = Flask(__name__)

# Global model instance
print("Loading model... this might take a minute...")
# In a real production app, we might load this lazily or in a separate worker
demand_model = MultimodalDemandModel(model_path="ml_service/model.h5") # Look here
print("Model loaded!")

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "service": "multimodal-demand-prediction"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # --- PATH A: Custom LSTM Model (from Colab) ---
        if hasattr(demand_model, 'is_custom') and demand_model.is_custom:
            sales_history = data.get('sales_history', [])
            # Fix format: LSTM expects (1, 30, 1)
            # Our web app might send raw numbers.
            # If empty, mock it for demo
            # INPUT SCALING TO MATCH COLAB TRAINING
            # The model was trained on 0-1 data. We must normalize the raw input (e.g. 1500 aka Sales).
            MAX_VAL = 2500.0  # Approximate max from dataset
            
            if not sales_history:
                input_seq = np.random.rand(1, 30, 1)
            else:
                seq = sales_history[-30:]
                if len(seq) < 30:
                    seq = [0]*(30-len(seq)) + seq
                
                # NORMALIZE HERE:
                normalized_seq = np.array(seq) / MAX_VAL 
                input_seq = normalized_seq.reshape(1, 30, 1)
            
            # Predict
            raw_pred = float(demand_model.model.predict(input_seq, verbose=0)[0][0])
            
            # OUTPUT SCALING (Denormalize)
            pred = raw_pred * MAX_VAL 
            
            return jsonify({
                "predicted_demand": int(pred),
                "confidence_interval": [int(pred * 0.9), int(pred * 1.1)],
                "model_version": "custom-hm-trained-v1",
                "note": "Using your Custom H&M Trained Model"
            })

        # --- PATH B: Default Multimodal Model (Legacy) ---
        # 1. Preprocess Text
        desc = data.get('description', "No description provided")
        tokenized = demand_model.preprocess_text([desc])
        
        # 2. Preprocess Image (Mocking image loading for now)
        # FIXED: Use Zero Placeholder
        image_data = np.zeros((1, 224, 224, 3), dtype=np.float32)
        
        # 3. Preprocess Time Series
        ts_data = np.random.rand(1, 30, 5).astype(np.float32)

        # Predict
        prediction = demand_model.model.predict({
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'image_input': image_data,
            'ts_input': ts_data
        }, verbose=0)
        
        # FIXED: De-normalize output
        MAX_SALES_SCALE = 2000000.0
        predicted_value = float(prediction[0][0]) * MAX_SALES_SCALE
        
        return jsonify({
            "predicted_demand": predicted_value,
            "confidence_interval": [predicted_value * 0.9, predicted_value * 1.1],
            "model_version": "v1.0-multimodal-lstm-bert-mobilenet"
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    try:
        items = request.json.get('items', [])
        if not items:
            return jsonify({"predictions": []})
        
        # Batch processing
        descriptions = [item.get('description', '') for item in items]
        
        # Text
        tokenized = demand_model.preprocess_text(descriptions)
        
        # Images (Mock/Placeholder for now as CSVs usually don't have direct image tensors)
        # FIXED: Zero Placeholder
        images = np.zeros((len(items), 224, 224, 3), dtype=np.float32)
        
        # TS
        # Parse sales_history better
        ts_batch = np.zeros((len(items), 30, 5), dtype=np.float32)
        for i, item in enumerate(items):
            hist = item.get('sales_history', [])
            # If string, try to parse
            if isinstance(hist, str):
                import ast
                try:
                    hist = ast.literal_eval(hist)
                except:
                    hist = []
            
            # Pad/Truncate
            hist_arr = np.array(hist[-30:]) if hist else np.zeros(30)
            if len(hist_arr) < 30:
                hist_arr = np.pad(hist_arr, (30-len(hist_arr), 0))
            
            # FIXED: Normalize
            ts_batch[i, :, 0] = hist_arr / 2000000.0

        # Predict
        predictions = demand_model.model.predict({
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'image_input': images,
            'ts_input': ts_batch
        }, verbose=0)
        
        # FIXED: De-normalize
        MAX_SALES_SCALE = 2000000.0
        results = [float(p[0]) * MAX_SALES_SCALE for p in predictions]
        return jsonify({"predictions": results})

    except Exception as e:
        print(f"Batch Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/predict_csv', methods=['POST'])
def predict_csv():
    import pandas as pd
    import ast
    try:
        file_path = request.json.get('file_path')
        if not file_path or not os.path.exists(file_path):
            return jsonify({"error": "File not found"}), 400
            
        print(f"Processing CSV: {file_path}")
        df = pd.read_csv(file_path)
        
        # Validate Columns (Relaxed for Partial Data)
        if 'sales_history' not in df.columns:
            return jsonify({"error": "CSV must have 'sales_history' column"}), 400

        # Prepare Batch
        items = []
        for _, row in df.iterrows():
            desc = row['description'] if 'description' in df.columns else "Generic Product"
            items.append({
                "description": str(desc),
                "sales_history": row['sales_history']
            })
            
        # Re-use batch logic (can refactor, but for now just call internal logic or copy-paste simplified)
        # Calling internal logic for DRY not easy with flask func but we can just use the model directly
        
        descriptions = [item['description'] for item in items]
        tokenized = demand_model.preprocess_text(descriptions)
        
        # Images (Deterministic Placeholder or Real URL)
        images = []
        for item in items:
            # 1. Try Real URL (if column exists)
            # if 'image_url' in item: ... (Future: Load URL)
            
            # 2. Deterministic "Embedding" based on Description Hash
            # FIXED: Use Zero Placeholder (avoid random noise bias)
            img_tensor = np.zeros((224, 224, 3), dtype=np.float32)
            images.append(img_tensor)
            
        images = np.array(images)
        
        # TS
        ts_batch = np.zeros((len(items), 30, 5), dtype=np.float32)
        for i, item in enumerate(items):
            hist = item['sales_history']
            if isinstance(hist, str):
                try:
                    hist = ast.literal_eval(hist)
                except:
                    hist = []
            
            hist_arr = np.array(hist[-30:]) if hist else np.zeros(30)
            if len(hist_arr) < 30:
                hist_arr = np.pad(hist_arr, (30-len(hist_arr), 0))
            
            # FIXED: Normalize
            ts_batch[i, :, 0] = hist_arr / 2000000.0
            
        predictions = demand_model.model.predict({
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'image_input': images,
            'ts_input': ts_batch
        }, verbose=0)
        
        # FIXED: De-normalize
        MAX_SALES_SCALE = 2000000.0
        results = [float(p[0]) * MAX_SALES_SCALE for p in predictions]
        
        return jsonify({
            "success": True, 
            "count": len(results),
            "predictions": results,
            "sample_first": results[0] if results else 0
        })

    except Exception as e:
        print(f"CSV Predict Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/train', methods=['POST'])
def train():
    # Run training in background to avoid timeout
    def training_task():
        print("Starting REAL training task...")
        try:
            history = demand_model.train_on_file(epochs=3)
            print(f"Training finished. Loss: {history['loss'][-1]}")
        except Exception as e:
            print(f"Training failed: {e}")
    
    thread = threading.Thread(target=training_task)
    thread.start()
    
    return jsonify({
        "message": "Training started on Synthetic High-Fidelity Data (500 samples)", 
        "mode": "Real-On-Device-Training",
        "estimated_duration": "30-60 seconds"
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)

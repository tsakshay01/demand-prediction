
import uvicorn
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from typing import List, Optional, Union, Any
import numpy as np
import os
import ast
import pandas as pd
import threading
import traceback
from advanced_model import AdvancedMultimodalModel

# Initialize FastAPI
app = FastAPI(
    title="Multimodal Demand Prediction Service",
    description="Advanced AI utilizing MobileNetV2, DistilBERT, LSTM and Attention Fusion.",
    version="3.0"
)

# CORS Middleware
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Model Instance
print("Loading Advanced Multimodal Model...")
demand_model = AdvancedMultimodalModel(model_dir='ml_service')
print("Model loaded!")

# --- Data Models ---

class PredictRequest(BaseModel):
    description: str
    sales_history: Union[List[float], str]
    image_url: Optional[str] = None

class BatchItem(BaseModel):
    description: str = "Generic Product"
    sales_history: Union[List[float], str]
    image_url: Optional[str] = None

class BatchPredictRequest(BaseModel):
    items: List[BatchItem]

class CSVPredictRequest(BaseModel):
    file_path: str

class TrainResponse(BaseModel):
    message: str
    mode: str
    estimated_duration: str

# --- Helper Functions ---

def parse_history(hist):
    if isinstance(hist, str):
        try:
            return ast.literal_eval(hist)
        except:
            return []
    return hist if hist is not None else []

# --- Routes ---

@app.get("/health")
def health():
    return {
        "status": "healthy", 
        "service": "multimodal-demand-prediction",
        "framework": "FastAPI",
        "model_type": "mobilenetv2-distilbert-lstm-attention",
        "is_trained": demand_model.is_trained
    }

@app.post("/predict")
def predict(data: PredictRequest):
    try:
        hist = parse_history(data.sales_history)
        result = demand_model.predict_single(data.description, hist, data.image_url)
        
        if 'error' in result:
            raise HTTPException(status_code=400, detail=result['error'])
            
        return {
            "predicted_demand": result['prediction'],
            "daily_forecast": result.get('daily_forecast', []),
            "confidence_interval": result['confidence_interval'],
            "model_version": "v3.0-fastapi-advanced",
            "modalities_used": result['modalities_used'],
            "text_features_active": result['text_features_active'],
            "ts_features_active": result['ts_features_active']
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch_predict")
def batch_predict(data: BatchPredictRequest):
    try:
        if not data.items:
            return {"predictions": []}
            
        items_processed = []
        for item in data.items:
            items_processed.append({
                "description": item.description,
                "sales_history": parse_history(item.sales_history),
                "image_url": item.image_url
            })
            
        results = demand_model.predict_batch(items_processed)
        
        predictions = []
        for r in results:
            predictions.append(r.get('prediction', 0) if 'error' not in r else 0)
            
        return {"predictions": predictions}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_csv")
def predict_csv(data: CSVPredictRequest):
    file_path = data.file_path
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=400, detail="File not found")
        
    print(f"Processing CSV: {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        if 'sales_history' not in df.columns:
            raise HTTPException(status_code=400, detail="CSV must have 'sales_history' column")
            
        items = []
        for _, row in df.iterrows():
            desc = row['description'] if 'description' in df.columns else "Generic Product"
            img_url = row['image_url'] if 'image_url' in df.columns else None
            hist = row['sales_history']
            
            items.append({
                "description": str(desc),
                "sales_history": parse_history(hist),
                "image_url": img_url
            })
            
        results = demand_model.predict_batch(items)
        
        detailed = []
        for i, result in enumerate(results):
            pred_value = result.get('prediction', 0) if 'error' not in result else 0
            
            product_id = df.iloc[i].get('product_id', f"P{i+1}") if 'product_id' in df.columns else f"P{i+1}"
            
            detailed.append({
                "product_id": product_id,
                "description": items[i]['description'],
                "history": items[i]['sales_history'],
                "prediction": pred_value,
                "daily_forecast": result.get('daily_forecast', []),
                "modalities_used": result.get('modalities_used', [])
            })
            
        return {
            "success": True,
            "count": len(results),
            "detailed_predictions": detailed
        }
        
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train")
def train():
    def training_task():
        print("Starting training task...")
        try:
            history = demand_model.train(csv_path='ml_service/dataset.csv')
            print(f"Training finished. MSE: {history['loss'][-1]:.2f}")
        except Exception as e:
            print(f"Training failed: {e}")
            
    thread = threading.Thread(target=training_task)
    thread.start()
    
    return {
        "message": "Training started (Advanced Multimodal)",
        "mode": "MobileNetV2-DistilBERT-LSTM",
        "estimated_duration": "Demo Mode: 5-10s"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)

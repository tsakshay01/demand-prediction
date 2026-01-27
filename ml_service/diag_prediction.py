
import sys
import os
sys.path.insert(0, 'ml_service')
import numpy as np
import pandas as pd
import ast
from advanced_model import AdvancedMultimodalModel

def diagnose():
    print("--- Diagnostic Tool ---")
    model = AdvancedMultimodalModel(model_dir='ml_service')
    
    # Check if weights loaded
    print(f"Model is_trained: {model.is_trained}")
    
    # Load akshay.csv
    df = pd.read_csv('akshay.csv')
    row = df.iloc[2] # Apple Watch
    description = row['description']
    history = ast.literal_eval(row['sales_history'])
    
    print(f"\nTesting product: {description}")
    print(f"History sample: {history[:5]}... (length {len(history)})")
    
    # Manually run prediction parts to see intermediate values
    img_input = model.preprocess_image(row['image_url'])
    img_feat = model.img_backbone.predict(img_input, verbose=0)
    img_emb = np.mean(img_feat, axis=(1, 2)).reshape(1, -1)
    
    txt_emb = model.text_model.encode([str(description)[:512]])
    
    hist_arr = np.array(history)
    if len(hist_arr) > 30: hist_arr = hist_arr[-30:]
    elif len(hist_arr) < 30: hist_arr = np.pad(hist_arr, (30-len(hist_arr), 0), 'constant')
    hist_log = np.log1p(hist_arr)
    ts_in = hist_log.reshape(1, 30, 1)
    
    # Raw prediction
    pred_log = model.model.predict([img_emb, txt_emb, ts_in], verbose=0)
    print(f"Raw log-space prediction: {pred_log}")
    
    pred_real = np.expm1(pred_log[0])
    print(f"Raw real-space prediction: {pred_real}")
    
    # Check gates
    import tensorflow as tf
    from tensorflow.keras.models import Model
    gate_model = Model(inputs=model.model.input, outputs=model.model.get_layer('modality_gates').output)
    gates = gate_model.predict([img_emb, txt_emb, ts_in], verbose=0)
    print(f"Modality Gates (Img, Txt, TS): {gates}")

if __name__ == "__main__":
    diagnose()

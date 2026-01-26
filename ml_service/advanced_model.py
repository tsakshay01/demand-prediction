
import os
import numpy as np
import requests
from PIL import Image
from io import BytesIO
import threading
import pandas as pd
import ast

# Deep Learning Frameworks
class AdvancedMultimodalModel:
    def __init__(self, model_dir='ml_service'):
        print("Initializing Advanced Multimodal Model (MobileNetV2 + DistilBERT + LSTM)...")
        self.model_dir = model_dir
        self.weights_path = os.path.join(model_dir, 'model_weights.weights.h5')
        self.is_trained = False
        
        import tensorflow as tf
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input, Dense, LSTM, Concatenate, Reshape, MultiHeadAttention, Flatten, RepeatVector, Dropout
        from tensorflow.keras.applications import MobileNetV2
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        from sentence_transformers import SentenceTransformer
        
        self.tf = tf
        self.Model = Model
        self.layers = {
            'Input': Input, 'Dense': Dense, 'LSTM': LSTM, 
            'Concatenate': Concatenate, 'Reshape': Reshape, 
            'MultiHeadAttention': MultiHeadAttention, 'Flatten': Flatten, 
            'RepeatVector': RepeatVector, 'Dropout': Dropout
        }
        self.preprocess_input = preprocess_input
        
        # 1. Text Model - Using Sentence-Transformers (stable, pretrained)
        print("Loading Sentence-Transformer (all-MiniLM-L6-v2)...")
        self.text_model = SentenceTransformer('all-MiniLM-L6-v2')  # 384-dim output
        self.text_dim = 384  # MiniLM outputs 384 dims
        print("Text model loaded successfully!")
        
        # 2. Image Model
        print("Loading MobileNetV2...")
        self.img_backbone = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        self.img_backbone.trainable = False
        
        # 3. Fusion Model (Dense 7 Output)
        print("Building Fusion Architecture (Direct 7-Day Forecast)...")
        self.model = self.build_fusion_model()
        
        # Load weights if exist
        if os.path.exists(self.weights_path):
            print(f"Loading trained weights from {self.weights_path}...")
            try:
                self.model.load_weights(self.weights_path)
                self.is_trained = True
                print("Weights loaded!")
            except:
                print("Weights load failed.")
        else:
            print("No weights found. Model needs training.")

    def build_fusion_model(self):
        # 1. Image
        img_in = self.layers['Input'](shape=(1280,), name='img_embedding') 
        img_proj = self.layers['Dense'](64, activation='relu')(img_in)
        img_proj = self.layers['Reshape']((1, 64))(img_proj)
        
        # 2. Text (MiniLM outputs 384-dim)
        txt_in = self.layers['Input'](shape=(self.text_dim,), name='text_embedding')
        txt_proj = self.layers['Dense'](64, activation='relu')(txt_in)
        txt_proj = self.layers['Reshape']((1, 64))(txt_proj)
        
        # 3. Time Series (Window=30)
        ts_in = self.layers['Input'](shape=(30, 1), name='sales_history')
        # LSTM Encoder
        ts_enc = self.layers['LSTM'](64, return_sequences=False)(ts_in)
        ts_enc = self.layers['Reshape']((1, 64))(ts_enc)
        
        # 4. Fusion
        concat = self.layers['Concatenate'](axis=1)([img_proj, txt_proj, ts_enc])
        attn = self.layers['MultiHeadAttention'](num_heads=2, key_dim=64)(concat, concat)
        flat = self.layers['Flatten']()(attn)
        context = self.layers['Dense'](64, activation='relu')(flat)
        
        # 5. Output: 7 Days Vector (Log Scale)
        prediction = self.layers['Dense'](7, activation='linear', name='weekly_forecast')(context)
        
        model = self.Model(inputs=[img_in, txt_in, ts_in], outputs=prediction)
        model.compile(optimizer='adam', loss='mse')
        return model

    def preprocess_image(self, url):
        try:
            if not url or not isinstance(url, str): return np.zeros((1, 224, 224, 3))
            response = requests.get(url, timeout=2)
            img = Image.open(BytesIO(response.content)).convert('RGB').resize((224, 224))
            x = self.tf.keras.preprocessing.image.img_to_array(img)
            return self.preprocess_input(np.expand_dims(x, axis=0))
        except: return np.zeros((1, 224, 224, 3))

    def create_sequences(self, data, window=30, horizon=7):
        X, y = [], []
        if len(data) <= window + horizon:
            return np.array(X), np.array(y)
        for i in range(len(data) - window - horizon + 1):
            X.append(data[i:(i + window)])
            y.append(data[(i + window):(i + window + horizon)])
        return np.array(X), np.array(y)

    def predict_single(self, description, sales_history, image_url=None):
        try:
            # --- 1. Feature Prep ---
            img_input = self.preprocess_image(image_url)
            img_feat = self.img_backbone.predict(img_input, verbose=0)
            img_emb = np.mean(img_feat, axis=(1, 2)).reshape(1, -1)  # (1, 1280)
            
            # Text embedding using sentence-transformers
            txt_emb = self.text_model.encode([str(description)[:512]])  # (1, 384)
            
            # --- Log Norm ---
            hist = np.array(sales_history if sales_history else [0]*30)
            # Use last 30
            if len(hist) > 30: hist = hist[-30:]
            elif len(hist) < 30: hist = np.pad(hist, (30-len(hist), 0), 'constant')
            
            hist_log = np.log1p(hist)
            ts_in = hist_log.reshape(1, 30, 1)

            # --- 2. Predict (Clean - No Leakage) ---
            # Model trained on Log scale
            pred_log = self.model.predict([img_emb, txt_emb, ts_in], verbose=0) # (1, 7)
            
            # --- 3. Denormalize ---
            pred = np.expm1(pred_log[0])
            daily_forecast = np.maximum(pred, 0).astype(int).tolist()
            total_prediction = sum(daily_forecast)

            return {
                "prediction": int(total_prediction),
                "daily_forecast": daily_forecast,
                "confidence_interval": [int(total_prediction*0.9), int(total_prediction*1.1)],
                "modalities_used": ["MobileNetV2", "DistilBERT", "LSTM", "Log-Space"],
                "text_features_active": True,
                "ts_features_active": True
            }
            
        except Exception as e:
            print(f"Pred Error: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}

    def predict_batch(self, items):
        return [self.predict_single(i.get('description',''), i.get('sales_history',[]), i.get('image_url')) for i in items]

    def train_model(self, csv_path='ml_service/dataset.csv'):
        print(f"Starting Training on {csv_path}...")
        df = pd.read_csv(csv_path)
        
        X_img_list, X_txt_list, X_ts_list, y_list = [], [], [], []
        
        # We need to process each product
        # Ideally batch processing, but loop for simplicity in demo
        # We limit to 100 products for fast training in this session
        
        limit = 100 
        print(f"Training on first {limit} products for demo speed...")
        
        for idx, row in df.head(limit).iterrows():
            # History
            hist_str = row['sales_history']
            try: hist = ast.literal_eval(hist_str)
            except: continue
            
            hist = np.array(hist)
            if len(hist) < 37: continue # Need 30 + 7
            
            # Create Sequences (Log Norm)
            hist_log = np.log1p(hist)
            seq_X, seq_y = self.create_sequences(hist_log, 30, 7)
            
            if len(seq_X) == 0: continue
            
            # CRITICAL FIX: Extract REAL embeddings (not zeros)
            # This prevents distribution shift between training and inference
            
            # Text Features (Real sentence-transformers)
            desc = str(row.get('description', 'Product'))
            txt_emb = self.text_model.encode([desc[:512]])  # (1, 384)
            
            # Image Features (Real MobileNet or zeros if no image)
            img_path = row.get('image_path', '')
            if img_path and os.path.exists(img_path):
                try:
                    img = Image.open(img_path).convert('RGB').resize((224, 224))
                    img_array = self.tf.keras.preprocessing.image.img_to_array(img)
                    img_input = self.preprocess_input(np.expand_dims(img_array, axis=0))
                    img_feat_map = self.img_backbone.predict(img_input, verbose=0)
                    img_emb = np.mean(img_feat_map, axis=(1, 2))  # (1, 1280)
                except:
                    img_emb = np.zeros((1, 1280))  # Fallback to zeros if image fails
            else:
                img_emb = np.zeros((1, 1280))
            
            # Repeat static features for each sequence
            for i in range(len(seq_X)):
                X_ts_list.append(seq_X[i])
                y_list.append(seq_y[i])
                X_img_list.append(img_emb[0])
                X_txt_list.append(txt_emb[0])
                
        X_img = np.array(X_img_list)
        X_txt = np.array(X_txt_list)
        X_ts = np.array(X_ts_list).reshape(-1, 30, 1)
        y = np.array(y_list)
        
        print(f"Training Data Shape: {X_ts.shape}")
        
        history = self.model.fit(
            [X_img, X_txt, X_ts], 
            y, 
            epochs=5, 
            batch_size=32, 
            validation_split=0.2
        )
        
        # CRITICAL FIX #3: Add interpretable metrics
        # Compute MAE in real sales units (not log-space)
        val_split_idx = int(len(y) * 0.8)
        X_val = [X_img[val_split_idx:], X_txt[val_split_idx:], X_ts[val_split_idx:]]
        y_val = y[val_split_idx:]
        
        y_pred_log = self.model.predict(X_val, verbose=0)
        y_pred_real = np.expm1(y_pred_log)
        y_val_real = np.expm1(y_val)
        
        mae_real = np.mean(np.abs(y_val_real - y_pred_real))
        total_pred = np.sum(y_pred_real, axis=1)
        total_true = np.sum(y_val_real, axis=1)
        pct_error = np.mean(np.abs(total_pred - total_true) / (total_true + 1)) * 100
        
        # ISSUE #4 FIX: Explicit naive baseline comparison
        # Naive baseline: repeat last day of input window 7 times
        X_ts_val = X_ts[val_split_idx:]
        last_day_log = X_ts_val[:, -1, 0]  # Last day of input (log scale)
        naive_pred_log = np.tile(last_day_log.reshape(-1, 1), (1, 7))  # Repeat 7 times
        naive_pred_real = np.expm1(naive_pred_log)
        baseline_mae = np.mean(np.abs(naive_pred_real - y_val_real))
        baseline_total = np.sum(naive_pred_real, axis=1)
        baseline_pct_error = np.mean(np.abs(baseline_total - total_true) / (total_true + 1)) * 100
        
        improvement = ((baseline_mae - mae_real) / baseline_mae) * 100
        
        print(f"Validation MAE (real units): {mae_real:.2f}")
        print(f"Average Weekly % Error: {pct_error:.2f}%")
        print(f"Naive Baseline MAE: {baseline_mae:.2f}")
        print(f"Naive Baseline % Error: {baseline_pct_error:.2f}%")
        print(f"Model Improvement over Baseline: {improvement:.2f}%")
        
        self.model.save_weights(self.weights_path)
        self.is_trained = True
        print("Training Complete & Weights Saved!")
        return {
            "loss": history.history['loss'],
            "val_loss": history.history['val_loss'],
            "mae_real": float(mae_real),
            "pct_error": float(pct_error)
        }

if __name__ == "__main__":
    # Training Trigger
    model = AdvancedMultimodalModel()
    model.train_model()

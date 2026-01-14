import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tensorflow as tf
from tensorflow.keras import layers, models, Model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from transformers import TFDistilBertModel, DistilBertTokenizer
import numpy as np
import os

class MultimodalDemandModel:
    def __init__(self, model_path="model.h5"):
        self.is_custom = False
        self.model = None
        self.transformer_layer = None
        self.scaler = None
        self.scaler_path = 'ml_service/scaler.pkl'
        
        try:
            import joblib
            if os.path.exists(self.scaler_path):
                self.scaler = joblib.load(self.scaler_path)
                print(f"✅ Loaded Scaler from {self.scaler_path}")
            else:
                from sklearn.preprocessing import MinMaxScaler
                self.scaler = MinMaxScaler(feature_range=(0, 1))
                print("⚠️ No scaler found. initialized new scaler.")
        except Exception as e:
            print(f"⚠️ Scaler load error: {e}")
            from sklearn.preprocessing import MinMaxScaler
            self.scaler = MinMaxScaler()

        try:
            self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            # FIXED: Actually load the BERT model
            pass # Placeholder to avoid indentation error if I just delete
            from transformers import TFDistilBertModel
            self.transformer_layer = TFDistilBertModel.from_pretrained('distilbert-base-uncased')
            print("✅ DistilBERT Loaded Successfully")
        except Exception as e:
            print(f"⚠️ BERT/Tokenizer framework load failed: {e}. Using mock/fallback.")
            self.tokenizer = None
            self.transformer_layer = None
        
        # --- ARCHITECTURE FIRST (Production Pattern) ---
        print("⚙️ Building Multimodal Architecture...")
        self.model = self.build_model()
        self.weights_path = 'ml_service/model_weights.h5'
        
        # Load Weights (Safe)
        if os.path.exists(self.weights_path):
            print(f"🔥 Loading trained weights from {self.weights_path}...")
            self.load_weights()
        else:
            print("⚠️ No weights found. Running with random initialization (Untrained).")

    def save_weights(self):
        try:
            self.model.save_weights(self.weights_path)
            print(f"Model weights saved to {self.weights_path}")
            # Save Scaler too
            import joblib
            joblib.dump(self.scaler, self.scaler_path)
            print(f"Scaler saved to {self.scaler_path}")
            return True
        except Exception as e:
            print(f"Failed to save state: {e}")
            return False

    def load_weights(self):
        if os.path.exists(self.weights_path):
            try:
                self.model.load_weights(self.weights_path)
                print(f"Model weights loaded from {self.weights_path}")
                return True
            except Exception as e:
                print(f"Failed to load weights: {e}")
                return False
        return False

    def build_model(self):
        # --- 1. Text Input (Description) ---
        # Input shape: (Batch, Sequence Length)
        input_ids = layers.Input(shape=(128,), dtype=tf.int32, name='input_ids')
        attention_mask = layers.Input(shape=(128,), dtype=tf.int32, name='attention_mask')
        
        # We freeze the BERT layer for speed in this demo, or fine-tune if needed
        if self.transformer_layer:
            self.transformer_layer.trainable = False
            bert_out = self.transformer_layer(input_ids=input_ids, attention_mask=attention_mask)[0] # (Batch, Seq, Hidden)
            text_features = layers.GlobalAveragePooling1D()(bert_out) # (Batch, Hidden)
        else:
            # Fallback: Simple Embedding if BERT failed
            print("Using Fallback Text Embedding")
            embed = layers.Embedding(input_dim=30522, output_dim=768)(input_ids)
            text_features = layers.GlobalAveragePooling1D()(embed)
        text_features = layers.Dense(64, activation='relu')(text_features)

        # --- 2. Image Input (Product Image) ---
        # Using MobileNetV2 for feature extraction
        # Input shape: (Batch, 224, 224, 3)
        image_input = layers.Input(shape=(224, 224, 3), name='image_input')
        base_mobilenet = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=False, 
            weights='imagenet'
        )
        base_mobilenet.trainable = False # Freeze for demo
        image_features = base_mobilenet(image_input)
        image_features = layers.GlobalAveragePooling2D()(image_features)
        image_features = layers.Dense(64, activation='relu')(image_features)

        # --- 3. Numerical/Time-Series Input (Sales History) ---
        # Input shape: (Batch, Timesteps, Features)
        ts_input = layers.Input(shape=(30, 5), name='ts_input') 
        
        # TRANSFORMATION: Upgrading LSTM to Transformer (TFT-style components)
        # 1. Projection to hidden dimension
        x_ts = layers.Dense(64)(ts_input) 
        
        # 2. Positional Encoding (Simplified)
        positions = tf.range(start=0, limit=30, delta=1)
        pos_embedding = layers.Embedding(input_dim=30, output_dim=64)(positions)
        x_ts = x_ts + pos_embedding

        # 3. Multi-Head Self-Attention (The Core of TFT/Transformers)
        # Captures long-range dependencies (e.g. yearly patterns) better than LSTM
        attention_output = layers.MultiHeadAttention(num_heads=4, key_dim=64)(x_ts, x_ts)
        x_ts = layers.Add()([x_ts, attention_output]) # Residual connection
        x_ts = layers.LayerNormalization(epsilon=1e-6)(x_ts)

        # 4. Feed Forward Network
        ffn = layers.Dense(64, activation="relu")(x_ts)
        ffn = layers.Dropout(0.1)(ffn)
        x_ts = layers.Add()([x_ts, ffn])
        x_ts = layers.LayerNormalization(epsilon=1e-6)(x_ts)

        # Global Pooling to flatten time dimension
        ts_features = layers.GlobalAveragePooling1D()(x_ts)

        # --- Fusion ---
        concat = layers.Concatenate()([text_features, image_features, ts_features])
        x = layers.Dense(128, activation='relu')(concat)
        x = layers.Dropout(0.3)(x)
        
        # Gated Residual Network (GRN) style block for final decision importance
        # This is a key component of TFT architecture
        gated_x = layers.Dense(64, activation='elu')(x)
        linear_x = layers.Dense(64)(x)
        x = layers.Multiply()([gated_x, linear_x]) # GLU-like gating
        
        # Output: Predicted Demand Volume
        output = layers.Dense(1, activation='linear', name='demand_output')(x)

        model = Model(inputs=[input_ids, attention_mask, image_input, ts_input], outputs=output)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model

    def preprocess_text(self, texts):
        if self.tokenizer is None:
            # Fallback: Return dummy tensors if tokenizer failed to load
            import tensorflow as tf
            batch_size = len(texts)
            return {
                'input_ids': tf.zeros((batch_size, 128), dtype=tf.int32),
                'attention_mask': tf.zeros((batch_size, 128), dtype=tf.int32)
            }
            
        return self.tokenizer(
            texts, 
            padding='max_length', 
            truncation=True, 
            max_length=128, 
            return_tensors='tf'
        )

    def train_on_file(self, csv_path='ml_service/dataset.csv', epochs=3):
        import pandas as pd
        import ast
        
        if not os.path.exists(csv_path):
            print("Dataset not found. Generating...")
            from dataset_generator import generate_dataset
            generate_dataset(output_file=csv_path)
            
        print(f"Loading dataset from {csv_path}...")
        df = pd.read_csv(csv_path)
        
        # Parse inputs
        descriptions = df['description'].tolist()
        sales_histories = [ast.literal_eval(h) for h in df['sales_history']]
        targets = df['demand_target'].values
        
        # 1. Text Process
        tokenized = self.preprocess_text(descriptions)
        
        # 2. Image Process (REAL + Synthetic Fallback)
        print("Loading real images...")
        num_samples = len(df)
        images = np.zeros((num_samples, 224, 224, 3), dtype=np.float32)
        
        if 'image_path' in df.columns:
            image_paths = df['image_path'].tolist()
            for idx, path in enumerate(image_paths):
                try:
                    # Handle NaN or empty strings
                    if path and isinstance(path, str) and os.path.exists(path):
                        # Load and resize using Keras utility
                        img = tf.keras.preprocessing.image.load_img(path, target_size=(224, 224))
                        img_arr = tf.keras.preprocessing.image.img_to_array(img)
                        # FIXED: Use MobileNetV2 standard preprocessing ([-1, 1])
                        images[idx] = preprocess_input(img_arr)
                    else:
                        # FIXED: Use Zero Placeholder instead of Random Noise
                        images[idx] = np.zeros((224, 224, 3), dtype=np.float32)
                except Exception as e:
                    print(f"⚠️ Image load error for {path}: {e}")
                    images[idx] = np.zeros((224, 224, 3), dtype=np.float32)
        else:
            print("⚠️ No 'image_path' column found. Using synthetic ZERO images (Fixed).")
            # FIXED: Use Zero Placeholder
            images = np.zeros((num_samples, 224, 224, 3), dtype=np.float32)
        
        # 3. TS Process
        # Ensure shape (Batch, 30, 5) - we only have 1 variable (sales), so we pad features
        ts_data = np.zeros((num_samples, 30, 5), dtype=np.float32)
        
        # Collect all sales data for scaling
        all_sales = []
        for hist in sales_histories:
            all_sales.extend(hist)
        # Also include targets in fitting to cover full range
        all_sales.extend(targets)
        
        # Fit Scaler
        print("Fitting Scaler on fresh data...")
        self.scaler.fit(np.array(all_sales).reshape(-1, 1))
        
        # Transform Inputs
        for i, hist in enumerate(sales_histories):
            hist_arr = np.array(hist[-30:]).reshape(-1, 1)
            if len(hist_arr) < 30:
                # Pad with zeros (which are 0.0 in scaled terms roughly if min is 0)
                # Better: Pad before scaling? For now, simplistic padding.
                scaled_hist = self.scaler.transform(hist_arr)
                pad_width = 30 - len(scaled_hist)
                scaled_hist = np.pad(scaled_hist, ((pad_width, 0), (0, 0)), mode='constant')
            else:
                scaled_hist = self.scaler.transform(hist_arr)
            
            ts_data[i, :, 0] = scaled_hist.flatten()
            
        # Transform Targets
        scaled_targets = self.scaler.transform(targets.reshape(-1, 1))
            
        print(f"Starting Training on {num_samples} samples for {epochs} epochs...")
        history = self.model.fit(
            {
                'input_ids': tokenized['input_ids'],
                'attention_mask': tokenized['attention_mask'],
                'image_input': images,
                'ts_input': ts_data
            },
            scaled_targets,
            epochs=epochs,
            batch_size=8,
            validation_split=0.2
        )
        print("Training complete.")
        self.save_weights()
        return history.history

if __name__ == "__main__":
    print("Initializing Multimodal Model...")
    dm = MultimodalDemandModel()
    dm.model.summary()
    print("Model built successfully.")

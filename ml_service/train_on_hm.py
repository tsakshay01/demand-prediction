"""
H&M Dataset → DemandAI Training Script
========================================
This script preprocesses the H&M Kaggle dataset and trains 
the Fusion Network (LSTM + Gated Fusion + Dense).

Pre-trained components (NO training needed):
  - MobileNetV2 (ImageNet) — extracts image features
  - SentenceTransformer (all-MiniLM-L6-v2) — extracts text features

What THIS script trains:
  - Fusion Network — learns to combine all 3 modalities → 7-day forecast

SETUP:
  1. Download H&M dataset from Kaggle
  2. Place articles.csv, transactions_train.csv, and images/ folder 
     in a directory (default: ./hm_data/)
  3. pip install tensorflow pandas numpy sentence-transformers Pillow scikit-learn
  4. Run: python train_on_hm.py
  5. Copy the output model_weights.weights.h5 to ml_service/ in the project

For Google Colab:
  - Upload files to /content/hm_data/
  - Enable GPU runtime for faster training
  - Change HM_DATA_DIR below to '/content/hm_data'
"""

import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
from datetime import datetime
import ast

# ============================================================
# CONFIGURATION — Adjust these for your setup
# ============================================================

HM_DATA_DIR = './hm_data'           # Where you placed H&M files
OUTPUT_WEIGHTS = 'model_weights.weights.h5'  # Output weights file
TOP_N_PRODUCTS = 2000               # Number of top-selling products to use
MIN_HISTORY_DAYS = 37               # Minimum 30 (input) + 7 (target)
EPOCHS = 50                         # Training epochs (50 recommended)
BATCH_SIZE = 32                     # Batch size
VALIDATION_SPLIT = 0.2              # 20% validation
USE_IMAGES = True                   # Set False if you don't have images/ folder
WINDOW_SIZE = 30                    # Input window (days)
HORIZON = 7                         # Forecast horizon (days)

# ============================================================
# STEP 1: PREPROCESS H&M DATA
# ============================================================

def preprocess_hm_data():
    """
    Convert H&M transactions + articles into training-ready format.
    Returns a DataFrame with: article_id, description, sales_history, image_path
    """
    print("\n" + "=" * 60)
    print("  STEP 1: PREPROCESSING H&M DATA")
    print("=" * 60)

    # Load articles
    articles_path = os.path.join(HM_DATA_DIR, 'articles.csv')
    print(f"\nLoading articles from {articles_path}...")
    articles = pd.read_csv(articles_path)
    print(f"  → {len(articles)} articles loaded")

    # Build description from available columns
    desc_cols = ['prod_name', 'product_type_name', 'product_group_name',
                 'colour_group_name', 'department_name', 'section_name']
    available_cols = [c for c in desc_cols if c in articles.columns]
    articles['description'] = articles[available_cols].fillna('').apply(
        lambda row: ' '.join(str(v) for v in row if str(v).strip()), axis=1
    )

    # Also use detail_desc if available
    if 'detail_desc' in articles.columns:
        articles['description'] = articles['description'] + ' ' + articles['detail_desc'].fillna('')

    articles['description'] = articles['description'].str.strip().str[:512]

    # Load transactions
    txn_path = os.path.join(HM_DATA_DIR, 'transactions_train.csv')
    print(f"\nLoading transactions from {txn_path}...")
    txn = pd.read_csv(txn_path,
                       usecols=['t_dat', 'article_id'],
                       parse_dates=['t_dat'])
    print(f"  → {len(txn)} transactions loaded")

    # Aggregate: daily sales count per article
    print("\nAggregating daily sales per article...")
    daily = txn.groupby(['article_id', 't_dat']).size().reset_index(name='sales')

    # Get top N products by total sales volume
    total_sales = daily.groupby('article_id')['sales'].sum().sort_values(ascending=False)
    top_articles = total_sales.head(TOP_N_PRODUCTS).index.tolist()
    print(f"  → Selected top {len(top_articles)} products by sales volume")

    daily_top = daily[daily['article_id'].isin(top_articles)]

    # Build continuous daily sales history for each product
    print("\nBuilding daily sales histories...")
    results = []
    date_range = pd.date_range(daily_top['t_dat'].min(), daily_top['t_dat'].max())

    for article_id in top_articles:
        product_data = daily_top[daily_top['article_id'] == article_id]
        product_daily = product_data.set_index('t_dat').reindex(date_range, fill_value=0)['sales']
        sales_list = product_daily.values.tolist()

        if len(sales_list) < MIN_HISTORY_DAYS:
            continue

        # Get description
        art_row = articles[articles['article_id'] == article_id]
        desc = art_row['description'].values[0] if len(art_row) > 0 else f"Product {article_id}"

        # Get image path
        img_path = ''
        if USE_IMAGES:
            # H&M images are organized as: images/0XXXXXXX/0XXXXXXX.jpg
            art_str = str(article_id).zfill(10)
            folder = art_str[:3]
            candidate = os.path.join(HM_DATA_DIR, 'images', folder, f'{art_str}.jpg')
            if os.path.exists(candidate):
                img_path = candidate

        results.append({
            'article_id': article_id,
            'description': desc,
            'sales_history': str(sales_list),
            'image_path': img_path
        })

    df = pd.DataFrame(results)
    print(f"  → {len(df)} products with sufficient history (>={MIN_HISTORY_DAYS} days)")

    # Save preprocessed data
    output_csv = os.path.join(HM_DATA_DIR, 'preprocessed_training_data.csv')
    df.to_csv(output_csv, index=False)
    print(f"  → Saved to {output_csv}")

    return df


# ============================================================
# STEP 2: BUILD AND TRAIN THE MODEL
# ============================================================

def train_model(df):
    """
    Train the Fusion Network on preprocessed H&M data.
    Uses pre-trained MobileNetV2 and SentenceTransformer as frozen feature extractors.
    """
    print("\n" + "=" * 60)
    print("  STEP 2: TRAINING FUSION NETWORK")
    print("=" * 60)

    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (Input, Dense, LSTM, Concatenate,
                                          Reshape, Flatten, Lambda,
                                          Softmax, Add, Multiply)
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    from sentence_transformers import SentenceTransformer

    # --- Load Pre-trained Feature Extractors ---
    print("\n[1/4] Loading pre-trained feature extractors...")
    print("  → Loading SentenceTransformer (all-MiniLM-L6-v2)...")
    text_model = SentenceTransformer('all-MiniLM-L6-v2')
    text_dim = 384

    print("  → Loading MobileNetV2 (ImageNet)...")
    img_backbone = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    img_backbone.trainable = False  # Frozen — no training

    # --- Build Fusion Network ---
    print("\n[2/4] Building Fusion Network architecture...")

    # Image branch
    img_in = Input(shape=(1280,), name='img_embedding')
    img_proj = Dense(64, activation='relu')(img_in)
    img_proj = Reshape((1, 64))(img_proj)

    # Text branch
    txt_in = Input(shape=(text_dim,), name='text_embedding')
    txt_proj = Dense(64, activation='relu')(txt_in)
    txt_proj = Reshape((1, 64))(txt_proj)

    # Time series branch
    ts_in = Input(shape=(WINDOW_SIZE, 1), name='sales_history')
    ts_enc = LSTM(64, return_sequences=False)(ts_in)
    ts_enc = Reshape((1, 64))(ts_enc)

    # Gated Fusion
    img_flat = Flatten()(img_proj)
    txt_flat = Flatten()(txt_proj)
    ts_flat = Flatten()(ts_enc)

    all_modalities = Concatenate(axis=-1)([img_flat, txt_flat, ts_flat])
    gate_logits = Dense(3, activation=None, name='gate_logits')(all_modalities)
    gate_weights = Softmax(name='modality_gates')(gate_logits)

    gate_img = Lambda(lambda x: x[:, 0:1])(gate_weights)
    gate_txt = Lambda(lambda x: x[:, 1:2])(gate_weights)
    gate_ts = Lambda(lambda x: x[:, 2:3])(gate_weights)

    weighted_img = Multiply()([img_flat, gate_img])
    weighted_txt = Multiply()([txt_flat, gate_txt])
    weighted_ts = Multiply()([ts_flat, gate_ts])

    context = Add()([weighted_img, weighted_txt, weighted_ts])

    # Output: 7-day forecast (log scale)
    prediction = Dense(HORIZON, activation='linear', name='weekly_forecast')(context)

    model = Model(inputs=[img_in, txt_in, ts_in], outputs=prediction)
    model.compile(optimizer='adam', loss='mse')
    model.summary()

    # --- Prepare Training Data ---
    print(f"\n[3/4] Preparing training data from {len(df)} products...")

    X_img_list, X_txt_list, X_ts_list, y_list = [], [], [], []

    for idx, row in df.iterrows():
        if idx % 100 == 0:
            print(f"  Processing product {idx+1}/{len(df)}...")

        # Parse sales history
        try:
            hist = ast.literal_eval(row['sales_history']) if isinstance(row['sales_history'], str) else list(row['sales_history'])
        except Exception:
            continue

        hist = np.array(hist, dtype=float)
        if len(hist) < WINDOW_SIZE + HORIZON:
            continue

        # Log-normalize
        hist_log = np.log1p(hist)

        # Create sliding window sequences
        for i in range(len(hist_log) - WINDOW_SIZE - HORIZON + 1):
            X_ts_list.append(hist_log[i:i + WINDOW_SIZE])
            y_list.append(hist_log[i + WINDOW_SIZE:i + WINDOW_SIZE + HORIZON])

            # Text embedding (same for all windows of this product)
            desc = str(row.get('description', 'Product'))[:512]
            txt_emb = text_model.encode([desc])[0]
            X_txt_list.append(txt_emb)

            # Image embedding
            img_path = str(row.get('image_path', ''))
            if img_path and os.path.exists(img_path):
                try:
                    img = Image.open(img_path).convert('RGB').resize((224, 224))
                    img_array = tf.keras.preprocessing.image.img_to_array(img)
                    img_input = preprocess_input(np.expand_dims(img_array, axis=0))
                    img_feat = img_backbone.predict(img_input, verbose=0)
                    img_emb = np.mean(img_feat, axis=(1, 2))[0]
                except Exception:
                    img_emb = np.zeros(1280)
            else:
                img_emb = np.zeros(1280)

            X_img_list.append(img_emb)

    # Convert to numpy arrays
    X_img = np.array(X_img_list)
    X_txt = np.array(X_txt_list)
    X_ts = np.array(X_ts_list).reshape(-1, WINDOW_SIZE, 1)
    y = np.array(y_list)

    print(f"\n  Training samples: {len(y)}")
    print(f"  X_img shape: {X_img.shape}")
    print(f"  X_txt shape: {X_txt.shape}")
    print(f"  X_ts  shape: {X_ts.shape}")
    print(f"  y     shape: {y.shape}")

    # --- Train ---
    print(f"\n[4/4] Training for {EPOCHS} epochs...")

    history = model.fit(
        [X_img, X_txt, X_ts],
        y,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        verbose=1
    )

    # --- Evaluate ---
    print("\n" + "=" * 60)
    print("  EVALUATION RESULTS")
    print("=" * 60)

    val_idx = int(len(y) * (1 - VALIDATION_SPLIT))
    X_val = [X_img[val_idx:], X_txt[val_idx:], X_ts[val_idx:]]
    y_val = y[val_idx:]

    y_pred_log = model.predict(X_val, verbose=0)
    y_pred_real = np.expm1(y_pred_log)
    y_val_real = np.expm1(y_val)

    mae = np.mean(np.abs(y_val_real - y_pred_real))
    total_pred = np.sum(y_pred_real, axis=1)
    total_true = np.sum(y_val_real, axis=1)
    mape = np.mean(np.abs(total_pred - total_true) / (total_true + 1)) * 100
    accuracy = 100 - mape

    # Naive baseline comparison (repeat last day)
    last_day = np.expm1(X_ts[val_idx:][:, -1, 0])
    naive = np.tile(last_day.reshape(-1, 1), (1, HORIZON))
    baseline_mae = np.mean(np.abs(naive - y_val_real))

    improvement = ((baseline_mae - mae) / baseline_mae) * 100

    print(f"  MAE (real units)         : {mae:.2f}")
    print(f"  MAPE                     : {mape:.2f}%")
    print(f"  Accuracy (100 - MAPE)    : {accuracy:.2f}%")
    print(f"  Naive Baseline MAE       : {baseline_mae:.2f}")
    print(f"  Improvement over Baseline: {improvement:.2f}%")
    print(f"  Final Training Loss      : {history.history['loss'][-1]:.4f}")
    print(f"  Final Validation Loss    : {history.history['val_loss'][-1]:.4f}")

    # --- Save Weights ---
    model.save_weights(OUTPUT_WEIGHTS)
    print(f"\n✅ Weights saved to: {OUTPUT_WEIGHTS}")
    print(f"   Copy this file to ml_service/ in your project.")
    print(f"   The ML service will auto-load it on startup.")

    return history


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    # Check if H&M data exists
    if not os.path.exists(HM_DATA_DIR):
        print(f"ERROR: Directory '{HM_DATA_DIR}' not found!")
        print(f"Please download the H&M dataset from Kaggle and place it there.")
        print(f"Needed files: articles.csv, transactions_train.csv")
        print(f"Optional: images/ folder for MobileNetV2")
        sys.exit(1)

    articles_csv = os.path.join(HM_DATA_DIR, 'articles.csv')
    txn_csv = os.path.join(HM_DATA_DIR, 'transactions_train.csv')

    if not os.path.exists(articles_csv):
        print(f"ERROR: {articles_csv} not found!")
        sys.exit(1)
    if not os.path.exists(txn_csv):
        print(f"ERROR: {txn_csv} not found!")
        sys.exit(1)

    # Check images folder
    images_dir = os.path.join(HM_DATA_DIR, 'images')
    if not os.path.exists(images_dir):
        print("WARNING: images/ folder not found. MobileNetV2 will receive zero inputs.")
        print("         The model will still work using Text + Time Series only.")
        globals()['USE_IMAGES'] = False

    # Run pipeline
    df = preprocess_hm_data()
    history = train_model(df)

    print("\n" + "=" * 60)
    print("  ✅ TRAINING COMPLETE!")
    print("=" * 60)
    print(f"  Output: {OUTPUT_WEIGHTS}")
    print(f"  Next: Copy {OUTPUT_WEIGHTS} → ml_service/ in your project")
    print("=" * 60)

# ============================================================
# DemandAI — COLAB TRAINING NOTEBOOK (Memory-Optimized)
# ============================================================
# INSTRUCTIONS:
#   1. Open Google Colab (colab.research.google.com)
#   2. Runtime → Change runtime type → GPU (T4)
#   3. Copy-paste this ENTIRE script into a single cell
#   4. Run the cell — it does EVERYTHING automatically
#   5. At the end it downloads model_weights.weights.h5 to your PC
#
# IMPORTANT: You MUST accept competition rules ONCE before running:
#   → https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/rules
#   → Click "I Understand and Accept" or "Late Submission"
#
# MEMORY OPTIMIZED: Works on free Colab (12GB RAM)
#   - Uses 500 products instead of 2000
#   - Reads transactions in chunks (not all at once)
#   - Skips sliding windows (every 7th step)
#   - Aggressive garbage collection
# ============================================================

# --- CELL 1: Install Dependencies ---
import subprocess, sys, gc
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
    'tensorflow', 'sentence-transformers', 'pandas', 'numpy',
    'Pillow', 'scikit-learn'])

# --- CELL 2: Download H&M Dataset from Kaggle ---
import os, json, requests

HM_DIR = '/content/hm_data'
os.makedirs(HM_DIR, exist_ok=True)

KAGGLE_TOKEN = "KGAT_e401379f7ade527f49e847c73efaf73a"
COMPETITION = "h-and-m-personalized-fashion-recommendations"
BASE_URL = "https://www.kaggle.com/api/v1"

headers = {"Authorization": f"Bearer {KAGGLE_TOKEN}"}

def download_kaggle_file(filename, dest_dir):
    """Download a single file from Kaggle competition using Bearer token"""
    url = f"{BASE_URL}/competitions/data/download/{COMPETITION}/{filename}"
    print(f"📥 Downloading {filename}...")

    response = requests.get(url, headers=headers, stream=True, allow_redirects=True)

    if response.status_code != 200:
        print(f"   ❌ HTTP {response.status_code}: {response.text[:200]}")
        return False

    content_type = response.headers.get('Content-Type', '')
    is_zip = 'zip' in content_type

    save_name = filename + ('.zip' if is_zip else '')
    save_path = os.path.join(dest_dir, save_name)

    total = int(response.headers.get('Content-Length', 0))
    downloaded = 0

    with open(save_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=1024*1024):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = downloaded / total * 100
                    mb = downloaded / 1024 / 1024
                    print(f"   {mb:.0f} MB ({pct:.1f}%)", end='\r')

    print(f"   ✅ {filename} — {downloaded/1024/1024:.1f} MB")

    if is_zip:
        import zipfile
        with zipfile.ZipFile(save_path, 'r') as z:
            z.extractall(dest_dir)
        os.remove(save_path)
        print(f"   📦 Unzipped")

    return True

# Download files
success1 = download_kaggle_file('articles.csv', HM_DIR)
success2 = download_kaggle_file('transactions_train.csv', HM_DIR)

if not success1 or not success2:
    print("\n⚠️ Direct download failed. Trying full zip download...")
    url = f"{BASE_URL}/competitions/data/download-all/{COMPETITION}"
    response = requests.get(url, headers=headers, stream=True, allow_redirects=True)
    if response.status_code == 200:
        zip_path = os.path.join(HM_DIR, 'hm_full.zip')
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024*1024):
                if chunk: f.write(chunk)
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(HM_DIR)
        os.remove(zip_path)
    else:
        raise Exception(f"Download failed! HTTP {response.status_code}")

print(f"\n✅ Dataset ready! Files: {os.listdir(HM_DIR)}")

# --- CELL 3: MEMORY-EFFICIENT Preprocessing ---
import numpy as np
import pandas as pd

# *** KEY MEMORY SAVINGS ***
TOP_N_PRODUCTS = 500      # Reduced from 2000 → saves ~75% RAM
WINDOW_SIZE = 30
HORIZON = 7
STEP_SIZE = 7             # Slide by 7 days instead of 1 → 7x fewer samples
MIN_HISTORY = WINDOW_SIZE + HORIZON

print("\n" + "="*60)
print("  STEP 1: PREPROCESSING (Memory-Optimized)")
print("="*60)

# Load articles (small file, ~30MB)
print("\nLoading articles.csv...")
articles = pd.read_csv(os.path.join(HM_DIR, 'articles.csv'))
print(f"  → {len(articles)} articles")

# Build description
desc_cols = ['prod_name', 'product_type_name', 'product_group_name',
             'colour_group_name', 'department_name', 'section_name']
available = [c for c in desc_cols if c in articles.columns]
articles['description'] = articles[available].fillna('').apply(
    lambda row: ' '.join(str(v) for v in row if str(v).strip()), axis=1
)
if 'detail_desc' in articles.columns:
    articles['description'] = articles['description'] + ' ' + articles['detail_desc'].fillna('')
articles['description'] = articles['description'].str.strip().str[:512]
# Keep only needed columns
articles = articles[['article_id', 'description']]
gc.collect()

# *** CHUNKED reading of transactions (key memory saver) ***
print("Loading transactions in chunks (memory-efficient)...")
txn_path = os.path.join(HM_DIR, 'transactions_train.csv')

# First pass: count sales per article to find top products
print("  Pass 1: Finding top products...")
article_counts = {}
for chunk in pd.read_csv(txn_path, usecols=['article_id'], chunksize=1_000_000):
    counts = chunk['article_id'].value_counts()
    for aid, cnt in counts.items():
        article_counts[aid] = article_counts.get(aid, 0) + cnt

# Sort and get top N
sorted_articles = sorted(article_counts.items(), key=lambda x: x[1], reverse=True)
top_article_ids = set(aid for aid, _ in sorted_articles[:TOP_N_PRODUCTS])
print(f"  → Top {len(top_article_ids)} products selected")
del article_counts, sorted_articles
gc.collect()

# Second pass: build daily sales for top products only
print("  Pass 2: Building daily sales...")
daily_sales = {}  # {article_id: {date_str: count}}
for chunk in pd.read_csv(txn_path, usecols=['t_dat', 'article_id'], chunksize=1_000_000):
    # Filter to top products only
    chunk = chunk[chunk['article_id'].isin(top_article_ids)]
    for _, row in chunk.groupby(['article_id', 't_dat']).size().reset_index(name='sales').iterrows():
        aid = row['article_id']
        if aid not in daily_sales:
            daily_sales[aid] = {}
        daily_sales[aid][row['t_dat']] = daily_sales[aid].get(row['t_dat'], 0) + row['sales']

print(f"  → Built daily sales for {len(daily_sales)} products")

# Delete the CSV from disk to free space
try:
    os.remove(txn_path)
    print("  → Deleted transactions CSV to free disk space")
except: pass
gc.collect()

# Build histories
print("Building sales histories...")
all_dates = set()
for dates in daily_sales.values():
    all_dates.update(dates.keys())
all_dates = sorted(all_dates)
date_range = pd.date_range(all_dates[0], all_dates[-1])
date_strs = [d.strftime('%Y-%m-%d') for d in date_range]

histories = []  # List of (article_id, description, sales_array)

for i, aid in enumerate(top_article_ids):
    if i % 100 == 0:
        print(f"  Product {i+1}/{len(top_article_ids)}...")

    sales_dict = daily_sales.get(aid, {})
    sales_array = np.array([sales_dict.get(d, 0) for d in date_strs], dtype=np.float32)

    if len(sales_array) < MIN_HISTORY:
        continue

    art_row = articles[articles['article_id'] == aid]
    desc = art_row['description'].values[0] if len(art_row) > 0 else f"Product {aid}"

    histories.append((aid, desc, sales_array))

print(f"  → {len(histories)} products with sufficient history")

# Keep date_strs for date feature computation
print(f"  → Date range: {date_strs[0]} to {date_strs[-1]} ({len(date_strs)} days)")

# Free memory (keep date_strs for date feature computation)
del daily_sales, all_dates, articles
gc.collect()

# --- CELL 4: Build & Train Model ---
print("\n" + "="*60)
print("  STEP 2: TRAINING FUSION NETWORK")
print("="*60)

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, LSTM, Concatenate,
                                      Reshape, Flatten, Lambda,
                                      Softmax, Add, Multiply)
from sentence_transformers import SentenceTransformer

# Load text model
print("\n[1/4] Loading SentenceTransformer...")
text_model = SentenceTransformer('all-MiniLM-L6-v2')
TEXT_DIM = 384
print("  → Text model loaded")

# Build Fusion Network
print("\n[2/4] Building Fusion Network...")

img_in = Input(shape=(1280,), name='img_embedding')
img_proj = Dense(64, activation='relu')(img_in)
img_proj = Reshape((1, 64))(img_proj)

txt_in = Input(shape=(TEXT_DIM,), name='text_embedding')
txt_proj = Dense(64, activation='relu')(txt_in)
txt_proj = Reshape((1, 64))(txt_proj)

ts_in = Input(shape=(WINDOW_SIZE, 1), name='sales_history')
ts_enc = LSTM(64, return_sequences=False)(ts_in)
ts_enc = Reshape((1, 64))(ts_enc)

# NEW: Date features input (day-of-week 7 + month 12 = 19 dims)
date_in = Input(shape=(19,), name='date_features')
date_proj = Dense(64, activation='relu')(date_in)

img_flat = Flatten()(img_proj)
txt_flat = Flatten()(txt_proj)
ts_flat = Flatten()(ts_enc)
date_flat = date_proj  # already (batch, 64)

all_mod = Concatenate(axis=-1)([img_flat, txt_flat, ts_flat, date_flat])  # (batch, 256)
gate_logits = Dense(4, activation=None, name='gate_logits')(all_mod)
gate_weights = Softmax(name='modality_gates')(gate_logits)  # (batch, 4)

gate_img = Lambda(lambda x: x[:, 0:1])(gate_weights)
gate_txt = Lambda(lambda x: x[:, 1:2])(gate_weights)
gate_ts = Lambda(lambda x: x[:, 2:3])(gate_weights)
gate_date = Lambda(lambda x: x[:, 3:4])(gate_weights)

w_img = Multiply()([img_flat, gate_img])
w_txt = Multiply()([txt_flat, gate_txt])
w_ts = Multiply()([ts_flat, gate_ts])
w_date = Multiply()([date_flat, gate_date])

context = Add()([w_img, w_txt, w_ts, w_date])  # (batch, 64)
prediction = Dense(HORIZON, activation='linear', name='weekly_forecast')(context)

model = Model(inputs=[img_in, txt_in, ts_in, date_in], outputs=prediction)
model.compile(optimizer='adam', loss='mse')
model.summary()

# Prepare training data (memory-efficient: build arrays directly)
print(f"\n[3/4] Preparing training data from {len(histories)} products...")

# Pre-compute text embeddings
print("  Computing text embeddings...")
all_descs = [h[1] for h in histories]
all_txt_embs = text_model.encode(all_descs, show_progress_bar=True, batch_size=64)

# Free the text model to save RAM
del text_model
gc.collect()

# Count total samples first
total_samples = 0
for _, _, sales in histories:
    n_windows = (len(sales) - WINDOW_SIZE - HORIZON) // STEP_SIZE + 1
    total_samples += max(0, n_windows)
print(f"  Total samples: {total_samples}")

# Pre-allocate arrays (much more memory efficient than lists)
X_img = np.zeros((total_samples, 1280), dtype=np.float32)
X_txt = np.zeros((total_samples, TEXT_DIM), dtype=np.float32)
X_ts = np.zeros((total_samples, WINDOW_SIZE, 1), dtype=np.float32)
X_date = np.zeros((total_samples, 19), dtype=np.float32)  # 7 day-of-week + 12 month
y = np.zeros((total_samples, HORIZON), dtype=np.float32)
is_train = np.ones(total_samples, dtype=bool)  # True=train, False=val

# Pre-compute date features for each day in date_range
from datetime import datetime as dt
date_objects = [dt.strptime(d, '%Y-%m-%d') for d in date_strs]
print(f"  Pre-computed date objects for {len(date_objects)} days")

idx = 0
for prod_idx, (aid, desc, sales) in enumerate(histories):
    if prod_idx % 100 == 0:
        print(f"  Building sequences: product {prod_idx+1}/{len(histories)}...")

    hist_log = np.log1p(sales)
    txt_emb = all_txt_embs[prod_idx]

    # Collect all window start positions for this product
    windows = list(range(0, len(hist_log) - WINDOW_SIZE - HORIZON + 1, STEP_SIZE))
    n_windows = len(windows)
    
    # TIME-BASED SPLIT: first 80% of windows = train, last 20% = validation
    # This ensures validation uses LATER dates (no lookahead bias)
    split_point = int(n_windows * 0.8)

    for w_idx, i in enumerate(windows):
        X_ts[idx, :, 0] = hist_log[i:i + WINDOW_SIZE]
        y[idx] = hist_log[i + WINDOW_SIZE:i + WINDOW_SIZE + HORIZON]
        X_txt[idx] = txt_emb
        is_train[idx] = (w_idx < split_point)
        
        # Date features: use the forecast start date (day after window ends)
        forecast_start_idx = i + WINDOW_SIZE
        if forecast_start_idx < len(date_objects):
            d = date_objects[forecast_start_idx]
            dow = np.zeros(7)
            dow[d.weekday()] = 1.0
            mon = np.zeros(12)
            mon[d.month - 1] = 1.0
            X_date[idx] = np.concatenate([dow, mon])
        
        idx += 1

# Trim if we over-counted
X_img = X_img[:idx]
X_txt = X_txt[:idx]
X_ts = X_ts[:idx]
X_date = X_date[:idx]
y = y[:idx]
is_train = is_train[:idx]

# Split into train and val sets (TIME-BASED)
train_mask = is_train
val_mask = ~is_train

X_img_train, X_img_val = X_img[train_mask], X_img[val_mask]
X_txt_train, X_txt_val = X_txt[train_mask], X_txt[val_mask]
X_ts_train, X_ts_val = X_ts[train_mask], X_ts[val_mask]
X_date_train, X_date_val = X_date[train_mask], X_date[val_mask]
y_train, y_val = y[train_mask], y[val_mask]

print(f"\n  Total samples: {idx}")
print(f"  Training samples (earlier dates): {train_mask.sum()}")
print(f"  Validation samples (later dates): {val_mask.sum()}")
print(f"  Split method: TIME-BASED (last 20% of each product's timeline)")
print(f"  Date features: 19-dim (7 day-of-week + 12 month one-hot)")
mem_gb = (X_img.nbytes + X_txt.nbytes + X_ts.nbytes + X_date.nbytes + y.nbytes) / 1024**3
print(f"  Arrays memory: {mem_gb:.2f} GB")

# Free intermediate data
del histories, all_txt_embs, all_descs, X_img, X_txt, X_ts, X_date, y, is_train, date_objects, date_strs
gc.collect()

# Train
EPOCHS = 50
BATCH_SIZE = 128

print(f"\n[4/4] Training for {EPOCHS} epochs (time-based split + date features)...")
history = model.fit(
    [X_img_train, X_txt_train, X_ts_train, X_date_train], y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=([X_img_val, X_txt_val, X_ts_val, X_date_val], y_val),
    verbose=1
)

# --- CELL 5: Evaluate ---
print("\n" + "="*60)
print("  EVALUATION RESULTS (Time-Based Split)")
print("="*60)

y_pred_log = model.predict([X_img_val, X_txt_val, X_ts_val, X_date_val], verbose=0)
y_pred_real = np.expm1(y_pred_log)
y_val_real = np.expm1(y_val)

mae = np.mean(np.abs(y_val_real - y_pred_real))
total_pred = np.sum(y_pred_real, axis=1)
total_true = np.sum(y_val_real, axis=1)
nonzero = total_true > 0
mape = np.mean(np.abs(total_pred[nonzero] - total_true[nonzero]) / total_true[nonzero]) * 100
accuracy = max(0, 100 - mape)

# Naive baseline (repeat last observed day)
last_day = np.expm1(X_ts_val[:, -1, 0])
naive = np.tile(last_day.reshape(-1, 1), (1, HORIZON))
baseline_mae = np.mean(np.abs(naive - y_val_real))
improvement = ((baseline_mae - mae) / baseline_mae) * 100

print(f"  Products Trained On   : {TOP_N_PRODUCTS}")
print(f"  Training Samples      : {len(y_train)}")
print(f"  Validation Samples    : {len(y_val)}")
print(f"  Split Method          : TIME-BASED (no lookahead bias)")
print(f"  MAE (real units)      : {mae:.2f}")
print(f"  MAPE                  : {mape:.2f}%")
print(f"  Accuracy (100-MAPE)   : {accuracy:.2f}%")
print(f"  Naive Baseline MAE    : {baseline_mae:.2f}")
print(f"  Improvement vs Naive  : {improvement:.2f}%")
print(f"  Final Training Loss   : {history.history['loss'][-1]:.4f}")
print(f"  Final Val Loss        : {history.history['val_loss'][-1]:.4f}")

# --- CELL 6: Save & Download Weights ---
OUTPUT_FILE = 'model_weights.weights.h5'
model.save_weights(OUTPUT_FILE)
print(f"\n✅ Weights saved to: {OUTPUT_FILE}")

# Auto-download in Colab
try:
    from google.colab import files
    print("📥 Downloading weights file to your PC...")
    files.download(OUTPUT_FILE)
    print("✅ Download started! Check your browser downloads.")
except ImportError:
    print(f"Not running in Colab. Copy {OUTPUT_FILE} to ml_service/ in your project.")

print("\n" + "="*60)
print("  ✅ TRAINING COMPLETE!")
print("="*60)
print(f"  Next: Place {OUTPUT_FILE} in ml_service/ folder")
print(f"  Then restart the ML service — it auto-loads the weights")
print("="*60)

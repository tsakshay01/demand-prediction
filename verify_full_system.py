import sys
import os
import requests
import numpy as np

# Add ml_service to path to import model
sys.path.append('ml_service')

print("=== STARTING FULL SYSTEM VERIFICATION ===")

def test_python_model_init():
    print("\n[1/4] Testing Model Initialization...")
    try:
        from model import MultimodalDemandModel
        dm = MultimodalDemandModel()
        if dm.model is None:
            raise Exception("Model is None (Factory Failed)")
        print("✅ Model Initialized Successfully")
        return dm
    except Exception as e:
        print(f"❌ Model Init Failed: {e}")
        return None

def test_training(dm):
    print("\n[2/4] Testing Training Loop...")
    if not dm: return False
    try:
        # Create dummy csv if not exists
        if not os.path.exists('dataset.csv'):
            with open('dataset.csv', 'w') as f:
                f.write('product_id,description,sales_history,demand_target\n')
                for i in range(10):
                    f.write(f'p{i},test product, "[10, 12, 14]", 20\n')
        
        history = dm.train_on_file(epochs=1)
        print("✅ Training Cycle Completed")
        return True
    except Exception as e:
        print(f"❌ Training Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_prediction(dm):
    print("\n[3/4] Testing Prediction (Inference)...")
    if not dm: return False
    try:
        # Mock inputs
        tokenized = dm.preprocess_text(["Blue Denim Jacket"])
        images = np.random.rand(1, 224, 224, 3).astype(np.float32)
        ts_data = np.random.rand(1, 30, 5).astype(np.float32)
        
        pred = dm.model.predict({
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'image_input': images,
            'ts_input': ts_data
        }, verbose=0)
        
        val = float(pred[0][0])
        print(f"✅ Prediction Success. Output: {val}")
        return True
    except Exception as e:
        print(f"❌ Prediction Failed: {e}")
        return False

def test_server_connection():
    print("\n[4/4] Testing Node.js Server Connection...")
    try:
        r = requests.get('http://localhost:3000/api/market-signals')
        if r.status_code == 200:
            print(f"✅ Server Reachable. Signals: {r.json()}")
        else:
            print(f"❌ Server Error: Status {r.status_code}")
    except Exception as e:
        print(f"❌ Server Unreachable: {e}")

if __name__ == "__main__":
    dm = test_python_model_init()
    if dm:
        test_training(dm)
        test_prediction(dm)
    test_server_connection()
    print("\n=== VERIFICATION COMPLETE ===")

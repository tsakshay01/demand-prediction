import pandas as pd
import numpy as np
import random
import os

def generate_desc():
    colors = ['Red', 'Blue', 'Green', 'Black', 'White', 'Yellow', 'Pink', 'Grey']
    materials = ['Cotton', 'Polyester', 'Denim', 'Silk', 'Wool', 'Linen']
    types = ['T-Shirt', 'Jeans', 'Summer Dress', 'Jacket', 'Sweater', 'Shorts', 'Skirt']
    adjectives = ['Slim Fit', 'Oversized', 'Casual', 'Formal', 'Vintage', 'Modern']
    
    return f"{random.choice(adjectives)} {random.choice(colors)} {random.choice(materials)} {random.choice(types)}"

def generate_sales_history(days=30):
    # Scale: Support small biz (50) to Enterprise (2M)
    magnitude = random.choice([100, 1000, 50000, 1000000]) 
    base = random.randint(magnitude // 2, magnitude * 2)
    
    # Seasonality (Sine wave)
    seq = np.linspace(0, np.pi * 4, days)
    seasonality = np.sin(seq) * (base * 0.1) # 10% fluctuation
    # Noise
    noise = np.random.normal(0, base * 0.05, days) # 5% noise
    
    history = base + seasonality + noise
    return np.maximum(history, 0).tolist() # Ensure no negative sales

def generate_dataset(num_samples=2000, output_file='ml_service/dataset.csv'):
    print(f"Generating {num_samples} synthetic products...")
    data = []

    # FIX: Scan for real images
    image_dir = 'public/images/learning_set'
    available_images = []
    if os.path.exists(image_dir):
        available_images = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"Found {len(available_images)} real images for training.")
    
    for _ in range(num_samples):
        desc = generate_desc()
        history = generate_sales_history(30)
        # Target: Demand for next week (simplified: average of history + random growth)
        target = np.mean(history) * (1 + random.choice([-0.1, 0.0, 0.1, 0.2])) 
        
        # Pick a random image if available
        img_path = random.choice(available_images) if available_images else ""

        data.append({
            'product_id': f"P{random.randint(10000, 99999)}",
            'description': desc,
            'sales_history': str(history), # Store as string representation of list
            'demand_target': round(target, 2),
            'image_path': img_path # New Column
        })
        
    df = pd.DataFrame(data)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"Dataset saved to {output_file}")
    return output_file

if __name__ == "__main__":
    generate_dataset()

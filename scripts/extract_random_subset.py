# extract_random_subset.py (comments in English)
import os
import shutil
import random
from pathlib import Path

# --------------------------
# Configuration (fixed for your project)
# --------------------------
SOURCE_DIR = r"E:\8016project\data\raw\CelebA-HQ\data256x256"
TARGET_DIR = r"E:\8016project\data\raw\rawsubset"
NUM_SAMPLES = 2000
RANDOM_SEED = 40  # Fixed seed for reproducibility

def extract_random_subset():
    # Create target directory if it doesn't exist
    Path(TARGET_DIR).mkdir(parents=True, exist_ok=True)
    
    # Get all image files in source folder (sorted by filename)
    img_files = sorted([
        f for f in os.listdir(SOURCE_DIR) 
        if f.endswith(('.jpg', '.jpeg', '.png'))
    ])
    
    # Check if we have enough images
    if len(img_files) < NUM_SAMPLES:
        raise ValueError(f"Only {len(img_files)} images found, need {NUM_SAMPLES}")
    
    # Randomly select 2000 samples (fixed seed for reproducibility)
    random.seed(RANDOM_SEED)
    selected_files = random.sample(img_files, NUM_SAMPLES)
    
    # Copy files to target directory
    for i, filename in enumerate(selected_files):
        src_path = os.path.join(SOURCE_DIR, filename)
        dst_path = os.path.join(TARGET_DIR, filename)
        shutil.copy2(src_path, dst_path)
        
        # Print progress every 100 files
        if (i + 1) % 100 == 0:
            print(f"Copied {i+1}/{NUM_SAMPLES} images")
    
    print(f"\n✅ Successfully extracted {NUM_SAMPLES} random images to:")
    print(f"   {TARGET_DIR}")
    print(f"Random seed: {RANDOM_SEED} (for reproducibility)")

if __name__ == "__main__":
    extract_random_subset()
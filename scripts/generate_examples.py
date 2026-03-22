import cv2
import numpy as np
import os
from pathlib import Path
import random

# Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --------------------------
# Visualization Config
# --------------------------
RAW_DIR = r"E:\8016project\data\raw\rawsubset"
PROCESSED_DIR = r"E:\8016project\data\processed\256x256"
DEGRADED_DIR = r"E:\8016project\data\degraded\256x256"
OUTPUT_DIR = r"E:\8016project\data\compare"
NUM_EXAMPLES = 10  # Generate 10 comparison images
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
FONT_COLOR = (255, 255, 255)  # White text
FONT_THICKNESS = 2

def create_comparison_image(raw_img, processed_img, degraded_img, filename):
    """Create side-by-side comparison: Raw → Preprocessed → Degraded"""
    # Resize raw image to 256x256 (match processed/degraded)
    raw_img_resized = cv2.resize(raw_img, (256, 256), interpolation=cv2.INTER_CUBIC)
    
    # Add text labels to each image
    cv2.putText(raw_img_resized, "Raw", (10, 30), FONT, FONT_SCALE, FONT_COLOR, FONT_THICKNESS)
    cv2.putText(processed_img, "Preprocessed (GT)", (10, 30), FONT, FONT_SCALE, FONT_COLOR, FONT_THICKNESS)
    cv2.putText(degraded_img, "Degraded (Input)", (10, 30), FONT, FONT_SCALE, FONT_COLOR, FONT_THICKNESS)
    
    # Concatenate side-by-side
    comparison_img = np.hstack((raw_img_resized, processed_img, degraded_img))
    return comparison_img

def generate_comparison_examples():
    """Generate visual comparison examples"""
    # Create output folder
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # Get all matched filenames
    all_files = [f for f in os.listdir(RAW_DIR) if f.endswith(('.jpg', '.png')) and os.path.exists(os.path.join(PROCESSED_DIR, f)) and os.path.exists(os.path.join(DEGRADED_DIR, f))]
    sample_files = random.sample(all_files, min(NUM_EXAMPLES, len(all_files)))
    
    print(f"🔥 Generating {len(sample_files)} comparison examples...")
    success_count = 0
    
    for idx, filename in enumerate(sample_files):
        try:
            # Read images
            raw_path = os.path.join(RAW_DIR, filename)
            processed_path = os.path.join(PROCESSED_DIR, filename)
            degraded_path = os.path.join(DEGRADED_DIR, filename)
            
            raw_img = cv2.imread(raw_path)
            processed_img = cv2.imread(processed_path)
            degraded_img = cv2.imread(degraded_path)
            
            if raw_img is None or processed_img is None or degraded_img is None:
                raise ValueError("One or more images missing/corrupted")
            
            # Create comparison image
            comparison_img = create_comparison_image(raw_img, processed_img, degraded_img, filename)
            
            # Save comparison
            save_path = os.path.join(OUTPUT_DIR, f"comparison_{idx+1}_{filename}")
            cv2.imwrite(save_path, comparison_img)
            
            success_count += 1
            print(f"✅ Generated: comparison_{idx+1}_{filename}")
        
        except Exception as e:
            print(f"❌ Failed to generate {filename}: {str(e)}")
    
    print(f"\n🎉 Comparison examples generated!")
    print(f"📊 Success: {success_count} | Failed: {len(sample_files) - success_count}")
    print(f"💾 Examples saved to: {OUTPUT_DIR}")
    print(f"💡 Open the folder to check preprocessing/degradation quality!")

if __name__ == "__main__":
    generate_comparison_examples()
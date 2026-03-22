import os
import cv2
import numpy as np
from pathlib import Path

# Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --------------------------
# Dataset Validation Config
# --------------------------
PROCESSED_DIR = r"E:\8016project\data\processed\256x256"  # Ground Truth (GT)
DEGRADED_DIR = r"E:\8016project\data\degraded\256x256"    # Input for model
TARGET_SIZE = (256, 256)
CHECK_SAMPLE = True  # Check 5 random samples for sanity

def validate_dataset_pairing():
    """Validate input (degraded) and GT (preprocessed) dataset pairing"""
    # Step 1: Get all filenames from both directories
    processed_files = set([f.lower() for f in os.listdir(PROCESSED_DIR) if f.endswith(('.jpg', '.png'))])
    degraded_files = set([f.lower() for f in os.listdir(DEGRADED_DIR) if f.endswith(('.jpg', '.png'))])
    
    # Step 2: Check for missing files
    missing_in_degraded = processed_files - degraded_files
    missing_in_processed = degraded_files - processed_files
    
    # Step 3: Basic stats
    total_processed = len(processed_files)
    total_degraded = len(degraded_files)
    matched_files = len(processed_files & degraded_files)
    
    # Print validation report
    print("="*50)
    print("📊 Dataset Validation Report")
    print("="*50)
    print(f"Total preprocessed (GT) images: {total_processed}")
    print(f"Total degraded (input) images: {total_degraded}")
    print(f"Matched (paired) images: {matched_files}")
    print(f"Missing in degraded folder: {len(missing_in_degraded)}")
    print(f"Missing in preprocessed folder: {len(missing_in_processed)}")
    
    # Step 4: Check for missing files (if any)
    if len(missing_in_degraded) > 0:
        print(f"\n❌ Missing in degraded folder (first 5): {list(missing_in_degraded)[:5]}")
    if len(missing_in_processed) > 0:
        print(f"\n❌ Missing in preprocessed folder (first 5): {list(missing_in_processed)[:5]}")
    
    # Step 5: Check sample image resolution/validity
    if CHECK_SAMPLE and matched_files > 0:
        print("\n" + "="*50)
        print("🔍 Sample Image Validation (5 random files)")
        print("="*50)
        sample_files = list(processed_files & degraded_files)[:5]
        for idx, filename in enumerate(sample_files):
            # Check GT image
            gt_path = os.path.join(PROCESSED_DIR, filename)
            gt_img = cv2.imread(gt_path)
            gt_valid = gt_img is not None and gt_img.shape[:2] == TARGET_SIZE
            
            # Check input image
            input_path = os.path.join(DEGRADED_DIR, filename)
            input_img = cv2.imread(input_path)
            input_valid = input_img is not None and input_img.shape[:2] == TARGET_SIZE
            
            print(f"Sample {idx+1} - {filename}:")
            print(f"  GT (preprocessed): {'✅ Valid (256x256)' if gt_valid else '❌ Invalid/Corrupted'}")
            print(f"  Input (degraded): {'✅ Valid (256x256)' if input_valid else '❌ Invalid/Corrupted'}")
    
    # Final pass/fail
    if matched_files == total_processed == total_degraded and all([gt_valid, input_valid]):
        print("\n✅ Dataset Validation PASSED (100% matched, all images valid)")
    else:
        print("\n⚠️  Dataset Validation WARNING (fix missing/corrupted files before training)")

if __name__ == "__main__":
    validate_dataset_pairing()
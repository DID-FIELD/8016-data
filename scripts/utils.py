# utils.py (comments in English)
import cv2
import numpy as np
from PIL import Image
import os
from pathlib import Path
import yaml

def load_config(config_path: str) -> dict:
    """Load config from yaml file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def read_image(img_path: str) -> np.ndarray:
    """Read image to RGB numpy array (uint8)"""
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb

def save_image(img: np.ndarray, save_path: str) -> None:
    """Save RGB numpy array to image file"""
    Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(save_path)

def visualize_result(raw_img: np.ndarray, processed_img: np.ndarray, degraded_img: np.ndarray, save_path: str) -> None:
    """Visualize raw/processed/degraded images side by side"""
    import matplotlib.pyplot as plt
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(raw_img)
    plt.title("Raw Image")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(processed_img)
    plt.title("Aligned + Normalized")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(degraded_img)
    plt.title("Degraded (Blur + Noise)")
    plt.axis('off')
    
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
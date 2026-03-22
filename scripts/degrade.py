import cv2
import numpy as np
import os
from pathlib import Path
import random

# Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --------------------------
# GFPGAN-Style Degradation Config (Match Official GFPGAN)
# --------------------------
TARGET_SIZE = (256, 256)
PROCESS_ALL = True  # Process all 2000 images (False = test 20)
TEST_NUM = 20

# GFPGAN core degradation parameters (tunable, match official defaults)
DEGRADATION_CONFIG = {
    # Gaussian blur (multi-scale, GFPGAN uses 1-7 odd kernels)
    'blur_kernel_sizes': [3, 5, 7],
    'blur_prob': 0.8,  # 80% chance to apply blur (GFPGAN default)
    
    # Gaussian noise (random intensity, GFPGAN uses 0-20)
    'noise_std_range': (5, 20),
    'noise_prob': 0.7,  # 70% chance to apply noise
    
    # JPEG compression (simulate compression artifacts, GFPGAN core)
    'jpeg_quality_range': (50, 90),
    'jpeg_prob': 0.6,   # 60% chance to apply compression
    
    # Downsampling (simulate low-res → upsampling, GFPGAN key step)
    'downscale_factors': [0.5, 0.75],  # Downscale then upscale back
    'downsample_prob': 0.7  # 70% chance to downsample
}

def random_gaussian_blur(image, config):
    """GFPGAN-style random Gaussian blur (multi-scale kernels)"""
    if random.random() < config['blur_prob']:
        kernel_size = random.choice(config['blur_kernel_sizes'])
        # GFPGAN uses sigma=0 (auto-calculate based on kernel)
        image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    return image

def random_gaussian_noise(image, config):
    """GFPGAN-style additive Gaussian noise (random intensity)"""
    if random.random() < config['noise_prob']:
        std = random.uniform(*config['noise_std_range'])
        noise = np.random.normal(0, std, image.shape).astype(np.float32)
        image = image.astype(np.float32) + noise
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image

def random_jpeg_compression(image, config):
    """GFPGAN-style JPEG compression (simulate real compression artifacts)"""
    if random.random() < config['jpeg_prob']:
        quality = random.randint(*config['jpeg_quality_range'])
        # Encode-decode to simulate JPEG compression (GFPGAN implementation)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, enc_img = cv2.imencode('.jpg', image, encode_param)
        image = cv2.imdecode(enc_img, 1)
    return image

def random_downsampling(image, target_size, config):
    """GFPGAN-style downsampling + upsampling (simulate low resolution)"""
    if random.random() < config['downsample_prob']:
        h, w = image.shape[:2]
        factor = random.choice(config['downscale_factors'])
        # Downscale (GFPGAN uses INTER_LINEAR for downsampling)
        down_h, down_w = int(h * factor), int(w * factor)
        image = cv2.resize(image, (down_w, down_h), interpolation=cv2.INTER_LINEAR)
        # Upscale back to target size (GFPGAN uses INTER_CUBIC for upsampling)
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_CUBIC)
    return image

def gfgan_degradation_pipeline(image, target_size, config):
    """Full GFPGAN degradation pipeline (match official order)"""
    # Step 1: Random Gaussian blur (GFPGAN first step)
    image = random_gaussian_blur(image, config)
    # Step 2: Random downsampling + upsampling (GFPGAN core)
    image = random_downsampling(image, target_size, config)
    # Step 3: Random Gaussian noise (GFPGAN third step)
    image = random_gaussian_noise(image, config)
    # Step 4: Random JPEG compression (GFPGAN final step)
    image = random_jpeg_compression(image, config)
    return image

def main():
    # Path configuration (match your preprocessing setup)
    processed_dir = r"E:\8016project\data\processed\256x256"
    degraded_output_dir = r"E:\8016project\data\degraded\256x256"
    Path(degraded_output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get all preprocessed images (match GFPGAN's file filtering)
    all_images = sorted([
        f for f in os.listdir(processed_dir)
        if f.lower().endswith(('.jpg', '.png', '.jpeg')) and os.path.isfile(os.path.join(processed_dir, f))
    ])
    images_to_process = all_images if PROCESS_ALL else all_images[:TEST_NUM]
    total = len(images_to_process)
    
    # Initialize random seed (GFPGAN uses fixed seed for reproducibility)
    random.seed(42)
    np.random.seed(42)
    
    print(f"🔥 GFPGAN-Style Degradation Started | Total images: {total}")
    print(f"⚙️  GFPGAN Params: Blur={DEGRADATION_CONFIG['blur_kernel_sizes']} | Noise={DEGRADATION_CONFIG['noise_std_range']} | JPEG={DEGRADATION_CONFIG['jpeg_quality_range']}")
    success_count = 0

    # Batch process (GFPGAN-style batch handling)
    for idx, filename in enumerate(images_to_process):
        try:
            # Step 1: Read preprocessed image (RGB format, GFPGAN standard)
            img_path = os.path.join(processed_dir, filename)
            img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img_bgr is None:
                raise ValueError("Preprocessed image read failed (corrupted file)")
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            # Step 2: Apply full GFPGAN degradation pipeline
            degraded_img = gfgan_degradation_pipeline(img_rgb, TARGET_SIZE, DEGRADATION_CONFIG)

            # Step 3: Save degraded image (BGR for OpenCV, match preprocessing)
            save_path = os.path.join(degraded_output_dir, filename)
            cv2.imwrite(save_path, cv2.cvtColor(degraded_img, cv2.COLOR_RGB2BGR))

            success_count += 1
            # Progress log (every 100 images, GFPGAN-style logging)
            if (idx + 1) % 100 == 0:
                print(f"📊 Progress: {idx+1}/{total} | Success: {success_count} | Failed: {idx+1 - success_count}")

        except Exception as e:
            print(f"❌ [{idx+1}/{total}] Failed: {filename} | Reason: {str(e)}")

    # Final summary (GFPGAN-style report)
    print(f"\n✅ GFPGAN Degradation Complete!")
    print(f"📊 Total Processed: {total} | Success: {success_count} | Failed: {total - success_count}")
    print(f"💾 Degraded Images Saved To: {degraded_output_dir}")
    print(f"💡 Note: Degradation matches GFPGAN's official low-quality face simulation (blur+downsample+noise+JPEG)")

if __name__ == "__main__":
    main()
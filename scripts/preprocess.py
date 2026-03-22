import cv2
import numpy as np
import os
import math
from pathlib import Path
from mtcnn import MTCNN

# Suppress TensorFlow/oneDNN warning messages (clean up terminal output)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --------------------------
# Core Configuration (English)
# --------------------------
TARGET_SIZE = (256, 256)  # Output image resolution (width, height)
TEST_NUM = 2000           # Number of test images to process (for quick validation)
CROP_MARGIN = 0.05         # Remove 5% of margin from all sides (keep core face)
ENABLE_FACE_HORIZONTAL_CORRECTION = False  # Disable face rotation correction
# Standard 5-point facial landmarks (academic standard for alignment)
STD_LANDMARKS = np.array([
    [76, 84],    # Left eye center
    [180, 84],   # Right eye center
    [128, 152],  # Nose tip
    [84, 208],   # Left mouth corner
    [172, 208]   # Right mouth corner
], dtype=np.float32)

def mild_tight_crop(aligned_face):
    """
    Safe mild tight crop to remove excess background (no impact on detection success)
    - Removes 5% margin from all sides of the aligned face (256x256)
    - Resizes back to target resolution (256x256) to maintain compatibility
    """
    h, w = aligned_face.shape[:2]
    crop_x1 = int(w * CROP_MARGIN)
    crop_y1 = int(h * CROP_MARGIN)
    crop_x2 = int(w * (1 - CROP_MARGIN))
    crop_y2 = int(h * (1 - CROP_MARGIN))
    cropped_face = aligned_face[crop_y1:crop_y2, crop_x1:crop_x2]
    cropped_face = cv2.resize(cropped_face, TARGET_SIZE, interpolation=cv2.INTER_CUBIC)
    return cropped_face

def correct_face_rotation(face_img, left_eye, right_eye):
    """
    Correct tilted face to horizontal using actual eye coordinates (not template)
    - Input: Cropped face + actual left/right eye coordinates from MTCNN
    - Output: Horizontally aligned face (no black borders, no distortion)
    """
    # Calculate tilt angle from actual eye line (radians → degrees)
    x1, y1 = left_eye
    x2, y2 = right_eye
    delta_x = x2 - x1
    delta_y = y2 - y1
    tilt_angle = math.atan2(delta_y, delta_x) * 180 / math.pi

    # Rotation center (center of the face image)
    h, w = face_img.shape[:2]
    center = (w // 2, h // 2)

    # Create rotation matrix (negative angle = correct tilt to horizontal)
    rotation_matrix = cv2.getRotationMatrix2D(center, -tilt_angle, 1.0)

    # Apply rotation with high-quality interpolation and border replication (no black edges)
    corrected_face = cv2.warpAffine(
        face_img,
        rotation_matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    return corrected_face

def main():
    # Initialize MTCNN face detector (default parameters)
    mtcnn = MTCNN()
    
    # Path configuration (update these paths if your project structure changes)
    raw_data_dir = r"E:\8016project\data\raw\rawsubset"
    processed_output_dir = r"E:\8016project\data\processed\256x256"
    Path(processed_output_dir).mkdir(parents=True, exist_ok=True)
    
    # Filter valid image files (only .jpg/.png) and limit to TEST_NUM
    image_files = sorted([
        f for f in os.listdir(raw_data_dir)
        if f.endswith(('.jpg', '.png')) and os.path.isfile(os.path.join(raw_data_dir, f))
    ])[:TEST_NUM]
    
    print(f"🔥 Preprocessing (Simplified): MTCNN Alignment + Normalization | Rotation Correction: {'ON' if ENABLE_FACE_HORIZONTAL_CORRECTION else 'OFF'} | Processing {len(image_files)} images")
    success_count = 0

    for idx, filename in enumerate(image_files):
        try:
            # Step 1: Read input image and convert to RGB format
            img_path = os.path.join(raw_data_dir, filename)
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                raise ValueError("Failed to read image (corrupted file or wrong path)")
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            # Step 2: Detect faces with MTCNN (select the largest face to avoid small background faces)
            face_detections = mtcnn.detect_faces(img_rgb)
            if not face_detections:
                raise ValueError("No face detected in the image")
            main_face = max(face_detections, key=lambda fd: fd['box'][2] * fd['box'][3])

            # Store actual eye coordinates (for rotation correction, if enabled)
            actual_left_eye = main_face['keypoints']['left_eye']
            actual_right_eye = main_face['keypoints']['right_eye']

            # Step 3: Extract 5 facial landmarks and perform affine alignment
            landmarks_np = np.array([
                main_face['keypoints']['left_eye'],
                main_face['keypoints']['right_eye'],
                main_face['keypoints']['nose'],
                main_face['keypoints']['mouth_left'],
                main_face['keypoints']['mouth_right']
            ], dtype=np.float32)
            M, _ = cv2.estimateAffinePartial2D(landmarks_np, STD_LANDMARKS)
            aligned_face = cv2.warpAffine(img_rgb, M, TARGET_SIZE, flags=cv2.INTER_CUBIC)

            # Step 4: Apply mild tight crop to remove excess background
            aligned_face = mild_tight_crop(aligned_face)

            # Step 5: Normalize pixel values to [0, 255] (avoid overexposure/black images)
            processed_face = (aligned_face.astype(np.float32) / 255.0 * 255).astype(np.uint8)

            # --------------------------
            # Optional: Correct face rotation (DISABLED in this version)
            # --------------------------
            if ENABLE_FACE_HORIZONTAL_CORRECTION:
                processed_face = correct_face_rotation(
                    processed_face,
                    actual_left_eye,
                    actual_right_eye
                )

            # Step 6: Save the preprocessed image (convert back to BGR for OpenCV)
            save_path = os.path.join(processed_output_dir, filename)
            cv2.imwrite(save_path, cv2.cvtColor(processed_face, cv2.COLOR_RGB2BGR))

            success_count += 1
            print(f"✅ [{idx+1}/{len(image_files)}] Success: {filename}")

        except Exception as e:
            print(f"❌ [{idx+1}/{len(image_files)}] Failed: {filename} | Reason: {str(e)}")

    # Final summary (clean output)
    print(f"\n✅ Preprocessing completed!")
    print(f"📊 Total: {len(image_files)} | Success: {success_count} | Failed: {len(image_files)-success_count}")
    print(f"💾 Results saved to: {processed_output_dir}")
    print(f"💡 5% margin removed (core face retained) + Rotation correction: {'ON' if ENABLE_FACE_HORIZONTAL_CORRECTION else 'OFF'}")

if __name__ == "__main__":
    main()
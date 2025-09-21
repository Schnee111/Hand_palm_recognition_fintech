# File: preprocess_dataset.py (Versi dengan Filter Kualitas Area)
import cv2
import mediapipe as mp
import numpy as np
import os
from tqdm import tqdm

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = 'hand_landmarker.task'
# --- KITA TAMBAHKAN FILTER ---
# Anggap segmentasi valid jika area telapak tangan > 10% dari total area gambar
MIN_PALM_AREA_RATIO = 0.10 

# --- FUNGSI SEGMENTASI DIMODIFIKASI DENGAN FILTER KUALITAS ---
def segment_palm_only(landmarker, image_path):
    if not os.path.exists(image_path):
        return None, "File tidak ditemukan"
        
    image_cv = cv2.imread(image_path)
    if image_cv is None:
        return None, "Gagal dibaca OpenCV"

    try:
        mp_image = mp.Image.create_from_file(image_path)
        hand_landmarker_result = landmarker.detect(mp_image)

        if not hand_landmarker_result.hand_landmarks:
            return None, "Tidak ada tangan terdeteksi"

        landmarks = hand_landmarker_result.hand_landmarks[0]
        palm_points_indices = [0, 1, 5, 9, 13, 17]
        
        image_height, image_width, _ = image_cv.shape
        palm_polygon_points = np.array(
            [[int(landmarks[i].x * image_width), int(landmarks[i].y * image_height)] for i in palm_points_indices],
            dtype=np.int32
        )

        custom_mask = np.zeros((image_height, image_width), dtype=np.uint8)
        cv2.fillPoly(custom_mask, [palm_polygon_points], (255))
        
        # --- FILTER KUALITAS DITERAPKAN DI SINI ---
        palm_area = cv2.countNonZero(custom_mask)
        total_area = image_height * image_width
        area_ratio = palm_area / total_area
        
        if area_ratio < MIN_PALM_AREA_RATIO:
            return None, f"Area telapak tangan terlalu kecil ({area_ratio:.2f})"
            
        segmented_palm = cv2.bitwise_and(image_cv, image_cv, mask=custom_mask)
        
        return segmented_palm, "Sukses"
    except Exception as e:
        return None, f"Error MediaPipe: {e}"


def process_dataset(source_base_dir, target_base_dir):
    print("Menginisialisasi model HandLandmarker...")
    BaseOptions = python.BaseOptions
    HandLandmarker = vision.HandLandmarker
    HandLandmarkerOptions = vision.HandLandmarkerOptions
    VisionRunningMode = vision.RunningMode
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=1
    )
    
    with HandLandmarker.create_from_options(options) as landmarker:
        print("Model berhasil diinisialisasi.")
        
        for subset in ['training', 'validation']:
            source_dir = os.path.join(source_base_dir, subset)
            
            if not os.path.isdir(source_dir):
                print(f"Warning: Folder sumber tidak ditemukan: {source_dir}")
                continue

            image_paths = []
            for root, _, files in os.walk(source_dir):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_paths.append(os.path.join(root, file))
            
            print(f"\nMemulai proses untuk subset '{subset}'. Total gambar: {len(image_paths)}")
            
            fail_count = 0
            for source_path in tqdm(image_paths, desc=f"Processing {subset}"):
                relative_path = os.path.relpath(source_path, source_base_dir)
                target_path = os.path.join(target_base_dir, relative_path)
                
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                
                if os.path.exists(target_path):
                    continue
                    
                segmented_image, status = segment_palm_only(landmarker, source_path)
                
                if segmented_image is not None:
                    cv2.imwrite(target_path, segmented_image)
                else:
                    fail_count += 1
            
            print(f"Proses '{subset}' selesai. Gagal diproses: {fail_count} gambar.")

if __name__ == '__main__':
    source_dataset_dir = 'palm_data_split'
    target_dataset_dir = 'palm_data_split_segmented_filtered'
    
    print("Memulai pra-pemrosesan seluruh dataset dengan filter kualitas...")
    process_dataset(source_dataset_dir, target_dataset_dir)
    print("\n✅ Pra-pemrosesan selesai!")
    print(f"Dataset bersih Anda tersedia di: {target_dataset_dir}")
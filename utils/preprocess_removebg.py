# File: preprocess_dataset_final.py
# Skrip ini memproses seluruh dataset untuk segmentasi TANGAN UTUH (telapak + jari).

import cv2
import mediapipe as mp
import numpy as np
import os
from tqdm import tqdm

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- KONFIGURASI ---
MODEL_PATH = 'selfie_multiclass_256x256.tflite'
SOURCE_BASE_DIR = 'palm_data_split'
# Folder baru untuk menyimpan hasil terbaik ini
TARGET_BASE_DIR = 'palm_data_split_full_hand_segmented' 

def segment_full_hand(segmenter, image_path):
    """
    Fungsi untuk melakukan segmentasi tangan utuh, dioptimalkan untuk memori.
    """
    try:
        # 1. Baca gambar asli dengan OpenCV
        image_cv = cv2.imread(image_path)
        if image_cv is None:
            return None, "Gagal dibaca OpenCV"

        # 2. Perkecil ukurannya untuk proses yang cepat dan hemat memori
        target_width = 640
        height, width, _ = image_cv.shape
        if width == 0 or height == 0: return None, "Gambar tidak valid"
        aspect_ratio = height / width
        target_height = int(target_width * aspect_ratio)
        resized_image = cv2.resize(image_cv, (target_width, target_height))
        
        # 3. Konversi ke format yang dibutuhkan MediaPipe
        resized_image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=resized_image_rgb)
        
        # 4. Jalankan segmentasi
        segmentation_result = segmenter.segment(mp_image)
        category_mask = segmentation_result.category_mask
        mask_np = category_mask.numpy_view()

        # 5. Buat background hitam dan terapkan mask pada gambar kecil
        black_background = np.zeros(resized_image.shape, dtype=np.uint8)
        condition = mask_np != 0
        condition_3_channels = np.stack((condition,) * 3, axis=-1)
        segmented_image = np.where(condition_3_channels, resized_image, black_background)
        
        return segmented_image, "Sukses"

    except Exception as e:
        return None, f"Error MediaPipe: {e}"

def process_dataset(source_base_dir, target_base_dir):
    """
    Fungsi utama untuk memproses seluruh dataset.
    """
    # Inisialisasi model SATU KALI di luar loop
    print("Menginisialisasi model ImageSegmenter...")
    BaseOptions = python.BaseOptions
    ImageSegmenter = vision.ImageSegmenter
    ImageSegmenterOptions = vision.ImageSegmenterOptions
    options = ImageSegmenterOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        output_category_mask=True
    )
    
    with ImageSegmenter.create_from_options(options) as segmenter:
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
            success_count = 0
            for source_path in tqdm(image_paths, desc=f"Processing {subset}"):
                relative_path = os.path.relpath(source_path, source_base_dir)
                target_path = os.path.join(target_base_dir, relative_path)
                
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                
                if os.path.exists(target_path):
                    success_count +=1
                    continue
                    
                segmented_image, status = segment_full_hand(segmenter, source_path)
                
                if segmented_image is not None:
                    cv2.imwrite(target_path, segmented_image)
                    success_count += 1
                else:
                    fail_count += 1
            
            print(f"Proses '{subset}' selesai. Berhasil: {success_count}, Gagal: {fail_count} gambar.")

if __name__ == '__main__':
    # Pastikan file model ada di folder yang sama dengan skrip ini
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: File model segmentasi '{MODEL_PATH}' tidak ditemukan.")
    else:
        print("Memulai pra-pemrosesan seluruh dataset...")
        process_dataset(SOURCE_BASE_DIR, TARGET_BASE_DIR)
        print("\n✅ Pra-pemrosesan FINAL selesai!")
        print(f"Dataset baru Anda tersedia di: {TARGET_BASE_DIR}")

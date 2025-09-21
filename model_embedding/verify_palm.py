# File: verify_image.py
import tensorflow as tf
import numpy as np
from scipy.spatial.distance import euclidean
import os

# --- KONFIGURASI ---
EMBEDDING_MODEL_PATH = 'palm_embedding_model.h5'
TEST_IMAGE_PATH = 'test/test.jpg' # GANTI dengan jalur gambar uji Anda
TEMPLATE_DB_PATH = 'palm_templates.npy'
IMG_SIZE = 128
CHANNELS = 3
DISTANCE_THRESHOLD = 15.0 # GANTI dengan ambang batas yang sudah Anda kalibrasi!
# --------------------

def load_and_preprocess_image(img_path, size, channels):
    """Memuat dan pra-proses satu gambar uji."""
    img = tf.keras.utils.load_img(
        img_path, target_size=(size, size), 
        color_mode='rgb' if channels == 3 else 'grayscale'
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0 # Normalisasi
    return img_array

def verify_and_identify(template_db, embedding_model, test_img_path, threshold):
    """Melakukan verifikasi dan identifikasi."""
    
    # 1. Ekstrak embedding dari gambar uji
    processed_image = load_and_preprocess_image(test_img_path, IMG_SIZE, CHANNELS)
    test_embedding = embedding_model.predict(processed_image, verbose=0)[0]
    
    min_distance = float('inf')
    best_match_id = "UNKNOWN"
    
    # 2. Bandingkan embedding uji dengan semua template
    distances = {}
    for id_name, template_vector in template_db.items(): # Mengakses item dari dictionary numpy
        distance = euclidean(test_embedding, template_vector)
        distances[id_name] = distance
        
        if distance < min_distance:
            min_distance = distance
            best_match_id = id_name

    print("\n=== Hasil Verifikasi Jarak ===")
    print(f"Gambar Uji: {os.path.basename(test_img_path)}")
    print(f"Ambang Batas (Threshold): {threshold:.4f}")

    # 3. Keputusan
    if min_distance <= threshold:
        print(f"✅ VERIFIKASI DITERIMA: ID {best_match_id}")
        print(f"   Jarak Terdekat: {min_distance:.4f} (DIBAWAH Threshold)")
    else:
        print(f"❌ VERIFIKASI DITOLAK: UNKNOWN ID")
        print(f"   Jarak Terdekat ke ID {best_match_id}: {min_distance:.4f} (DIATAS Threshold)")

    print("\nJarak ke Semua ID (Terurut):")
    sorted_distances = sorted(distances.items(), key=lambda item: item[1])
    for id_name, dist in sorted_distances:
         print(f"   ID {id_name}: {dist:.4f}")

if __name__ == "__main__":
    if not os.path.exists(TEMPLATE_DB_PATH):
        print(f"❌ Error: Database template '{TEMPLATE_DB_PATH}' tidak ditemukan. Jalankan create_templates.py terlebih dahulu!")
    elif not os.path.exists(EMBEDDING_MODEL_PATH):
        print(f"❌ Error: Model Embedding '{EMBEDDING_MODEL_PATH}' tidak ditemukan.")
    else:
        # Muat Model dan Template
        embedding_model = tf.keras.models.load_model(EMBEDDING_MODEL_PATH)
        # Muat array NumPy, lalu ambil item [()] untuk mendapatkan dict
        template_db_loaded = np.load(TEMPLATE_DB_PATH, allow_pickle=True)[()] 
        
        # Uji Verifikasi
        verify_and_identify(template_db_loaded, embedding_model, TEST_IMAGE_PATH, DISTANCE_THRESHOLD)
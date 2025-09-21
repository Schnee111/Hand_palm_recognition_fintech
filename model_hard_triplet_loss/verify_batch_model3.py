# File: verify_folder_triplet_no_seg.py
# Skrip untuk verifikasi batch TANPA segmentasi, khusus untuk Model 3.
import tensorflow as tf
import numpy as np
from scipy.spatial.distance import euclidean
import os
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
from tqdm import tqdm

# --- DEFINISI CUSTOM LAYER (HARUS ADA SAAT MEMUAT MODEL) ---
class L2NormalizeLayer(Layer):
    def __init__(self, **kwargs):
        super(L2NormalizeLayer, self).__init__(**kwargs)
    def call(self, inputs):
        return K.l2_normalize(inputs, axis=1)
    def get_config(self):
        config = super(L2NormalizeLayer, self).get_config()
        return config

# --- KONFIGURASI ---
# PENTING: Pastikan path ini menunjuk ke model & template DARI HASIL TRAINING TANPA SEGMENTASI
EMBEDDING_MODEL_PATH = 'triplet_encoder_best_mobilenetv2.h5'
TEMPLATE_DB_PATH = 'palm_templates_triplet_mobilenetv2.npy'

TEST_FOLDER_PATH = '../test_unverified' # Ganti dengan folder berisi gambar uji Anda
IMG_SIZE = 160
DISTANCE_THRESHOLD = 1.0 # Sesuaikan nilai ini setelah kalibrasi

# --- FUNGSI PRA-PROSES STANDAR ---
def load_and_preprocess_image(img_path, size):
    """Memuat dan pra-proses satu gambar uji tanpa segmentasi."""
    try:
        img = tf.keras.utils.load_img(
            img_path, target_size=(size, size), color_mode='rgb'
        )
        img_array = tf.keras.utils.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        return img_array
    except Exception:
        return None

# --- FUNGSI VERIFIKASI ---
def verify_single_image(template_db, embedding_model, test_img_path, threshold):
    """Fungsi ini memverifikasi satu gambar dan mengembalikan hasilnya."""
    processed_image = load_and_preprocess_image(test_img_path, IMG_SIZE)
    if processed_image is None:
        print(f"[{os.path.basename(test_img_path)}] ❌ GAGAL: Gambar tidak bisa dibaca.")
        return None, "Gagal Membaca"

    test_embedding = embedding_model.predict(processed_image, verbose=0)[0]
    
    # Logika verifikasi terpadu L/R
    grouped_templates = {}
    for id_name, template_vector in template_db.items():
        main_id = id_name.split('_')[0]
        if main_id not in grouped_templates: grouped_templates[main_id] = []
        grouped_templates[main_id].append(template_vector)
        
    min_overall_distance = float('inf')
    best_match_id = "UNKNOWN"
    for main_id, templates in grouped_templates.items():
        dist_to_templates = [euclidean(test_embedding, t) for t in templates]
        min_dist_for_id = min(dist_to_templates)
        if min_dist_for_id < min_overall_distance:
            min_overall_distance = min_dist_for_id
            best_match_id = main_id

    # Keputusan
    if min_overall_distance <= threshold:
        print(f"[{os.path.basename(test_img_path)}] ✅ DITERIMA sebagai ID {best_match_id} (Jarak: {min_overall_distance:.4f})")
        return True, best_match_id
    else:
        print(f"[{os.path.basename(test_img_path)}] ❌ DITOLAK. Terdekat dengan ID {best_match_id} (Jarak: {min_overall_distance:.4f})")
        return False, best_match_id

if __name__ == "__main__":
    if not all([os.path.exists(p) for p in [TEMPLATE_DB_PATH, EMBEDDING_MODEL_PATH]]):
        print("❌ Error: Pastikan file model dan template sudah ada.")
    elif not os.path.isdir(TEST_FOLDER_PATH):
        print(f"❌ Error: Folder uji '{TEST_FOLDER_PATH}' tidak ditemukan.")
    else:
        # Muat semua model sekali saja
        print("Memuat model...")
        custom_objects = {'L2NormalizeLayer': L2NormalizeLayer}
        embedding_model = tf.keras.models.load_model(EMBEDDING_MODEL_PATH, custom_objects=custom_objects, compile=False)
        template_db_loaded = np.load(TEMPLATE_DB_PATH, allow_pickle=True).item()
        
        print("Model siap. Memulai verifikasi batch...")
        
        image_files = [f for f in os.listdir(TEST_FOLDER_PATH) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        accepted_count = 0
        rejected_count = 0
        failed_count = 0
        
        for filename in (image_files):
            full_path = os.path.join(TEST_FOLDER_PATH, filename)
            is_accepted, _ = verify_single_image(template_db_loaded, embedding_model, full_path, DISTANCE_THRESHOLD)
            
            if is_accepted is None:
                failed_count += 1
            elif is_accepted:
                accepted_count += 1
            else:
                rejected_count += 1
        
        # Tampilkan Ringkasan
        print("\n\n--- RINGKASAN VERIFIKASI BATCH (TANPA SEGMENTASI) ---")
        print(f"Total Gambar Diuji      : {len(image_files)}")
        print(f"✅ Diterima (Accepted)    : {accepted_count}")
        print(f"❌ Ditolak (Rejected)     : {rejected_count}")
        print(f"⚠️ Gagal Dibaca          : {failed_count}")
        print("----------------------------------------------------")
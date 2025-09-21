# File: verify_folder.py
# Skrip untuk verifikasi semua gambar dalam satu folder.
import cv2
import mediapipe as mp
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
from scipy.spatial.distance import euclidean
from tqdm import tqdm

# --- BAGIAN 1: DEFINISI CUSTOM LAYER DAN FUNGSI SEGMENTASI ---
class L2NormalizeLayer(Layer):
    def __init__(self, **kwargs):
        super(L2NormalizeLayer, self).__init__(**kwargs)
    def call(self, inputs):
        return K.l2_normalize(inputs, axis=1)
    def get_config(self):
        config = super(L2NormalizeLayer, self).get_config()
        return config

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH_SEGMENTER = 'selfie_multiclass_256x256.tflite'

def segment_full_hand(segmenter, image_path):
    try:
        image_cv = cv2.imread(image_path)
        if image_cv is None: return None
        target_width = 640
        height, width, _ = image_cv.shape
        if width == 0 or height == 0: return None
        aspect_ratio = height / width
        target_height = int(target_width * aspect_ratio)
        resized_image = cv2.resize(image_cv, (target_width, target_height))
        resized_image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=resized_image_rgb)
        segmentation_result = segmenter.segment(mp_image)
        category_mask = segmentation_result.category_mask
        mask_np = category_mask.numpy_view()
        black_background = np.zeros(resized_image.shape, dtype=np.uint8)
        condition = mask_np != 0
        condition_3_channels = np.stack((condition,) * 3, axis=-1)
        segmented_image = np.where(condition_3_channels, resized_image, black_background)
        return segmented_image
    except Exception:
        return None

# --- BAGIAN 2: KONFIGURASI VERIFIKASI ---
EMBEDDING_MODEL_PATH = 'triplet_encoder_best_mobilenetv2.h5'
TEMPLATE_DB_PATH = 'palm_templates_triplet_mobilenetv2.npy'
# --- PERUBAHAN: Path ke FOLDER, bukan FILE ---
TEST_FOLDER_PATH = '../test_unverified' # Ganti dengan folder berisi gambar uji Anda
IMG_SIZE = 160
DISTANCE_THRESHOLD = 0.6 # Sesuaikan nilai ini setelah kalibrasi

# --- BAGIAN 3: LOGIKA VERIFIKASI ---
def preprocess_for_model(img_array, size):
    img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    img_resized = tf.image.resize(img_rgb, [size, size])
    img_array_final = tf.keras.utils.img_to_array(img_resized)
    img_array_final = np.expand_dims(img_array_final, axis=0)
    img_array_final = img_array_final / 255.0
    return img_array_final

def verify_single_image(template_db, embedding_model, segmenter, test_img_path, threshold):
    """Fungsi ini memverifikasi satu gambar dan mengembalikan hasilnya."""
    segmented_image = segment_full_hand(segmenter, test_img_path)
    if segmented_image is None:
        print(f"[{os.path.basename(test_img_path)}] ❌ GAGAL: Segmentasi tidak berhasil.")
        return None, "Gagal Segmentasi"

    processed_image = preprocess_for_model(segmented_image, IMG_SIZE)
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
    if not all([os.path.exists(p) for p in [TEMPLATE_DB_PATH, EMBEDDING_MODEL_PATH, MODEL_PATH_SEGMENTER]]):
        print("❌ Error: Pastikan file model, template, dan segmenter sudah ada.")
    elif not os.path.isdir(TEST_FOLDER_PATH):
        print(f"❌ Error: Folder uji '{TEST_FOLDER_PATH}' tidak ditemukan.")
    else:
        # Muat semua model sekali saja
        print("Memuat model...")
        custom_objects = {'L2NormalizeLayer': L2NormalizeLayer}
        embedding_model = tf.keras.models.load_model(EMBEDDING_MODEL_PATH, custom_objects=custom_objects, compile=False)
        template_db_loaded = np.load(TEMPLATE_DB_PATH, allow_pickle=True).item()
        
        BaseOptions = python.BaseOptions
        ImageSegmenter = vision.ImageSegmenter
        ImageSegmenterOptions = vision.ImageSegmenterOptions
        options = ImageSegmenterOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH_SEGMENTER),
            output_category_mask=True)
        
        with ImageSegmenter.create_from_options(options) as segmenter:
            print("Model siap. Memulai verifikasi batch...")
            
            # Dapatkan daftar gambar dari folder
            image_files = [f for f in os.listdir(TEST_FOLDER_PATH) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            # Inisialisasi penghitung
            accepted_count = 0
            rejected_count = 0
            failed_count = 0
            
            # Loop melalui semua gambar
            for filename in (image_files):
                full_path = os.path.join(TEST_FOLDER_PATH, filename)
                is_accepted, _ = verify_single_image(template_db_loaded, embedding_model, segmenter, full_path, DISTANCE_THRESHOLD)
                
                if is_accepted is None:
                    failed_count += 1
                elif is_accepted:
                    accepted_count += 1
                else:
                    rejected_count += 1
            
            # Tampilkan Ringkasan
            print("\n\n--- RINGKASAN VERIFIKASI BATCH ---")
            print(f"Total Gambar Diuji      : {len(image_files)}")
            print(f"✅ Diterima (Accepted)    : {accepted_count}")
            print(f"❌ Ditolak (Rejected)     : {rejected_count}")
            print(f"⚠️ Gagal Segmentasi      : {failed_count}")
            print("------------------------------------")
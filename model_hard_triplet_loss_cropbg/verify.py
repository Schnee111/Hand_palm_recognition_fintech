# File: verify.py (Final dengan Tampilan Jarak Lengkap)
import cv2
import mediapipe as mp
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
from scipy.spatial.distance import euclidean

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

def segment_full_hand(image_path):
    if not os.path.exists(image_path):
        print(f"❌ Error: File tidak ditemukan di path: {image_path}")
        return None
    image_cv = cv2.imread(image_path)
    if image_cv is None:
        print(f"❌ Error: OpenCV gagal membaca gambar dari {image_path}")
        return None
    target_width = 640
    height, width, _ = image_cv.shape
    if width == 0 or height == 0: return None
    aspect_ratio = height / width
    target_height = int(target_width * aspect_ratio)
    resized_image = cv2.resize(image_cv, (target_width, target_height))
    resized_image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    try:
        BaseOptions = python.BaseOptions
        ImageSegmenter = vision.ImageSegmenter
        ImageSegmenterOptions = vision.ImageSegmenterOptions
        options = ImageSegmenterOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH_SEGMENTER),
            output_category_mask=True)
        with ImageSegmenter.create_from_options(options) as segmenter:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=resized_image_rgb)
            segmentation_result = segmenter.segment(mp_image)
            category_mask = segmentation_result.category_mask
            mask_np = category_mask.numpy_view()
            black_background = np.zeros(resized_image.shape, dtype=np.uint8)
            condition = mask_np != 0
            condition_3_channels = np.stack((condition,) * 3, axis=-1)
            segmented_image = np.where(condition_3_channels, resized_image, black_background)
            return segmented_image
    except Exception as e:
        print(f"❌ Terjadi error saat segmentasi: {e}")
        return None

# --- BAGIAN 2: KONFIGURASI VERIFIKASI ---
EMBEDDING_MODEL_PATH = 'triplet_encoder_best_mobilenetv2.h5'
TEMPLATE_DB_PATH = 'palm_templates_triplet_mobilenetv2.npy'
TEST_IMAGE_PATH = '../test_unverified/IMG_20250919_164206_228.jpg'
IMG_SIZE = 160
DISTANCE_THRESHOLD = 0.6

# --- BAGIAN 3: LOGIKA VERIFIKASI FINAL ---
def preprocess_for_model(img_array, size):
    img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    img_resized = tf.image.resize(img_rgb, [size, size])
    img_array_final = tf.keras.utils.img_to_array(img_resized)
    img_array_final = np.expand_dims(img_array_final, axis=0)
    img_array_final = img_array_final / 255.0
    return img_array_final

def verify_and_identify(template_db, embedding_model, test_img_path, threshold):
    print(f"\nMemulai proses verifikasi untuk: {os.path.basename(test_img_path)}")
    print("1. Melakukan segmentasi tangan utuh...")
    segmented_image = segment_full_hand(test_img_path)
    if segmented_image is None:
        print("❌ GAGAL: Proses segmentasi tidak menghasilkan gambar.")
        return
    print("   ✅ Segmentasi berhasil.")

    print("2. Melakukan pra-pemrosesan untuk model...")
    processed_image = preprocess_for_model(segmented_image, IMG_SIZE)
    print("   ✅ Pra-pemrosesan berhasil.")

    print("3. Mengekstrak fitur embedding...")
    test_embedding = embedding_model.predict(processed_image, verbose=0)[0]
    print("   ✅ Ekstraksi embedding berhasil.")

    print("4. Membandingkan dengan database template (Logika Terpadu L/R)...")
    grouped_templates = {}
    for id_name, template_vector in template_db.items():
        main_id = id_name.split('_')[0]
        if main_id not in grouped_templates: grouped_templates[main_id] = []
        grouped_templates[main_id].append(template_vector)
    min_overall_distance = float('inf')
    best_match_id = "UNKNOWN"
    distances_per_id = {}
    for main_id, templates in grouped_templates.items():
        dist_to_templates = [euclidean(test_embedding, t) for t in templates]
        min_dist_for_id = min(dist_to_templates)
        distances_per_id[main_id] = min_dist_for_id
        if min_dist_for_id < min_overall_distance:
            min_overall_distance = min_dist_for_id
            best_match_id = main_id

    print("\n=== HASIL VERIFIKASI FINAL ===")
    print(f"Ambang Batas (Threshold): {threshold:.4f}")

    if min_overall_distance <= threshold:
        print(f"✅ VERIFIKASI DITERIMA: ID {best_match_id}")
        print(f"   Jarak Terdekat: {min_overall_distance:.4f} (DIBAWAH Threshold)")
    else:
        print(f"❌ VERIFIKASI DITOLAK: UNKNOWN ID")
        print(f"   Jarak Terdekat ke ID {best_match_id}: {min_overall_distance:.4f} (DIATAS Threshold)")

    # --- PENAMBAHAN KODE: TAMPILKAN SEMUA JARAK ---
    print("\nJarak ke Semua ID (Terurut dari terdekat):")
    sorted_distances = sorted(distances_per_id.items(), key=lambda item: item[1])
    for id_name, dist in sorted_distances:
         print(f"   - Jarak ke ID {id_name}: {dist:.4f}")
    # --- AKHIR PENAMBAHAN KODE ---

if __name__ == "__main__":
    if not os.path.exists(TEMPLATE_DB_PATH) or not os.path.exists(EMBEDDING_MODEL_PATH):
        print("❌ Error: Pastikan file model dan template database sudah ada.")
    elif not os.path.exists(MODEL_PATH_SEGMENTER):
        print(f"❌ Error: File model segmentasi '{MODEL_PATH_SEGMENTER}' tidak ditemukan.")
    else:
        custom_objects = {'L2NormalizeLayer': L2NormalizeLayer}
        embedding_model = tf.keras.models.load_model(EMBEDDING_MODEL_PATH, custom_objects=custom_objects, compile=False)
        template_db_loaded = np.load(TEMPLATE_DB_PATH, allow_pickle=True).item()
        verify_and_identify(template_db_loaded, embedding_model, TEST_IMAGE_PATH, DISTANCE_THRESHOLD)

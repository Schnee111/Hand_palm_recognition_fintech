# File: verify.py (Final dengan Visualisasi Segmentasi)
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

MODEL_PATH_SEGMENTER = 'hand_landmarker.task'

def segment_palm_only(image_path):
    if not os.path.exists(image_path):
        print(f"❌ Error: File tidak ditemukan di path: {image_path}")
        return None
    image_cv = cv2.imread(image_path)
    if image_cv is None:
        print(f"❌ Error: OpenCV gagal membaca gambar dari {image_path}")
        return None
    try:
        BaseOptions = python.BaseOptions
        HandLandmarker = vision.HandLandmarker
        HandLandmarkerOptions = vision.HandLandmarkerOptions
        VisionRunningMode = vision.RunningMode
        options = HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=MODEL_PATH_SEGMENTER), running_mode=VisionRunningMode.IMAGE, num_hands=1)
        with HandLandmarker.create_from_options(options) as landmarker:
            mp_image = mp.Image.create_from_file(image_path)
            hand_landmarker_result = landmarker.detect(mp_image)
            if not hand_landmarker_result.hand_landmarks: return None
            landmarks = hand_landmarker_result.hand_landmarks[0]
            palm_points_indices = [0, 1, 5, 9, 13, 17]
            image_height, image_width, _ = image_cv.shape
            palm_polygon_points = np.array([[int(landmarks[i].x * image_width), int(landmarks[i].y * image_height)] for i in palm_points_indices], dtype=np.int32)
            custom_mask = np.zeros((image_height, image_width), dtype=np.uint8)
            cv2.fillPoly(custom_mask, [palm_polygon_points], (255))
            segmented_palm = cv2.bitwise_and(image_cv, image_cv, mask=custom_mask)
            return segmented_palm
    except Exception as e:
        print(f"❌ Terjadi error saat segmentasi: {e}")
        return None

# --- BAGIAN 2: KONFIGURASI VERIFIKASI ---
EMBEDDING_MODEL_PATH = 'triplet_encoder_best_mobilenetv2.h5'
TEMPLATE_DB_PATH = 'palm_templates_triplet_mobilenetv2.npy'
TEST_IMAGE_PATH = '../test/test.jpg'
IMG_SIZE = 160
DISTANCE_THRESHOLD = 0.8

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
    
    # LANGKAH 1: Lakukan Segmentasi Palm-Only
    print("1. Melakukan segmentasi telapak tangan...")
    segmented_image = segment_palm_only(test_img_path)
    if segmented_image is None:
        print("❌ GAGAL: Proses segmentasi tidak menghasilkan gambar.")
        return
    print("   ✅ Segmentasi berhasil.")
    
    # --- PENAMBAHAN KODE VISUALISASI ---
    print("   Visualisasi hasil segmentasi... Tekan tombol apa saja untuk melanjutkan.")
    try:
        original_image = cv2.imread(test_img_path)
        cv2.imshow('Gambar Asli', original_image)
        cv2.imshow('Hasil Segmentasi', segmented_image)
        cv2.waitKey(0) # Program akan berhenti di sini sampai Anda menekan tombol
        cv2.destroyAllWindows() # Menutup semua jendela gambar
    except Exception as e:
        print(f"Gagal menampilkan gambar: {e}")
    # --- AKHIR PENAMBAHAN KODE ---

    # LANGKAH 2: Pra-proses gambar hasil segmentasi
    print("2. Melakukan pra-pemrosesan untuk model...")
    processed_image = preprocess_for_model(segmented_image, IMG_SIZE)
    print("   ✅ Pra-pemrosesan berhasil.")

    # LANGKAH 3: Ekstrak embedding
    print("3. Mengekstrak fitur embedding...")
    test_embedding = embedding_model.predict(processed_image, verbose=0)[0]
    print("   ✅ Ekstraksi embedding berhasil.")

    # LANGKAH 4: Bandingkan dengan database (Logika Terpadu L/R)
    print("4. Membandingkan dengan database template...")
    # ... (sisa fungsi ini sama persis seperti sebelumnya)
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

if __name__ == "__main__":
    if not os.path.exists(TEMPLATE_DB_PATH) or not os.path.exists(EMBEDDING_MODEL_PATH):
        print("❌ Error: Pastikan file model dan template database sudah ada.")
    else:
        custom_objects = {'L2NormalizeLayer': L2NormalizeLayer}
        embedding_model = tf.keras.models.load_model(EMBEDDING_MODEL_PATH, custom_objects=custom_objects, compile=False)
        template_db_loaded = np.load(TEMPLATE_DB_PATH, allow_pickle=True).item()
        verify_and_identify(template_db_loaded, embedding_model, TEST_IMAGE_PATH, DISTANCE_THRESHOLD)
# File: verify.py (FINAL: MENGGUNAKAN CUSTOM LAYER)
import tensorflow as tf
import numpy as np
from scipy.spatial.distance import euclidean
import os
from tensorflow.keras.layers import Layer # Import Layer
import tensorflow.keras.backend as K

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
EMBEDDING_MODEL_PATH = 'palm_triplet_encoder_mobilenetv2.h5'
TEMPLATE_DB_PATH = 'palm_templates_triplet_mobilenetv2.npy'
TEST_IMAGE_PATH = '../test/R_clear.jpg' # GANTI dengan jalur gambar uji Anda
IMG_SIZE = 160
CHANNELS = 3
DISTANCE_THRESHOLD = 1.0 # INGAT: Nilai ini perlu dikalibrasi
# --------------------

def load_and_preprocess_image(img_path, size, channels):
    """Memuat dan pra-proses satu gambar uji."""
    img = tf.keras.utils.load_img(
        img_path, target_size=(size, size),
        color_mode='rgb' if channels == 3 else 'grayscale'
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def verify_and_identify(template_db, embedding_model, test_img_path, threshold):
    """Melakukan verifikasi dan identifikasi."""
    if not os.path.exists(test_img_path):
        print(f"❌ Error: Gambar uji '{test_img_path}' tidak ditemukan.")
        return

    processed_image = load_and_preprocess_image(test_img_path, IMG_SIZE, CHANNELS)
    test_embedding = embedding_model.predict(processed_image, verbose=0)[0]

    min_distance = float('inf')
    best_match_id = "UNKNOWN"

    distances = {}
    for id_name, template_vector in template_db.items():
        distance = euclidean(test_embedding, template_vector)
        distances[id_name] = distance

        if distance < min_distance:
            min_distance = distance
            best_match_id = id_name

    print("\n=== Hasil Verifikasi Jarak (Model MobileNetV2) ===")
    print(f"Gambar Uji: {os.path.basename(test_img_path)}")
    print(f"Ambang Batas (Threshold): {threshold:.4f}")

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
        print(f"❌ Error: Database template '{TEMPLATE_DB_PATH}' tidak ditemukan. Jalankan create_templates.py.")
    elif not os.path.exists(EMBEDDING_MODEL_PATH):
        print(f"❌ Error: Model Embedding '{EMBEDDING_MODEL_PATH}' tidak ditemukan.")
    else:
        # Memberitahu load_model tentang Custom Layer kita
        custom_objects = {'L2NormalizeLayer': L2NormalizeLayer}
        embedding_model = tf.keras.models.load_model(
            EMBEDDING_MODEL_PATH,
            custom_objects=custom_objects,
            compile=False
        )
        template_db_loaded = np.load(TEMPLATE_DB_PATH, allow_pickle=True).item()
        
        verify_and_identify(template_db_loaded, embedding_model, TEST_IMAGE_PATH, DISTANCE_THRESHOLD)
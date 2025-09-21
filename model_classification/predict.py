# File: predict.py
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

# --- KONFIGURASI PREDIKSI ---
MODEL_PATH = 'palm_print_cnn_prototipe.h5' # Nama file model yang tersimpan
TEST_IMAGE_PATH = 'test/test1.jpg'       # GANTI dengan jalur gambar baru yang ingin diuji
IMG_SIZE = 128
CHANNELS = 3 # Harus sama dengan CHANNELS yang digunakan saat pelatihan
# ----------------------------

def load_and_preprocess_image(img_path, size, channels):
    """Memuat dan pra-proses satu gambar."""
    img = image.load_img(
        img_path, 
        target_size=(size, size), 
        color_mode='rgb' if channels == 3 else 'grayscale'
    )
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) # Tambah dimensi batch
    img_array /= 255.0 # Normalisasi (harus sama seperti saat training)
    return img_array

def get_class_labels(data_dir):
    """Mengambil daftar label kelas (ID folder) dari folder training."""
    training_dir = os.path.join(data_dir, 'training')
    if not os.path.exists(training_dir):
        return None
    # Daftar folder ID (001, 002, dst.)
    labels = sorted([d for d in os.listdir(training_dir) if os.path.isdir(os.path.join(training_dir, d))])
    return labels

def predict_new_image():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: File model '{MODEL_PATH}' tidak ditemukan. Jalankan train_model.py terlebih dahulu.")
        return
    
    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"❌ Error: Gambar uji '{TEST_IMAGE_PATH}' tidak ditemukan. Letakkan gambar yang ingin diuji di sini.")
        return

    # Muat Model
    model = tf.keras.models.load_model(MODEL_PATH)

    # Muat dan Pra-proses Gambar
    processed_image = load_and_preprocess_image(TEST_IMAGE_PATH, IMG_SIZE, CHANNELS)
    
    # Ambil Label Kelas
    labels = get_class_labels('palm_data_split')
    if not labels:
        print("❌ Gagal mendapatkan label kelas. Pastikan palm_data_split ada.")
        return

    # Lakukan Prediksi
    predictions = model.predict(processed_image)[0]
    predicted_class_index = np.argmax(predictions)
    confidence = predictions[predicted_class_index]
    
    predicted_id = labels[predicted_class_index]

    print("\n=== Hasil Prediksi ===")
    print(f"Gambar yang diuji: {TEST_IMAGE_PATH}")
    print(f"Kelas yang diprediksi: ID {predicted_id}")
    print(f"Tingkat Keyakinan (Confidence): {confidence*100:.2f}%")
    print("\nTop 3 Prediksi:")
    
    # Menampilkan 3 prediksi teratas
    top_3_indices = np.argsort(predictions)[::-1][:3]
    for i in top_3_indices:
        print(f"- ID {labels[i]}: {predictions[i]*100:.2f}%")

if __name__ == "__main__":
    predict_new_image()
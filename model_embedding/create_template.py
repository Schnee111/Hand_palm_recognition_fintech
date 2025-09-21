# File: create_templates.py
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# --- KONFIGURASI ---
EMBEDDING_MODEL_PATH = 'palm_embedding_model_50_epochs.h5'
DATA_DIR = 'palm_data_split' 
TEMPLATE_DB_PATH = 'palm_templates.npy' # Nama file untuk menyimpan template
IMG_SIZE = 128
CHANNELS = 3
# --------------------

def create_and_save_template_database(model_path, data_dir, template_save_path):
    if not os.path.exists(model_path):
        print(f"❌ Error: Model Embedding '{model_path}' tidak ditemukan.")
        return

    embedding_model = tf.keras.models.load_model(model_path)
    
    # Siapkan Data Generator
    template_datagen = ImageDataGenerator(rescale=1./255)
    template_generator = template_datagen.flow_from_directory(
        os.path.join(data_dir, 'training'),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=32, 
        class_mode='categorical',
        shuffle=False, 
        color_mode='rgb' if CHANNELS == 3 else 'grayscale'
    )
    
    class_indices = template_generator.class_indices
    idx_to_class = {v: k for k, v in class_indices.items()}

    print("\nMembuat Template Database (Fitur Rata-rata)...")
    
    # 1. Ekstrak semua embeddings dari data training
    # Memprediksi semua gambar training sekaligus (lebih cepat daripada iterasi satu per satu)
    all_embeddings = embedding_model.predict(template_generator, steps=len(template_generator), verbose=1)
    
    # 2. Hitung embedding rata-rata (template) untuk setiap ID
    template_db = {}
    all_labels = template_generator.classes # Label diurutkan sesuai urutan generator

    for idx, class_name in idx_to_class.items():
        # Pilih semua embedding yang memiliki label yang sama
        class_embeddings = all_embeddings[all_labels == idx]
        
        # Hitung rata-rata
        template_db[class_name] = np.mean(class_embeddings, axis=0)
        print(f"Template {class_name} dibuat dari {len(class_embeddings)} sampel.")
    
    # 3. Simpan database template menggunakan NumPy
    np.save(template_save_path, template_db)
    print(f"\n✅ Database Template berhasil disimpan di: {template_save_path}")

if __name__ == "__main__":
    create_and_save_template_database(EMBEDDING_MODEL_PATH, DATA_DIR, TEMPLATE_DB_PATH)
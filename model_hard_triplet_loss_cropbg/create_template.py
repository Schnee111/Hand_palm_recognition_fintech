# File: create_templates.py (FINAL: MENGGUNAKAN CUSTOM LAYER)
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Layer
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
EMBEDDING_MODEL_PATH = 'triplet_encoder_best_mobilenetv2.h5'
TEMPLATE_DB_PATH = 'palm_templates_triplet_mobilenetv2.npy'
DATA_DIR = '../palm_data_split_full_hand_segmented'
IMG_SIZE = 160
CHANNELS = 3
# --------------------

def create_and_save_template_database(model_path, data_dir, template_save_path):
    if not os.path.exists(model_path):
        print(f"❌ Error: Model Embedding '{model_path}' tidak ditemukan.")
        return

    # Memberitahu load_model tentang Custom Layer kita
    custom_objects = {'L2NormalizeLayer': L2NormalizeLayer}
    embedding_model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
    
    template_datagen = ImageDataGenerator(rescale=1./255)
    template_generator = template_datagen.flow_from_directory(os.path.join(data_dir, 'training'), target_size=(IMG_SIZE, IMG_SIZE), batch_size=32, class_mode='categorical', shuffle=False, color_mode='rgb')
    
    class_indices = template_generator.class_indices
    idx_to_class = {v: k for k, v in class_indices.items()}
    
    print("\nMembuat Template Database (Fitur Rata-rata)...")
    all_embeddings = embedding_model.predict(template_generator, steps=len(template_generator), verbose=1)
    all_labels = template_generator.classes
    
    template_db = {}
    for idx, class_name in idx_to_class.items():
        class_embeddings = all_embeddings[all_labels == idx]
        template_db[class_name] = np.mean(class_embeddings, axis=0)
        print(f"Template {class_name} dibuat dari {len(class_embeddings)} sampel.")
        
    np.save(template_save_path, template_db)
    print(f"\n✅ Database Template berhasil disimpan di: {template_save_path}")

if __name__ == "__main__":
    create_and_save_template_database(EMBEDDING_MODEL_PATH, DATA_DIR, TEMPLATE_DB_PATH)
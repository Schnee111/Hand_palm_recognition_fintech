# File: evaluate_models.py (Versi Final dengan Logika Evaluasi Terpadu)
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from scipy.spatial.distance import euclidean
from itertools import combinations
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
from tqdm import tqdm

# --- KONFIGURASI ---
DATA_DIR = 'palm_data_split_full_hand_segmented'
TEST_DIR = os.path.join(DATA_DIR, 'validation')

# Model 1: Klasifikasi
MODEL_CLASSIFIER_PATH = 'palm_print_cnn_prototipem.h5' 

# Model 2: Verifikasi dari Classifier
MODEL_EMBEDDING_PATH = 'palm_embedding_modelm.h5'

# Model 3: Verifikasi dengan Triplet Loss
MODEL_TRIPLET_PATH = 'model_hard_triplet_loss_cropbg/triplet_encoder_best_mobilenetv2.h5'

# --- DEFINISI CUSTOM LAYER (untuk Model 3) ---
class L2NormalizeLayer(Layer):
    def __init__(self, **kwargs):
        super(L2NormalizeLayer, self).__init__(**kwargs)
    def call(self, inputs):
        return K.l2_normalize(inputs, axis=1)
    def get_config(self):
        config = super(L2NormalizeLayer, self).get_config()
        return config

# --- FUNGSI EVALUASI MODEL 1: KLASIFIKASI ---
# Di dalam file evaluate_models.py, ganti fungsi ini:

def evaluate_classifier(model_path, test_dir):
    print("\n--- Mengevaluasi Model 1: Klasifikasi Standar (L/R Digabung) ---")
    if not os.path.exists(model_path):
        print(f"Model {model_path} tidak ditemukan. Melewati evaluasi.")
        return

    model = tf.keras.models.load_model(model_path)
    img_size = model.input_shape[1]
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_size, img_size),
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )

    # Prediksi seperti biasa
    y_pred_proba = model.predict(test_generator)
    y_pred_int = np.argmax(y_pred_proba, axis=1) # Prediksi dalam bentuk integer
    y_true_int = test_generator.classes          # Label asli dalam bentuk integer

    # --- PERUBAHAN LOGIKA DIMULAI DI SINI ---
    
    # 1. Buat pemetaan dari integer ke nama kelas (misal: 0 -> '001_L')
    idx_to_class = {v: k for k, v in test_generator.class_indices.items()}
    
    # 2. Konversi label integer ke ID utama (string, misal: '001')
    y_pred_main_id = [idx_to_class[i].split('_')[0] for i in y_pred_int]
    y_true_main_id = [idx_to_class[i].split('_')[0] for i in y_true_int]
    
    # 3. Dapatkan daftar label ID utama yang unik untuk plot
    main_id_labels = sorted(list(set(y_true_main_id)))

    # Kalkulasi Akurasi BARU berdasarkan ID utama
    acc = accuracy_score(y_true_main_id, y_pred_main_id)
    print(f"✅ Akurasi Model Klasifikasi (ID Orang): {acc*100:.2f}%")

    # Plot Confusion Matrix BARU yang lebih sederhana
    cm = confusion_matrix(y_true_main_id, y_pred_main_id, labels=main_id_labels)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=main_id_labels, yticklabels=main_id_labels, cmap='Blues')
    plt.title('Confusion Matrix - Model Klasifikasi (ID Orang)')
    plt.ylabel('Kelas Asli (True)')
    plt.xlabel('Kelas Prediksi')
    plt.tight_layout()
    plt.savefig('eval_plot_confusion_matrix_simplified.png')
    print("📈 Plot Confusion Matrix yang disederhanakan disimpan sebagai 'eval_plot_confusion_matrix_simplified.png'")
    plt.close()

# --- FUNGSI EVALUASI MODEL 2 & 3: VERIFIKASI (LOGIKA DIPERBARUI) ---
def evaluate_verification_model(model_name, model_path, test_dir, custom_objects=None):
    print(f"\n--- Mengevaluasi Model Verifikasi: {model_name} (Logika Terpadu L/R) ---")
    if not os.path.exists(model_path):
        print(f"Model {model_path} tidak ditemukan. Melewati evaluasi.")
        return

    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
    img_size = model.input_shape[1]

    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        test_dir, target_size=(img_size, img_size), batch_size=32,
        class_mode='categorical', shuffle=False
    )
    
    embeddings = model.predict(test_generator)
    labels = test_generator.classes
    idx_to_class = {v: k for k, v in test_generator.class_indices.items()}
    
    # --- PERUBAHAN LOGIKA: Gunakan ID utama (misal: '001') bukan '001_L' ---
    main_ids = np.array([idx_to_class[l].split('_')[0] for l in labels])
    
    genuine_distances = []
    imposter_distances = []

    # Kelompokkan indeks gambar berdasarkan ID utama
    unique_main_ids = np.unique(main_ids)
    id_indices = {uid: np.where(main_ids == uid)[0] for uid in unique_main_ids}

    print("Membuat pasangan genuine dan imposter...")
    for uid in tqdm(unique_main_ids, desc="Processing IDs"):
        indices = id_indices[uid]
        # Buat pasangan genuine (semua kombinasi gambar dari orang yang sama, termasuk L vs R)
        if len(indices) > 1:
            for i, j in combinations(indices, 2):
                dist = euclidean(embeddings[i], embeddings[j])
                genuine_distances.append(dist)
        
        # Buat pasangan imposter (dari ID ini vs ID lain)
        other_indices = np.where(main_ids != uid)[0]
        # Ambil sampel acak agar jumlahnya seimbang dan proses cepat
        imposter_sample_size = min(len(other_indices), len(indices) * 2) # Batasi jumlah pasangan imposter
        if imposter_sample_size > 0:
            imposter_sample_indices = np.random.choice(other_indices, size=imposter_sample_size, replace=False)
            for i in indices:
                for j in imposter_sample_indices:
                    dist = euclidean(embeddings[i], embeddings[j])
                    imposter_distances.append(dist)

    y_true = np.concatenate([np.ones(len(genuine_distances)), np.zeros(len(imposter_distances))])
    y_scores = np.concatenate([genuine_distances, imposter_distances])
    
    fpr, tpr, thresholds = roc_curve(y_true, -y_scores)
    roc_auc = auc(fpr, tpr)
    
    print(f"✅ Area Under Curve (AUC) Model {model_name}: {roc_auc:.4f}")

    # Plot ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic (ROC) - {model_name}')
    plt.legend(loc="lower right"); plt.grid(True)
    plt.savefig(f'eval_plot_roc_{model_name.replace(" ", "_")}.png')
    print(f"📈 Plot ROC Curve disimpan sebagai 'eval_plot_roc_{model_name.replace(' ', '_')}.png'")
    plt.close()

if __name__ == '__main__':
    evaluate_classifier(MODEL_CLASSIFIER_PATH, TEST_DIR)
    
    evaluate_verification_model("Verifikasi Jarak", MODEL_EMBEDDING_PATH, TEST_DIR)

    custom_objects = {'L2NormalizeLayer': L2NormalizeLayer}
    evaluate_verification_model("Triplet Loss full-hand segmented", MODEL_TRIPLET_PATH, TEST_DIR, custom_objects=custom_objects)
    
    print("\n✅ Semua evaluasi selesai. Periksa file .png yang dihasilkan.")

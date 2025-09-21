# File: train_embedding_model.py (FINAL: METODE TRANSFER BOBOT)
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
import os
import numpy as np

# --- KONFIGURASI MODEL ---
DATA_DIR = 'palm_data_split' 
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 50 
CHANNELS = 3 
EMBEDDING_SIZE = 128
CLASSIFIER_MODEL_NAME = 'palm_classifier_50_epochs.h5' # Model klasifikasi penuh
EMBEDDING_MODEL_NAME = 'palm_embedding_model_50_epochs.h5' # Model fitur untuk verifikasi
# --------------------------

def build_classifier_model(input_shape, num_classes):
    """Membangun arsitektur CNN dasar (menggunakan Sequential)."""
    # Beri nama lapisan agar bobotnya mudah diakses
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, name='conv1'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu', name='conv2'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu', name='conv3'),
        MaxPooling2D((2, 2)),
        
        Flatten(),
        Dropout(0.5), 
        
        # Lapisan Embedding / Fitur
        Dense(EMBEDDING_SIZE, name='embedding_layer', 
              kernel_regularizer=l2(0.001), activation='relu'),
        
        # Lapisan Softmax Akhir 
        Dense(num_classes, activation='softmax', name='output_softmax') 
    ])
    return model

def build_embedding_architecture(input_shape):
    """Membangun arsitektur model embedding (hanya kerangka, tanpa bobot)."""
    # Lapisan harus sesuai dengan build_classifier_model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, name='conv1'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu', name='conv2'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu', name='conv3'),
        MaxPooling2D((2, 2)),
        
        Flatten(),
        Dropout(0.5), 
        
        # Lapisan Embedding / Fitur (TANPA REGULARIZER DAN SOFTMAX)
        Dense(EMBEDDING_SIZE, name='embedding_layer', activation='relu') 
    ])
    return model

def train_and_save_model():
    print("Memuat Data...")
    
    # 1. Pemuatan dan Pra-pemrosesan Data
    train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, horizontal_flip=True)
    val_datagen = ImageDataGenerator(rescale=1./255)

    try:
        train_generator = train_datagen.flow_from_directory(
            os.path.join(DATA_DIR, 'training'), target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE, class_mode='categorical', color_mode='rgb'
        )
        validation_generator = val_datagen.flow_from_directory(
            os.path.join(DATA_DIR, 'validation'), target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE, class_mode='categorical', color_mode='rgb'
        )
    except Exception as e:
        print(f"❌ Gagal memuat data. Error: {e}")
        return

    NUM_CLASSES = train_generator.num_classes
    print(f"Jumlah ID (kelas) terdeteksi: {NUM_CLASSES}")
    
    # 2. Inisialisasi dan Kompilasi Model Klasifikasi
    INPUT_SHAPE = (IMG_SIZE, IMG_SIZE, CHANNELS)
    classifier_model = build_classifier_model(INPUT_SHAPE, NUM_CLASSES)
    
    classifier_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    print("\n=== Ringkasan Model Klasifikasi ===")
    classifier_model.summary()

    # 3. Pelatihan Model
    print(f"\n=== Memulai Pelatihan ({EPOCHS} Epochs) ===")
    classifier_model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator
    )

    # 4. Simpan Model Klasifikasi Dasar
    classifier_model.save(CLASSIFIER_MODEL_NAME)
    
    # 5. EKSTRAKSI MODEL EMBEDDING (TANPA MENGAKSES .INPUT)
    
    print("\n[Mengekstrak Model Embedding via Transfer Bobot]")
    
    # 5a. Bangun Arsitektur Model Embedding BARU
    embedding_architecture = build_embedding_architecture(INPUT_SHAPE)
    
    # Panggil arsitektur baru sekali untuk mendefinisikan bobotnya
    dummy_input = np.zeros((1, IMG_SIZE, IMG_SIZE, CHANNELS))
    _ = embedding_architecture.predict(dummy_input, verbose=0) 
    
    # 5b. Transfer bobot dari model yang sudah dilatih (classifier_model)
    
    # Dapatkan bobot dari setiap lapisan di classifier_model dan salin
    for layer in embedding_architecture.layers:
        try:
            # Ambil lapisan yang sesuai dari model klasifikasi
            source_layer = classifier_model.get_layer(layer.name)
            
            # Pindahkan bobot
            layer.set_weights(source_layer.get_weights())
        except Exception:
             # Lewati lapisan yang tidak cocok (misalnya Dropout atau Pooling)
            pass

    # 5c. Simpan Model Embedding yang sudah memiliki bobot teruji
    embedding_architecture.save(EMBEDDING_MODEL_NAME)
    
    print(f"\n✅ Model Embedding (Fitur) berhasil disimpan sebagai {EMBEDDING_MODEL_NAME}")
    print("Lanjutkan ke skrip verifikasi untuk menguji ambang batas jarak.")

if __name__ == "__main__":
    train_and_save_model()
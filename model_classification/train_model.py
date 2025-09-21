# File: train_model.py
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# --- KONFIGURASI MODEL ---
DATA_DIR = 'palm_data_split' 
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 20 # Anda bisa menambah atau mengurangi jumlah epoch
CHANNELS = 3 # 3 untuk RGB (warna), ganti 1 jika semua data Anda grayscale
MODEL_NAME = 'palm_print_cnn_prototipe.h5'
# --------------------------

def build_model(input_shape, num_classes):
    """Membangun arsitektur model CNN sederhana."""
    model = Sequential([
        # Input Layer dan Konvolusi
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        
        # Konvolusi Kedua
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        # Konvolusi Ketiga
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        Flatten(),
        
        # Dropout untuk mencegah Overfitting
        Dropout(0.5), 
        
        # Layer Fully Connected
        Dense(512, activation='relu'),
        
        # Layer Output
        Dense(num_classes, activation='softmax')
    ])
    return model

def train_and_save_model():
    print("Memuat Data...")
    
    # 1. Pemuatan dan Pra-pemrosesan Data
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        horizontal_flip=True
    )
    val_datagen = ImageDataGenerator(rescale=1./255)

    try:
        train_generator = train_datagen.flow_from_directory(
            os.path.join(DATA_DIR, 'training'),
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            color_mode='rgb' if CHANNELS == 3 else 'grayscale'
        )
        validation_generator = val_datagen.flow_from_directory(
            os.path.join(DATA_DIR, 'validation'),
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            color_mode='rgb' if CHANNELS == 3 else 'grayscale'
        )
    except Exception as e:
        print(f"❌ Gagal memuat data. Pastikan folder '{DATA_DIR}' sudah ada dan terstruktur dengan benar.")
        print(f"Error: {e}")
        return

    NUM_CLASSES = train_generator.num_classes
    print(f"Jumlah ID (kelas) terdeteksi: {NUM_CLASSES}")
    
    # 2. Inisialisasi dan Kompilasi Model
    INPUT_SHAPE = (IMG_SIZE, IMG_SIZE, CHANNELS)
    model = build_model(INPUT_SHAPE, NUM_CLASSES)
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\n=== Ringkasan Model ===")
    model.summary()

    # 3. Pelatihan Model
    print(f"\n=== Memulai Pelatihan ({EPOCHS} Epochs) ===")
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator
    )

    # 4. Evaluasi Akhir dan Simpan
    print("\n=== Evaluasi Akhir ===")
    loss, accuracy = model.evaluate(validation_generator)
    print(f"Loss Validasi Akhir: {loss:.4f}")
    print(f"Akurasi Validasi Akhir: {accuracy*100:.2f}%")

    model.save(MODEL_NAME)
    print(f"\n Model berhasil disimpan sebagai {MODEL_NAME}")

if __name__ == "__main__":
    train_and_save_model()
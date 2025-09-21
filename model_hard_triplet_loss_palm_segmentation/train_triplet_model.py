# File: train_triplet_model.py (FINAL: MENGGUNAKAN CUSTOM LAYER)
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout, Layer
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import AdamW
import os
import numpy as np
import tensorflow.keras.backend as K

# --- KONFIGURASI MODEL ---
DATA_DIR = '../palm_data_split_segmented'
IMG_SIZE = 160
BATCH_SIZE = 32
EPOCHS = 50
CHANNELS = 3
EMBEDDING_SIZE = 128
MODEL_NAME = 'palm_triplet_encoder_mobilenetv2.h5'
ALPHA = 0.6
# --------------------------

# --- DEFINISI CUSTOM LAYER (SOLUSI ANTI GAGAL) ---
class L2NormalizeLayer(Layer):
    def __init__(self, **kwargs):
        super(L2NormalizeLayer, self).__init__(**kwargs)

    def call(self, inputs):
        return K.l2_normalize(inputs, axis=1)
    
    def get_config(self):
        config = super(L2NormalizeLayer, self).get_config()
        return config

# --- CUSTOM TRIPLET LOSS FUNCTION ---
def batch_hard_triplet_loss(y_true, y_pred, margin=ALPHA):
    labels = y_true
    dot_product = tf.matmul(y_pred, tf.transpose(y_pred))
    square_norm = tf.reduce_sum(tf.square(y_pred), axis=1)
    distances = tf.expand_dims(square_norm, 1) - 2.0 * dot_product + tf.expand_dims(square_norm, 0)
    distances = tf.maximum(distances, 0.0)
    labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
    mask_positive = tf.cast(labels_equal, tf.float32)
    mask_negative = tf.cast(tf.logical_not(labels_equal), tf.float32)
    diag_mask = tf.eye(tf.shape(distances)[0], dtype=tf.float32)
    mask_positive = mask_positive - diag_mask
    max_positive_dist = tf.reduce_max(tf.multiply(distances, mask_positive), axis=1, keepdims=True)
    large_val = 1e9
    neg_distances = tf.where(tf.equal(mask_negative, 1.0), distances, large_val * tf.ones_like(distances))
    min_negative_dist = tf.reduce_min(neg_distances, axis=1, keepdims=True)
    triplet_loss = tf.maximum(max_positive_dist - min_negative_dist + margin, 0.0)
    num_positive_triplets = tf.reduce_sum(tf.cast(tf.greater(triplet_loss, 1e-16), tf.float32))
    return K.sum(triplet_loss) / (num_positive_triplets + 1e-16)

# --- ARSITEKTUR ENCODER ---
def build_embedding_encoder(input_shape):
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape, pooling=None)
    for layer in base_model.layers[:-40]:
        layer.trainable = False

    inputs = base_model.input
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    embedding = Dense(EMBEDDING_SIZE, name='embedding_layer', kernel_regularizer=l2(0.001), activation=None)(x)
    
    # Mengganti Lambda dengan Custom Layer yang solid
    embedding = L2NormalizeLayer(name='l2_normalize')(embedding)
    
    return Model(inputs, embedding, name='Embedding_Encoder')

# --- FUNGSI PELATIHAN ---
def train_and_save_model():
    print("Memuat Data...")
    train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, horizontal_flip=False, width_shift_range=0.1, height_shift_range=0.1, brightness_range=[0.8, 1.2], zoom_range=0.1)
    val_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory(os.path.join(DATA_DIR, 'training'), target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, class_mode='sparse', color_mode='rgb')
    validation_generator = val_datagen.flow_from_directory(os.path.join(DATA_DIR, 'validation'), target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, class_mode='sparse', color_mode='rgb')
    
    INPUT_SHAPE = (IMG_SIZE, IMG_SIZE, CHANNELS)
    encoder_model = build_embedding_encoder(INPUT_SHAPE)
    encoder_model.compile(optimizer=AdamW(learning_rate=0.0001), loss=batch_hard_triplet_loss)
    encoder_model.summary()
    
    print(f"\n=== Memulai Pelatihan ({EPOCHS} Epochs) ===")
    checkpoint_cb = ModelCheckpoint(filepath='triplet_encoder_best_mobilenetv2.h5', monitor='val_loss', save_best_only=True, mode='min', verbose=1)
    lr_scheduler_cb = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1, mode='min', min_lr=1e-7)
    encoder_model.fit(train_generator, epochs=EPOCHS, steps_per_epoch=len(train_generator), validation_data=validation_generator, validation_steps=len(validation_generator), callbacks=[checkpoint_cb, lr_scheduler_cb])
    
    encoder_model.save(MODEL_NAME)
    print(f"\n✅ Model Embedding (Encoder) berhasil disimpan sebagai {MODEL_NAME}")

if __name__ == "__main__":
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    train_and_save_model()
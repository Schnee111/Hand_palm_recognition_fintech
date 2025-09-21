# File: split_data.py
import os
import shutil
import random

# --- KONFIGURASI JALUR ---
SOURCE_ROOT_DIR = 'data_separated'        # Folder yang berisi 001, 002, 003, dst.
TARGET_ROOT_DIR = 'palm_data_split' # Folder output yang akan dibuat
TRAIN_RATIO = 0.8               # Rasio data Training
# --------------------

def split_data(source_dir, target_dir, ratio):
    # Hapus folder output lama dan buat yang baru
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir, exist_ok=True)

    TRAIN_DIR = os.path.join(target_dir, 'training')
    VAL_DIR = os.path.join(target_dir, 'validation')
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(VAL_DIR, exist_ok=True)

    total_files_moved = 0
    total_classes = 0

    for class_id in os.listdir(source_dir):
        class_path = os.path.join(source_dir, class_id)
        if not os.path.isdir(class_path) or class_id.startswith('.'): # Lewati file non-folder
            continue
        
        total_classes += 1

        all_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        if not all_files:
            continue
            
        random.shuffle(all_files) # Acak urutan file

        # Hitung titik split
        split_point = int(len(all_files) * ratio)
        train_files = all_files[:split_point]
        val_files = all_files[split_point:]

        # Buat folder ID di dalam training dan validation
        os.makedirs(os.path.join(TRAIN_DIR, class_id), exist_ok=True)
        os.makedirs(os.path.join(VAL_DIR, class_id), exist_ok=True)

        # Salin file ke folder yang dituju
        for filename in train_files:
            shutil.copyfile(os.path.join(class_path, filename), os.path.join(TRAIN_DIR, class_id, filename))
        for filename in val_files:
            shutil.copyfile(os.path.join(class_path, filename), os.path.join(VAL_DIR, class_id, filename))
        
        total_files_moved += len(train_files) + len(val_files)
        
    print(f"Selesai membagi data. Jumlah Kelas/ID: {total_classes}. Total file: {total_files_moved}")

if __name__ == "__main__":
    split_data(SOURCE_ROOT_DIR, TARGET_ROOT_DIR, TRAIN_RATIO)
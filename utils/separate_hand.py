# File: separate_hands.py
import os
import shutil
import re

# --- KONFIGURASI ---
SOURCE_ROOT_DIR = 'data'      # Folder yang berisi 001, 002, 042, dst.
TARGET_ROOT_DIR = 'data_separated' # Folder output baru untuk data mentah yang sudah dipisah
# --------------------

def separate_hands(source_dir, target_dir):
    """Memindai folder data mentah dan memisahkan file L dan R ke folder baru."""
    print(f"Memulai proses pemisahan tangan dari {source_dir}...")
    
    # 1. Hapus folder output lama jika ada dan buat yang baru
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir, exist_ok=True)
    
    total_files_moved = 0
    total_classes_processed = 0

    # 2. Iterasi melalui setiap folder ID di data mentah (misalnya, 042)
    for original_class_id in os.listdir(source_dir):
        original_path = os.path.join(source_dir, original_class_id)
        
        if not os.path.isdir(original_path) or original_class_id.startswith('.'):
            continue
        
        total_classes_processed += 1
        
        # 3. Proses file di dalam folder ID
        for filename in os.listdir(original_path):
            file_path = os.path.join(original_path, filename)
            
            if not os.path.isfile(file_path) or not filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                continue

            # 4. Deteksi Sisi Tangan (L atau R) dari nama file
            
            # Cari pola _L_ atau _R_ di nama file
            match = re.search(r'_[LR]_', filename.upper())
            
            if match:
                hand_side = match.group(0).strip('_') # Ambil 'L' atau 'R'
                
                # Buat nama folder ID baru: 042_L atau 042_R
                new_class_id = f"{original_class_id}_{hand_side}"
                new_class_path = os.path.join(target_dir, new_class_id)
                
                # Pastikan folder baru ada
                os.makedirs(new_class_path, exist_ok=True)
                
                # Jalur file tujuan
                dest_path = os.path.join(new_class_path, filename)
                
                # Pindahkan/Salin file
                shutil.copyfile(file_path, dest_path)
                total_files_moved += 1
            
            else:
                print(f"Peringatan: File {filename} di {original_class_id} tidak memiliki label L/R yang jelas. Dilewati.")

    print(f"\n✅ Pemisahan selesai. Total Kelas Mentah Diproses: {total_classes_processed}")
    print(f"Total File Dipindahkan/Disalin: {total_files_moved}")
    print(f"Struktur data baru dibuat di: {TARGET_ROOT_DIR}")

if __name__ == "__main__":
    separate_hands(SOURCE_ROOT_DIR, TARGET_ROOT_DIR)
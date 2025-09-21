from PIL import Image
import os

# --- KONFIGURASI ---
TARGET_DIMENSION = (2448, 3264) # Sesuaikan target dimensi data lain
FOLDER_ID = '042' # Folder ID Anda

# --- JALANKAN RESIZE ---
print(f"Memulai resize untuk folder {FOLDER_ID}...")
target_path = os.path.join('data', FOLDER_ID)

if not os.path.exists(target_path):
    print(f"Error: Folder {target_path} tidak ditemukan.")
else:
    for filename in os.listdir(target_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            file_path = os.path.join(target_path, filename)
            try:
                img = Image.open(file_path)
                
                # Ubah ukuran ke dimensi target
                img = img.resize(TARGET_DIMENSION, Image.LANCZOS)
                
                # Simpan kembali, menimpa file asli
                img.save(file_path) 
                # print(f"Resize berhasil: {filename}")
            except Exception as e:
                print(f"Gagal memproses {filename}: {e}")

    print("✅ Semua gambar telah di-resize ke 2400x3000. Anda bisa melanjutkan ke split_data.py.")
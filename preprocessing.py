# ... (kode impor dan fungsi preprocess_data tetap sama) ...
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os # Tambahkan impor ini

def preprocess_data(df):
    # ... (isi fungsi Anda tidak berubah) ...
    df_processed = df.copy()
    # ... (semua langkah preprocessing) ...
    print("Preprocessing data selesai.")
    return df_processed

if __name__ == '__main__':
    # Tentukan path input dan output
    input_path = '../namadataset_raw/train.csv'
    output_folder = '../namadataset_preprocessing'
    output_path = os.path.join(output_folder, 'train_clean.csv')
    
    # Buat folder output jika belum ada
    os.makedirs(output_folder, exist_ok=True)

    try:
        raw_data = pd.read_csv(input_path)
        clean_data = preprocess_data(raw_data)
        
        # --- TAMBAHAN PENTING ---
        # Simpan dataframe yang sudah bersih ke file CSV
        clean_data.to_csv(output_path, index=False)
        print(f"Data bersih berhasil disimpan di: {output_path}")
        # -------------------------

    except FileNotFoundError:
        print(f"Error: File tidak ditemukan di {input_path}")
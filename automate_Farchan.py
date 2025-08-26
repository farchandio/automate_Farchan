import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

def preprocess_heart_disease_data(csv_path, output_dir=None):
    try:
        df = pd.read_csv(csv_path)
        
        # Cek nama kolom
        print(">>> Nama kolom di file CSV Anda adalah:", df.columns.tolist())
        
        # Tentukan kolom target
        target_col = None
        if 'output' in df.columns:
            target_col = 'output'
        else:
            # Cari kandidat target umum
            candidates = [col for col in df.columns if col.lower() in 
                          ['target', 'class', 'label', 'diagnosis', 'disease']]
            
            if len(candidates) == 1:
                target_col = candidates[0]
                print(f"Kolom 'output' tidak ditemukan, menggunakan kandidat: {target_col}")
            elif len(candidates) > 1:
                print("Beberapa kandidat kolom target ditemukan:")
                for i, col in enumerate(candidates, 1):
                    print(f"{i}. {col}")
                choice = input("Pilih nomor kolom target (atau tekan Enter untuk pilih terakhir): ")
                if choice.isdigit() and 1 <= int(choice) <= len(candidates):
                    target_col = candidates[int(choice)-1]
                else:
                    target_col = df.columns[-1]
                    print(f"Tidak ada pilihan valid. Menggunakan kolom terakhir: {target_col}")
            else:
                target_col = df.columns[-1]
                print(f"Kolom target tidak jelas. Menggunakan kolom terakhir: {target_col}")
        
    except FileNotFoundError:
        print(f"Error: File tidak ditemukan di {csv_path}")
        return
    
    # Pisahkan X dan y
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # --- Deteksi & encoding kolom kategorikal ---
    categorical_cols = X.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        print("Kolom kategorikal terdeteksi:", categorical_cols.tolist())
        X = pd.get_dummies(X, drop_first=True)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Standardisasi fitur numerik
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    
    print("Preprocessing data Heart Disease selesai.")
    
    # Simpan hasil
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        X_train.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
        X_test.to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
        y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
        y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)
        print(f"Data yang telah diproses disimpan di folder: {output_dir}")

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    input_csv_path = 'namadataset_raw/heart_disease_uci.csv'  # ganti sesuai dataset
    output_directory = 'namadataset_preprocessing'
    
    preprocess_heart_disease_data(input_csv_path, output_directory)

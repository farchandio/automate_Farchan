# Import library yang diperlukan
import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):

    # Membuat salinan dataframe untuk menghindari perubahan pada data asli
    df_processed = df.copy()

    # --- 1. Menangani Data Kosong (Missing Values) ---
    # Mengisi 'Age' dengan median
    age_median = df_processed['Age'].median()
    df_processed['Age'].fillna(age_median, inplace=True)

    # Mengisi 'Embarked' dengan modus
    embarked_mode = df_processed['Embarked'].mode()[0]
    df_processed['Embarked'].fillna(embarked_mode, inplace=True)

    # Menghapus kolom 'Cabin' karena terlalu banyak nilai kosong (>70%)
    df_processed.drop('Cabin', axis=1, inplace=True)

    # --- 2. Menghapus Data Duplikat ---
    df_processed.drop_duplicates(inplace=True)

    # --- 3. Encoding Data Kategorikal ---
    # Mengubah 'Sex' menjadi numerik (0 untuk male, 1 untuk female)
    df_processed['Sex'] = df_processed['Sex'].map({'male': 0, 'female': 1})

    # Menggunakan One-Hot Encoding untuk 'Embarked' karena tidak ada urutan tingkatan
    df_processed = pd.get_dummies(df_processed, columns=['Embarked'], prefix='Embarked')

    # --- 4. Menghapus Kolom yang Tidak Relevan ---
    # PassengerId, Name, dan Ticket tidak digunakan untuk pemodelan dasar
    df_processed.drop(['Name', 'Ticket', 'PassengerId'], axis=1, inplace=True)

    # --- 5. Standarisasi Fitur Numerik ---
    # Menyamakan skala untuk 'Age' dan 'Fare'
    scaler = StandardScaler()
    numerical_features = ['Age', 'Fare']
    df_processed[numerical_features] = scaler.fit_transform(df_processed[numerical_features])

    print("Preprocessing data selesai.")
    return df_processed

# Blok ini hanya akan berjalan jika file ini dieksekusi secara langsung
# Berguna untuk melakukan tes cepat pada fungsi di atas
if __name__ == '__main__':
    # Muat data mentah (sesuaikan path jika perlu)
    try:
        raw_data = pd.read_csv('/content/dataset_ekstrak/train.csv') # Corrected path

        # Panggil fungsi preprocessing
        clean_data = preprocess_data(raw_data)

        # Tampilkan hasil untuk verifikasi
        print("\nData setelah preprocessing:")
        print(clean_data.head())

        print("\nInformasi Dataframe setelah preprocessing:")
        clean_data.info()

    except FileNotFoundError:
        print("Error: File 'train.csv' tidak ditemukan. Pastikan path file sudah benar.") # Corrected print statement
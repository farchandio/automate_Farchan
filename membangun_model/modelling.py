import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

# 1. Muat Dataset yang Sudah Dibersihkan
# Pastikan path ini benar sesuai struktur folder Anda
try:
    df = pd.read_csv('titanic_preprocessing/train.csv')
except FileNotFoundError:
    print("Error: Pastikan file 'train_clean.csv' ada di folder 'namadataset_preprocessing/'")
    exit()

# 2. Pisahkan Fitur (X) dan Target (y)
X = df.drop('Survived', axis=1)
y = df['Survived']

# 3. Bagi Data menjadi Data Latih dan Data Uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Pengaturan Eksperimen MLflow
# Menyimpan hasil tracking di database SQLite lokal agar lebih rapi
mlflow.set_tracking_uri("sqlite:///mlflow.db")
# Atur nama eksperimen Anda
experiment_name = "Titanic Survival Classification - Basic"
mlflow.set_experiment(experiment_name)

# 5. Latih Model di dalam MLflow Run
with mlflow.start_run():
    # Mengaktifkan autologging untuk Scikit-learn
    # MLflow akan secara otomatis mencatat parameter, metrik, dan model
    mlflow.sklearn.autolog()
    
    print("Melatih model Logistic Regression...")
    
    # Inisialisasi dan latih model
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    
    # Lakukan prediksi pada data uji
    y_pred = model.predict(X_test)
    
    # Hitung akurasi (meskipun autolog sudah mencatat, ini untuk verifikasi)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Model training selesai.")
    print(f"Akurasi Model: {accuracy:.4f}")
    
    run_id = mlflow.active_run().info.run_id
    print(f"MLflow Run ID: {run_id}")

print("\nProses selesai. Untuk melihat hasil, jalankan 'mlflow ui --backend-store-uri sqlite:///mlflow.db' di terminal.")
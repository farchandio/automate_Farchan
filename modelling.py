import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer # Library untuk mengisi nilai kosong

# --- 1. Pengaturan Eksperimen MLflow ---
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Heart Disease Prediction - Basic")

# --- 2. Muat Dataset ---
try:
    X_train = pd.read_csv('namadataset_preprocessing/X_train.csv')
    X_test = pd.read_csv('namadataset_preprocessing/X_test.csv')
    y_train = pd.read_csv('namadataset_preprocessing/y_train.csv')
    y_test = pd.read_csv('namadataset_preprocessing/y_test.csv')
except FileNotFoundError as e:
    print(f"Error: Salah satu file dataset tidak ditemukan. Pastikan semua file ada.")
    exit()

# --- 3. Penanganan Nilai Kosong (Imputation) ---
# Langkah ini untuk memastikan tidak ada nilai NaN yang masuk ke model
imputer = SimpleImputer(strategy='median')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# --- 4. Latih Model di dalam Sesi MLflow ---
with mlflow.start_run():
    mlflow.sklearn.autolog()
    print("Melatih model Logistic Regression...")
    
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("Model training selesai.")
    print(f"Akurasi Model: {accuracy:.4f}")

print("\nProses selesai. Untuk melihat hasil, jalankan 'mlflow ui --backend-store-uri sqlite:///mlflow.db' di terminal.")
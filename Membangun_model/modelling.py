import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Konfigurasi MLflow pada lokal
mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("La-Liga-Baseline-Model")

# Mengaktifkan MLflow autologging
mlflow.autolog()

def run_modelling():
    # Load dataset
    print("Memuat Dataset.....")
    data = pd.read_csv("la_liga_cleaned.csv")

    # Memisahkan fitur dan target
    X = data.drop("FTR", axis=1)
    y = data["FTR"]

    # Membagi data menjadi set pelatihan dan pengujian
    print("Membagi dataset menjadi set pelatihan dan pengujian.....")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Memulai sesi pelacakan MLflow
    print("Mulai melatih model dan merekam ke mlflow.....")
    with mlflow.start_run(run_name="RandomForest_Basic"):
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        print("[INFO] Pelatihan selesai. Proses pencatatan telah direkam pada MLflow UI.")

if __name__ == "__main__":
    run_modelling()
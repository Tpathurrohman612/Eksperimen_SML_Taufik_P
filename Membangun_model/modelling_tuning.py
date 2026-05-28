import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings

# mengabaikan peringatan
warnings.filterwarnings("ignore")

# Konfigurasi integrasi DagsHub untuk melacak eksperimen secara online
dagshub.init(repo_owner="Tpathurrohman612", repo_name="Eksperimen_SML_Taufik_P", mlflow=True)

# Menonaktifkan autologging MLflow
mlflow.sklearn.autolog(disable=True)

def run_tuning_and_logging():
    # Load dataset
    print("Memuat Dataset.....")
    data = pd.read_csv("la_liga_cleaned.csv")

    # Memisahkan fitur dan target
    X = data.drop("FTR", axis=1)
    y = data["FTR"]

    # Membagi data menjadi set pelatihan dan pengujian
    print("Membagi dataset menjadi set pelatihan dan pengujian.....")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Inisialsi model dasar Random Forest
    print("Memulai proses Hyperparameter Tuning menggunakan GridSearch.....")
    rf_model = RandomForestClassifier(random_state=42)

    # Mendefinisikan parameter grid untuk tuning
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [None, 10, 20]
    }

    # Melaksanakan pencarian kombinasi parameter optimal menggunakan Cross Validation
    grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Mengekstrak model dengan performa terbaik dari hasil evaluasi GridSearchCV
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    # Memulai sesi pelacakan MLflow
    print("Memulai proses pencatatan (logging) metrik dan parameter secara manual ke DagsHub.....")
    with mlflow.start_run(run_name="RandomForest_Tuning"):
        # Mencatat hyperparameter optimal yang ditemukan oleh GridSearch
        mlflow.log_params(grid_search.best_params_)

        # Mengkalkulasi dan mencatat metrik evaluasi klasifikasi secara manual
        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
        mlflow.log_metric("precision_macro", precision_score(y_test, y_pred, average='macro'))
        mlflow.log_metric("recall_macro", recall_score(y_test, y_pred, average='macro'))
        mlflow.log_metric("f1_macro", f1_score(y_test, y_pred, average='macro'))

        print("Membangkitkan dan menyimpan visualisasi evaluasi model.....")
        # Membuat Artefak visualisasi confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix La Liga')
        plt.ylabel('Label Aktual')
        plt.xlabel('Label Prediksi')
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png") # Mengunggah gambar ke pelacak (Tracker)
        plt.close() # Menutup plot untuk menghindari tumpang tindih pada visualisasi berikutnya

        # Membuat Artefak visualisasi feature importance
        importances = best_model.feature_importances_
        feature = X.columns
        plt.figure(figsize=(10, 6))
        plt.barh(feature[:10], importances[:10])
        plt.title('Top 10 Feature Importance')
        plt.savefig("feature_importance.png")
        mlflow.log_artifact("feature_importance.png") # Mengunggah gambar ke pelacak (Tracker)
        plt.close() # Menutup plot untuk menghindari tumpang tindih pada visualisasi berikutnya

        # Menyimpan dan mencatat arsitek model akhir
        mlflow.sklearn.log_model(best_model, "model")

        print("[INFO] Proses tuning dan pencatatan telah selesai. Semua metrik, parameter, dan artefak telah direkam pada DagsHub.")

if __name__ == "__main__":
    run_tuning_and_logging()
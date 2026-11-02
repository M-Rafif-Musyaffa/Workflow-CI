import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def run_model():
    print("Memuat dataset yang sudah diproses...")
    df = pd.read_csv("heart-disease-dataset_preprocessing/heart_ready.csv")
    X = df.drop("target", axis=1)
    y = df["target"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Memulai MLflow run
    with mlflow.start_run() as run:
        print("Melatih model RandomForest...")
        rf = RandomForestClassifier(random_state=42)
        rf.fit(X_train, y_train)
        
        # Log model dengan format yang aman (skops) agar diizinkan MLflow
        mlflow.sklearn.log_model(
            sk_model=rf, 
            artifact_path="model",
            skops_trusted_types=["sklearn.tree._tree.Tree"]
        )
        
        # Simpan Run ID ke dalam file teks untuk dibaca GitHub Actions nanti
        with open("run_id.txt", "w") as f:
            f.write(run.info.run_id)
            
        print("Selesai! Model dan Run ID telah tersimpan secara lokal di folder mlruns.")

if __name__ == "__main__":
    run_model()

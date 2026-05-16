import mlflow
import mlflow.sklearn
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Credit Scoring")

# Load dataset hasil preprocessing
df = pd.read_csv("../preprocessing/dataset_preprocessing.csv")

# Pisahkan fitur dan target
X = df.drop("Credit_Score", axis=1)
y = df["Credit_Score"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Autolog MLflow
mlflow.sklearn.autolog(log_models=False)

# Training model
with mlflow.start_run():
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("Accuracy:", accuracy)
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    joblib.dump(model, "credit_score_model.pkl")
    mlflow.log_artifact("credit_score_model.pkl")

    print("\nModel berhasil dilatih dan disimpan")
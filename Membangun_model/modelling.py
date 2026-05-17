import mlflow
import mlflow.sklearn
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("../preprocessing/dataset_preprocessing.csv")

X = df.drop("Credit_Score", axis=1)
y = df["Credit_Score"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

mlflow.log_metric("accuracy", accuracy)

joblib.dump(model, "credit_score_model.pkl")
mlflow.log_artifact("credit_score_model.pkl")

mlflow.sklearn.log_model(model, "model")

print("\nModel berhasil dilatih dan disimpan")

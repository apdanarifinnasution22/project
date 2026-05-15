import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv('train.csv')

# Hapus kolom yang tidak diperlukan
drop_cols = ['ID', 'Customer_ID', 'Name', 'SSN']
df = df.drop(columns=drop_cols)

# Missing value handling
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].median())

# Encoding categorical data
label_encoder = LabelEncoder()

categorical_cols = df.select_dtypes(include=['object']).columns

for col in categorical_cols:
    df[col] = df[col].astype(str)
    df[col] = label_encoder.fit_transform(df[col])

# Pisahkan fitur dan target
X = df.drop('Credit_Score', axis=1)
y = df['Credit_Score']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Normalisasi data
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Training model
model = RandomForestClassifier(random_state=42)

model.fit(X_train, y_train)

# Prediksi
y_pred = model.predict(X_test)

# Evaluasi
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy :", accuracy)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Simpan model
joblib.dump(model, 'credit_score_model.pkl')

print("\nModel berhasil disimpan")
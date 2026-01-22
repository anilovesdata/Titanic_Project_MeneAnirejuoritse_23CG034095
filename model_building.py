# model_building.py
"""
Titanic Survival Prediction Model Builder (Script Version)
Features used: Pclass, Sex, Age, SibSp, Fare
Algorithm: Logistic Regression
Persistence: Joblib
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer
import joblib
import os

# ────────────────────────────────────────────────
# Optional: Temporary SSL fix (remove after running Install Certificates.command)
# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context
# ────────────────────────────────────────────────

# 1. Load dataset
print("Loading Titanic dataset...")
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

try:
    df = pd.read_csv(url)
except Exception as e:
    print(f"Failed to load from URL: {e}")
    print("Make sure to run 'Install Certificates.command' from your Python 3.13 folder in Applications.")
    print("Or run: python3 -m pip install --upgrade certifi")
    exit(1)

# Select relevant columns
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Fare']
target = 'Survived'
df = df[features + [target]].copy()

print(f"Dataset shape: {df.shape}")
print(df.head())

# 2. Preprocessing
print("\nPreprocessing...")

# Handle missing values
imputer = SimpleImputer(strategy='median')
df['Age'] = imputer.fit_transform(df[['Age']])

# Encode Sex
label_encoder = LabelEncoder()
df['Sex'] = label_encoder.fit_transform(df['Sex'])

# Scale numerical features
scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

# Features & target
X = df[features]
y = df[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# 3. Train model
print("\nTraining Logistic Regression...")
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)

# 4. Evaluate
print("\nEvaluation:")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=["Did Not Survive", "Survived"]))

# 5. Save artifacts
model_dir = "model"
os.makedirs(model_dir, exist_ok=True)

joblib.dump(model,         os.path.join(model_dir, "titanic_survival_model.pkl"))
joblib.dump(scaler,        os.path.join(model_dir, "scaler.pkl"))
joblib.dump(label_encoder, os.path.join(model_dir, "label_encoder.pkl"))
joblib.dump(imputer,       os.path.join(model_dir, "imputer.pkl"))

print(f"\nSaved to: {model_dir}/")
print("Files: titanic_survival_model.pkl, scaler.pkl, label_encoder.pkl, imputer.pkl")

# 6. Demo reload & predict
print("\nTesting reloaded model...")
loaded_model   = joblib.load(os.path.join(model_dir, "titanic_survival_model.pkl"))
loaded_scaler  = joblib.load(os.path.join(model_dir, "scaler.pkl"))
loaded_encoder = joblib.load(os.path.join(model_dir, "label_encoder.pkl"))
loaded_imputer = joblib.load(os.path.join(model_dir, "imputer.pkl"))

example = pd.DataFrame({
    'Pclass': [3],
    'Sex':    ['male'],
    'Age':    [25],
    'SibSp':  [0],
    'Fare':   [7.25]
})

example['Age'] = loaded_imputer.transform(example[['Age']])
example['Sex'] = loaded_encoder.transform(example['Sex'])
example[['Age', 'Fare']] = loaded_scaler.transform(example[['Age', 'Fare']])

pred = loaded_model.predict(example)[0]
prob = loaded_model.predict_proba(example)[0][1]

print(f"Prediction: {'Survived' if pred == 1 else 'Did Not Survive'}")
print(f"Survival probability: {prob:.3f}")

print("\nDone. You can now run the Flask app.")
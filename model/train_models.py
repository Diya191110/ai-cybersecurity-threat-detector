import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# Step 1: Load cleaned data
df = pd.read_csv("data/cleaned_phishing_data.csv")

# Step 2: Drop the 'Index' column if it exists
if 'Index' in df.columns:
    df = df.drop(columns=['Index'])

# Step 3: Split into features and labels
X = df.drop('StatsReport', axis=1)
y = df['StatsReport']

# Step 4: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train multiple models
models = {
    "RandomForest": RandomForestClassifier(random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

best_model = None
best_accuracy = 0.0
best_model_name = ""

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {acc:.4f}")

    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model
        best_model_name = name

# Step 6: Save the best model
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "best_model.pkl")
joblib.dump(best_model, model_path)

print(f"\n✅ Best Model: {best_model_name} with Accuracy: {best_accuracy:.4f}")
print(f"📁 Model saved at: {model_path}")

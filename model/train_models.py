import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import joblib
import os

#Step1: # Load the dataset
data_path = os.path.join("data", "cleaned_phishing_data.csv")
df = pd.read_csv(data_path)


# Step 2: Split the dataset into features and target variable
X= df.drop(['class'],axis=1)
y=df['class'].replace(-1,0)

# Step 3: Split data into train and test
X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.2,random_state=42)

# Step 4: Define models:
models = {
    'RandomForest': RandomForestClassifier(),
    'LogisticRegression': LogisticRegression(max_iter=1000),
    'XGBoost': XGBClassifier(eval_metric='mlogloss',verbosity=0) 
    }

best_model= None
best_acc = 0
best_model_name=""

#Step 5: Train and evaluate the model:

for model_name, model in models.items():
    model.fit(X_train,y_train)
    preds=model.predict(X_test)
    acc=accuracy_score(y_test,preds)
    print(f"{model_name} Accuracy: {acc:.4f}")

    if acc > best_acc:
        best_acc=acc
        best_model = model
        best_model_name = model_name


# Step 6: Save the best model:
model_dir = "models"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

model_path = os.path.join(model_dir, "best_model.pkl")
joblib.dump(best_model, model_path)


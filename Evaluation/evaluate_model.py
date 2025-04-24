import pandas as pd
import joblib
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import train_test_split

# Step 1: Load the dataset:
df = pd.read_csv('data/cleaned_phishing_data.csv')
X=df.drop('class',axis=1)
y=df['class'].replace({-1:0})

# Step 2: Split again to get the test set:
_,X_test,_, y_test = train_test_split(X,y,test_size=0.2 , random_state=42)

# Step 3: Load the best model:
model=joblib.load('models/best_model.pkl')


# Step 4: Predict and evaluate:
y_pred = model.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test,y_pred))
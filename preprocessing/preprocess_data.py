import pandas as pd
from sklearn.preprocessing import LabelEncoder,StandardScaler

# Step 1:Load the dataset:
data_path=data_path = 'C:\\Users\\KIIT\\Documents\\GitHub\\ai-cybersecurity-threat-detector\\data\\phishing_data.csv'
df=pd.read_csv(data_path)

#Step 2: Show basic inforamtion:
print("Dataset shape:", df.shape)
print("First 5 rows:\n",df.head())
print("\n Missing values: \n", df.isnull().sum())

#Step 3: Drop rows and columns with missing values:
df.dropna(inplace=True)

# Step 4:Separate features and labels:
X=df.drop('class',axis=1)
y=df['class']

# Step 5:Encode categorical columns:
label_encoders={}
for col in X.columns:
    if X[col].dtype == 'object':
        le=LabelEncoder()
        X[col]=le.fit_transform(X[col])
        label_encoders[col] = le


# Step 6: Feature scaling(Standardization):
scaler= StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 7: Save the cleaned and scaled dataset:
processed_df=pd.DataFrame(X_scaled, columns=X.columns)
processed_df['class']= y.values

save_path= r'C:\Users\KIIT\Documents\GitHub\ai-cybersecurity-threat-detector\data\cleaned_phishing_data.csv'
processed_df.to_csv(save_path, index=False)

print("\n Data cleaned and saved to 'cleaned_phising_data.csv' ")


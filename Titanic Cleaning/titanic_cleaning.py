import pandas as pd

df = pd.read_csv("Titanic Cleaning/Titanic_Dataset.csv") 
df.head()
# 1. View first rows of the dataset
df.head()

# 2. General information about columns and data types
df.info()

# 3. Descriptive statistics of numeric columns
df.describe()

# 4. Count of null values per column
df.isna().sum()

# --- Step 1: Rename columns to lowercase for consistency ---
df.columns = df.columns.str.lower()
df.columns

# --- Step 2: Drop irrelevant columns or columns with too many nulls ---
df = df.drop(['name', 'ticket', 'cabin'], axis=1)
df.head()

# --- Step 3: Fill missing values ---
# Fill Age with median
df['age'].fillna(df['age'].median(), inplace=True)

# Fill Embarked with the most frequent value
df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)

# Verify no nulls remain
df.isnull().sum()

# Convert 'sex' to binary variable: male=0, female=1
df['sex'] = df['sex'].map({'male': 0, 'female': 1})

# Create dummy variables for 'embarked'
df = pd.get_dummies(df, columns=['embarked'], drop_first=True)

# Verify the result
df.head()

from sklearn.preprocessing import StandardScaler

# 1️⃣ Target variable
y = df['survived']   # what we want to predict

# 2️⃣ Features (drop 'survived' so it does not mix)
X = df.drop(columns=['survived'])

# 3️⃣ Scale ONLY numeric columns
scaler = StandardScaler()
X_scaled = X.copy()  # to not overwrite in case we want to compare
num_cols = ['pclass', 'age', 'sibsp', 'parch', 'fare']
X_scaled[num_cols] = scaler.fit_transform(X_scaled[num_cols])

# Check first rows to verify changes
X_scaled.head()

from sklearn.model_selection import train_test_split

# Split data: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Check dataset sizes
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1️⃣ Create the model
model = RandomForestClassifier(random_state=42)

# 2️⃣ Train the model with training data
model.fit(X_train, y_train)

# 3️⃣ Predict with test data
y_pred = model.predict(X_test)

# 4️⃣ Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Feature importance
importances = model.feature_importances_

# Associate with column names
feature_importances = pd.Series(importances, index=X_train.columns)

# Sort from highest to lowest importance
feature_importances.sort_values(ascending=False)

df.to_csv("titanic_cleaned.csv", index=False)
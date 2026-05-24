# Ankylosing-Spondalytis# Ankylosing Spondylitis Prediction using Machine Learning

# Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Machine Learning Algorithms
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# ---------------------------------------------------
# Load Dataset
# ---------------------------------------------------

df = pd.read_csv("ankylosing_spondylitis_dataset_500_rows.csv")

# Display first 5 rows
print("First 5 Rows:\n")
print(df.head())

# ---------------------------------------------------
# Dataset Information
# ---------------------------------------------------

print("\nDataset Info:\n")
print(df.info())

# ---------------------------------------------------
# Check Missing Values
# ---------------------------------------------------

print("\nMissing Values:\n")
print(df.isnull().sum())

# ---------------------------------------------------
# Convert Categorical Data into Numerical Data
# ---------------------------------------------------

le = LabelEncoder()

categorical_columns = [
    'Gender',
    'Morning_Stiffness',
    'Spinal_Flexibility',
    'HLA_B27_Status',
    'Fatigue_Level',
    'Ankylosing_Spondylitis'
]

for col in categorical_columns:
    df[col] = le.fit_transform(df[col])

# ---------------------------------------------------
# Feature Selection
# ---------------------------------------------------

# Input Features
X = df.drop(['Patient_ID', 'Ankylosing_Spondylitis'], axis=1)

# Target Variable
y = df['Ankylosing_Spondylitis']

# ---------------------------------------------------
# Split Dataset into Training and Testing
# ---------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# ---------------------------------------------------
# 1. Decision Tree Classifier
# ---------------------------------------------------

dt_model = DecisionTreeClassifier(random_state=42)

dt_model.fit(X_train, y_train)

dt_pred = dt_model.predict(X_test)

print("\nDecision Tree Accuracy:")
print(accuracy_score(y_test, dt_pred))

# ---------------------------------------------------
# 2. Random Forest Classifier
# ---------------------------------------------------

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

print("\nRandom Forest Accuracy:")
print(accuracy_score(y_test, rf_pred))

# ---------------------------------------------------
# 3. Logistic Regression
# ---------------------------------------------------

lr_model = LogisticRegression(max_iter=1000)

lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)

print("\nLogistic Regression Accuracy:")
print(accuracy_score(y_test, lr_pred))

# ---------------------------------------------------
# 4. K-Nearest Neighbors
# ---------------------------------------------------

knn_model = KNeighborsClassifier(n_neighbors=5)

knn_model.fit(X_train, y_train)

knn_pred = knn_model.predict(X_test)

print("\nKNN Accuracy:")
print(accuracy_score(y_test, knn_pred))

# ---------------------------------------------------
# Best Model Evaluation (Random Forest)
# ---------------------------------------------------

print("\nClassification Report:\n")
print(classification_report(y_test, rf_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, rf_pred))

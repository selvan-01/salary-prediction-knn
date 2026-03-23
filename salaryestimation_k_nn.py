"""
📊 Salary Estimation using K-Nearest Neighbors (KNN)

This project predicts whether an employee's salary is >50K or <=50K 
based on features like Age, Education, Capital Gain, and Hours per Week.
"""

# =========================
# 1. Import Libraries
# =========================
import pandas as pd          # Data handling
import numpy as np           # Numerical operations
import matplotlib.pyplot as plt  # Visualization

# =========================
# 2. Load Dataset
# =========================
# Make sure 'salary.csv' is in the same directory
dataset = pd.read_csv('salary.csv')

# =========================
# 3. Dataset Overview
# =========================
print("Dataset Shape:", dataset.shape)
print("\nFirst 5 Rows:\n", dataset.head())

# =========================
# 4. Data Preprocessing
# =========================

# Convert salary labels to binary values
# <=50K → 0, >50K → 1
dataset['income'] = dataset['income'].map({
    '<=50K': 0,
    '>50K': 1
}).astype(int)

# =========================
# 5. Split Features & Target
# =========================
X = dataset.iloc[:, :-1].values   # Independent variables
y = dataset.iloc[:, -1].values    # Dependent variable (income)

# =========================
# 6. Train-Test Split
# =========================
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=0
)

# =========================
# 7. Feature Scaling
# =========================
# Standardization ensures all features contribute equally

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)   # Fit + Transform training data
X_test = sc.transform(X_test)         # Only transform test data

# =========================
# 8. Find Optimal K Value
# =========================
from sklearn.neighbors import KNeighborsClassifier

error = []

# Try K values from 1 to 39
for i in range(1, 40):
    model = KNeighborsClassifier(n_neighbors=i)
    model.fit(X_train, y_train)
    
    pred_i = model.predict(X_test)
    error.append(np.mean(pred_i != y_test))

# Plot Error Rate vs K
plt.figure(figsize=(10, 5))
plt.plot(range(1, 40), error, linestyle='dashed', marker='o')
plt.title('Error Rate vs K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
plt.show()

# =========================
# 9. Train Final Model
# =========================
# Choose optimal K (example: 2)
model = KNeighborsClassifier(n_neighbors=2, metric='minkowski', p=2)
model.fit(X_train, y_train)

# =========================
# 10. Predict New Employee
# =========================
print("\n🔍 Enter New Employee Details:")

age = int(input("Age: "))
edu = int(input("Education (numeric): "))
cg = int(input("Capital Gain: "))
wh = int(input("Hours per Week: "))

new_emp = [[age, edu, cg, wh]]

# Apply same scaling before prediction
result = model.predict(sc.transform(new_emp))

if result[0] == 1:
    print("✅ Employee likely earns >50K")
else:
    print("❌ Employee likely earns <=50K")

# =========================
# 11. Predict Test Data
# =========================
y_pred = model.predict(X_test)

print("\n📊 Predictions vs Actual:")
print(np.concatenate(
    (y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)),
    axis=1
))

# =========================
# 12. Model Evaluation
# =========================
from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:\n", cm)

accuracy = accuracy_score(y_test, y_pred)
print(f"\n🎯 Model Accuracy: {accuracy * 100:.2f}%")
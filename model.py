"""
Train KNN Model and Save It
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import pickle

# Load dataset
dataset = pd.read_csv('salary.csv')

# Convert labels
dataset['income'] = dataset['income'].map({
    '<=50K': 0,
    '>50K': 1
}).astype(int)

# Split data
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0
)

# Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)

# Train model
model = KNeighborsClassifier(n_neighbors=2)
model.fit(X_train, y_train)

# Save model & scaler
pickle.dump(model, open('knn_model.pkl', 'wb'))
pickle.dump(sc, open('scaler.pkl', 'wb'))

print("✅ Model & Scaler saved successfully!")
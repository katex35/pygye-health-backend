import ssl
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# SSL certificate verification disabled
ssl._create_default_https_context = ssl._create_unverified_context

print("Loading dataset...")
heart_disease = fetch_ucirepo(id=45)
X = heart_disease.data.features
y = heart_disease.data.targets

print(f"Number of instances: {len(X)}")
print(f"Number of features: {len(X.columns)}")
print(f"Dataset size: {X.memory_usage(deep=True).sum() / (1024 * 1024):.2f} MB")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
print("Training scaler...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train the model
print("Training model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train.values.ravel())

# Save the model and scaler
print("Saving model and scaler...")
joblib.dump(model, 'model.joblib')
joblib.dump(scaler, 'scaler.joblib')

print("Model and scaler have been saved successfully!")
print(f"Model size: {model.n_estimators} trees")
print(f"Number of features: {X.shape[1]}")
print(f"Training set size: {X_train.shape[0]} samples") 
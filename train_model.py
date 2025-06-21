import ssl
import pandas as pd
import joblib
import numpy as np
from ucimlrepo import fetch_ucirepo
from lightweight_model import SimpleRandomForest, SimpleStandardScaler

# SSL certificate verification disabled
ssl._create_default_https_context = ssl._create_unverified_context

print("Fetching UCI Heart Disease dataset...")
# Fetch dataset directly from UCI
heart_disease = fetch_ucirepo(id=45)
X = heart_disease.data.features
y = heart_disease.data.targets

print(f"Dataset shape: {X.shape}")
print("Dataset columns:", list(X.columns))

# Convert to numpy arrays
X_numpy = X.values
y_numpy = y.values.ravel()

# Convert target to binary (0 = no disease, 1 = disease)
y_binary = (y_numpy > 0).astype(int)

print(f"Features shape: {X_numpy.shape}")
print(f"Target distribution: {np.bincount(y_binary)}")

# Train scaler
print("Training scaler...")
scaler = SimpleStandardScaler()
X_scaled = scaler.fit_transform(X_numpy)

# Train model
print("Training model...")
model = SimpleRandomForest(
    n_estimators=20,  
    max_depth=10      
)
model.fit(X_scaled, y_binary)

# Save the model and scaler
print("Saving model and scaler...")
joblib.dump(model, 'model.joblib')
joblib.dump(scaler, 'scaler.joblib')

print("Model and scaler saved successfully!")
print(f"Model trees: {model.n_estimators}")
print(f"Features: {X_numpy.shape[1]}")

# Quick test
print("\nTesting model...")
test_predictions = model.predict(X_scaled[:5])
test_probabilities = model.predict_proba(X_scaled[:5])
print(f"Sample predictions: {test_predictions}")
print(f"Sample probabilities: {test_probabilities[:, 1].round(3)}")
print(f"Actual values: {y_binary[:5]}") 
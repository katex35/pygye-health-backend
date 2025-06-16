import ssl
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# SSL verification for development, disable 
ssl._create_default_https_context = ssl._create_unverified_context

# load dataset
print("Loading dataset...")
heart_disease = fetch_ucirepo(id=45)
X = heart_disease.data.features
y = heart_disease.data.targets

# split data
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# scale data
print("Scaling data...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# train model with reduced size
print("Training model...")
model = RandomForestClassifier(
    n_estimators=50,  # reduced from 100
    max_depth=5,      # added max depth
    random_state=42
)
model.fit(X_train_scaled, y_train.values.ravel())

# save model and scaler
print("Saving model and scaler...")
joblib.dump(model, 'model.joblib')
joblib.dump(scaler, 'scaler.joblib')

print("Done!") 
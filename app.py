from flask import Flask, jsonify, request
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": ["http://127.0.0.1:5500", "http://localhost:5500", "https://pygye-health-frontend.vercel.app"],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Load the saved model and scaler
print("Loading saved model and scaler...")
model = joblib.load('model.joblib')
scaler = joblib.load('scaler.joblib')
print("Model and scaler loaded successfully!")

@app.route('/')
def home():
    return jsonify({
        "message": "works fine!",
        "status": "success"
    })

@app.route('/api/health')
def health_check():
    return jsonify({
        "status": "healthy",
        "version": "1.0.0"
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        required_fields = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                         'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        
        for field in required_fields:
            if field not in data:
                return jsonify({
                    "error": f"Missing required field: {field}",
                    "status": "error"
                }), 400

        input_data = np.array([[
            data['age'],
            data['sex'],
            data['cp'],
            data['trestbps'],
            data['chol'],
            data['fbs'],
            data['restecg'],
            data['thalach'],
            data['exang'],
            data['oldpeak'],
            data['slope'],
            data['ca'],
            data['thal']
        ]])

        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)
        probability = model.predict_proba(input_scaled)

        return jsonify({
            "prediction": int(prediction[0]),
            "probability": float(probability[0][1]),
            "status": "success"
        })

    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

@app.route('/api/features', methods=['GET'])
def get_features():
    return jsonify({
        "features": {
            "age": "Age in years",
            "sex": "Gender (1 = male, 0 = female)",
            "cp": "Chest pain type (0-3)",
            "trestbps": "Resting blood pressure in mm Hg",
            "chol": "Serum cholesterol in mg/dl",
            "fbs": "Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)",
            "restecg": "Resting electrocardiographic results (0-2)",
            "thalach": "Maximum heart rate achieved",
            "exang": "Exercise induced angina (1 = yes, 0 = no)",
            "oldpeak": "ST depression induced by exercise relative to rest",
            "slope": "Slope of the peak exercise ST segment (0-2)",
            "ca": "Number of major vessels colored by fluoroscopy (0-3)",
            "thal": "Thalassemia test result (0-3)"
        },
        "status": "success"
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)

application = app 
from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load model
with open('heart.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [float(x) for x in request.form.values()]
        features = [np.array(features)]
        prediction = model.predict(features)
        risk_level = 'High Risk' if prediction[0] == 1 else 'Low Risk'
        return render_template('index.html', prediction_text=f'{risk_level}')
    except Exception as e:
        return f"Error: {e}"


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # default to 5000 for local
    app.run(host="0.0.0.0", port=port)

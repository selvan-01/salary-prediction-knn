"""
Flask App for Salary Prediction
"""

from flask import Flask, render_template, request
import pickle
import numpy as np

# Load model & scaler
model = pickle.load(open('knn_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

app = Flask(__name__)

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        age = int(request.form['age'])
        education = int(request.form['education'])
        capital_gain = int(request.form['capital_gain'])
        hours = int(request.form['hours'])

        data = np.array([[age, education, capital_gain, hours]])
        scaled_data = scaler.transform(data)

        prediction = model.predict(scaled_data)

        if prediction[0] == 1:
            result = "Salary > 50K 💰"
        else:
            result = "Salary <= 50K 📉"

        return render_template('index.html', prediction_text=result)

    except Exception as e:
        return render_template('index.html', prediction_text="Error: " + str(e))


if __name__ == "__main__":
    app.run(debug=True)
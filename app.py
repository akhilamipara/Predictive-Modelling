from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("heart_failure_model.pkl")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            features = [
                float(request.form['age']),
                int(request.form['anaemia']),
                int(request.form['creatinine_phosphokinase']),
                int(request.form['diabetes']),
                int(request.form['ejection_fraction']),
                int(request.form['high_blood_pressure']),
                float(request.form['platelets']),
                float(request.form['serum_creatinine']),
                int(request.form['serum_sodium']),
                int(request.form['sex']),
                int(request.form['smoking']),
                int(request.form['time']),
            ]
            prediction = model.predict([features])[0]
            result = "High Risk of Death" if prediction == 1 else "Low Risk of Death"
        except:
            result = "Invalid input. Please check your values."
        return render_template("index.html", prediction=result)

if __name__ == '__main__':
    app.run(debug=True)

# app.py
from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load model & preprocessors
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model')

model         = joblib.load(os.path.join(MODEL_DIR, 'titanic_survival_model.pkl'))
scaler        = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
label_encoder = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.pkl'))
imputer       = joblib.load(os.path.join(MODEL_DIR, 'imputer.pkl'))

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    probability = None
    error = None

    if request.method == 'POST':
        try:
            pclass = int(request.form['pclass'])
            sex    = request.form['sex']
            age    = float(request.form['age'])
            sibsp  = int(request.form['sibsp'])
            fare   = float(request.form['fare'])

            input_data = pd.DataFrame({
                'Pclass': [pclass],
                'Sex':    [sex],
                'Age':    [age],
                'SibSp':  [sibsp],
                'Fare':   [fare]
            })

            input_data['Age'] = imputer.transform(input_data[['Age']])
            input_data['Sex'] = label_encoder.transform(input_data['Sex'])
            input_data[['Age', 'Fare']] = scaler.transform(input_data[['Age', 'Fare']])

            pred = model.predict(input_data)[0]
            prob = model.predict_proba(input_data)[0][1]

            prediction = "Survived" if pred == 1 else "Did Not Survive"
            probability = round(prob * 100, 1)

        except Exception as e:
            error = str(e)

    return render_template(
        'index.html',
        prediction=prediction,
        probability=probability,
        error=error
    )

if __name__ == '__main__':
    app.run(debug=True)
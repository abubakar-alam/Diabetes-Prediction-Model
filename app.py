from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained machine learning model
model_path = r'C:\Users\CC\Desktop\diabetes\models\trained_models.pkl'

with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Define the home route
@app.route('/')
def home():
    return render_template('index.html')

# Define the predict route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form input values
        pregnancies = float(request.form['pregnancies'])
        glucose = float(request.form['glucose'])
        blood_pressure = float(request.form['blood_pressure'])
        skin_thickness = float(request.form['skin_thickness'])
        insulin = float(request.form['insulin'])
        bmi = float(request.form['bmi'])
        diabetes_pedigree_function = float(request.form['diabetes_pedigree_function'])
        age = float(request.form['age'])

        # Create a DataFrame for consistent reshaping
        input_data = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi,
                                     diabetes_pedigree_function, age]])

        # Make a prediction
        prediction = model.predict(input_data.values.reshape(1, -1))

        # Display the prediction result on the webpage
        if prediction == 0:
            result = 'Not Diabetic'
        else:
            result = 'Diabetic'

        return render_template('result.html', prediction_result=result)

    except Exception as e:
        return render_template('result.html', prediction_result='Error: Please check your inputs.')

if __name__ == '__main__':
    app.run(debug=True)

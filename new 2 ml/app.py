from flask import Flask, render_template, request
import joblib
import os
from model import (
    generate_dataset,
    train_model,
    generate_charts
)

app = Flask(__name__)

# Generate dataset if not exists
if not os.path.exists("student_data.csv"):
    generate_dataset()

# Train model if not exists
if not os.path.exists("student_model.pkl"):
    train_model()

# Generate charts
generate_charts()

# Load trained model
model = joblib.load("student_model.pkl")

# Load accuracy
with open("accuracy.txt", "r") as file:
    accuracy = file.read()


@app.route('/')
def home():
    return render_template("index.html", accuracy=accuracy)


@app.route('/predict', methods=['POST'])
def predict():
    study_hours = float(request.form['study_hours'])
    attendance = float(request.form['attendance'])
    assignments_score = float(request.form['assignments_score'])
    exam_score = float(request.form['exam_score'])
    sleep_hours = float(request.form['sleep_hours'])

    # Prepare data
    data = [[
        study_hours,
        attendance,
        assignments_score,
        exam_score,
        sleep_hours
    ]]

    # Predict
    prediction = model.predict(data)[0]

    # Probability
    probability = model.predict_proba(data)[0]

    pass_probability = round(probability[1] * 100, 2)
    fail_probability = round(probability[0] * 100, 2)

    result = "Pass" if prediction == 1 else "Fail"

    return render_template(
        "result.html",
        result=result,
        pass_probability=pass_probability,
        fail_probability=fail_probability,
        accuracy=accuracy,
        study_hours=study_hours,
        attendance=attendance,
        assignments_score=assignments_score,
        exam_score=exam_score,
        sleep_hours=sleep_hours
    )


if __name__ == '__main__':
    app.run(debug=True)
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib


# ==========================================
# Generate Realistic Student Dataset
# ==========================================
def generate_dataset():

    data = []

    # Generate 5000 records
    for i in range(5000):

        # Random realistic student data
        study_hours = round(random.uniform(1, 12), 1)

        attendance = random.randint(30, 100)

        assignments_score = random.randint(20, 100)

        exam_score = random.randint(20, 100)

        sleep_hours = round(random.uniform(3, 10), 1)

        # ----------------------------------
        # Smart scoring logic
        # ----------------------------------

        total_score = (
            study_hours * 4 +
            attendance * 0.25 +
            assignments_score * 0.30 +
            exam_score * 0.45 +
            sleep_hours * 1.5
        )

        # Add randomness/noise
        noise = random.randint(-20, 20)

        final_score = total_score + noise

        # More realistic prediction logic
        if final_score >= 65:
            final_result = 1
        else:
            final_result = 0

        data.append([
            study_hours,
            attendance,
            assignments_score,
            exam_score,
            sleep_hours,
            final_result
        ])

    columns = [
        "study_hours",
        "attendance",
        "assignments_score",
        "exam_score",
        "sleep_hours",
        "final_result"
    ]

    df = pd.DataFrame(data, columns=columns)

    # Save dataset
    df.to_csv("student_data.csv", index=False)

    print("Dataset Generated Successfully!")
    print(df.head())


# ==========================================
# Train Machine Learning Model
# ==========================================
def train_model():

    # Load dataset
    df = pd.read_csv("student_data.csv")

    # Features
    X = df.drop("final_result", axis=1)

    # Target
    y = df["final_result"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # ----------------------------------
    # Random Forest Model
    # ----------------------------------

    model = RandomForestClassifier(
        n_estimators=50,
        max_depth=5,
        random_state=42
    )

    # Train model
    model.fit(X_train, y_train)

    # Predict
    predictions = model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, predictions)

    accuracy_percent = round(accuracy * 100, 2)

    # Save accuracy
    with open("accuracy.txt", "w") as file:
        file.write(str(accuracy_percent))

    # Save model
    joblib.dump(model, "student_model.pkl")

    print("Model Trained Successfully!")
    print("Accuracy:", accuracy_percent, "%")


# ==========================================
# Generate Charts
# ==========================================
def generate_charts():

    # Create charts folder
    if not os.path.exists("static/charts"):
        os.makedirs("static/charts")

    # Load dataset
    df = pd.read_csv("student_data.csv")

    # ======================================
    # Pass vs Fail Pie Chart
    # ======================================

    result_counts = df["final_result"].value_counts()

    plt.figure(figsize=(6, 6))

    plt.pie(
        result_counts,
        labels=["Pass", "Fail"],
        autopct='%1.1f%%'
    )

    plt.title("Pass vs Fail Distribution")

    plt.savefig("static/charts/pie_chart.png")

    plt.close()

    # ======================================
    # Accuracy Bar Chart
    # ======================================

    with open("accuracy.txt", "r") as file:
        accuracy = float(file.read())

    plt.figure(figsize=(6, 5))

    plt.bar(
        ["Accuracy"],
        [accuracy]
    )

    plt.ylim(0, 100)

    plt.title("AI Model Accuracy")

    plt.ylabel("Accuracy %")

    plt.savefig("static/charts/accuracy_chart.png")

    plt.close()

    # ======================================
    # Attendance vs Exam Score
    # ======================================

    plt.figure(figsize=(7, 5))

    plt.scatter(
        df["attendance"],
        df["exam_score"]
    )

    plt.xlabel("Attendance")

    plt.ylabel("Exam Score")

    plt.title("Attendance vs Exam Score")

    plt.savefig("static/charts/scatter_chart.png")

    plt.close()

    # ======================================
    # Study Hours Histogram
    # ======================================

    plt.figure(figsize=(7, 5))

    plt.hist(
        df["study_hours"],
        bins=12
    )

    plt.xlabel("Study Hours")

    plt.ylabel("Students")

    plt.title("Study Hours Distribution")

    plt.savefig("static/charts/histogram_chart.png")

    plt.close()

    print("Charts Generated Successfully!")
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

os.makedirs("model", exist_ok=True)

df = pd.read_csv("data/customers.csv")

df["gender"] = df["gender"].map({"M":0, "F":1})

X = df.drop("is_churn", axis=1)
y = df["is_churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

joblib.dump(model, "model/model.pkl")

print("Model trained successfully 🚀")
import joblib

model = joblib.load("model/model.pkl")

def predict_customer(data):
    probs = model.predict_proba([data])[0]

    stay = probs[0]
    leave = probs[1]

    return {
        "stay": float(stay),
        "leave": float(leave)
    }
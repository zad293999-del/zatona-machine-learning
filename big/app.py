from flask import Flask, request, render_template_string
from src.predict import predict_customer
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

@app.route("/")
def home():
    return render_template_string("""
<!DOCTYPE html>
<html>
<head>
<title>Customer Behavior AI</title>

<style>
body {
    font-family: Arial;
    background: linear-gradient(135deg, #1e3c72, #2a5298);
    display: flex;
    justify-content: center;
    align-items: center;
    height: 100vh;
}

.container {
    background: white;
    padding: 20px;
    border-radius: 12px;
    width: 380px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.3);
}

h2 { text-align:center; }

input {
    width: 100%;
    padding: 7px;
    margin: 5px 0;
    border-radius: 6px;
    border: 1px solid #ccc;
}

button {
    width: 100%;
    padding: 10px;
    background: #2a5298;
    color: white;
    border: none;
    border-radius: 8px;
    cursor: pointer;
}

button:hover {
    background: #1e3c72;
}

.result {
    margin-top: 10px;
    text-align: center;
    font-weight: bold;
}

img {
    width: 100%;
    margin-top: 15px;
}
</style>

</head>

<body>

<div class="container">

<h2>Customer AI</h2>

<input id="age" placeholder="Age">
<input id="gender" placeholder="Gender (0/1)">
<input id="total_orders" placeholder="Total Orders">
<input id="total_spent" placeholder="Total Spent">
<input id="last_purchase_days" placeholder="Last Purchase Days">
<input id="avg_order_value" placeholder="Avg Order Value">
<input id="visited_pages" placeholder="Visited Pages">
<input id="cart_abandoned" placeholder="Cart Abandoned">

<button onclick="predict()">Predict</button>

<div class="result" id="result"></div>

<img id="chart"/>

</div>

<script>
function predict(){

    fetch("/predict_api", {
        method: "POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify({
            customer_id: 0,
            age: document.getElementById("age").value,
            gender: document.getElementById("gender").value,
            total_orders: document.getElementById("total_orders").value,
            total_spent: document.getElementById("total_spent").value,
            last_purchase_days: document.getElementById("last_purchase_days").value,
            avg_order_value: document.getElementById("avg_order_value").value,
            visited_pages: document.getElementById("visited_pages").value,
            cart_abandoned: document.getElementById("cart_abandoned").value
        })
    })

    .then(res => res.json())
    .then(data => {

        let stay = data.stay.toFixed(2);
        let leave = data.leave.toFixed(2);

        document.getElementById("result").innerHTML =
        "Stay: " + stay + "% | Leave: " + leave + "%";

        document.getElementById("chart").src =
        data.chart_url + "?t=" + new Date().getTime();

    });

}
</script>

</body>
</html>
""")

@app.route("/predict_api", methods=["POST"])
def predict_api():
    data = request.json

    features = [
        data["customer_id"],
        int(data["age"]),
        int(data["gender"]),
        int(data["total_orders"]),
        float(data["total_spent"]),
        int(data["last_purchase_days"]),
        float(data["avg_order_value"]),
        int(data["visited_pages"]),
        int(data["cart_abandoned"])
    ]

    result = predict_customer(features)

    stay = result["stay"] * 100
    leave = result["leave"] * 100

    # إنشاء static folder
    if not os.path.exists("static"):
        os.makedirs("static")

    # رسم chart
    plt.figure(figsize=(4,4))
    plt.pie(
        [stay, leave],
        labels=["Stay", "Leave"],
        autopct="%1.1f%%",
        colors=["green", "red"]
    )
    plt.title("Customer Prediction")

    chart_path = "static/chart.png"
    plt.savefig(chart_path)
    plt.close()

    return {
        "stay": stay,
        "leave": leave,
        "chart_url": "/" + chart_path
    }

if __name__ == "__main__":
    app.run(debug=True)
import os
import socket
import numpy as np
import pandas as pd
import pickle
from flask import Flask, request
from sklearn.ensemble import RandomForestClassifier

# Initialize Flask app
app = Flask(__name__)

MODEL_FILE = "model.pkl"

def train_model():
    print("Training model...")
    data = pd.DataFrame({
        "amount": [100, 200, 5000, 50, 7000, 120, 3000, 80, 9000, 150],
        "transaction_time": [10, 12, 2, 14, 1, 16, 3, 18, 0, 11],
        "location_risk": [1, 1, 5, 1, 5, 1, 4, 1, 5, 1],
        "frequency": [2, 3, 10, 1, 15, 2, 8, 1, 20, 3],
        "fraud": [0, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    })

    X = data.drop("fraud", axis=1)
    y = data["fraud"]
    model = RandomForestClassifier()
    model.fit(X, y)
    pickle.dump(model, open(MODEL_FILE, "wb"))
    print("Model saved!")

if not os.path.exists(MODEL_FILE):
    train_model()

model = pickle.load(open(MODEL_FILE, "rb"))

def find_free_port():
    s = socket.socket()
    s.bind(('', 0))
    port = s.getsockname()[1]
    s.close()
    return port

@app.route("/")
def home():
    return """
    <html>
    <head>
        <title>AI Fraud Detection System</title>
        <style>
            body {
                font-family: 'Arial', sans-serif;
                background-color: #f4f4f9;
                margin: 0;
                padding: 0;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                min-height: 100vh;
            }
            h1 {
                color: #333;
                margin-bottom: 20px;
            }
            form {
                background-color: #fff;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                max-width: 400px;
                width: 100%;
            }
            label {
                display: block;
                margin-top: 10px;
                font-weight: bold;
            }
            input[type=number], select {
                width: 100%;
                padding: 10px;
                margin-top: 8px;
                margin-bottom: 16px;
                border: 1px solid #ccc;
                border-radius: 4px;
                box-sizing: border-box;
                font-size: 16px;
            }
            input[type=submit] {
                width: 100%;
                padding: 12px;
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-size: 16px;
                transition: background-color 0.3s ease;
            }
            input[type=submit]:hover {
                background-color: #45a049;
            }
            .footer {
                margin-top: 20px;
                font-size: 14px;
                color: #777;
            }
        </style>
    </head>
    <body>
        <h1>AI Fraud Detection System</h1>
        <form action="/predict" method="post">
            <label for="amount">Amount:</label>
            <input type="number" id="amount" name="amount" required>

            <label for="transaction_time">Transaction Time (0-24):</label>
            <input type="number" id="transaction_time" name="transaction_time" min="0" max="24" required>

            <label for="country">Country:</label>
            <select id="country" name="country" required>
                <option value="UAE">UAE</option>
                <option value="USA">USA</option>
                <option value="UK">UK</option>
                <option value="India">India</option>
                <option value="Nigeria">Nigeria</option>
            </select>

            <label for="frequency">Frequency:</label>
            <input type="number" id="frequency" name="frequency" required>

            <input type="submit" value="Check Fraud">
        </form>
        <div class="footer">
            &copy; 2027 AI Fraud Detection System
        </div>
    </body>
    </html>
    """

@app.route("/predict", methods=["POST"])
def predict():
    try:
        amount = float(request.form["amount"])
        transaction_time = float(request.form["transaction_time"])
        frequency = float(request.form["frequency"])
        country = request.form["country"]

        country_risk_map = {
            "UAE": 1,
            "USA": 2,
            "UK": 2,
            "India": 3,
            "Nigeria": 5
        }

        location_risk = country_risk_map.get(country, 1)

        features = np.array([[amount, transaction_time, location_risk, frequency]])
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]

        if prediction == 1:
            result = "⚠️ Fraudulent Transaction Detected!"
        else:
            result = "✅ Transaction is Safe."

        return f"""
        <html>
        <head>
            <title>Prediction Result</title>
            <style>
                body {{
                    font-family: 'Arial', sans-serif;
                    background-color: #f4f4f9;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                    min-height: 100vh;
                    margin: 0;
                }}
                .container {{
                    background-color: #fff;
                    padding: 40px;
                    border-radius: 10px;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                    text-align: center;
                }}
                h2 {{
                    margin-bottom: 20px;
                }}
                p {{
                    font-size: 18px;
                }}
                a {{
                    display: inline-block;
                    margin-top: 20px;
                    padding: 10px 20px;
                    background-color: #4CAF50;
                    color: white;
                    text-decoration: none;
                    border-radius: 4px;
                    transition: background-color 0.3s ease;
                }}
                a:hover {{
                    background-color: #45a049;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h2>{result}</h2>
                <p><strong>Country Selected:</strong> {country}</p>
                <p><strong>Fraud Probability:</strong> {probability:.2f}</p>
                <a href="/">Check Another Transaction</a>
            </div>
        </body>
        </html>
        """

    except Exception as e:
        return f"Error in input or processing: {e}"

if __name__ == "__main__":
    port = find_free_port()
    print(f"Running on http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port)
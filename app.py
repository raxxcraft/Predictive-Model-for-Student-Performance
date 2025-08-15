from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

app = Flask(__name__, static_folder="frontend", static_url_path="/")
CORS(app)  # allow frontend calls

MODEL = None

# -----------------------------
# Utilities
# -----------------------------
def load_dataset(csv_path: str) -> pd.DataFrame:
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        # Fallback synthetic dataset (if CSV missing)
        rng = np.random.default_rng(42)
        n = 120
        attendance = rng.uniform(40, 100, n)               # 40-100%
        cgpa_pct = rng.uniform(50, 95, n)                  # 50-95% (if CGPA out of 10, *10)
        # Create a next percentage influenced by attendance & CGPA with some noise
        next_pct = 0.45 * attendance + 0.55 * cgpa_pct + rng.normal(0, 3, n)
        next_pct = np.clip(next_pct, 0, 100)
        df = pd.DataFrame({
            "StudentName": [f"Student {i+1}" for i in range(n)],
            "RegisterNo": [f"REG{i+1:04d}" for i in range(n)],
            "Attendance": attendance.round(2),
            "CGPA_Percentage": cgpa_pct.round(2),
            "Next_Percentage": next_pct.round(2)
        })
    return df

def train_model(df: pd.DataFrame):
    X = df[["Attendance", "CGPA_Percentage"]].values
    y = df["Next_Percentage"].values
    model = LinearRegression()
    model.fit(X, y)
    return model

def to_percentage_scale(value: float) -> float:
    """
    Accept CGPA either on 10-point scale or in percentage.
    If the number looks like <=10, assume CGPA out of 10 and convert to percent.
    """
    try:
        v = float(value)
    except Exception:
        return 0.0
    return v * 10.0 if v <= 10 else v

def classify_performance(p: float):
    """
    Your requested thresholds and messages:
    100–85: High
     85–60: Average
     60–45: Low
     <45  : Very Poor
    """
    if 85 <= p <= 100:
        return "High", "Congratulations! You got a great percentage—outstanding work!"
    elif 60 <= p < 85:
        return "Average", "Well done! You’ve got a decent percentage—keep pushing forward!"
    elif 45 <= p < 60:
        return "Low", "Don’t be discouraged—you’ve got potential. Let’s aim higher next time!"
    else:
        return "Very Poor", "Don’t be discouraged—you’ve got potential. Let’s aim higher next time!"

# -----------------------------
# Startup: load data & train
# -----------------------------
DATA_PATH = os.path.join("data", "students.csv")
os.makedirs("data", exist_ok=True)
df_data = load_dataset(DATA_PATH)
MODEL = train_model(df_data)

# -----------------------------
# Routes / API
# -----------------------------
@app.route("/")
def root():
    # Serve the frontend
    return send_from_directory(app.static_folder, "index.html")

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify(status="ok")

@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Request JSON:
    {
      "studentName": "...",        # optional
      "registerNo": "...",         # optional
      "attendance": 92.5,          # 0-100
      "cgpaPercentage": 78.0       # either 0-100 or <=10 means CGPA on 10-point
    }
    """
    data = request.get_json(force=True) or {}

    student_name = data.get("studentName", "").strip()
    register_no = data.get("registerNo", "").strip()

    try:
        attendance = float(data.get("attendance", 0))
    except Exception:
        attendance = 0.0

    cgpa_raw = data.get("cgpaPercentage", 0)
    cgpa_pct = to_percentage_scale(cgpa_raw)

    # Clamp sensible ranges
    attendance = float(np.clip(attendance, 0, 100))
    cgpa_pct = float(np.clip(cgpa_pct, 0, 100))

    X = np.array([[attendance, cgpa_pct]])
    pred_pct = float(MODEL.predict(X)[0])
    pred_pct = float(np.clip(pred_pct, 0, 100))  # ensure 0..100

    category, message = classify_performance(pred_pct)

    return jsonify({
        "studentName": student_name,
        "registerNo": register_no,
        "inputs": {
            "attendance": attendance,
            "cgpa_percentage": cgpa_pct
        },
        "predicted_percentage": round(pred_pct, 2),
        "category": category,
        "message": message
    })

if __name__ == "__main__":
    # Run local dev server
    app.run(host="127.0.0.1", port=5000, debug=True)

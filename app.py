"""Aplikasi Flask: Peramalan Harga Penutupan Bitcoin (BTC-USD) Harian dengan LSTM (TA-04)."""
import json
import os

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from tensorflow.keras.models import load_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
DATA_DIR = os.path.join(BASE_DIR, "data")

app = Flask(__name__)

model = load_model(os.path.join(MODEL_DIR, "lstm_btc.keras"), compile=False)
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
with open(os.path.join(MODEL_DIR, "meta.json"), encoding="utf-8") as f:
    meta = json.load(f)
series = pd.read_csv(os.path.join(DATA_DIR, "bitcoin_close_clean.csv"), parse_dates=["date"])
test_pred = pd.read_csv(os.path.join(MODEL_DIR, "test_predictions.csv"), parse_dates=["date"])

TIMESTEPS = int(meta["timesteps"])
CLIP_LO, CLIP_HI = float(meta["clip_lo"]), float(meta["clip_hi"])
MAX_DAYS = 14


def fmt_dates(s):
    return [d.strftime("%Y-%m-%d") for d in s]


def forecast(days: int) -> dict:
    """Peramalan rekursif atas log-return, lalu dikonversi ke harga."""
    closes = series["close"].values.astype(float)
    rets = np.clip(np.log(closes[1:] / closes[:-1])[-TIMESTEPS:], CLIP_LO, CLIP_HI)
    window = scaler.transform(rets.reshape(-1, 1)).flatten().tolist()
    price = float(closes[-1])
    prices, returns = [], []
    for _ in range(days):
        x = np.array(window[-TIMESTEPS:], dtype="float32").reshape(1, TIMESTEPS, 1)
        p = float(model.predict(x, verbose=0)[0, 0])
        r = float(scaler.inverse_transform([[p]])[0, 0])
        price *= float(np.exp(r))
        prices.append(round(price, 2))
        returns.append(round(r * 100, 3))
        window.append(p)
    dates = pd.date_range(series["date"].iloc[-1] + pd.Timedelta(days=1), periods=days)
    recent = series.tail(30)
    return {
        "dates": fmt_dates(dates), "prices": prices, "returns_pct": returns,
        "recent_dates": fmt_dates(recent["date"]), "recent_prices": recent["close"].round(2).tolist(),
    }


@app.route("/")
def index():
    return render_template("index.html", meta=meta, max_days=MAX_DAYS)


@app.route("/api/history")
def api_history():
    return jsonify({"dates": fmt_dates(series["date"]), "values": series["close"].round(2).tolist()})


@app.route("/api/evaluation")
def api_evaluation():
    return jsonify({
        "metrics": meta["metrics"], "dates": fmt_dates(test_pred["date"]),
        "actual": test_pred["actual"].round(2).tolist(),
        "predicted": test_pred["predicted"].round(2).tolist(),
        "naive": test_pred["naive"].round(2).tolist(),
    })


@app.route("/api/forecast")
def api_forecast():
    try:
        days = int(request.args.get("days", 7))
    except ValueError:
        return jsonify({"error": "Parameter 'days' harus berupa angka."}), 400
    if not 1 <= days <= MAX_DAYS:
        return jsonify({"error": f"'days' harus antara 1 dan {MAX_DAYS}."}), 400
    return jsonify(forecast(days))


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)

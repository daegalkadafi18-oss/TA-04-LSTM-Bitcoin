"""TA-04: Latih LSTM harga Bitcoin (BTC-USD). Jalankan dari VS Code: python train.py
Hasil: model/ , data/ , figures/ (grafik PNG untuk laporan)."""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use("Agg")   # simpan grafik ke file, tidak membuka jendela
os.makedirs("figures", exist_ok=True)
_fig = [0]
def show(name):
    import matplotlib.pyplot as _plt
    _plt.tight_layout()
    _plt.savefig(f"figures/{name}.png", dpi=150)
    _plt.close()
    print("Grafik disimpan: figures/%s.png" % name)

import json, glob, warnings, zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import tensorflow as tf
import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings("ignore")
np.random.seed(42); tf.random.set_seed(42)
print("TensorFlow:", tf.__version__, "| scikit-learn:", sklearn.__version__)

try:
    import kagglehub
    DATA_DIR = kagglehub.dataset_download("sudalairajkumar/cryptocurrencypricehistory")
except Exception as e:
    print("kagglehub gagal:", e)
    DATA_DIR = "data"          # fallback: taruh coin_Bitcoin.csv di folder data/

files = [f for f in glob.glob(os.path.join(DATA_DIR, "**", "*.csv"), recursive=True)
         if os.path.basename(f).lower() == "coin_bitcoin.csv"]
assert files, "coin_Bitcoin.csv tidak ditemukan. Unduh manual dari Kaggle dan taruh di folder data/."
print("File:", files[0])

raw = pd.read_csv(files[0])
raw.columns = [c.strip().lower() for c in raw.columns]
raw["date"] = pd.to_datetime(raw["date"]).dt.normalize()
raw = raw.drop_duplicates("date").sort_values("date").set_index("date")
raw = raw[["open", "high", "low", "close", "volume", "marketcap"]]
print(raw.shape)
print(raw.head())

raw.info()
print(raw.describe())

print("Missing value:\n", raw.isna().sum())
print("\nRentang:", raw.index.min().date(), "s.d.", raw.index.max().date())
full = pd.date_range(raw.index.min(), raw.index.max(), freq="D")
print("Tanggal hilang:", len(full.difference(raw.index)))
print("Harga <= 0:", int((raw['close'] <= 0).sum()))

fig, ax = plt.subplots(1, 2, figsize=(14, 4))
raw["close"].plot(ax=ax[0]); ax[0].set_title("Harga penutupan BTC (skala linear)"); ax[0].set_ylabel("USD"); ax[0].grid(True)
raw["close"].plot(ax=ax[1], logy=True); ax[1].set_title("Harga penutupan BTC (skala log)"); ax[1].grid(True)
plt.tight_layout(); show("01_eda_harga")

r_eda = np.log(raw["close"]).diff().dropna()
fig, ax = plt.subplots(1, 2, figsize=(14, 4))
r_eda.plot(ax=ax[0], linewidth=.6); ax[0].set_title("Log-return harian"); ax[0].grid(True)
r_eda.rolling(30).std().plot(ax=ax[1]); ax[1].set_title("Volatilitas bergulir 30 hari"); ax[1].grid(True)
plt.tight_layout(); show("02_eda_return")
print(r_eda.describe())

df = raw.asfreq("D").interpolate(method="time")
assert (df["close"] > 0).all()
print("Jumlah hari:", len(df), "| sisa missing:", int(df.isna().sum().sum()))

TIMESTEPS = 30
close = df["close"]
ret = np.log(close).diff().dropna()

split = int(len(ret) * 0.8)
CLIP_LO, CLIP_HI = [float(x) for x in ret.iloc[:split].quantile([0.005, 0.995])]
ret_clip = ret.clip(CLIP_LO, CLIP_HI)
print(f"Batas clip return: [{CLIP_LO:.4f}, {CLIP_HI:.4f}]")

scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(ret_clip.iloc[:split].values.reshape(-1, 1))
scaled = scaler.transform(ret_clip.values.reshape(-1, 1))
print("Data latih:", split, "hari | Data uji:", len(ret) - split, "hari")

def make_sequences(arr, ts):
    X, y = [], []
    for i in range(ts, len(arr)):
        X.append(arr[i - ts:i, 0]); y.append(arr[i, 0])
    return np.array(X).reshape(-1, ts, 1), np.array(y)

X_train, y_train = make_sequences(scaled[:split], TIMESTEPS)
X_test,  y_test  = make_sequences(scaled[split - TIMESTEPS:], TIMESTEPS)
test_idx = ret.index[split:]
print("X_train:", X_train.shape, "| X_test:", X_test.shape)
assert len(test_idx) == len(y_test)

model = Sequential([
    Input(shape=(TIMESTEPS, 1)),
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(16, activation="relu"),
    Dense(1),
])
model.compile(optimizer="adam", loss="mean_squared_error")
model.summary()

early = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=100, batch_size=32,
                    validation_split=0.1, callbacks=[early], verbose=1)

plt.figure(figsize=(10, 4))
plt.plot(history.history["loss"], label="Loss latih")
plt.plot(history.history["val_loss"], label="Loss validasi")
plt.title("Kurva Pelatihan"); plt.xlabel("Epoch"); plt.ylabel("MSE (skala 0-1)")
plt.legend(); plt.grid(True); show("03_kurva_loss")

pred_ret = scaler.inverse_transform(model.predict(X_test, verbose=0)).flatten()
actual_ret = ret.iloc[split:].values

prev_price   = close.shift(1).loc[test_idx].values
actual_price = close.loc[test_idx].values
pred_price   = prev_price * np.exp(pred_ret)
naive_price  = prev_price

def metrics(a, p):
    return (float(np.sqrt(mean_squared_error(a, p))), float(mean_absolute_error(a, p)),
            float(np.mean(np.abs((a - p) / a)) * 100))

rmse, mae, mape = metrics(actual_price, pred_price)
n_rmse, n_mae, n_mape = metrics(actual_price, naive_price)
dir_acc = float(np.mean(np.sign(pred_ret) == np.sign(actual_ret)) * 100)
up_ratio = float(np.mean(actual_ret > 0) * 100)

print(f"LSTM  -> RMSE {rmse:,.2f} | MAE {mae:,.2f} | MAPE {mape:.2f}%")
print(f"Naif  -> RMSE {n_rmse:,.2f} | MAE {n_mae:,.2f} | MAPE {n_mape:.2f}%")
print(f"Akurasi arah (naik/turun) LSTM: {dir_acc:.2f}% | proporsi hari naik: {up_ratio:.2f}%")

fig, ax = plt.subplots(1, 2, figsize=(15, 5))
ax[0].plot(test_idx, actual_price, label="Aktual", color="blue")
ax[0].plot(test_idx, pred_price, label="Prediksi LSTM", color="orange", alpha=.8)
ax[0].set_title("Harga BTC: Aktual vs Prediksi (Data Uji)"); ax[0].set_ylabel("USD"); ax[0].legend(); ax[0].grid(True)
ax[1].plot(test_idx[-90:], actual_price[-90:], label="Aktual", color="blue")
ax[1].plot(test_idx[-90:], pred_price[-90:], label="Prediksi LSTM", color="orange")
ax[1].set_title("90 hari terakhir (zoom)"); ax[1].legend(); ax[1].grid(True)
plt.tight_layout(); show("04_aktual_vs_prediksi")

os.makedirs("model", exist_ok=True); os.makedirs("data", exist_ok=True)
model.save("model/lstm_btc.keras")
try:
    model.save("model/lstm_btc.h5")
except Exception as e:
    print("Simpan .h5 dilewati:", e)
joblib.dump(scaler, "model/scaler.joblib")

df[["close"]].reset_index().to_csv("data/bitcoin_close_clean.csv", index=False)
pd.DataFrame({"date": test_idx, "actual": actual_price, "predicted": pred_price, "naive": naive_price}
             ).to_csv("model/test_predictions.csv", index=False)

meta = {
    "timesteps": TIMESTEPS, "clip_lo": CLIP_LO, "clip_hi": CLIP_HI,
    "n_days": int(len(df)), "data_start": str(df.index.min().date()), "data_end": str(df.index.max().date()),
    "metrics": {"rmse": rmse, "mae": mae, "mape": mape, "naive_rmse": n_rmse, "naive_mae": n_mae,
                "naive_mape": n_mape, "direction_acc": dir_acc, "up_ratio": up_ratio},
}
json.dump(meta, open("model/meta.json", "w", encoding="utf-8"), indent=2)
open("model/versions.txt", "w").write(
    f"tensorflow=={tf.__version__}\nscikit-learn=={sklearn.__version__}\nnumpy=={np.__version__}\npandas=={pd.__version__}\n")

files = ["model/lstm_btc.keras", "model/lstm_btc.h5", "model/scaler.joblib", "model/meta.json",
         "model/versions.txt", "model/test_predictions.csv", "data/bitcoin_close_clean.csv"]
with zipfile.ZipFile("artifacts.zip", "w") as z:
    for p in files:
        if os.path.exists(p): z.write(p)
print(open("model/versions.txt").read())
try:
    from google.colab import files as colab_files
    colab_files.download("artifacts.zip")
except ImportError:
    pass

from tensorflow.keras.models import load_model
m2 = load_model("model/lstm_btc.keras", compile=False)
sc2 = joblib.load("model/scaler.joblib")

closes = df["close"].values
r = np.clip(np.log(closes[1:] / closes[:-1])[-TIMESTEPS:], CLIP_LO, CLIP_HI)
window = sc2.transform(r.reshape(-1, 1)).flatten().tolist()
price, out = float(closes[-1]), []
for _ in range(7):
    p = float(m2.predict(np.array(window[-TIMESTEPS:]).reshape(1, TIMESTEPS, 1), verbose=0)[0, 0])
    price *= float(np.exp(sc2.inverse_transform([[p]])[0, 0]))
    out.append(price); window.append(p)
dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=7)
print(pd.DataFrame({"tanggal": dates.date, "prediksi_harga_USD": np.round(out, 2)}))


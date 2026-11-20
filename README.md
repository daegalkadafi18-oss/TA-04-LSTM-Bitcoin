# TA-04: Peramalan Harga Penutupan Bitcoin (BTC-USD) Harian dengan LSTM

Daegal Fauzan Kadafi (301240005), Praktikum Pembelajaran Mesin

Model LSTM memprediksi harga penutupan Bitcoin hari berikutnya dari 30 hari terakhir. Model dilatih pada **log-return** harian (bukan harga mentah) karena harga Bitcoin tidak stasioner dan data uji berada di luar rentang data latih. Hasilnya dikonversi kembali ke harga: `harga_t = harga_(t-1) x exp(return_t)`. Aplikasi web Flask menampilkan grafik historis (skala linear/log), aktual vs prediksi vs baseline, dan peramalan 3-14 hari.

**Dataset (Kaggle):** https://www.kaggle.com/datasets/sudalairajkumar/cryptocurrencypricehistory
(file `coin_Bitcoin.csv`; kolom `SNo, Name, Symbol, Date, High, Low, Open, Close, Volume, Marketcap`; harian sejak 28 April 2013, berakhir sekitar Juli 2021)

## Struktur
```
train.py                         # latih model dari terminal VS Code (hasil: model/, data/, figures/)
notebook/LSTM_Bitcoin.ipynb      # versi notebook (opsional, untuk dijalankan sel demi sel)
app.py, templates/index.html     # aplikasi Flask
model/  data/  figures/          # hasil training (dibuat otomatis)
requirements.txt                 # untuk aplikasi + deploy
requirements-train.txt           # tambahan untuk training lokal (matplotlib, kagglehub, ipykernel)
Procfile, render.yaml
```

## Langkah 1: Siapkan VS Code
1. Pasang ekstensi **Python** dan **Jupyter** (Microsoft).
2. File, Open Folder, pilih folder proyek.
3. Ctrl+Shift+P, ketik **Python: Select Interpreter**, pilih yang berlokasi di `.venv`.
4. Buka terminal (Ctrl+`), pastikan prompt berawalan `(.venv)`. Jika drive C: sempit, jalankan dulu:
   `$env:TEMP="D:\temp"; $env:TMP="D:\temp"`
5. `pip install --no-cache-dir -r requirements.txt` lalu `pip install --no-cache-dir -r requirements-train.txt`

## Langkah 2: Latih model di VS Code
**Cara A (disarankan): skrip**
```powershell
python train.py
```
Dataset diunduh otomatis dari Kaggle (kagglehub). Jika gagal, unduh `coin_Bitcoin.csv` manual dari halaman dataset dan taruh di folder `data/`, lalu jalankan lagi. Hasil: `model/` (model, scaler, meta), `data/bitcoin_close_clean.csv`, dan grafik PNG di `figures/` untuk laporan. Metrik tercetak di terminal (screenshot untuk laporan).

**Cara B: notebook.** Buka `notebook/LSTM_Bitcoin.ipynb`, pilih kernel `.venv` (pojok kanan atas), klik Run All.

Setelah selesai, salin versi dari `model/versions.txt` ke `requirements.txt` (mis. `tensorflow-cpu==2.21.0`, `scikit-learn==1.9.1`) agar cocok saat deploy.

## Langkah 3: Jalankan aplikasi lokal
```powershell
python app.py        # buka http://localhost:5000
```
Endpoint: `/`, `/api/history`, `/api/evaluation`, `/api/forecast?days=7`, `/health`.

## Langkah 4: GitHub
```bash
git init && git add . && git commit -m "TA-04 LSTM Bitcoin"
git branch -M main && git remote add origin https://github.com/<username>/<repo>.git && git push -u origin main
```
Pastikan `model/` dan `data/` ikut ter-commit.

## Langkah 5: Deploy ke Render
1. render.com, New, Web Service, hubungkan repo GitHub.
2. Build `pip install -r requirements.txt`, Start `gunicorn app:app --workers 1 --timeout 120`, env `PYTHON_VERSION=3.11.9` (sudah ada di `render.yaml`).
3. Salin URL aplikasi untuk laporan. Jika RAM gratis 512 MB tidak cukup untuk TensorFlow, gunakan Railway.

## Catatan metodologi
- Split 80/20 kronologis; scaler dan batas clip return (kuantil 0,5% dan 99,5%) hanya dihitung dari data latih (tanpa data leakage).
- Evaluasi satu langkah: harga prediksi = harga aktual kemarin x exp(return prediksi). Dibandingkan dengan baseline naif (harga kemarin) dan akurasi arah naik/turun.
- Harga kripto mendekati *random walk*, jadi baseline naif sulit dikalahkan. Itu temuan yang sah untuk dibahas di laporan dan video.
- Dataset berakhir Juli 2021, sehingga peramalan di aplikasi dimulai dari tanggal setelahnya (bukan harga pasar saat ini). Hanya untuk akademik, bukan saran investasi.

## Kerangka video YouTube (5-7 menit)
1. Masalah dan dataset (45 dtk), 2. EDA: skala log, volatilitas (1 mnt), 3. Mengapa log-return + windowing 3D (1,5 mnt), 4. Hasil: LSTM vs baseline naif dan akurasi arah (1,5 mnt), 5. Demo aplikasi (1 mnt), 6. Temuan unik dan kesimpulan (45 dtk).

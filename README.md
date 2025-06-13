<!-- filepath: e:\aiml\MLFLOW\README.md -->
# ML-TRADING-SIGNALS
PROYEK Sistem Trading ML dengan Anaconda & MLflow
# Saluran Pembelajaran Mesin Perdagangan dengan MLflow

Proyek ini adalah sistem pembelajaran mesin lengkap untuk analisis data perdagangan dan prediksi sinyal menggunakan MLflow untuk pelacakan eksperimen. Dirancang khusus untuk data backtest XAUUSD dengan fokus pada probabilitas tingkat kemenangan yang tinggi, rasio risiko-imbalan yang jelas, dan stabilitas jangka panjang.

## 🎯 **Tujuan Proyek**

1. **Probabilitas Tingkat Kemenangan Tinggi**: Mengidentifikasi pola perdagangan dengan tingkat kemenangan > 60%
2. **Risiko Imbalan yang Jelas**: Mengoptimalkan rasio risiko-imbalan untuk profitabilitas maksimum
3. **Stabilitas Jangka Panjang**: Model yang konsisten dan andal
4. **Keluaran Sinyal Berkualitas**: Menghasilkan sinyal perdagangan yang dapat ditindaklanjuti

## ✅ **Status Saat Ini**

- ✅ **Saluran Lanjutan Diperbaiki**: Semua kesalahan indentasi telah diselesaikan
- ✅ **UI MLflow Berjalan**: Pelacakan eksperimen di http://127.0.0.1:5000
- ✅ **Beberapa Model**: Random Forest, Gradient Boosting, Regresi Logistik
- ✅ **Rekayasa Fitur**: 12+ fitur khusus perdagangan telah diimplementasikan
- ✅ **Pelacakan Eksperimen**: Integrasi MLflow penuh berfungsi
- ✅ **Skrip yang Mudah Digunakan**: Alat otomatisasi yang ramah bagi pemula

## 🚀 **Mulai Cepat - Pilih Metode Anda**

### Metode 1: File Batch Windows (Paling Mudah untuk Pemula)
```batch
# Klik dua kali file ini di Windows Explorer
start_experiments.bat
```

### Metode 2: Skrip Python Interaktif
```bash
python run_experiments.py
```

### Metode 3: Eksekusi Saluran Langsung
```bash
# Saluran sederhana (ML dasar)
python simple_ml_pipeline.py

# Saluran lanjutan dengan MLflow
python -c "from advanced_ml_pipeline import AdvancedTradingPipeline; p=AdvancedTradingPipeline(); p.run_complete_pipeline(10)"
```

### Metode 4: Periksa Status Proyek
```bash
python check_status.py
```

## 📊 **Lihat Hasil Anda**

1. **Mulai UI MLflow** (jika belum berjalan):
   ```bash
   python -m mlflow ui --port 5000
   ```

2. **Buka di browser**: http://127.0.0.1:5000

3. **Jelajahi eksperimen Anda**:
   - Bandingkan kinerja model
   - Lihat pentingnya fitur
   - Analisis sinyal perdagangan
   - Lacak riwayat eksperimen

### **Menjalankan Saluran Lengkap**
```bash
python main.py
```

### **Menjalankan MLflow UI**
```bash
mlflow ui --backend-store-uri ./mlruns --port 5000
```
Buka browser ke `http://localhost:5000` untuk melihat pelacakan eksperimen.

## 📊 **Fitur yang Dibuat**

### **1. Indikator Teknis**
- Simple Moving Average (SMA): Periode 5, 10, 20, 50
- Exponential Moving Average (EMA): Periode 12, 26
- RSI (Relative Strength Index): Periode 14
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands dengan deviasi standar 2

### **2. Fitur Berbasis Waktu**
- Pengkodean jam, hari dalam seminggu, bulan
- Pengkodean siklus (transformasi sin/cos)
- Indikator sesi perdagangan (Asia, Eropa, AS)
- Periode tumpang tindih pasar

### **3. Fitur Risiko**
- Volatilitas historis (periode 5, 10, 20)
- Statistik MAE/MFE
- Rasio risiko-imbalan
- Skewness dan kurtosis

### **4. Fitur Statistik**
- Statistik bergulir (rata-rata, median, std, min, maks)
- Fitur jeda (periode 1, 2, 3, 5, 10)
- Fitur kuantil (25%, 75%)

### **5. Fitur Interaksi**
- Interaksi harga-volume
- Interaksi sesi-volatilitas
- Interaksi waktu-harga

## 🤖 **Model yang Digunakan**

1. **XGBoost**: Gradient boosting dengan optimasi untuk perdagangan
2. **LightGBM**: Gradient boosting cepat dengan efisiensi memori
3. **Random Forest**: Model ansambel untuk perbandingan dasar

Semua model dilacak menggunakan MLflow dengan:
- Pencatatan parameter
- Pelacakan metrik
- Pembuatan versi model
- Analisis pentingnya fitur

## 📈 **Metrik Evaluasi**

### **Metrik Klasifikasi**
- Akurasi, Presisi, Perolehan Kembali, Skor-F1
- AUC-ROC untuk prediksi probabilitas

### **Metrik Khusus Perdagangan**
- **Tingkat Kemenangan**: Persentase perdagangan yang menguntungkan
- **Faktor Keuntungan**: Rasio total keuntungan vs total kerugian
- **Rasio Risiko-Imbalan**: Rata-rata keuntungan vs rata-rata kerugian
- **Penarikan Maksimum**: Kerugian maksimum dari puncak
- **Rasio Sharpe**: Imbalan yang disesuaikan dengan risiko
- **Harapan**: Nilai yang diharapkan per perdagangan

### **Metrik Kualitas Sinyal**
- Presisi pada berbagai ambang probabilitas
- Frekuensi dan cakupan sinyal
- Metrik stabilitas lintas periode waktu

## 🎯 **Variabel Target**

1. **is_profitable**: Perdagangan menghasilkan keuntungan (biner)
2. **is_winning_trade**: Keuntungan perdagangan > 10 pips (biner)

## 📁 **Struktur Data**

### **Data Masukan (dataBT)**
- Simbol, Stempel Waktu, Jenis (BELI/JUAL)
- HargaBuka, HargaTutup, Volume
- Keuntungan, MAE_pips, MFE_pips
- AlasanKeluar, Sesi perdagangan

### **Fitur yang Dihasilkan**
- 100+ indikator teknis
- Fitur berbasis waktu
- Metrik risiko
- Fitur statistik

## 🔧 **Konfigurasi Model**

```yaml
models:
  - name: "xgboost"
    params:
      n_estimators: [100, 200, 300]
      max_depth: [3, 5, 7]
      learning_rate: [0.01, 0.1, 0.2]
  
  - name: "lightgbm"
    params:
      n_estimators: [100, 200, 300]
      max_depth: [3, 5, 7]
      learning_rate: [0.01, 0.1, 0.2]
```

## 📊 **Pelacakan MLflow**

MLflow melacak semua eksperimen dengan informasi:

### **Parameter**
- Hiperparameter model
- Pengaturan rekayasa fitur
- Parameter pra-pemrosesan data

### **Metrik**
- Metrik pelatihan dan validasi
- Metrik kinerja perdagangan
- Skor validasi silang

### **Artefak**
- Model terlatih
- Plot pentingnya fitur
- Laporan kinerja

### **Registri Model**
- Pembuatan versi model
- Penerapan pentahapan dan produksi
- Perbandingan model

## 🔍 **Analisis dan Interpretasi**

### **Pentingnya Fitur**
- Mengidentifikasi fitur yang paling berpengaruh
- Menganalisis kontribusi setiap kategori fitur
- Menghilangkan fitur yang tidak relevan

### **Kinerja Model**
- Membandingkan kinerja antar model
- Menganalisis stabilitas lintas waktu
- Mengidentifikasi overfitting

### **Analisis Sinyal Perdagangan**
- Kualitas sinyal pada berbagai ambang batas
- Pertukaran frekuensi sinyal vs akurasi
- Metrik kinerja yang disesuaikan dengan risiko

## 📋 **Praktik Terbaik**

### **Manajemen Data**
1. **Caching Data**: Caching otomatis untuk iterasi yang lebih cepat
2. **Validasi Data**: Pemeriksaan kualitas untuk integritas data
3. **Pembuatan Versi Fitur**: Melacak perubahan rekayasa fitur

### **Pengembangan Model**
1. **Validasi Sadar Waktu**: TimeSeriesSplit untuk evaluasi yang realistis
2. **Pemilihan Fitur**: Analisis pentingnya fitur secara sistematis
3. **Penyetelan Hiperparameter**: Pencarian grid dengan validasi silang

### **Manajemen Risiko**
1. **Pemantauan Penarikan**: Pelacakan berkelanjutan terhadap penarikan maksimum
2. **Penyaringan Sinyal**: Ambang kualitas untuk pembuatan sinyal
3. **Pemantauan Kinerja**: Evaluasi kinerja model secara berkala

## 🚨 **Peringatan dan Risiko**

1. **Kinerja Masa Lalu**: Hasil historis tidak menjamin kinerja di masa depan
2. **Kondisi Pasar**: Kinerja model dapat berubah seiring kondisi pasar
3. **Manajemen Risiko**: Selalu gunakan stop loss dan ukuran posisi yang tepat
4. **Overfitting**: Pantau tanda-tanda overfitting pada data yang tidak terlihat

## 📈 **Rekomendasi Penggunaan**

### **Untuk Tingkat Kemenangan Tinggi**
- Gunakan ambang probabilitas > 0,7 untuk sinyal dengan keyakinan tinggi
- Fokus pada periode tumpang tindih sesi (Eropa-AS)
- Pertimbangkan konfirmasi volume untuk validasi sinyal

### **Untuk Manajemen Risiko**
- Tetapkan batas penarikan maksimum 10%
- Gunakan rasio risiko-imbalan minimal 2:1
- Terapkan ukuran posisi berdasarkan keyakinan sinyal

### **Untuk Stabilitas Jangka Panjang**
- Latih ulang model setiap 3-6 bulan
- Pantau penyimpangan fitur dan perubahan rezim pasar
- Terapkan prediksi ansambel untuk stabilitas

## 🔄 **Pemantauan dan Pemeliharaan**

1. **Pelatihan Ulang Reguler**: Jadwalkan pelatihan ulang model
2. **Pemantauan Kinerja**: Lacak kinerja langsung vs backtest
3. **Pemantauan Fitur**: Pantau penyimpangan fitur
4. **Kualitas Data**: Pemeriksaan kualitas data berkelanjutan

## 📞 **Dukungan dan Pengembangan**

Untuk pengembangan lebih lanjut:
1. Implementasi saluran prediksi waktu-nyata
2. Integrasi dengan platform perdagangan
3. Metode ansambel lanjutan
4. Analisis multi-kerangka waktu
5. Optimasi portofolio

---

**Catatan**: Proyek ini dirancang untuk tujuan pendidikan dan penelitian. Selalu gunakan manajemen risiko yang tepat dalam perdagangan nyata.

# ML-TRADING-SIGNALS
PROYEK Sistem Trading ML dengan Anaconda &amp; MLflow

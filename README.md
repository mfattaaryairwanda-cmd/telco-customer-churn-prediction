# Prediksi Customer Churn Telco

Proyek *end-to-end* klasifikasi biner untuk memprediksi pelanggan telekomunikasi yang berhenti berlangganan (*churn*): pemahaman bisnis, EDA, pembersihan data, rekayasa fitur, dan perbandingan enam algoritma pada tiga skenario fitur.

📄 **Laporan lengkap:** [`reports/laporan_proyek.md`](reports/laporan_proyek.md)

## Ringkasan

| Item | Keterangan |
|---|---|
| Dataset | [IBM Telco Customer Churn (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn), 7.043 pelanggan, 21 kolom |
| Target | `Churn` (Yes = 1, No = 0), proporsi churn 26,54% |
| Pembagian data | 80:20 terstratifikasi, `random_state=42` → 5.634 train, 1.409 test (374 churn) |
| Algoritma | KNN, SVM, Logistic Regression, Decision Tree, XGBoost, Random Forest |
| Tuning | `RandomizedSearchCV`, 50 kombinasi, 5-fold CV, skor F1 |
| Metrik | Precision, Recall, F1, ROC-AUC, Average Precision (AP) |

## Hasil Utama (data test)

| Skenario | Fitur | XGBoost (ROC-AUC / AP) | Logistic Regression (ROC-AUC / AP) |
|---|---|---|---|
| Phase 1 | 19 fitur asli | 0,846 / 0,654 | 0,840 / 0,633 |
| Phase 2 | 10 fitur hasil rekayasa | 0,839 / 0,641 | 0,834 / 0,607 |
| Phase 3 | Phase 2 + polynomial | 0,838 / 0,639 | 0,834 / 0,607 |

1. **Pelanggan paling berisiko churn:** kontrak bulanan (42,7%), internet fiber optic (41,9%), pembayaran electronic check (45,3%), dan berlangganan 0-12 bulan (47,4%).
2. **Model teratas hampir setara.** Selisih antar XGBoost, Random Forest, dan Logistic Regression berada dalam batas variasi sampel, sehingga tidak ada pemenang yang pasti.
3. **Rekayasa fitur menyederhanakan, bukan meningkatkan.** Fitur berkurang dari 19 menjadi 10 dengan skor nyaris setara.
4. **KNN overfit** pada Phase 2 dan 3 (F1 train 0,993 vs test sekitar 0,57), dan **SVM** paling lemah pada AP.

## Struktur Repo

```
├── data/
│   ├── raw/Telco-Customer-Churn.csv     data asli
│   └── processed/                       hasil olahan (tidak di-commit)
├── images/                              grafik untuk laporan
├── models/                              model terlatih .pkl (tidak di-commit)
├── notebooks/
├── reports/laporan_proyek.md
├── requirements.txt
└── README.md
```

## Alur Notebook

| No | Notebook | Isi | Keluaran |
|---|---|---|---|
| 01 | business_understanding | Masalah, tujuan, metrik | - |
| 02 | eda | Struktur, distribusi, komposisi churn | - |
| 03 | data_cleaning | Tipe data, missing value, duplikasi | `telco_clean.csv` |
| 04 | preprocessing | Split train/test (19 fitur) | `data_split.pkl` |
| 05 | model_experiments (phase 1) | Enam model, semua fitur | `best_models_1.pkl` |
| 06 | Feature_engineering | Dokumentasi fitur baru dan alasannya | - |
| 07 | preprocess_Feature_Engineer | Pembuatan fitur dan split (10 fitur) | `data_split_FE.pkl` |
| 08 | model_experiments (phase 2) | Enam model, fitur hasil rekayasa | `best_models_2.pkl` |
| 09 | model_experiments (phase 3) | Phase 2 + polynomial features | `best_models_3.pkl` |

Jalankan berurutan dari 01 ke 09.

## Cara Menjalankan

```bash
git clone https://github.com/[USERNAME]/[NAMA-REPO].git
cd [NAMA-REPO]
python -m venv .venv
.venv\Scripts\activate            # Windows (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
jupyter lab
```

## Teknologi

Python, pandas, NumPy, scikit-learn, XGBoost, jcopml, matplotlib, seaborn, Jupyter.

## Penulis

M. Fatta Arya Irwanda • mfattaaryairwanda@gmail.com
# ⚽ Eksperimen SML: Automasi Preprocessing Dataset La Liga

![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat-square&logo=python)
![Pandas](https://img.shields.io/badge/Pandas-Data_Manipulation-150458?style=flat-square&logo=pandas)
![Scikit-Learn](https://img.shields.io/badge/Scikit_Learn-Machine_Learning-F7931E?style=flat-square&logo=scikit-learn)
![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-CI%2FCD-2088FF?style=flat-square&logo=github-actions)

## 📖 Deskripsi Proyek

Repositori ini memuat skrip otomasi preprocessing untuk dataset statistik pertandingan sepak bola La Liga (2014-2020). Proyek ini dirancang untuk membersihkan data mentah, merekayasa fitur baru, serta mentransformasi dataset agar siap digunakan untuk pemodelan Machine Learning. Selain itu, repositori ini telah dilengkapi dengan pipeline CI/CD menggunakan GitHub Actions untuk menjalankan pemrosesan data secara otomatis di cloud.

## ✨ Fitur Utama Preprocessing

Skrip [preprocessing/automate_Taufik_P.py](preprocessing/automate_Taufik_P.py) secara otomatis melakukan serangkaian transformasi data berikut:

1. 🎯 Rekayasa Fitur (Feature Engineering)
   - Membuat variabel target klasifikasi `FTR` (Full Time Result) dengan nilai `H` (Home Win), `A` (Away Win), atau `D` (Draw) berdasarkan perbandingan jumlah gol yang dicetak masing-masing tim.

2. 🧹 Pembersihan Data & Eliminasi Duplikat
   - Menghapus baris observasi yang terduplikasi secara otomatis.
   - Menangani missing values pada atribut numerik menggunakan metode median dari `SimpleImputer`.

3. 🛡️ Pencegahan Kebocoran Data (Data Leakage)
   - Menghapus atribut-atribut yang tidak bersifat prediktif di awal pertandingan, seperti skor akhir, skor babak pertama, gol kebobolan, dan rating tim, untuk memastikan model akhir tidak mengalami overfitting.

4. 🔢 Representasi Data (Encoding & Scaling)
   - Melakukan Label Encoding pada atribut kategorikal seperti nama tim (`Home Team`, `Away Team`) dan target prediksi.
   - Melakukan standardisasi distribusi pada semua atribut numerik yang tersisa menggunakan `StandardScaler`.

## 📂 Struktur Direktori Utama

```text
Eksperimen_SML_Taufik_P/
├── .github/
│   └── workflows/
│       └── preprocessing.yml
├── la-liga-2014-2020_raw/
│   └── combined_data_laliga.csv
├── la-liga-2014-2020_preprocessing/
│   └── la_liga_cleaned.csv
└── preprocessing/
    └── automate_Taufik_P.py
```

## 📁 Penjelasan File

- [preprocessing/automate_Taufik_P.py](preprocessing/automate_Taufik_P.py)
  - Skrip utama yang menjalankan seluruh tahapan preprocessing.

- [la-liga-2014-2020_raw/combined_data_laliga.csv](la-liga-2014-2020_raw/combined_data_laliga.csv)
  - File dataset mentah yang digunakan sebagai input.

- [.github/workflows/preprocessing.yml](.github/workflows/preprocessing.yml)
  - Workflow GitHub Actions untuk menjalankan preprocessing secara otomatis.

## 🧰 Prasyarat

Pastikan perangkat Anda telah menginstal:

- Python 3.12
- pip
- Git

## ⚙️ Instalasi Lokal

Ikuti langkah berikut di terminal:

```bash
git clone <url-repository>
cd Eksperimen_SML_Taufik_P
python -m venv venv
source venv/Scripts/activate
pip install --upgrade pip
pip install pandas numpy scikit-learn
```

> Jika Anda menggunakan Git Bash pada Windows, perintah aktivasi virtual environment biasanya:
>
> ```bash
> source venv/Scripts/activate
> ```

## ▶️ Cara Penggunaan (Lokal)

### 1. Menjalankan preprocessing

Gunakan perintah berikut dari root repository:

```bash
python preprocessing/automate_Taufik_P.py --input la-liga-2014-2020_raw/combined_data_laliga.csv --output la-liga-2014-2020_preprocessing/la_liga_cleaned.csv
```

### 2. Hasil output

Skrip akan menghasilkan file CSV hasil preprocessing pada path:

```text
la-liga-2014-2020_preprocessing/la_liga_cleaned.csv
```

### 3. Alur yang dijalankan

Skrip akan melakukan tahapan berikut:

1. Memuat dataset mentah.
2. Membuat label target FTR.
3. Menghapus data duplikat.
4. Menghapus kolom yang tidak sesuai untuk prediksi.
5. Mengisi nilai kosong pada fitur numerik.
6. Melakukan encoding pada data kategorikal.
7. Memisahkan fitur dan target.
8. Melakukan standardisasi fitur numerik.
9. Menyimpan hasil preprocessing ke file CSV.

## 🔄 Otomasi CI/CD Pipeline

Repository ini juga dilengkapi dengan pipeline otomatis menggunakan GitHub Actions melalui file [.github/workflows/preprocessing.yml](.github/workflows/preprocessing.yml).

### Trigger pipeline

Pipeline akan berjalan saat:

- ada push ke branch `main`
- dilakukan manual run melalui GitHub Actions (`workflow_dispatch`)

### Langkah kerja pipeline

1. Checkout repository
2. Setup Python 3.12
3. Install dependency:
   - pandas
   - numpy
   - scikit-learn
4. Jalankan skrip preprocessing
5. Upload hasil dataset sebagai artifact

### Hasil pipeline

Dataset hasil preprocessing akan dihasilkan secara otomatis dan disimpan sebagai artifact pada GitHub Actions sehingga dapat diunduh langsung setelah pipeline selesai.

## 📊 Output yang Dihasilkan

File hasil preprocessing berisi:

- fitur hasil transformasi,
- target `FTR`,
- data yang sudah dibersihkan dan siap digunakan untuk pelatihan model.

## 💡 Catatan Penting

- Pastikan file input tersedia sebelum menjalankan skrip.
- Jika path file input atau output berbeda, sesuaikan argumen pada perintah.
- Jika terjadi error, cek pesan yang ditampilkan oleh skrip untuk mengetahui bagian proses yang gagal.

## 🤖 Kesimpulan

Repository ini memberikan solusi sederhana namun efektif untuk otomatisasi preprocessing data sepak bola La Liga, baik secara lokal maupun melalui pipeline CI/CD di GitHub Actions.

## 👤 Author
**Taufik Pathurrohman**
* 🎓 **Sistem Informasi (FST) – Universitas Terbuka (Bandung)**
* 🤖 **AI & Machine Learning Enthusiast** 
* 🚀 **Pijak in Collaboration with IBM SkillsBuild**
* 🐙 **GitHub:** [@tpathurrohman612](https://github.com/tpathurrohman612)

---
*Jika Anda memiliki pertanyaan atau saran terkait proyek ini, jangan ragu untuk menghubungi atau membuat issue di repositori ini!*

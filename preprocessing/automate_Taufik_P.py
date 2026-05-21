import pandas as pd
import numpy as np
import os
import argparse
import warnings
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# Mengabaikan peringatan (warnings) guna menjaga kebersihan output terminal
warnings.filterwarnings('ignore')

def run_preprocessing(input_path, output_path):
    try:
        # Validasi file input
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"File input tidak ditemukan: {input_path}")
        
        # Memuat Dataset
        print(f"Memuat dataset dari: {input_path}")
        df = pd.read_csv(input_path)
        observasi_awal = df.shape[0]
        
        if observasi_awal == 0:
            raise ValueError("Dataset kosong atau tidak memiliki data.")
        
        print(f"Memuat dataset telah selesai. Jumlah data awal: {observasi_awal} observasi.")
        
        # Validasi kolom penting
        kolom_penting = ['Home Team Goals Scored', 'Away Team Goals Scored']
        kolom_hilang = [col for col in kolom_penting if col not in df.columns]
        if kolom_hilang:
            raise ValueError(f"Kolom penting tidak ditemukan: {kolom_hilang}")

        # Rekayasa Fitur (Target FTR)
        def get_ftr(row):
            if row['Home Team Goals Scored'] > row['Away Team Goals Scored']:
                return 'H'
            elif row['Home Team Goals Scored'] < row['Away Team Goals Scored']:
                return 'A'
            else:
                return 'D'
        df['FTR'] = df.apply(get_ftr, axis=1)
                
        # Eliminasi Data Duplikat
        df.drop_duplicates(inplace=True)
        print(f"Jumlah observasi duplikat yang telah dieliminasi: {observasi_awal - df.shape[0]}")

        # Pencegahan Kebocoran Data & Pembersihan Atribut Non-Prediktif
        kolom_bocor = [
            'Home Team Goals Scored', 'Away Team Goals Scored',
            'Home Team Goals Conceeded', 'Away Team Goals Conceeded',
            'Match Excitement', 'Home Team Rating', 'Away Team Rating',
            'Unnamed: 0', 'Score', 'Half Time Score', 'Year', 'year', 'Date',
            'Match ID', 'Referee'
        ]

        kolom_dihapus = [col for col in kolom_bocor if col in df.columns]
        df.drop(columns=kolom_dihapus, inplace=True, errors='ignore')
        print(f"Jumlah atribut penyebab kebocoran data (leakage) yang dihapus: {len(kolom_dihapus)}")

        # Penanganan Nilai Kosong (Missing Values)
        kolom_numerik = df.select_dtypes(include=[np.number]).columns
        imputer = SimpleImputer(strategy='median')
        df[kolom_numerik] = imputer.fit_transform(df[kolom_numerik])
        print("Penanganan nilai kosong pada atribut numerik telah selesai.")

        # Representasi Data Kategorikal (Encoding)
        le_team = LabelEncoder()
        le_target = LabelEncoder()

        if 'Home Team' in df.columns and 'Away Team' in df.columns:
            all_entitas_team = pd.concat([df['Home Team'], df['Away Team']]).unique()
            le_team.fit(all_entitas_team)
            df['Home Team'] = le_team.transform(df['Home Team'])
            df['Away Team'] = le_team.transform(df['Away Team'])

        if 'FTR' in df.columns:
            df['FTR'] = le_target.fit_transform(df['FTR'])
        print("Proses label encoding untuk atribut kategorikal telah selesai.")
        
        # Pemisahan Fitur (X) dan Target (y)
        X = df.drop(columns=['FTR'])
        y = df['FTR']
        # Memastikan hanya atribut numerik yang diproses pada tahap standardisasi
        X = X.select_dtypes(include=[np.number])
        print(f"Pemisahan fitur dan target selesai. Jumlah fitur prediktif: {X.shape[1]}")

        # Standardisasi Atribut (Feature Scaling)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_final = pd.DataFrame(X_scaled, columns=X.columns)
        print("Proses standardisasi atribut telah selesai dieksekusi.")

        # Pengecekan kolom non-numerik yang tersisa
        kolom_non_numerik = df.select_dtypes(exclude=[np.number]).columns.tolist()
        if kolom_non_numerik:
            print(f"Peringatan: Kolom non-numerik yang tersisa: {kolom_non_numerik}")
        
        # Integrasi dan Penyimpanan Dataset
        df_hasil_akhir = X_final.copy()
        df_hasil_akhir['FTR'] = y.values

        # Membuat direktori
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        df_hasil_akhir.to_csv(output_path, index=False)
        print(f"Dataset hasil preprocessing berhasil disimpan di: {output_path}")
        
        # Ringkasan Hasil Akhir
        print("\n" + "="*45)
        print("RINGKASAN HASIL PREPROCESSING")
        print("="*45)
        print(f"Jumlah observasi awal: {observasi_awal}")
        print(f"Jumlah observasi akhir: {df_hasil_akhir.shape[0]}")
        print(f"Jumlah fitur akhir: {df_hasil_akhir.shape[1] - 1}")
        print(f"Nama fitur: {list(X_final.columns)}")
        print(f"Distribusi target (FTR): {dict(df_hasil_akhir['FTR'].value_counts())}")
        print("="*45 + "\n")
        
        # Return informasi statistik
        return {
            'success': True,
            'observasi_awal': observasi_awal,
            'observasi_akhir': df_hasil_akhir.shape[0],
            'jumlah_fitur': df_hasil_akhir.shape[1] - 1,
            'output_path': output_path
        }
    
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return {'success': False, 'error': str(e)}
    except ValueError as e:
        print(f"ERROR: {e}")
        return {'success': False, 'error': str(e)}
    except KeyError as e:
        print(f"ERROR: Kolom tidak ditemukan: {e}")
        return {'success': False, 'error': f'Kolom tidak ditemukan: {e}'}
    except Exception as e:
        print(f"ERROR: Terjadi kesalahan yang tidak terduga: {e}")
        return {'success': False, 'error': str(e)}

# Eksekusi Program Utama dan Pengaturan Dinamis
if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description='Otomatisasi Preprocessing Dataset Sepak Bola')
        parser.add_argument('--input', type=str, default='../la-liga-2014-2020_raw/combined_data_laliga.csv', 
                            help='Jalur akses menuju file dataset mentah')
        parser.add_argument('--output', type=str, default='./la-liga-2014-2020_preprocessing/la_liga_cleaned.csv', 
                            help='Jalur penyimpanan untuk file hasil preprocessing')
        
        args = parser.parse_args()
        
        print(f"Input path: {args.input}")
        print(f"Output path: {args.output}\n")
        
        # Menjalankan fungsi utama dengan penanganan hasil
        hasil = run_preprocessing(args.input, args.output)
        
        if hasil and hasil.get('success'):
            print("Preprocessing berhasil diselesaikan!")
            exit(0)
        else:
            error_msg = hasil.get('error', 'Kesalahan tidak diketahui') if hasil else 'Kesalahan tidak diketahui'
            print(f"Preprocessing gagal: {error_msg}")
            exit(1)
    
    except Exception as e:
        print(f"ERROR: Kesalahan fatal saat menjalankan program: {e}")
        exit(1)
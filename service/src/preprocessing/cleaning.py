import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

class DataCleaning:
    def __init__(self, file_path='data/data_fitur.csv'):
        self.file_path = file_path
        self.df = None
        self.cleaned_df = None

    def load_data(self):
        """Load data from csv"""
        try:
            self.df = pd.read_csv(self.file_path)
            print(f"Data berhasil dimuat. Shape: {self.df.head}")
            # print(self.df.head())
        except FileNotFoundError:
            print(f"File {self.file_path} tidak ditemukan")
            return None
        
    def select_features(self):
        """Select features based on requirements"""
        if self.df is None:
            print("Data belum dimuat")
            return None
        
        selected_features = [
            'PRODI', 'STATUS', 'ASAL_SEKOLAH', 'SEKOLAH_JURUSAN', 'ALAMAT_SEKOLAH',
            'KERJA_AYAH', 'KETERANGAN_AYAH', 'JENJANG_SEBELUMNYA', 'JENIS_SELEKSI',
            'ASAL_KOTA', 'ASAL_PROVINSI', 'DEPARTEMEN', 'PENGHASILAN_KATEGORI',
            'TAHUN ANGKATAN', 'IPK_2023_GENAP', 'IPK_2023_GANJIL', 'IPK_2022_GENAP',
            'IPK_2022_GANJIL', 'IPK_2021_GENAP', 'IPK_2021_GANJIL',
            'BEASISWA_LAIN', 'BEASISWA_KIPK', 'PRESTASI', 'Prosentase_Kehadiran'
        ]

        # filter hanya kolom yang ada di dataset
        available_features = [col for col in selected_features if col in self.df.columns]
        self.df = self.df[available_features]
        print(f"Selected {len(available_features)} features")
        return self.df
    
    def handle_missing_values(self):
        """Handle missing values in the dataset"""
        if self.df is None:
            print("Data belum dimuat!")
            return None
        
        print("Menangani missing values...")

        # Informasi missing values sebelum cleaning
        missing_before = self.df.isnull().sum().sum()
        print(f"Total missing values sebelum cleaning: {missing_before}")

        # 1. Kolom IPK, ganti NaN dengan 0
        ipk_columns = [col for col in self.df.columns if 'IPK' in col]
        for col in ipk_columns:
            self.df[col] = self.df[col].fillna(0)

        # 2. Untuk kolom kategorikal, ganti dengan 'Tidak diketahui'
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            self.df[col] = self.df[col].fillna('Tidak Diketahui')

        # 3. Untuk kolom numerik selain IPK, ganti dengan median
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if 'IPK' not in col]
        for col in numeric_cols:
            self.df[col] = self.df[col].fillna(self.df[col].median())

        # Missing values after cleaning
        missing_after = self.df.isnull().sum().sum()
        print(f"Total missing values setelah cleaning: {missing_after}")
        return self.df
    
    def handle_outliers(self):
        """Handle outliers in numerical columns"""
        if self.df is None:
            print("Data belum dimuat")
            return None
        
        print("Menangani outliers...")

        # Only process numerical columns that are not IDs or similar
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns

        for col in numerical_cols:
            if col != 'TAHUN ANGKATAN':
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1

                # Define bounds
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                # Cap outliers
                self.df[col] = np.where(self.df[col] < lower_bound, lower_bound, self.df[col])
                self.df[col] = np.where(self.df[col] > upper_bound, upper_bound, self.df[col])

                return self.df
            
            def create_ipk_rata_rata(self):
                """Create IPK_RATA_RATA columns for all IPK columns"""
                if self.df is None:
                    print("Data belum dimuat")
                    return None
                
                print("Membuat kolom IPK_RATA_RATA...")

                # Find all IPK columns
                ipk_columns = [col for col in self.df.columns if 'IPK' in col]

                if ipk_columns:
                    ipk_data = self.df[ipk_columns]
                    ipk_data_replaced = ipk_data.replace(0, np.nan)
                    self.df['IPK_RATA_RATA'] = ipk_data_replaced.mean(axis = 1)
                    self.df['IPK_RATA_RATA']= self.df['IPK_RATA_RATA'].fillna(0)
                    print(f"Kolom IPK_RATA_RATA berhasil dibuat. Rata-rata: {self.df['IPK_RATA_RATA'].mean():.2f}")
                else:
                    print("Tidak ditemukan kolom IPK")
                    self.df['IPK_RATA_RATA'] = 0

                return self.df 
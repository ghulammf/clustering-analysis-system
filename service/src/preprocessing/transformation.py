import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import joblib
import os


class DataTransformation:
    def __init__(self, df=None):
        # self.file_path = file_path
        self.df = df
        self.transformed_df = None
        self.label_encoders = {}
        self.onehot_encoders = {}
        self.messages = []

    def load_cleaned_data(self, df):
        """Load cleaned data"""
        if df is None:
            self.messages.append("Data kosong!")
            return None

        self.df = df.copy()
        self.messages.append(f"Memuat data cleaning")
        return self.df

    def encode_categorical_variables(self):
        """Encode categorical variables but keep original columns"""
        if self.df is None:
            print("Data belum dimuat!")
            return None

        self.messages.append("Encoding variabel kategorikal")

        # Identify categorical columns (excluding IPK_RATA_RATA and BEASISWA_IPK)
        categorical_cols = self.df.select_dtypes(include=["object"]).columns.tolist()

        # Juga tambahkan kolom yang perlu di-encode
        categorical_cols_to_encode = [
            "ASAL_PROVINSI",
            "ASAL_KOTA",
            "KERJA_AYAH",
            "PENGHASILAN_KATEGORI",
            "PRODI",
            "SEKOLAH_JURUSAN",
            "ASAL_SEKOLAH",
            "JENIS_SELEKSI",
        ]

        for col in categorical_cols_to_encode:
            if col in self.df.columns:
                le = LabelEncoder()
                # Handle missing values before encoding
                self.df[col] = self.df[col].fillna("Unknown")
                self.df[col + "_ENCODED"] = le.fit_transform(self.df[col].astype(str))
                self.label_encoders[col] = le
                print(f"{col}: {len(le.classes_)} categories")

        return self.df

    def create_additional_features_for_analysis_only(self):
        """Create additional features for analysis, but we'll keep them separate"""
        self.messages.append("Membuat fitur analisis")

        # Fitur biner untuk analisis
        if "ASAL_PROVINSI" in self.df.columns:
            self.df["ASAL_JAWA_TIMUR"] = (
                self.df["ASAL_PROVINSI"] == "Jawa Timur"
            ).astype(int)

        if "JENIS_SELEKSI" in self.df.columns:
            self.df["SELEKSI_PRESTASI"] = (
                self.df["JENIS_SELEKSI"]
                .str.contains("Prestasi", case=False, na=False)
                .astype(int)
            )

        if "ASAL_SEKOLAH" in self.df.columns:
            self.df["SEKOLAH_NEGERI"] = (
                self.df["ASAL_SEKOLAH"]
                .str.contains("SMAN|SMKN|MAN", case=False, na=False)
                .astype(int)
            )

        # Fitur interaksi untuk analisis
        if all(
            col in self.df.columns
            for col in ["PENGHASILAN_KATEGORI_ENCODED", "BEASISWA_KIPK"]
        ):
            self.df["PENGHASILAN_BEASISWA_INTERACTION"] = (
                self.df["PENGHASILAN_KATEGORI_ENCODED"] * self.df["BEASISWA_KIPK"]
            )

        if all(
            col in self.df.columns for col in ["SEKOLAH_NEGERI", "SELEKSI_PRESTASI"]
        ):
            self.df["SEKOLAH_SELEKSI_INTERACTION"] = (
                self.df["SEKOLAH_NEGERI"] * self.df["SELEKSI_PRESTASI"]
            )

        # Fitur binned untuk analisis
        if "IPK_RATA_RATA" in self.df.columns:
            bins = [0, 2.0, 2.5, 3.0, 3.5, 4.0]
            labels = ["Sangat Rendah", "Rendah", "Cukup", "Baik", "Sangat Baik"]
            self.df["IPK_KATEGORI"] = pd.cut(
                self.df["IPK_RATA_RATA"], bins=bins, labels=labels, include_lowest=True
            )
            le_ipk = LabelEncoder()
            self.df["IPK_KATEGORI_ENCODED"] = le_ipk.fit_transform(
                self.df["IPK_KATEGORI"].astype(str)
            )
            self.label_encoders["IPK_KATEGORI"] = le_ipk

        # Aggregasi untuk analisis
        if "ASAL_SEKOLAH" in self.df.columns:
            school_counts = self.df["ASAL_SEKOLAH"].value_counts()
            self.df["SEKOLAH_JUMLAH_SISWA"] = self.df["ASAL_SEKOLAH"].map(school_counts)

        if "ASAL_KOTA" in self.df.columns:
            city_counts = self.df["ASAL_KOTA"].value_counts()
            self.df["KOTA_JUMLAH_SISWA"] = self.df["ASAL_KOTA"].map(city_counts)

        print(f"Total kolom setelah fitur tambahan: {len(self.df.columns)}")

        return self.df

    def create_final_datasets(self):
        """Create three separate datasets for different purposes"""
        if self.df is None:
            print("Data belum dimuat!")
            return None

        self.messages.append("Membuat dataset akhir")

        # 1. Dataset dengan 10 fitur asli (untuk referensi/display)
        original_features = [
            "ASAL_PROVINSI",
            "ASAL_KOTA",
            "KERJA_AYAH",
            "PENGHASILAN_KATEGORI",
            "BEASISWA_KIPK",
            "PRODI",
            "SEKOLAH_JURUSAN",
            "ASAL_SEKOLAH",
            "JENIS_SELEKSI",
            "IPK_RATA_RATA",
        ]

        dataset_original = self.df[original_features].copy()
        # print(f"Dataset 1 (10 fitur asli): {dataset_original.shape}")

        # 2. Dataset dengan 10 fitur encoded (untuk modeling)
        encoded_features = [
            "ASAL_PROVINSI_ENCODED",
            "ASAL_KOTA_ENCODED",
            "KERJA_AYAH_ENCODED",
            "PENGHASILAN_KATEGORI_ENCODED",
            "BEASISWA_KIPK",
            "PRODI_ENCODED",
            "SEKOLAH_JURUSAN_ENCODED",
            "ASAL_SEKOLAH_ENCODED",
            "JENIS_SELEKSI_ENCODED",
            "IPK_RATA_RATA",
        ]

        # Hanya ambil yang ada
        existing_encoded = [f for f in encoded_features if f in self.df.columns]
        dataset_encoded = self.df[existing_encoded].copy()
        # print(f"Dataset 2 (10 fitur encoded): {dataset_encoded.shape}")

        # 3. Dataset dengan semua fitur (untuk analisis lengkap)
        dataset_full = self.df.copy()
        # print(f"Dataset 3 (semua fitur): {dataset_full.shape}")

        # Simpan semua dataset
        self.transformed_df = dataset_encoded  # Untuk lanjut ke normalization

        # return {
        #     'original': dataset_original,
        #     'encoded': dataset_encoded,
        #     'full': dataset_full
        # }
        return self.transformed_df

    def save_transformed_data(self, datasets):
        """Save all transformed datasets"""
        print("\nMenyimpan dataset...")

        # Save original dataset (10 fitur asli)
        # datasets['original'].to_csv('transformed_original_10features.csv', index=False)
        # print(f"✅ Dataset asli (10 fitur) disimpan ke transformed_original_10features.csv")

        # Save encoded dataset (10 fitur encoded)
        # datasets['encoded'].to_csv('transformed_encoded_10features.csv', index=False)
        # print(f"✅ Dataset encoded (10 fitur) disimpan ke transformed_encoded_10features.csv")

        # Save full dataset
        # datasets['full'].to_csv('transformed_full_features.csv', index=False)
        # print(f"✅ Dataset lengkap disimpan ke transformed_full_features.csv")

        # Save encoders
        if self.label_encoders:
            joblib.dump(self.label_encoders, "label_encoders.pkl")
            print(f"✅ Label encoders disimpan ke label_encoders.pkl")

        return True

    def run_full_transformation(self, df, return_result=False):
        """Run the complete data transformation pipeline"""
        self.messages.append("MEMULAI TRANSFORMASI DATA")

        self.load_cleaned_data(df)

        if self.df is not None:
            self.encode_categorical_variables()
            self.create_additional_features_for_analysis_only()
            self.create_final_datasets()
            # datasets = self.create_final_datasets()
            # self.save_transformed_data(datasets)

            print("DATA TRANSFORMATION SELESAI")

            # return datasets['encoded']  # Return encoded dataset untuk normalization
            if return_result:
                return self.transformed_df

            return self.response()
        else:
            print("Gagal memuat data untuk transformasi!")
            return None

    def response(self):
        return {
            "status": "success",
            "messages": self.messages,
            "data_preview": self.transformed_df.head(5).to_dict(orient="records"),
        }

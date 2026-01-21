import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os


class DataCleaning:
    def __init__(self, file_path="data/data_fitur.csv"):
        self.file_path = file_path
        self.df = None
        self.cleaned_df = None
        self.messages = []

    def load_data(self):
        """Load data from csv"""
        try:
            self.df = pd.read_csv(self.file_path)
            self.messages.append(f"Memuat data fitur")
            return self.df
        except FileNotFoundError:
            self.messages.append(f"File {self.file_path} tidak ditemukan")
            return None

    def select_features(self):
        """Select features based on requirements"""
        if self.df is None:
            self.messages.append("Data belum dimuat")
            return None

        selected_features = [
            "PRODI",
            "STATUS",
            "ASAL_SEKOLAH",
            "SEKOLAH_JURUSAN",
            "ALAMAT_SEKOLAH",
            "KERJA_AYAH",
            "KETERANGAN_AYAH",
            "JENJANG_SEBELUMNYA",
            "JENIS_SELEKSI",
            "ASAL_KOTA",
            "ASAL_PROVINSI",
            "DEPARTEMEN",
            "PENGHASILAN_KATEGORI",
            "TAHUN ANGKATAN",
            "IPK_2023_GENAP",
            "IPK_2023_GANJIL",
            "IPK_2022_GENAP",
            "IPK_2022_GANJIL",
            "IPK_2021_GENAP",
            "IPK_2021_GANJIL",
            "BEASISWA_LAIN",
            "BEASISWA_KIPK",
            "PRESTASI",
            "Prosentase_Kehadiran",
        ]

        # filter hanya kolom yang ada di dataset
        available_features = [
            col for col in selected_features if col in self.df.columns
        ]
        self.df = self.df[available_features]

        # Filter tahun angkatan 2023
        if "TAHUN ANGKATAN" in self.df.columns:
            self.df = self.df[self.df["TAHUN ANGKATAN"] == 2023]
        else:
            self.messages.append("Data mahasiswa tahun 2023 tidak ditemukan")

        self.messages.append(f"Selected {len(available_features)} features")
        return self.df

    def handle_missing_values(self):
        """Handle missing values in the dataset"""
        if self.df is None:
            self.messages.append("Data belum dimuat!")
            return None

        self.messages.append("Menangani missing values")

        # Informasi missing values sebelum cleaning
        missing_before = self.df.isnull().sum().sum()
        # self.messages.append(f"Total missing values sebelum cleaning: {missing_before}")

        # 1. Kolom IPK, ganti NaN dengan 0
        ipk_columns = [col for col in self.df.columns if "IPK" in col]
        for col in ipk_columns:
            self.df[col] = self.df[col].fillna(0)

        # 2. Untuk kolom kategorikal, ganti dengan 'Tidak diketahui'
        categorical_cols = self.df.select_dtypes(include=["object"]).columns
        for col in categorical_cols:
            self.df[col] = self.df[col].fillna("Tidak Diketahui")

        # 3. Untuk kolom numerik selain IPK, ganti dengan median
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if "IPK" not in col]
        for col in numeric_cols:
            self.df[col] = self.df[col].fillna(self.df[col].median())

        # Missing values after cleaning
        missing_after = self.df.isnull().sum().sum()
        # self.messages.append(f"Total missing values setelah cleaning: {missing_after}")
        # self.messages.append(f"Total missing values: sebelum: {missing_before}, sesudah: {missing_after}")
        return self.df

    def handle_outliers(self):
        """Handle outliers in numerical columns"""
        if self.df is None:
            self.messages.append("Data belum dimuat")
            return None

        self.messages.append("Menangani outliers")

        # Only process numerical columns that are not IDs or similar
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns

        for col in numerical_cols:
            if col != "TAHUN ANGKATAN":
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1

                # Define bounds
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                # Cap outliers
                self.df[col] = np.where(
                    self.df[col] < lower_bound, lower_bound, self.df[col]
                )
                self.df[col] = np.where(
                    self.df[col] > upper_bound, upper_bound, self.df[col]
                )

        return self.df

    def create_ipk_rata_rata(self):
        """Create IPK_RATA_RATA columns for all IPK columns"""
        if self.df is None:
            self.messages.append("Data belum dimuat")
            return None

        print("Membuat kolom IPK_RATA_RATA")

        # Find all IPK columns
        ipk_columns = [col for col in self.df.columns if "IPK" in col]

        if ipk_columns:
            ipk_data = self.df[ipk_columns]
            ipk_data_replaced = ipk_data.replace(0, np.nan)
            self.df["IPK_RATA_RATA"] = ipk_data_replaced.mean(axis=1)
            self.df["IPK_RATA_RATA"] = self.df["IPK_RATA_RATA"].fillna(0)
            # self.messages.append(f"Kolom IPK_RATA_RATA berhasil dibuat. Rata-rata: {self.df['IPK_RATA_RATA'].mean():.2f}")
        else:
            self.messages.append("Tidak ditemukan kolom IPK")
            self.df["IPK_RATA_RATA"] = 0

        return self.df

    def finalize_selected_features(self):
        """Select final features as specified"""
        if self.df is None:
            self.messages.append("Data belum dimuat!")
            return None

        final_features = [
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

        # Buat BEASISWA_IPK jika belum ada
        if "BEASISWA_LAIN" in self.df.columns and "BEASISWA_KIPK" in self.df.columns:
            self.df["BEASISWA_IPK"] = self.df[["BEASISWA_LAIN", "BEASISWA_KIPK"]].max(
                axis=1
            )
        elif "BEASISWA_IPK" not in self.df.columns:
            self.df["BEASISWA_IPK"] = 0

        # Buat IPK_RATA_RATA jika belum ada
        if "IPK_RATA_RATA" not in self.df.columns:
            self.create_ipk_rata_rata()

        # Pilih hanya fitur yang diminta
        existing_features = []
        for feature in final_features:
            if feature in self.df.columns:
                existing_features.append(feature)
            else:
                print(f"⚠️ Fitur {feature} tidak ditemukan, membuat default value")
                if feature == "BEASISWA_IPK":
                    self.df[feature] = 0
                elif feature == "IPK_RATA_RATA":
                    self.df[feature] = 0
                elif feature in ["PRODI", "ASAL_SEKOLAH"]:
                    self.df[feature] = "Tidak Diketahui"
                else:
                    self.df[feature] = "Unknown"
                existing_features.append(feature)

        self.cleaned_df = self.df[existing_features].copy()

        # self.messages.append(f"Final features: {existing_features}")
        self.messages.append(f"Final dataset shape: {self.cleaned_df.shape}")

        return self.cleaned_df

    def run_cleaning(self, return_result=False):
        self.messages.append("MEMULAI DATA CLEANING")

        df = self.load_data()
        if df is None:
            return self.response()

        self.select_features()
        self.handle_missing_values()
        self.handle_outliers()
        self.create_ipk_rata_rata()
        result = self.finalize_selected_features()

        print("clenaing selesai")

        if return_result:
            return result

        return self.response()

    def response(self):
        return {
            "status": "success",
            "messages": self.messages,
            "data_preview": self.cleaned_df.head(5).to_dict(orient="records"),
        }

    # def run_ws(self):
    #     yield self._step("load_data", "memuat data", self.load_data)
    #     yield self._step("select_features", "Menyeleksi fitur", self.select_features)
    #     yield self._step("handle_missing_values", "Menangani missing values", self.handle_missing_values)
    #     yield self._step("handle_outliers", "Menangani outliers", "berhasil Menangani outliers")
    #     yield self._step("create_ipk_rata-rata", "Membuat rata-rata IPK", self.create_ipk_rata_rata)
    #     yield self._step("finalize_selected_features", "Finalisasi data", "Berhasil melakukan Finalisasi data")

    # def _step(self, key, label, func):
    #     result = func()
    #     return {
    #         "key": key,
    #         "label": label,
    #         "message": result
    #     }

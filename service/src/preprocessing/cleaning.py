import pandas as pd
import numpy as np


class DataCleaning:
    def __init__(self, file_path="data/data_fitur.csv"):
        self.file_path = file_path
        self.df = None
        self.cleaned_df = None

    def load_data(self):
        self.df = pd.read_csv(self.file_path)
        return self.df

    def select_features(self):
        selected_features = [
            "PRODI",  # METADATA
            "ASAL_PROVINSI",
            "ASAL_KOTA",
            "KERJA_AYAH",
            "PENGHASILAN_KATEGORI",
            "BEASISWA_KIPK",
            "SEKOLAH_JURUSAN",
            "ASAL_SEKOLAH",
            "JENIS_SELEKSI",
            "TAHUN ANGKATAN",
            "IPK_2021_GANJIL",
            "IPK_2021_GENAP",
            "IPK_2022_GANJIL",
            "IPK_2022_GENAP",
            "IPK_2023_GANJIL",
            "IPK_2023_GENAP",
        ]

        self.df = self.df[[c for c in selected_features if c in self.df.columns]]
        self.df = self.df[self.df["TAHUN ANGKATAN"] == 2023]
        return self.df

    def handle_missing_values(self):
        ipk_cols = [c for c in self.df.columns if "IPK" in c]
        self.df[ipk_cols] = self.df[ipk_cols].fillna(0)

        cat_cols = self.df.select_dtypes(include="object").columns
        self.df[cat_cols] = self.df[cat_cols].fillna("Tidak Diketahui")

        if "BEASISWA_KIPK" in self.df.columns:
            self.df["BEASISWA_KIPK"] = self.df["BEASISWA_KIPK"].fillna(0).astype(int)

        return self.df

    def create_ipk_rata_rata(self):
        ipk_cols = [c for c in self.df.columns if "IPK" in c]
        ipk = self.df[ipk_cols].replace(0, np.nan)
        self.df["IPK_RATA_RATA"] = ipk.mean(axis=1).fillna(0)
        return self.df

    def run(self):
        self.load_data()
        self.select_features()
        self.handle_missing_values()
        self.create_ipk_rata_rata()
        self.cleaned_df = self.df.copy()
        return self.cleaned_df

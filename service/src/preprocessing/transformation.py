from sklearn.preprocessing import LabelEncoder
import pandas as pd
import os
import joblib


class DataTransformation:
    def __init__(self, df):
        self.df = df.copy()
        self.X_clustering = None
        self.df_metadata = None
        self.encoders = {}

    def split_metadata(self):
        self.df_metadata = self.df[["PRODI"]].copy()
        return self.df_metadata

    def encode_features(self):
        clustering_features = [
            "ASAL_PROVINSI",
            "ASAL_KOTA",
            "KERJA_AYAH",
            "PENGHASILAN_KATEGORI",
            "SEKOLAH_JURUSAN",
            "ASAL_SEKOLAH",
            "JENIS_SELEKSI",
        ]

        df_encoded = pd.DataFrame(index=self.df.index)

        for col in clustering_features:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(self.df[col].astype(str))
            self.encoders[col] = le

        df_encoded["BEASISWA_KIPK"] = self.df["BEASISWA_KIPK"]
        df_encoded["IPK_RATA_RATA"] = self.df["IPK_RATA_RATA"]

        self.X_clustering = df_encoded
        return self.X_clustering

    def save(self, path="data_processed"):
        os.makedirs(path, exist_ok=True)

        metadata_path = f"{path}/metadata.csv"

        self.df_metadata.to_csv(metadata_path, index=False)

    def run(self):
        self.split_metadata()
        self.encode_features()
        self.save()
        return self.X_clustering

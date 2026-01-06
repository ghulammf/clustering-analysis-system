import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import joblib
import os

class DataTransformation:
    def __init__(self, file_path=''):
        self.file_path = file_path
        self.df = None
        self.transformed_df = None
        self.label_encoders = {}
        self.onehot_encoders = {}

    def load_cleaned_data(self):
        """Load cleaned data"""
        try:
            self.df = pd.read_csv(self.file_path)
            print("Data berhasil dimuat")
            return self.df
        except FileNotFoundError:
            print(f"File {self.file_path} tidak ditemukan")
            return None
        
    def encode_categorical_variables(self, method='label'):
        """Encode categorical variables"""
        if self.df is None:
            print("Data belum dimuat")
            return None
        
        categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()

        if method == 'label':
            for col in categorical_cols:
                le = LabelEncoder()
                self.df[col + '_ENCODED'] = le.fit_transform(self.df[col].astype(str))
                self.label_encoders[col] = le
                
        elif method == 'onehot':
            for col in categorical_cols:
                dummies = pd.get_dummies(self.df[col], prefix=col, dtype=int)
                self.df = pd.concat([self.df, dummies], axis=1)
                self.onehot_encoders[col] = dummies.columns.tolist()

        return self.df
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

    def load_cleaned_data(self, df):
        """Load cleaned data"""
        if df is None:
            print("Data kosong!")
            return None
        
        self.df = df.copy()
        print(f"Data berhasil dimuat. Shape: {self.df}")
        print(f"Kolom: {list(self.df.columns)}")
        return self.df
        # try:
        #     self.df = pd.read_csv(self.file_path)
        #     print("Data berhasil dimuat")
        #     return self.df
        # except FileNotFoundError:
        #     print(f"File {self.file_path} tidak ditemukan")
        #     return None
        
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

    def create_binary_features(self):
        """Create binary features from categorical data"""
        if self.df is None:
            print("Data belum dimuat!")
            return None
        
        if 'ASAL_PROVINSI' in self.columns:
            self.df['ASAL_JAWA_TIMUR'] = (self.df['ASAL_PROVINSI'] == 'Jawa Timur').astype(int)
            print(f"ASAL_JAWA_TIMUR: {self.df['ASAL_JAWA_TIMUR'].sum()} mahasiswa dari Jawa Timur")
        
        if 'JENIS_SELEKSI' in self.df.columns:
            self.df['SELEKSI_PRESTASI'] = self.df['JENIS_SELEKSI'].str.contains('Prestasi', case=False, na=False).astype(int)
            print(f"SELEKSI_PRESTASI: {self.df['SELEKSI_PRESTASI'].sum()} mahasiswa jalur prestasi")

        if 'ASAL_SEKOLAH' in self.df.columns:
            self.df['SEKOLAH_NEGERI'] = self.df['ASAL_SEKOLAH'].str.contains('SMAN|SMKN|MAN', case=False, na=False).astype(int)
            print(f"SEKOLAH_NEGERI: {self.df['SEKOLAH_NEGERI'].sum()} mahasiswa dari sekolah negeri")

            return self.df
        
    def create_interaction_features(self):
        """Create interaction features"""
        if self.df is None:
            print("Data belum dimuat!")
            return None
            
        print("Membuat fitur interaksi...")

        if all(col in self.df.columns for col in ['PENGHASILAN_KATEGORI_ENCODED', 'BEASISWA_KIPK']):
            self.df['PENGHASILAN_BEASISWA_INTERACTION'] = self.df['PENGHASILAN_KATEGORI_ENCODED'] * self.df['BEASISWA_KIPK']
        
        if all(col in self.df.columns for col in ['SEKOLAH_NEGERI', 'SELEKSI_PRESTASI']):
            self.df['SEKOLAH_SELEKSI_INTERACTION'] = self.df['SEKOLAH_NEGERI'] * self.df['SELEKSI_PRESTASI']
        
        return self.df
    
    def create_binned_features(self):
        """Create binned/grouped features"""
        if self.df is None:
            print("Data belum dimuat!")
            return None
        
        # Bin IPK into categories
        if 'IPK_RATA_RATA' in self.df.columns:
            bins = [0, 2.0, 2.5, 3.0, 3.5, 4.0]
            labels = ['Sangat Rendah', 'Rendah', 'Cukup', 'Baik', 'Sangat Baik']
            self.df['IPK_KATEGORI'] = pd.cut(self.df['IPK_RATA_RATA'], bins=bins, labels=labels, include_lowest=True)
            
            # Encode the binned IPK
            le_ipk = LabelEncoder()
            self.df['IPK_KATEGORI_ENCODED'] = le_ipk.fit_transform(self.df['IPK_KATEGORI'].astype(str))
            self.label_encoders['IPK_KATEGORI'] = le_ipk
            
            print("IPK binned into categories:")
            print(self.df['IPK_KATEGORI'].value_counts())
        
        # Bin income categories into groups
        if 'PENGHASILAN_KATEGORI' in self.df.columns:
            # Define income groups
            income_mapping = {
                '0-1 Juta': 'Rendah',
                '1 Juta-2.5 Juta': 'Rendah',
                '2.5-4 Juta': 'Menengah',
                '4-6 Juta': 'Menengah',
                '6-10 Juta': 'Tinggi',
                '10-20 Juta': 'Tinggi',
                '>20 Juta': 'Sangat Tinggi'
            }
            
            self.df['PENGHASILAN_GROUP'] = self.df['PENGHASILAN_KATEGORI'].map(income_mapping)
            self.df['PENGHASILAN_GROUP'] = self.df['PENGHASILAN_GROUP'].fillna('Tidak Diketahui')
            
            # Encode income group
            le_income = LabelEncoder()
            self.df['PENGHASILAN_GROUP_ENCODED'] = le_income.fit_transform(self.df['PENGHASILAN_GROUP'])
            self.label_encoders['PENGHASILAN_GROUP'] = le_income
        
        return self.df
    
    def aggregate_school_features(self):
        """Aggregate school-related features"""
        if self.df is None:
            print("Data belum dimuat!")
            return None
            
        print("Meng-agregasi fitur sekolah...")
        
        # Count students from each school
        if 'ASAL_SEKOLAH' in self.df.columns:
            school_counts = self.df['ASAL_SEKOLAH'].value_counts()
            self.df['SEKOLAH_JUMLAH_SISWA'] = self.df['ASAL_SEKOLAH'].map(school_counts)
            print(f"Rata-rata siswa per sekolah: {self.df['SEKOLAH_JUMLAH_SISWA'].mean():.2f}")
        
        # Count students from each city
        if 'ASAL_KOTA' in self.df.columns:
            city_counts = self.df['ASAL_KOTA'].value_counts()
            self.df['KOTA_JUMLAH_SISWA'] = self.df['ASAL_KOTA'].map(city_counts)
        
        return self.df
    
    def finalize_transformed_data(self):
        """Create final transformed dataset"""
        if self.df is None:
            print("Data belum dimuat!")
            return None
            
        # Select only numerical columns for modeling
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        
        self.transformed_df = self.df[numerical_cols].copy()
        
        print(f"Dataset setelah transformasi: {self.transformed_df.shape}")
        print(f"Kolom numerik: {list(self.transformed_df.columns)}")
        
        return self.transformed_df
    
    def save_artifacts(self):
        """Save ML artifacs (encoders, scaler, model)"""

        if self.label_encoders:
            joblib.dump(self.label_encoders, 'label_encoders.pkl')

        if hasattr(self, 'scaler') and self.scaler is not None:
            joblib.dump(self.scaler, 'scaler.pkl')

        if hasattr(self, 'model') and self.model is not None:
            joblib.dump(self.model, 'model.pkl')
        
    def run_full_transformation(self, encoding_method='label'):
        """Run the complete data transformation pipeline"""
        print("="*50)
        print("MEMULAI TRANSFORMASI DATA")
        print("="*50)
        
        # Step 1: Load cleaned data
        self.load_cleaned_data()
        
        if self.df is not None:
            # Step 2: Encode categorical variables
            self.encode_categorical_variables(method=encoding_method)
            
            # Step 3: Create binary features
            self.create_binary_features()
            
            # Step 4: Create interaction features
            self.create_interaction_features()
            
            # Step 5: Create binned features
            self.create_binned_features()
            
            # Step 6: Aggregate school features
            self.aggregate_school_features()
            
            # Step 7: Finalize transformed data
            self.finalize_transformed_data()
            
            # Step 8: Save transformed data
            self.save_transformed_data()
            
            print("="*50)
            print("TRANSFORMASI DATA SELESAI")
            print("="*50)
            
            return self.transformed_df
        else:
            print("Gagal memuat data untuk transformasi!")
            return None

if __name__ == "__main__":
    transformer = DataTransformation()
    transformed_data = transformer.run_full_transformation(encoding_method='label')
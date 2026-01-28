from sklearn.preprocessing import MinMaxScaler
import joblib
import os


class DataNormalization:
    def __init__(self, X):
        self.X = X.copy()
        self.scaler = MinMaxScaler()
        self.X_normalized = None

    def normalize(self):
        exclude = ["BEASISWA_KIPK"]
        cols = [c for c in self.X.columns if c not in exclude]

        self.X_normalized = self.X.copy()
        self.X_normalized[cols] = self.scaler.fit_transform(self.X[cols])
        return self.X_normalized

    def save(self, path="data_processed"):
        os.makedirs(path, exist_ok=True)

        x_path = f"{path}/X_clustering.csv"
        scaler_path = f"{path}/scaler.pkl"

        self.X_normalized.to_csv(x_path, index=False)
        joblib.dump(self.scaler, scaler_path)

    def run(self):
        self.normalize()
        self.save()
        print("Preprocessing selesai")
        return self.X_normalized

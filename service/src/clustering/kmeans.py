import pandas as pd
from sklearn.cluster import KMeans
from src.clustering.cluster_utils import evaluate_clustering


class KMeansClustering:
    def __init__(self, X: pd.DataFrame):
        self.X = X

    def auto_search(self, cluster_range=range(2, 8), random_state=42):
        results = []

        for k in cluster_range:
            model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            labels = model.fit_predict(self.X)

            scores = evaluate_clustering(self.X, labels)
            scores["n_clusters"] = k
            results.append(scores)

        best = max(results, key=lambda x: x["silhouette_score"])
        return results, best

    def fit_final(self, n_clusters, random_state=42):
        model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        labels = model.fit_predict(self.X)
        return labels

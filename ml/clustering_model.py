from sklearn.cluster import MiniBatchKMeans
import numpy as np
import pickle


class Clusterizer:
    def __init__(self):
        self.model = MiniBatchKMeans(n_clusters=5, random_state=42)

    def fit(self, X) -> None:
        self.model.fit(X)

    def predict(self, X) -> np.ndarray:
        preds = self.model.predict(X)
        return preds

    def save(self, path) -> None:
        with open(path, 'wb') as fout:
            pickle.dump(self.model, fout)

    def load(self, path):
        with open(path, 'rb') as fin:
            self.model = pickle.load(fin)

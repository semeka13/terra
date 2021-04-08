from catboost import CatBoostClassifier
import numpy as np
from os import listdir
import pickle
from ml.errors import NotFittedError


class CatBoostRecomender:
    def __init__(self, model=CatBoostClassifier()):
        self.model = model

    def fit(self, X: np.ndarray, y: np.ndarray, path=None) -> None:
        if path == None:
            self.model.fit(X, y)
        else:
            if path in listdir("trained_models"):
                self.model.fit(X, y, init_model=path)
            else:
                raise

    def predict(self, X) -> np.ndarray:
        if not self.model.is_fitted():
            raise NotFittedError()

        preds = self.model.predict_proba(X)[:, 1]
        return preds

    def save(self, path) -> None:
        if not self.model.is_fitted():
            raise NotFittedError()

        with open(path, 'wb') as fout:
            pickle.dump(self.model, fout)

    def load(self, path):
        with open(path, 'rb') as fin:
            self.model = pickle.load(fin)

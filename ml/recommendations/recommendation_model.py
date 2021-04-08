from catboost import CatBoostClassifier
import numpy as np
from os import listdir
import pickle
from ml.errors import NotFittedError


class CatBoostRecommender:
    def __init__(self, model=CatBoostClassifier()):
        self.model = model

    def fit(self, X: np.ndarray, y: np.ndarray, model_file=None, model_dir="None") -> None:
        if model_file == None:
            self.model.fit(X, y)
        else:
            if model_file in listdir(model_dir):
                self.model.fit(X, y, init_model=model_file)
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

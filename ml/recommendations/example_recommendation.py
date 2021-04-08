from ml.recommendations.recommendation_model import CatBoostRecomender
from sklearn.datasets import load_breast_cancer
X, y = load_breast_cancer(return_X_y=True)

cbr = CatBoostRecomender()
cbr.fit(X[:200], y[:200])
cbr.save("cb.pickle")

cbr2 = CatBoostRecomender()
cbr2.load("cb.pickle")
print(cbr2.predict(X[200:]))

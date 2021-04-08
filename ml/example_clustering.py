from ml.clustering_model import Clusterizer
from sklearn.datasets import load_boston

X, y = load_boston(return_X_y=True)

cl = Clusterizer()
cl.fit(X[:400])
cl.save("cluterizer.joblib")

cl2 = Clusterizer()
cl2.load("cluterizer.joblib")
print(cl2.predict(X[400:]))
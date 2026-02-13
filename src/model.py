"""
O arquivo model.py cria um pipeline de escalonamento e classificação usando o KNN.
"""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


def knn_pipeline():
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(weights="distance"))
    ])

    return pipeline
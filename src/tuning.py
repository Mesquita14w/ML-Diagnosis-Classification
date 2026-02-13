"""
O arquivo tuning.py ajusta os hiperparâmetros do knn usando validação cruzada, com foco em maximizar recall.
"""

from sklearn.model_selection import GridSearchCV

def tune_knn(model, X_train, y_train):

    param_grid = {
        "knn__n_neighbors": list(range(1, 21)),
        "knn__weights": ["uniform", "distance"]
    }

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="recall",
        cv=5,
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    return grid
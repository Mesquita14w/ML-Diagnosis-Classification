import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
from src.load import load_data
from src.train import split_data
from src.model import knn_pipeline
from src.tuning import tune_knn


def main():
    X, y = load_data("data/raw/data.csv")

    X_train, X_test, y_train, y_test = split_data(X, y)

    model = knn_pipeline()
    grid = tune_knn(model, X_train, y_train)
    
    print(f"Melhor número de vizinhos: {grid.best_params_}, Melhor recall médio (Validação Cruzada): {grid.best_score_}")

    """
    A partir daqui melhoramos o modelo a partir da melhor avaliação da Validação Cruzada e o melhor número de vizinhos, 
    e avaliamos o modelo usando a Curva ROC, AUC score e o Índice de Youden para encontrar o melhor threshold para equilibrar 
    o recall e a precisão.
    """

    best_model = grid.best_estimator_
    y_proba = best_model.predict_proba(X_test)[:, 1]

    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    auc_score = roc_auc_score(y_test, y_proba)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("Taxa de Falsos Positivos")
    plt.ylabel("Taxa de Verdadeiros Positivos")
    plt.title("Curva ROC")
    plt.legend()
    plt.show()

    youden_index = tpr - fpr
    best_index = np.argmax(youden_index)
    best_threshold = thresholds[best_index]

    print("\nAUC Score:", auc_score)
    print("Melhor threshold:", best_threshold)

    y_pred_optimal = (y_proba >= best_threshold).astype(int)

    print("\nMatriz de Confusão:")
    print(confusion_matrix(y_test, y_pred_optimal))

    print("\nAvaliação do Modelo:")
    print(classification_report(y_test, y_pred_optimal))

    # Salvando o Modelo Final
    os.makedirs("models", exist_ok=True)

    model_package = {
        "model": best_model,
        "threshold": best_threshold
    }

    joblib.dump(model_package, "models/knn_model_final.pkl")

if __name__ == "__main__":
    main()
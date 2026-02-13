"""
O arquivo train.py divide os dados em treino e teste através do train_test_split do Scikit-learn.
"""

from sklearn.model_selection import train_test_split

def split_data(X, y, test_size=0.3, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)

    return X_train, X_test, y_train, y_test
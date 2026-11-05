"""Carga de dados, divisão estratificada sem vazamento e mascaramento de rótulos."""

from typing import Tuple
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_digits_dataset():
    """Carrega o dataset Digits completo (com imagens 8x8 e matriz de atributos)."""
    return load_digits()


def split_and_scale(
    X: np.ndarray,
    y: np.ndarray,
    seed: int = 42,
    test_size: float = 0.30,
    val_size_total: float = 0.10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Divide os dados de forma estratificada em Treino (60%), Validação (10%) e Teste (30%),
    e ajusta o StandardScaler exclusivamente no conjunto de treino (evitando vazamento de dados).
    """
    # 1. Separar Teste (30%) do restante (70%)
    X_tmp, X_te, y_tmp, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )

    # 2. Separar Validação (10% do total = 1/7 dos 70% restantes) do Treino (60% do total)
    val_rel_size = val_size_total / (1.0 - test_size)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_rel_size, stratify=y_tmp, random_state=seed
    )

    # 3. Ajustar StandardScaler apenas no conjunto de Treino
    scaler = StandardScaler().fit(X_tr)
    X_tr_sc = scaler.transform(X_tr)
    X_val_sc = scaler.transform(X_val)
    X_te_sc = scaler.transform(X_te)

    return X_tr_sc, X_val_sc, X_te_sc, y_tr, y_val, y_te, scaler


def mask_labels(y_train: np.ndarray, frac: float, seed: int) -> np.ndarray:
    """
    Mantém `frac` dos rótulos de forma estratificada (garantindo pelo menos 1 exemplo por classe).
    Os rótulos escondidos recebem o valor -1 (convenção do scikit-learn).
    """
    rng = np.random.default_rng(seed)
    y_semi = np.full_like(y_train, -1)
    for c in np.unique(y_train):
        idx = np.where(y_train == c)[0]
        k = max(1, int(round(frac * len(idx))))
        keep = rng.choice(idx, size=k, replace=False)
        y_semi[keep] = y_train[keep]
    return y_semi


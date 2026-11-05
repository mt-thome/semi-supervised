"""Fábricas dos modelos supervisionados e semissupervisionados."""

from typing import Optional, Union
from sklearn.svm import SVC
from sklearn.semi_supervised import SelfTrainingClassifier, LabelSpreading, LabelPropagation


def make_baseline(seed: int = 42, C: float = 1.0, gamma: Union[str, float] = "scale") -> SVC:
    """Retorna o classificador base SVC supervisionado com estimativa de probabilidade."""
    return SVC(kernel="rbf", C=C, gamma=gamma, probability=True, random_state=seed)


def make_self_training(
    seed: int = 42,
    threshold: float = 0.75,
    criterion: str = "threshold",
    k_best: int = 10,
    max_iter: Optional[int] = 10,
) -> SelfTrainingClassifier:
    """Retorna o classificador Self-Training usando SVC(probability=True) como estimador base."""
    base_svc = make_baseline(seed=seed)
    return SelfTrainingClassifier(
        estimator=base_svc,
        threshold=threshold,
        criterion=criterion,
        k_best=k_best,
        max_iter=max_iter,
    )


def make_label_spreading(
    kernel: str = "knn",
    n_neighbors: int = 7,
    gamma: float = 0.01,
    alpha: float = 0.2,
    max_iter: int = 1000,
) -> LabelSpreading:
    """Retorna o classificador Label Spreading com kernel 'knn' ou 'rbf'."""
    return LabelSpreading(
        kernel=kernel,
        n_neighbors=n_neighbors,
        gamma=gamma,
        alpha=alpha,
        max_iter=max_iter,
    )


def make_label_propagation(
    kernel: str = "knn",
    n_neighbors: int = 7,
    gamma: float = 0.01,
    max_iter: int = 1000,
) -> LabelPropagation:
    """Retorna o classificador Label Propagation (clamping rígido dos rótulos originais)."""
    return LabelPropagation(
        kernel=kernel,
        n_neighbors=n_neighbors,
        gamma=gamma,
        max_iter=max_iter,
    )


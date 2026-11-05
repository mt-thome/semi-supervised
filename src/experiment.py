"""Execução do protocolo experimental, busca de hiperparâmetros e avaliação."""

import time
import warnings
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from scipy.stats import wilcoxon

from .data import load_digits_dataset, split_and_scale, mask_labels
from .models import (
    make_baseline,
    make_self_training,
    make_label_spreading,
    make_label_propagation,
)

warnings.filterwarnings("ignore")

DEFAULT_FRACTIONS = [0.02, 0.05, 0.10, 0.20]
DEFAULT_SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]


def run_hyperparameter_search(
    X: np.ndarray,
    y: np.ndarray,
    fractions: List[float] = DEFAULT_FRACTIONS,
    seeds: List[int] = DEFAULT_SEEDS,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Executa a grade completa de hiperparâmetros no conjunto de VALIDAÇÃO
    para Self-Training e Label Spreading em todas as frações e sementes.

    Retorna:
        df_st_grid: DataFrame com resultados na validação e teste do Self-Training
        df_ls_grid: DataFrame com resultados na validação e teste do Label Spreading
    """
    st_rows = []
    ls_rows = []

    st_configs = [
        {"criterion": "threshold", "threshold": 0.60, "k_best": 10, "label": "thr=0.60"},
        {"criterion": "threshold", "threshold": 0.75, "k_best": 10, "label": "thr=0.75"},
        {"criterion": "threshold", "threshold": 0.90, "k_best": 10, "label": "thr=0.90"},
        {"criterion": "k_best", "threshold": 0.75, "k_best": 10, "label": "k_best=10"},
        {"criterion": "k_best", "threshold": 0.75, "k_best": 30, "label": "k_best=30"},
    ]

    ls_configs = []
    for k in [3, 5, 7, 10]:
        for alpha in [0.1, 0.2, 0.5, 0.8]:
            ls_configs.append(
                {
                    "kernel": "knn",
                    "n_neighbors": k,
                    "gamma": 0.01,
                    "alpha": alpha,
                    "param_str": f"knn (k={k})",
                    "param_val": k,
                }
            )
    for gamma in [0.001, 0.005, 0.01, 0.05]:
        for alpha in [0.1, 0.2, 0.5, 0.8]:
            ls_configs.append(
                {
                    "kernel": "rbf",
                    "n_neighbors": 7,
                    "gamma": gamma,
                    "alpha": alpha,
                    "param_str": f"rbf (γ={gamma})",
                    "param_val": gamma,
                }
            )

    for frac in fractions:
        for seed in seeds:
            X_tr, X_val, X_te, y_tr, y_val, y_te, _ = split_and_scale(X, y, seed=seed)
            y_semi = mask_labels(y_tr, frac=frac, seed=seed)
            unlab_mask = y_semi == -1

            # 1. Avaliar grade do Self-Training
            for cfg in st_configs:
                t0 = time.perf_counter()
                clf = make_self_training(
                    seed=seed,
                    threshold=cfg["threshold"],
                    criterion=cfg["criterion"],
                    k_best=cfg["k_best"],
                    max_iter=10,
                )
                clf.fit(X_tr, y_semi)
                elapsed = time.perf_counter() - t0

                val_pred = clf.predict(X_val)
                te_pred = clf.predict(X_te)

                # Pseudo-rótulos atribuídos (labeled_iter_ > 0 indica pseudo-rotulado)
                pseudo_mask = clf.labeled_iter_ > 0
                n_pseudo = int(np.sum(pseudo_mask))
                pseudo_acc = (
                    accuracy_score(y_tr[pseudo_mask], clf.transduction_[pseudo_mask])
                    if n_pseudo > 0
                    else np.nan
                )
                trans_all_acc = accuracy_score(y_tr[unlab_mask], clf.transduction_[unlab_mask])

                st_rows.append(
                    {
                        "frac": frac,
                        "seed": seed,
                        "config_label": cfg["label"],
                        "criterion": cfg["criterion"],
                        "threshold": cfg["threshold"],
                        "k_best": cfg["k_best"],
                        "val_acc": accuracy_score(y_val, val_pred),
                        "val_f1": f1_score(y_val, val_pred, average="macro"),
                        "test_acc": accuracy_score(y_te, te_pred),
                        "test_f1": f1_score(y_te, te_pred, average="macro"),
                        "n_iter": clf.n_iter_,
                        "n_pseudo": n_pseudo,
                        "pseudo_acc": pseudo_acc,
                        "trans_all_acc": trans_all_acc,
                        "train_time_s": elapsed,
                    }
                )

            # 2. Avaliar grade do Label Spreading
            for cfg in ls_configs:
                t0 = time.perf_counter()
                clf = make_label_spreading(
                    kernel=cfg["kernel"],
                    n_neighbors=cfg["n_neighbors"],
                    gamma=cfg["gamma"],
                    alpha=cfg["alpha"],
                )
                clf.fit(X_tr, y_semi)
                elapsed = time.perf_counter() - t0

                val_pred = clf.predict(X_val)
                te_pred = clf.predict(X_te)
                # Lidar com eventuais NaNs se grafo desconexo
                if np.isnan(val_pred).any():
                    val_pred = np.nan_to_num(val_pred, nan=0).astype(int)
                if np.isnan(te_pred).any():
                    te_pred = np.nan_to_num(te_pred, nan=0).astype(int)

                trans_unlab = clf.transduction_[unlab_mask]
                pseudo_acc = accuracy_score(y_tr[unlab_mask], trans_unlab)

                ls_rows.append(
                    {
                        "frac": frac,
                        "seed": seed,
                        "kernel": cfg["kernel"],
                        "param_str": cfg["param_str"],
                        "param_val": cfg["param_val"],
                        "n_neighbors": cfg["n_neighbors"],
                        "gamma": cfg["gamma"],
                        "alpha": cfg["alpha"],
                        "val_acc": accuracy_score(y_val, val_pred),
                        "val_f1": f1_score(y_val, val_pred, average="macro"),
                        "test_acc": accuracy_score(y_te, te_pred),
                        "test_f1": f1_score(y_te, te_pred, average="macro"),
                        "pseudo_acc": pseudo_acc,
                        "train_time_s": elapsed,
                    }
                )

    return pd.DataFrame(st_rows), pd.DataFrame(ls_rows)


def select_best_configs_per_fraction(
    df_st_grid: pd.DataFrame, df_ls_grid: pd.DataFrame
) -> Tuple[Dict[float, Dict[str, Any]], Dict[float, Dict[str, Any]]]:
    """
    Seleciona a melhor configuração de hiperparâmetros por fração rotulada
    com base na média de acurácia no conjunto de VALIDAÇÃO (sem tocar no teste).
    """
    best_st = {}
    st_summary = (
        df_st_grid.groupby(["frac", "config_label", "criterion", "threshold", "k_best"])[
            ["val_acc", "val_f1"]
        ]
        .mean()
        .reset_index()
    )
    for frac, grp in st_summary.groupby("frac"):
        top = grp.sort_values(by=["val_acc", "val_f1"], ascending=False).iloc[0]
        best_st[frac] = {
            "config_label": top["config_label"],
            "criterion": top["criterion"],
            "threshold": float(top["threshold"]),
            "k_best": int(top["k_best"]),
            "val_acc_mean": float(top["val_acc"]),
        }

    best_ls = {}
    ls_summary = (
        df_ls_grid.groupby(["frac", "kernel", "param_str", "n_neighbors", "gamma", "alpha"])[
            ["val_acc", "val_f1"]
        ]
        .mean()
        .reset_index()
    )
    for frac, grp in ls_summary.groupby("frac"):
        top = grp.sort_values(by=["val_acc", "val_f1"], ascending=False).iloc[0]
        best_ls[frac] = {
            "kernel": top["kernel"],
            "param_str": top["param_str"],
            "n_neighbors": int(top["n_neighbors"]),
            "gamma": float(top["gamma"]),
            "alpha": float(top["alpha"]),
            "val_acc_mean": float(top["val_acc"]),
        }

    return best_st, best_ls


def run_main_experiment(
    X: np.ndarray,
    y: np.ndarray,
    best_st: Dict[float, Dict[str, Any]],
    best_ls: Dict[float, Dict[str, Any]],
    fractions: List[float] = DEFAULT_FRACTIONS,
    seeds: List[int] = DEFAULT_SEEDS,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Executa o loop experimental principal (4 frações x 10 sementes) comparando:
    - Baseline (SVC só com rotulados)
    - Self-Training (melhor config da validação por fração)
    - Label Spreading (melhor config da validação por fração)
    - Label Propagation (comparação extra com clamping rígido, mesmo kernel/vizinhos do LS)
    - Supervisionado 100% (teto de desempenho usando todos os rótulos de treino)

    Também coleta estatísticas por iteração do Self-Training para análise de confirmation bias.
    """
    rows = []
    iter_rows = []

    for frac in fractions:
        st_cfg = best_st[frac]
        ls_cfg = best_ls[frac]

        for seed in seeds:
            X_tr, X_val, X_te, y_tr, y_val, y_te, _ = split_and_scale(X, y, seed=seed)
            y_semi = mask_labels(y_tr, frac=frac, seed=seed)
            lab_mask = y_semi != -1
            unlab_mask = y_semi == -1
            n_labeled = int(np.sum(lab_mask))

            # 1. Baseline Supervisionado (apenas amostras rotuladas)
            t0 = time.perf_counter()
            clf_base = make_baseline(seed=seed).fit(X_tr[lab_mask], y_tr[lab_mask])
            t_base = time.perf_counter() - t0
            pred_base_te = clf_base.predict(X_te)
            pred_base_unlab = clf_base.predict(X_tr[unlab_mask])
            rows.append(
                {
                    "seed": seed,
                    "frac": frac,
                    "n_labeled": n_labeled,
                    "model": "Baseline (SVC)",
                    "hyperparams": "kernel=rbf, C=1.0",
                    "test_acc": accuracy_score(y_te, pred_base_te),
                    "test_f1": f1_score(y_te, pred_base_te, average="macro"),
                    "val_acc": accuracy_score(y_val, clf_base.predict(X_val)),
                    "pseudo_acc": accuracy_score(y_tr[unlab_mask], pred_base_unlab),
                    "train_time_s": t_base,
                }
            )

            # 2. Self-Training (com melhor hiperparâmetro da validação)
            t0 = time.perf_counter()
            clf_st = make_self_training(
                seed=seed,
                threshold=st_cfg["threshold"],
                criterion=st_cfg["criterion"],
                k_best=st_cfg["k_best"],
                max_iter=10,
            ).fit(X_tr, y_semi)
            t_st = time.perf_counter() - t0
            pred_st_te = clf_st.predict(X_te)

            # Acurácia transdutiva nos não-rotulados (transduction_ em todos os não-rotulados)
            pseudo_mask = clf_st.labeled_iter_ > 0
            st_pseudo_acc = (
                accuracy_score(y_tr[pseudo_mask], clf_st.transduction_[pseudo_mask])
                if np.any(pseudo_mask)
                else accuracy_score(y_tr[unlab_mask], clf_st.predict(X_tr[unlab_mask]))
            )
            rows.append(
                {
                    "seed": seed,
                    "frac": frac,
                    "n_labeled": n_labeled,
                    "model": "Self-Training",
                    "hyperparams": st_cfg["config_label"],
                    "test_acc": accuracy_score(y_te, pred_st_te),
                    "test_f1": f1_score(y_te, pred_st_te, average="macro"),
                    "val_acc": accuracy_score(y_val, clf_st.predict(X_val)),
                    "pseudo_acc": st_pseudo_acc,
                    "train_time_s": t_st,
                }
            )

            # Registrar evolução por iteração para análise detalhada (thresholds e k_best)
            for cfg_it in [
                {"label": "thr=0.60", "criterion": "threshold", "threshold": 0.60, "k_best": 10},
                {"label": "thr=0.75", "criterion": "threshold", "threshold": 0.75, "k_best": 10},
                {"label": "thr=0.90", "criterion": "threshold", "threshold": 0.90, "k_best": 10},
                {"label": "k_best=10", "criterion": "k_best", "threshold": 0.75, "k_best": 10},
                {"label": "k_best=30", "criterion": "k_best", "threshold": 0.75, "k_best": 30},
            ]:
                clf_st_it = make_self_training(
                    seed=seed,
                    threshold=cfg_it["threshold"],
                    criterion=cfg_it["criterion"],
                    k_best=cfg_it["k_best"],
                    max_iter=10,
                ).fit(X_tr, y_semi)
                for it in range(1, clf_st_it.n_iter_ + 1):
                    it_mask = clf_st_it.labeled_iter_ == it
                    cum_mask = (clf_st_it.labeled_iter_ > 0) & (clf_st_it.labeled_iter_ <= it)
                    n_it = int(np.sum(it_mask))
                    n_cum = int(np.sum(cum_mask))
                    if n_it > 0:
                        it_acc = accuracy_score(y_tr[it_mask], clf_st_it.transduction_[it_mask])
                        cum_acc = accuracy_score(y_tr[cum_mask], clf_st_it.transduction_[cum_mask])
                        iter_rows.append(
                            {
                                "seed": seed,
                                "frac": frac,
                                "config_label": cfg_it["label"],
                                "threshold": cfg_it["threshold"],
                                "iteration": it,
                                "n_added": n_it,
                                "n_cumulative": n_cum,
                                "iter_pseudo_acc": it_acc,
                                "cum_pseudo_acc": cum_acc,
                            }
                        )

            # 3. Label Spreading (com melhor hiperparâmetro da validação)
            t0 = time.perf_counter()
            clf_ls = make_label_spreading(
                kernel=ls_cfg["kernel"],
                n_neighbors=ls_cfg["n_neighbors"],
                gamma=ls_cfg["gamma"],
                alpha=ls_cfg["alpha"],
            ).fit(X_tr, y_semi)
            t_ls = time.perf_counter() - t0
            pred_ls_te = clf_ls.predict(X_te)
            ls_pseudo_acc = accuracy_score(y_tr[unlab_mask], clf_ls.transduction_[unlab_mask])
            rows.append(
                {
                    "seed": seed,
                    "frac": frac,
                    "n_labeled": n_labeled,
                    "model": "Label Spreading",
                    "hyperparams": f"{ls_cfg['param_str']}, α={ls_cfg['alpha']}",
                    "test_acc": accuracy_score(y_te, pred_ls_te),
                    "test_f1": f1_score(y_te, pred_ls_te, average="macro"),
                    "val_acc": accuracy_score(y_val, clf_ls.predict(X_val)),
                    "pseudo_acc": ls_pseudo_acc,
                    "train_time_s": t_ls,
                }
            )

            # 4. Label Propagation (comparação opcional com alpha fixo / hard clamping)
            t0 = time.perf_counter()
            clf_lp = make_label_propagation(
                kernel=ls_cfg["kernel"],
                n_neighbors=ls_cfg["n_neighbors"],
                gamma=ls_cfg["gamma"],
            ).fit(X_tr, y_semi)
            t_lp = time.perf_counter() - t0
            pred_lp_te = clf_lp.predict(X_te)
            if np.isnan(pred_lp_te).any():
                pred_lp_te = np.nan_to_num(pred_lp_te, nan=0).astype(int)
            lp_pseudo_acc = accuracy_score(y_tr[unlab_mask], clf_lp.transduction_[unlab_mask])
            rows.append(
                {
                    "seed": seed,
                    "frac": frac,
                    "n_labeled": n_labeled,
                    "model": "Label Propagation",
                    "hyperparams": f"{ls_cfg['param_str']} (clamping rígido)",
                    "test_acc": accuracy_score(y_te, pred_lp_te),
                    "test_f1": f1_score(y_te, pred_lp_te, average="macro"),
                    "val_acc": accuracy_score(y_val, clf_lp.predict(X_val)),
                    "pseudo_acc": lp_pseudo_acc,
                    "train_time_s": t_lp,
                }
            )

            # 5. Teto Supervisionado (100% dos rótulos do treino)
            t0 = time.perf_counter()
            clf_ceil = make_baseline(seed=seed).fit(X_tr, y_tr)
            t_ceil = time.perf_counter() - t0
            pred_ceil_te = clf_ceil.predict(X_te)
            rows.append(
                {
                    "seed": seed,
                    "frac": frac,
                    "n_labeled": len(y_tr),
                    "model": "Supervisionado (100%)",
                    "hyperparams": "100% rótulos (teto)",
                    "test_acc": accuracy_score(y_te, pred_ceil_te),
                    "test_f1": f1_score(y_te, pred_ceil_te, average="macro"),
                    "val_acc": accuracy_score(y_val, clf_ceil.predict(X_val)),
                    "pseudo_acc": 1.0,
                    "train_time_s": t_ceil,
                }
            )

    return pd.DataFrame(rows), pd.DataFrame(iter_rows)


def compute_wilcoxon_tests(df_results: pd.DataFrame) -> pd.DataFrame:
    """
    Executa o teste pareado de Wilcoxon entre os métodos principais (10 sementes por fração)
    tanto para Acurácia quanto para F1 Macro.
    """
    comparisons = [
        ("Label Spreading", "Baseline (SVC)"),
        ("Self-Training", "Baseline (SVC)"),
        ("Label Spreading", "Self-Training"),
    ]
    stat_rows = []

    for frac in sorted(df_results["frac"].unique()):
        sub = df_results[df_results["frac"] == frac]
        for m1, m2 in comparisons:
            s1 = sub[sub["model"] == m1].sort_values("seed")
            s2 = sub[sub["model"] == m2].sort_values("seed")

            diff_acc = s1["test_acc"].values - s2["test_acc"].values
            diff_f1 = s1["test_f1"].values - s2["test_f1"].values

            try:
                _, p_acc = wilcoxon(diff_acc, zero_method="zsplit")
            except Exception:
                p_acc = 1.0

            try:
                _, p_f1 = wilcoxon(diff_f1, zero_method="zsplit")
            except Exception:
                p_f1 = 1.0

            stat_rows.append(
                {
                    "frac": frac,
                    "comparison": f"{m1} vs {m2}",
                    "mean_diff_acc": float(np.mean(diff_acc)),
                    "std_diff_acc": float(np.std(diff_acc, ddof=1)),
                    "p_value_acc": float(p_acc),
                    "significant_acc_005": bool(p_acc < 0.05),
                    "mean_diff_f1": float(np.mean(diff_f1)),
                    "p_value_f1": float(p_f1),
                    "significant_f1_005": bool(p_f1 < 0.05),
                }
            )

    return pd.DataFrame(stat_rows)


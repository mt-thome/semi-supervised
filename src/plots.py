"""Funções de visualização e geração de figuras para o experimento Digits."""

from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

from .data import split_and_scale, mask_labels
from .models import make_baseline, make_self_training, make_label_spreading

sns.set_theme(style="whitegrid", context="notebook", palette="deep")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 150


def plot_exploratory_analysis(
    digits, save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Figura 1: Amostras do dataset Digits (8x8), distribuição de classes e projeção PCA 2D.
    """
    X, y, images = digits.data, digits.target, digits.images
    fig = plt.figure(figsize=(16, 5.2))
    gs = fig.add_gridspec(2, 7, width_ratios=[1, 1, 1, 1, 1, 2.5, 2.8], wspace=0.45, hspace=0.35)

    # 10 imagens (2x5 nas primeiras 5 colunas)
    for c in range(10):
        row = c // 5
        col = c % 5
        ax = fig.add_subplot(gs[row, col])
        idx = np.where(y == c)[0][0]
        ax.imshow(images[idx], cmap="binary", interpolation="nearest")
        ax.set_title(f"Dígito {c}", fontsize=10, fontweight="bold")
        ax.axis("off")

    # Distribuição das classes
    ax_bar = fig.add_subplot(gs[:, 5])
    unique, counts = np.unique(y, return_counts=True)
    bars = ax_bar.bar(unique, counts, color=sns.color_palette("Blues_d", 10), edgecolor="black", alpha=0.85)
    ax_bar.set_xticks(unique)
    ax_bar.set_xlabel("Classe (Dígito)")
    ax_bar.set_ylabel("Número de amostras")
    ax_bar.set_title("Distribuição das Classes (Total = 1.797)", fontweight="bold")
    ax_bar.set_ylim(0, max(counts) * 1.15)
    for bar, count in zip(bars, counts):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 3,
            str(count),
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # PCA 2D
    ax_pca = fig.add_subplot(gs[:, 6])
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    scatter = ax_pca.scatter(
        X_pca[:, 0], X_pca[:, 1], c=y, cmap="tab10", s=14, alpha=0.75, edgecolors="none"
    )
    var_exp = pca.explained_variance_ratio_ * 100
    ax_pca.set_xlabel(f"PC1 ({var_exp[0]:.1f}% var.)")
    ax_pca.set_ylabel(f"PC2 ({var_exp[1]:.1f}% var.)")
    ax_pca.set_title("Projeção PCA 2D (Estrutura de Clusters)", fontweight="bold")
    cbar = plt.colorbar(scatter, ax=ax_pca, ticks=range(10))
    cbar.set_label("Classe")

    fig.suptitle("Análise Exploratória do Dataset Digits (8x8 pixels, 64 atributos, 10 classes)", fontsize=13, fontweight="bold", y=1.02)
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_learning_curves(
    df_results: pd.DataFrame, save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Figura 2: Curva de desempenho (Acurácia e F1 Macro no Teste) x Fração rotulada
    com barras de erro (desvio padrão nas 10 sementes).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharex=True)
    main_models = ["Baseline (SVC)", "Self-Training", "Label Spreading", "Label Propagation"]
    palette = {
        "Baseline (SVC)": "#d95f02",
        "Self-Training": "#1b9e77",
        "Label Spreading": "#7570b3",
        "Label Propagation": "#e7298a",
    }
    markers = {
        "Baseline (SVC)": "o",
        "Self-Training": "s",
        "Label Spreading": "^",
        "Label Propagation": "D",
    }

    ceil_acc = df_results[df_results["model"] == "Supervisionado (100%)"]["test_acc"].mean()
    ceil_f1 = df_results[df_results["model"] == "Supervisionado (100%)"]["test_f1"].mean()

    for ax, metric, title, ceil_val in zip(
        axes,
        ["test_acc", "test_f1"],
        ["Acurácia no Teste", "F1-Score Macro no Teste"],
        [ceil_acc, ceil_f1],
    ):
        for model in main_models:
            sub = df_results[df_results["model"] == model]
            if sub.empty:
                continue
            stats = sub.groupby("frac")[metric].agg(["mean", "std"]).reset_index()
            pct = stats["frac"] * 100
            ax.errorbar(
                pct,
                stats["mean"],
                yerr=stats["std"],
                label=model,
                color=palette[model],
                marker=markers[model],
                linewidth=2.2,
                markersize=7,
                capsize=4,
            )

        ax.axhline(
            ceil_val,
            color="black",
            linestyle="--",
            linewidth=1.5,
            label=f"Teto Supervisionado 100% ({ceil_val:.3f})",
        )
        ax.set_xticks([2, 5, 10, 20])
        ax.set_xticklabels(["2%\n(~22 rót.)", "5%\n(~54 rót.)", "10%\n(~108 rót.)", "20%\n(~215 rót.)"])
        ax.set_xlabel("Fração de Rótulos Mantidos no Treino")
        ax.set_ylabel(title)
        ax.set_title(f"{title} vs. Fração Rotulada (10 sementes)", fontweight="bold")
        ax.set_ylim(0.50, 1.01)
        ax.legend(loc="lower right", frameon=True)

    fig.suptitle(
        "Comparação de Desempenho Indutivo: Baseline vs. Semissupervisionado",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_label_spreading_heatmaps(
    df_ls_grid: pd.DataFrame, frac_focus: float = 0.05, save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Figura 4: Heatmap de sensibilidade do Label Spreading (k_neighbors/gamma x alpha)
    no conjunto de validação para uma fração representativa (default 5%) e média geral.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))

    sub_knn = df_ls_grid[(df_ls_grid["frac"] == frac_focus) & (df_ls_grid["kernel"] == "knn")]
    piv_knn = sub_knn.pivot_table(index="n_neighbors", columns="alpha", values="val_acc", aggfunc="mean")
    sns.heatmap(
        piv_knn,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        vmin=0.75,
        vmax=0.96,
        ax=axes[0],
        cbar_kws={"label": "Acurácia na Validação"},
    )
    axes[0].set_title(f"Label Spreading (kernel='knn') — Fração {int(frac_focus*100)}%", fontweight="bold")
    axes[0].set_xlabel("Hiperparâmetro α (clamping factor)")
    axes[0].set_ylabel("Número de vizinhos (n_neighbors)")

    sub_rbf = df_ls_grid[(df_ls_grid["frac"] == frac_focus) & (df_ls_grid["kernel"] == "rbf")]
    piv_rbf = sub_rbf.pivot_table(index="gamma", columns="alpha", values="val_acc", aggfunc="mean")
    sns.heatmap(
        piv_rbf,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        vmin=0.75,
        vmax=0.96,
        ax=axes[1],
        cbar_kws={"label": "Acurácia na Validação"},
    )
    axes[1].set_title(f"Label Spreading (kernel='rbf') — Fração {int(frac_focus*100)}%", fontweight="bold")
    axes[1].set_xlabel("Hiperparâmetro α (clamping factor)")
    axes[1].set_ylabel("Largura do kernel RBF (γ)")

    fig.suptitle(
        "Análise de Sensibilidade de Hiperparâmetros do Label Spreading (Validação)",
        fontsize=13,
        fontweight="bold",
        y=1.03,
    )
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_self_training_sensitivity(
    df_st_grid: pd.DataFrame,
    df_st_iters: pd.DataFrame,
    frac_focus: float = 0.05,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Figura 5: Sensibilidade do Self-Training ao limiar/critério e evolução de pseudo-rótulos
    e acurácia por iteração (confirmation bias).
    """
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))

    # Painel 1: Desempenho no teste por config e fração
    sns.lineplot(
        data=df_st_grid,
        x="frac",
        y="test_acc",
        hue="config_label",
        style="config_label",
        markers=True,
        dashes=False,
        errorbar="sd",
        ax=axes[0],
        linewidth=2,
    )
    axes[0].set_xticks([0.02, 0.05, 0.10, 0.20])
    axes[0].set_xticklabels(["2%", "5%", "10%", "20%"])
    axes[0].set_xlabel("Fração Rotulada")
    axes[0].set_ylabel("Acurácia no Teste")
    axes[0].set_title("Desempenho vs. Configuração", fontweight="bold")
    axes[0].legend(title="Critério / Limiar", fontsize=8.5)

    # Painel 2: Quantidade acumulada de pseudo-rótulos por iteração (na fração de foco)
    sub_it = df_st_iters[df_st_iters["frac"] == frac_focus]
    hue_col = "config_label" if "config_label" in sub_it.columns else "threshold"
    sns.lineplot(
        data=sub_it,
        x="iteration",
        y="n_cumulative",
        hue=hue_col,
        style=hue_col,
        markers=True,
        dashes=False,
        ax=axes[1],
        linewidth=2,
    )
    axes[1].set_xlabel("Iteração do Self-Training")
    axes[1].set_ylabel("Total Acumulado de Pseudo-Rótulos")
    axes[1].set_title(f"Pseudo-Rótulos Aceitos por Iteração ({int(frac_focus*100)}% rót.)", fontweight="bold")
    axes[1].legend(title="Critério / Limiar", fontsize=8.5)

    # Painel 3: Acurácia dos pseudo-rótulos acumulados por iteração (Confirmation Bias)
    sns.lineplot(
        data=sub_it,
        x="iteration",
        y="cum_pseudo_acc",
        hue=hue_col,
        style=hue_col,
        markers=True,
        dashes=False,
        ax=axes[2],
        linewidth=2,
    )
    axes[2].set_xlabel("Iteração do Self-Training")
    axes[2].set_ylabel("Acurácia Acumulada dos Pseudo-Rótulos")
    axes[2].set_title(f"Degradação / Confirmation Bias ({int(frac_focus*100)}% rót.)", fontweight="bold")
    axes[2].legend(title="Critério / Limiar", fontsize=8.5)

    fig.suptitle(
        "Análise do Self-Training: Sensibilidade ao Limiar e Acúmulo de Erros por Iteração",
        fontsize=13,
        fontweight="bold",
        y=1.03,
    )
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_transductive_accuracy(
    df_results: pd.DataFrame, save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Figura 6: Acurácia dos pseudo-rótulos (avaliação transdutiva nos dados não rotulados do treino)
    por método e fração.
    """
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    sub = df_results[
        df_results["model"].isin(["Baseline (SVC)", "Self-Training", "Label Spreading", "Label Propagation"])
    ]
    sns.barplot(
        data=sub,
        x="frac",
        y="pseudo_acc",
        hue="model",
        errorbar="sd",
        capsize=0.08,
        ax=ax,
        palette=["#d95f02", "#1b9e77", "#7570b3", "#e7298a"],
    )
    ax.set_xticklabels(["2% (~22 rót.)", "5% (~54 rót.)", "10% (~108 rót.)", "20% (~215 rót.)"])
    ax.set_xlabel("Fração Rotulada no Treino")
    ax.set_ylabel("Acurácia nos Dados Não Rotulados (Pseudo-Rótulos)")
    ax.set_ylim(0.5, 1.02)
    ax.set_title("Avaliação Transdutiva: Qualidade dos Pseudo-Rótulos por Método e Fração", fontweight="bold")
    ax.legend(title="Método", loc="lower right")

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_confusion_matrices_and_changes(
    digits,
    best_st: Dict[float, Dict[str, Any]],
    best_ls: Dict[float, Dict[str, Any]],
    frac: float = 0.05,
    seed: int = 42,
    save_path_cm: Optional[Path] = None,
    save_path_examples: Optional[Path] = None,
) -> Tuple[plt.Figure, plt.Figure]:
    """
    Figura 7: Matrizes de confusão (Baseline vs Self-Training vs Label Spreading na fração 5%)
    e exemplos de imagens de teste que mudaram de erro para acerto (e vice-versa).
    """
    X, y, images = digits.data, digits.target, digits.images
    # Recuperar também os índices do conjunto de teste para exibir as imagens 8x8 originais
    indices = np.arange(len(y))
    idx_tmp, idx_te, y_tmp, y_te = train_test_split(
        indices, y, test_size=0.30, stratify=y, random_state=seed
    )
    X_tr, X_val, X_te, y_tr, y_val, y_te_check, _ = split_and_scale(X, y, seed=seed)
    y_semi = mask_labels(y_tr, frac=frac, seed=seed)
    lab_mask = y_semi != -1

    clf_base = make_baseline(seed=seed).fit(X_tr[lab_mask], y_tr[lab_mask])
    st_cfg = best_st[frac]
    clf_st = make_self_training(
        seed=seed,
        threshold=st_cfg["threshold"],
        criterion=st_cfg["criterion"],
        k_best=st_cfg["k_best"],
    ).fit(X_tr, y_semi)
    ls_cfg = best_ls[frac]
    clf_ls = make_label_spreading(
        kernel=ls_cfg["kernel"],
        n_neighbors=ls_cfg["n_neighbors"],
        gamma=ls_cfg["gamma"],
        alpha=ls_cfg["alpha"],
    ).fit(X_tr, y_semi)

    pred_base = clf_base.predict(X_te)
    pred_st = clf_st.predict(X_te)
    pred_ls = clf_ls.predict(X_te)

    # Figura 7A: Matrizes de Confusão
    fig_cm, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for ax, pred, name, cmap in zip(
        axes,
        [pred_base, pred_st, pred_ls],
        ["Baseline (SVC)", "Self-Training", "Label Spreading"],
        ["Oranges", "Greens", "Purples"],
    ):
        cm = confusion_matrix(y_te, pred)
        acc = accuracy_score(y_te, pred)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap=cmap,
            cbar=False,
            ax=ax,
             linewidths=0.5,
        )
        ax.set_title(f"{name}\n(Acurácia = {acc*100:.1f}%)", fontweight="bold")
        ax.set_xlabel("Classe Predita")
        ax.set_ylabel("Classe Real")

    fig_cm.suptitle(
        f"Matrizes de Confusão no Teste — Fração Rotulada = {int(frac*100)}% (Seed {seed})",
        fontsize=13,
        fontweight="bold",
        y=1.03,
    )
    if save_path_cm:
        fig_cm.savefig(save_path_cm, bbox_inches="tight")

    # Figura 7B: Exemplos onde o Label Spreading corrigiu erros do Baseline (e onde errou)
    fixed_idx = np.where((pred_base != y_te) & (pred_ls == y_te))[0]
    broken_idx = np.where((pred_base == y_te) & (pred_ls != y_te))[0]

    fig_ex, axes_ex = plt.subplots(2, 5, figsize=(13, 5.2))
    for i in range(5):
        ax = axes_ex[0, i]
        if i < len(fixed_idx):
            k = fixed_idx[i]
            orig_img = images[idx_te[k]]
            ax.imshow(orig_img, cmap="binary", interpolation="nearest")
            ax.set_title(
                f"Real: {y_te[k]}\nBase: {pred_base[k]} → LS: {pred_ls[k]}",
                fontsize=9.5,
                color="#1b7837",
                fontweight="bold",
            )
        ax.axis("off")

    for i in range(5):
        ax = axes_ex[1, i]
        if i < len(broken_idx):
            k = broken_idx[i]
            orig_img = images[idx_te[k]]
            ax.imshow(orig_img, cmap="binary", interpolation="nearest")
            ax.set_title(
                f"Real: {y_te[k]}\nBase: {pred_base[k]} → LS: {pred_ls[k]}",
                fontsize=9.5,
                color="#b2182b",
                fontweight="bold",
            )
        else:
            ax.text(0.5, 0.5, "(Sem mais casos)", ha="center", va="center", fontsize=9)
        ax.axis("off")

    axes_ex[0, 0].text(
        -0.35,
        0.5,
        "Corrigidos pelo\nLabel Spreading\n(Erro → Acerto)",
        transform=axes_ex[0, 0].transAxes,
        fontsize=10,
        fontweight="bold",
        color="#1b7837",
        va="center",
        ha="right",
    )
    axes_ex[1, 0].text(
        -0.35,
        0.5,
        "Degradados pelo\nLabel Spreading\n(Acerto → Erro)",
        transform=axes_ex[1, 0].transAxes,
        fontsize=10,
        fontweight="bold",
        color="#b2182b",
        va="center",
        ha="right",
    )

    fig_ex.suptitle(
        f"Exemplos Qualitativos no Teste ({int(frac*100)}% rotulados): Mudanças entre Baseline e Label Spreading",
        fontsize=12.5,
        fontweight="bold",
        y=1.02,
    )
    if save_path_examples:
        fig_ex.savefig(save_path_examples, bbox_inches="tight")

    return fig_cm, fig_ex


def plot_training_time(
    df_results: pd.DataFrame, save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Figura 8: Tempo de treinamento (em milissegundos) por método e fração rotulada.
    """
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    df_plot = df_results.copy()
    df_plot["train_time_ms"] = df_plot["train_time_s"] * 1000.0
    sub = df_plot[
        df_plot["model"].isin(["Baseline (SVC)", "Self-Training", "Label Spreading", "Label Propagation"])
    ]
    sns.barplot(
        data=sub,
        x="frac",
        y="train_time_ms",
        hue="model",
        errorbar="sd",
        capsize=0.08,
        ax=ax,
        palette=["#d95f02", "#1b9e77", "#7570b3", "#e7298a"],
    )
    ax.set_xticklabels(["2%", "5%", "10%", "20%"])
    ax.set_xlabel("Fração Rotulada no Treino")
    ax.set_ylabel("Tempo de Treino (ms) — Escala Log")
    ax.set_yscale("log")
    ax.set_title("Custo Computacional: Tempo de Treinamento por Método e Fração", fontweight="bold")
    ax.legend(title="Método", loc="upper left")

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig

"""Utilidades compartidas para interpretación de métricas y visualizaciones.

Usado por evaluate.py, classifier.py y presentation_report.py para generar
gráficos y textos contextualizados para la presentación del TP.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

PRESENTATION_DIR = Path("exports/presentacion")
PALETTE = {"semantic": "#2E75B6", "bm25": "#1F4E79", "accent": "#E07B39", "neutral": "#6B7280"}
LEVEL_NAMES = {1: "Constitución", 2: "Ley/Cód.", 3: "D.Supremo", 4: "Resolución", 5: "Otros"}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 120,
})


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


# ─── Interpretaciones contextualizadas (búsqueda legal) ─────────────────────

def interpret_precision(value: float, k: int) -> str:
    pct = value * 100
    if value >= 0.7:
        quality = "alta"
        impl = (
            f"De cada {k} leyes mostradas, en promedio {pct:.0f}% son realmente pertinentes. "
            "Un abogado pierde poco tiempo revisando resultados irrelevantes."
        )
    elif value >= 0.5:
        quality = "moderada"
        impl = (
            f"Aproximadamente la mitad de los {k} resultados son relevantes. "
            "El usuario debe filtrar manualmente varios títulos antes de encontrar la norma correcta."
        )
    else:
        quality = "baja"
        impl = (
            f"Solo {pct:.0f}% de los resultados son pertinentes. "
            "La búsqueda genera ruido: muchas normas mostradas no responden a la consulta."
        )
    return f"Precisión@{k} = {value:.3f} ({quality}). {impl}"


def interpret_recall(value: float, k: int) -> str:
    pct = value * 100
    if value >= 0.6:
        quality = "alta cobertura"
        impl = (
            f"El motor recupera {pct:.0f}% de las palabras clave jurídicas esperadas en el top-{k}. "
            "Útil cuando el abogado usa vocabulario distinto al del título formal de la ley."
        )
    elif value >= 0.35:
        quality = "cobertura parcial"
        impl = (
            f"Se cubren {pct:.0f}% de los términos relevantes. "
            "Puede haber normas pertinentes fuera del top-{k} que el abogado no verá."
        ).replace("{k}", str(k))
    else:
        quality = "cobertura limitada"
        impl = (
            f"Solo {pct:.0f}% de los conceptos jurídicos buscados aparecen. "
            "Alto riesgo de no encontrar la norma aplicable si el título usa formulación distinta."
        )
    return f"Recall@{k} = {value:.3f} ({quality}). {impl}"


def interpret_mrr(value: float, k: int) -> str:
    if value >= 0.75:
        pos = "en la primera o segunda posición"
        impl = "El abogado encuentra la ley relevante casi de inmediato, sin desplazarse por la lista."
    elif value >= 0.5:
        pos = "en las primeras posiciones del ranking"
        impl = "La norma correcta suele aparecer pronto, pero no siempre en el primer resultado."
    else:
        pos = "lejos del inicio del ranking"
        impl = (
            "La ley pertinente aparece tarde o no aparece. "
            "En la práctica, el usuario podría abandonar la búsqueda antes de encontrarla."
        )
    rank_equiv = 1 / value if value > 0 else float("inf")
    rank_str = f"posición {rank_equiv:.1f}" if value > 0 else "sin acierto"
    return (
        f"MRR@{k} = {value:.3f} (equivalente a acertar en {rank_str}). "
        f"Indica si el primer resultado relevante aparece {pos}. {impl}"
    )


def interpret_ndcg(semantic: float, bm25: float, k: int) -> str:
    diff = semantic - bm25
    winner = "semántico" if diff >= 0 else "BM25"
    loser = "BM25" if diff >= 0 else "semántico"
    better = max(semantic, bm25)
    worse = min(semantic, bm25)

    if better >= 0.75:
        quality = "excelente ordenamiento"
        user_exp = "Las leyes más pertinentes aparecen arriba; la experiencia de búsqueda es fluida."
    elif better >= 0.55:
        quality = "ordenamiento aceptable"
        user_exp = "Los resultados relevantes están presentes pero no siempre en las mejores posiciones."
    else:
        quality = "ordenamiento deficiente"
        user_exp = "Los resultados útiles quedan enterrados; el abogado debe revisar muchos títulos."

    gap = abs(diff)
    if gap < 0.05:
        comp = f"Ambos motores ordenan de forma similar (diferencia {gap:.3f})."
    else:
        comp = (
            f"El motor {winner} ordena mejor (+{gap:.3f} NDCG): "
            f"coloca las normas pertinentes más arriba que {loser} "
            f"({better:.3f} vs {worse:.3f})."
        )

    return (
        f"NDCG@{k} mide calidad del ranking (1.0 = perfecto). "
        f"{quality.capitalize()}. {user_exp} {comp}"
    )


def interpret_f_beta(f1: float, f2: float, context: str = "vigencia") -> str:
    if context == "vigencia":
        risk = (
            "Un falso negativo (marcar derogada como vigente) es crítico: "
            "el abogado podría citar una ley sin efecto. "
            "F2 penaliza más los falsos negativos que F1."
        )
    else:
        risk = "F2 da mayor peso al recall, útil cuando omitir una clase es más costoso."

    f2_vs_f1 = "superior" if f2 > f1 else "similar" if abs(f2 - f1) < 0.02 else "inferior"
    return (
        f"F1 = {f1:.4f} (balance precisión-recall). "
        f"F2 = {f2:.4f} (recall {f2_vs_f1} a F1). {risk}"
    )


# ─── Gráficos para presentación ──────────────────────────────────────────────

def plot_metrics_comparison(
    metrics_semantic: Dict[str, float],
    metrics_bm25: Dict[str, float],
    k: int,
    out_path: Path,
) -> None:
    """Diapositiva: barras agrupadas Semántico vs BM25."""
    labels = [f"Prec@{k}", f"Rec@{k}", f"MRR@{k}", f"NDCG@{k}"]
    keys = ["precision", "recall", "mrr", "ndcg"]
    sem_vals = [metrics_semantic[k] for k in keys]
    bm25_vals = [metrics_bm25[k] for k in keys]

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, sem_vals, width, label="Semántico (FAISS)", color=PALETTE["semantic"])
    bars2 = ax.bar(x + width / 2, bm25_vals, width, label="BM25 (léxico)", color=PALETTE["bm25"])
    ax.bar_label(bars1, fmt="%.2f", padding=3, fontsize=9)
    ax.bar_label(bars2, fmt="%.2f", padding=3, fontsize=9)
    ax.set_ylabel("Valor de métrica (0–1)")
    ax.set_title("Comparación de motores de búsqueda legal", fontsize=14, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right")
    ax.axhline(0.5, color=PALETTE["neutral"], linestyle="--", alpha=0.4, label="_umbral referencia")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_per_query_heatmap(
    query_labels: List[str],
    sem_matrix: np.ndarray,
    bm25_matrix: np.ndarray,
    metric_names: List[str],
    out_path: Path,
) -> None:
    """Diapositiva: heatmap por consulta (diferencia semántico − BM25)."""
    diff = sem_matrix - bm25_matrix
    fig, axes = plt.subplots(1, 2, figsize=(14, max(6, len(query_labels) * 0.35)))

    for ax, data, title in [
        (axes[0], sem_matrix, "Semántico"),
        (axes[1], diff, "Ventaja semántico vs BM25"),
    ]:
        sns.heatmap(
            data,
            annot=True,
            fmt=".2f",
            cmap="Blues" if title == "Semántico" else "RdBu_r",
            center=0 if title != "Semántico" else None,
            xticklabels=metric_names,
            yticklabels=[q[:40] + "…" if len(q) > 40 else q for q in query_labels],
            ax=ax,
            cbar_kws={"shrink": 0.8},
        )
        ax.set_title(title, fontweight="bold")

    fig.suptitle("Métricas por consulta jurídica", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_error_summary(
    fp_counts: Dict[str, int],
    fn_counts: Dict[str, int],
    out_path: Path,
) -> None:
    """Diapositiva: palabras frecuentes en falsos positivos vs falsos negativos."""
    top_fp = sorted(fp_counts.items(), key=lambda x: -x[1])[:12]
    top_fn = sorted(fn_counts.items(), key=lambda x: -x[1])[:12]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, data, title, color in [
        (axes[0], top_fp, "Falsos positivos\n(resultados irrelevantes mostrados)", PALETTE["accent"]),
        (axes[1], top_fn, "Falsos negativos\n(conceptos no recuperados)", PALETTE["bm25"]),
    ]:
        if not data:
            ax.text(0.5, 0.5, "Sin datos", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title, fontweight="bold")
            continue
        words, counts = zip(*data)
        ax.barh(words[::-1], counts[::-1], color=color)
        ax.set_xlabel("Frecuencia en títulos")
        ax.set_title(title, fontweight="bold")

    fig.suptitle("Análisis de errores — búsqueda legal", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_cv_scores(
    fold_scores: Dict[str, List[float]],
    out_path: Path,
    title: str = "Validación cruzada estratificada",
) -> None:
    """Diapositiva: boxplot de scores por fold y métrica."""
    metrics = list(fold_scores.keys())
    data = [fold_scores[m] for m in metrics]
    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(data, labels=metrics, patch_artist=True)
    colors = [PALETTE["semantic"], PALETTE["bm25"], PALETTE["accent"], PALETTE["neutral"]]
    for patch, color in zip(bp["boxes"], colors[: len(metrics)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("Score")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_logreg_coefficients(
    feature_names: np.ndarray,
    coef: np.ndarray,
    class_idx: int,
    out_path: Path,
    legal_annotations: Optional[Dict[str, str]] = None,
) -> None:
    """Diapositiva XAI: coeficientes LogReg con anotaciones jurídicas."""
    idx_pos = np.argsort(coef)[-15:][::-1]
    idx_neg = np.argsort(coef)[:10]
    idx = np.concatenate([idx_pos, idx_neg])
    names = feature_names[idx]
    values = coef[idx]
    colors = [PALETTE["semantic"] if v > 0 else PALETTE["accent"] for v in values]

    fig, ax = plt.subplots(figsize=(11, 7))
    y_pos = np.arange(len(names))
    ax.barh(y_pos, values, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Coeficiente (peso hacia Vigente)")
    ax.set_title(
        "Explicabilidad — palabras que el modelo aprendió para vigencia",
        fontsize=13,
        fontweight="bold",
    )

    if legal_annotations:
        for i, name in enumerate(names):
            if name in legal_annotations:
                ax.annotate(
                    legal_annotations[name],
                    xy=(values[i], i),
                    xytext=(5, 0),
                    textcoords="offset points",
                    fontsize=7,
                    color="#374151",
                    va="center",
                )

    from matplotlib.patches import Patch
    ax.legend(
        handles=[
            Patch(color=PALETTE["semantic"], label="→ Predice VIGENTE"),
            Patch(color=PALETTE["accent"], label="→ Predice DEROGADA"),
        ],
        loc="lower right",
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_enhanced(
    cm: np.ndarray,
    labels: List[str],
    out_path: Path,
    title: str,
    interpretations: Optional[List[str]] = None,
) -> None:
    """Matriz de confusión con interpretación por celda."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=labels, yticklabels=labels, cbar=False)
    ax.set_xlabel("Predicción del modelo")
    ax.set_ylabel("Etiqueta real (scraper SPIJ)")
    ax.set_title(title, fontweight="bold")

    if interpretations:
        fig.text(0.5, -0.02, "\n".join(interpretations), ha="center", fontsize=8,
                 wrap=True, transform=ax.transAxes)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


LEGAL_TERM_ANNOTATIONS = {
    "derog": "Término del scraper: norma sin efecto",
    "derogado": "Alerta SPIJ de derogación",
    "derogada": "Estado explícito de invalidez",
    "sin efecto": "Frase jurídica de caducidad",
    "sustitúyese": "Modificación que puede implicar derogación implícita",
    "sustituyese": "Variante ortográfica en textos legales",
    "abrog": "Abrogación = derogación total",
    "vigente": "Confirma validez normativa",
    "modifica": "Cambio parcial (sigue vigente)",
    "aprueba": "Acto de promulgación (suele ser vigente)",
    "repeal": "Derogación en metadatos EN",
    "invalid": "Marcador de invalidez",
}

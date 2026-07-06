"""Modelos supervisados para clasificación de normas SPIJ.

Entrena y evalúa dos modelos:
    1. Vigencia (binario):      Regresión Logística sobre TF-IDF
       is_valid ∈ {1=Vigente, 0=Derogada}

    2. Jerarquía (multiclase):  Random Forest sobre features de texto y metadatos
       level ∈ {1=Constitución, 2=Ley, 3=D.Supremo, 4=Resolución, 5=Otros}

También genera análisis SHAP para explicar las predicciones.

Uso:
    python classifier.py --task vigencia
    python classifier.py --task jerarquia
    python classifier.py --task ambos --shap

Archivos generados en data/models/:
    vigencia_model.pkl      — pipeline scikit-learn (TF-IDF + LogReg)
    jerarquia_model.pkl     — pipeline scikit-learn (TF-IDF + RandomForest)
    vigencia_report.txt     — classification report
    jerarquia_report.txt    — classification report
    shap_vigencia.png       — gráfico SHAP beeswarm
    shap_jerarquia.png      — gráfico SHAP summary
"""

import argparse
import logging
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from db import get_session
from models import Norm, NormVersion
from utils import setup_logging

setup_logging()
LOG = logging.getLogger(__name__)

MODEL_DIR = Path("data/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


# ─── Extracción de features ──────────────────────────────────────────────────

def _strip_html(text: str) -> str:
    if not text:
        return ""
    if "<" in text and ">" in text:
        return BeautifulSoup(text, "html.parser").get_text(separator=" ", strip=True)
    return text


def load_dataset(session) -> pd.DataFrame:
    """Carga el dataset de normas desde la BD con sus textos completos."""
    norms = session.query(Norm).filter(Norm.is_valid.isnot(None)).all()
    rows = []
    for norm in norms:
        latest = (
            session.query(NormVersion)
            .filter_by(norm_id=norm.id)
            .order_by(NormVersion.version_date.desc())
            .first()
        )
        raw = _strip_html(latest.raw_text if latest else "")
        # Texto combinado para TF-IDF: tipo + título + primeros 2000 chars
        combined_text = " ".join(filter(None, [
            norm.type or "",
            norm.title or "",
            raw[:2000],
        ]))
        rows.append({
            "norm_id":        norm.id,
            "type":           norm.type or "",
            "title":          norm.title or "",
            "text":           combined_text,
            "is_valid":       int(norm.is_valid) if norm.is_valid is not None else None,
            "level":          int(norm.level) if norm.level else 5,
            "has_derogation": int("derog" in raw.lower() or "sin efecto" in raw.lower()),
            "text_length":    len(raw),
        })
    df = pd.DataFrame(rows)
    LOG.info("Dataset cargado: %d normas con etiqueta de vigencia.", len(df))
    return df


# ─── Modelo 1: Clasificación de Vigencia ────────────────────────────────────

def train_vigencia(df: pd.DataFrame, run_shap: bool = False):
    """Entrena Regresión Logística (TF-IDF) para predecir is_valid."""
    df_v = df.dropna(subset=["is_valid"]).copy()
    if len(df_v) < 20:
        LOG.error("Datos insuficientes para entrenar el modelo de vigencia (mínimo 20 normas etiquetadas).")
        return

    X, y = df_v["text"], df_v["is_valid"].astype(int)
    LOG.info("Distribución de clases — Vigencia: %s", dict(y.value_counts()))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=RANDOM_STATE
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=20_000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=2,
        )),
        ("clf", LogisticRegression(
            C=1.0,
            class_weight="balanced",  # maneja desbalance de clases
            max_iter=1000,
            random_state=RANDOM_STATE,
        )),
    ])

    # Validación cruzada 5-fold en train
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1")
    LOG.info("CV F1 (5-fold) — Vigencia: %.3f ± %.3f", cv_scores.mean(), cv_scores.std())

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    report = classification_report(
        y_test, y_pred,
        target_names=["Derogada (0)", "Vigente (1)"],
        digits=4,
    )
    print("\n" + "=" * 55)
    print("  MODELO DE VIGENCIA — Regresión Logística (TF-IDF)")
    print("=" * 55)
    print(f"  CV F1 (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print("\n" + report)

    # Guardar reporte
    report_path = MODEL_DIR / "vigencia_report.txt"
    report_path.write_text(
        f"CV F1 (5-fold): {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}\n\n{report}"
    )

    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Derogada", "Vigente"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Matriz de Confusión — Vigencia", fontweight="bold")
    plt.tight_layout()
    cm_path = MODEL_DIR / "vigencia_confusion_matrix.png"
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    LOG.info("Matriz de confusión guardada: %s", cm_path)

    # Guardar modelo
    model_path = MODEL_DIR / "vigencia_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)
    LOG.info("Modelo guardado: %s", model_path)

    # SHAP
    if run_shap:
        _shap_logreg(pipeline, X_test, y_test, "vigencia")

    return pipeline


# ─── Modelo 2: Clasificación Jerárquica ─────────────────────────────────────

def train_jerarquia(df: pd.DataFrame, run_shap: bool = False):
    """Entrena Random Forest (TF-IDF) para predecir el nivel jerárquico."""
    df_j = df.dropna(subset=["level"]).copy()
    df_j = df_j[df_j["level"].between(1, 5)]

    # Filtrar clases con muy pocos ejemplos (≥5 para poder hacer split)
    class_counts = df_j["level"].value_counts()
    valid_classes = class_counts[class_counts >= 5].index
    df_j = df_j[df_j["level"].isin(valid_classes)]

    if len(df_j) < 20:
        LOG.error("Datos insuficientes para entrenar el modelo de jerarquía (mínimo 20 normas).")
        return

    X, y = df_j["text"], df_j["level"].astype(int)
    LOG.info("Distribución de clases — Jerarquía: %s", dict(y.value_counts().sort_index()))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=RANDOM_STATE
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=15_000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=2,
        )),
        ("clf", RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1_macro")
    LOG.info("CV F1-macro (5-fold) — Jerarquía: %.3f ± %.3f", cv_scores.mean(), cv_scores.std())

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    level_names = {1: "Constitución", 2: "Ley/Cód.", 3: "D.Supremo", 4: "Resolución", 5: "Otros"}
    target_names = [level_names.get(l, str(l)) for l in sorted(valid_classes)]

    report = classification_report(y_test, y_pred, target_names=target_names, digits=4)
    print("\n" + "=" * 55)
    print("  MODELO DE JERARQUÍA — Random Forest (TF-IDF)")
    print("=" * 55)
    print(f"  CV F1-macro (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print("\n" + report)

    report_path = MODEL_DIR / "jerarquia_report.txt"
    report_path.write_text(
        f"CV F1-macro (5-fold): {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}\n\n{report}"
    )

    # Importancia de características (top 20)
    tfidf = pipeline.named_steps["tfidf"]
    rf    = pipeline.named_steps["clf"]
    feature_names = tfidf.get_feature_names_out()
    importances   = rf.feature_importances_
    top_idx = np.argsort(importances)[-20:][::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(feature_names[top_idx][::-1], importances[top_idx][::-1], color="#1F4E79")
    ax.set_xlabel("Importancia (Gini)")
    ax.set_title("Top 20 features — Random Forest Jerarquía", fontweight="bold")
    plt.tight_layout()
    feat_path = MODEL_DIR / "jerarquia_feature_importance.png"
    fig.savefig(feat_path, dpi=150)
    plt.close(fig)
    LOG.info("Importancia de features guardada: %s", feat_path)

    model_path = MODEL_DIR / "jerarquia_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)
    LOG.info("Modelo guardado: %s", model_path)

    if run_shap:
        _shap_rf(pipeline, X_test, "jerarquia")

    return pipeline


# ─── SHAP ────────────────────────────────────────────────────────────────────

def _shap_logreg(pipeline, X_test, y_test, task_name: str):
    """Genera análisis SHAP para el modelo de Regresión Logística."""
    try:
        import shap
        tfidf = pipeline.named_steps["tfidf"]
        clf   = pipeline.named_steps["clf"]
        X_test_tfidf = tfidf.transform(X_test)
        explainer = shap.LinearExplainer(clf, X_test_tfidf, feature_perturbation="interventional")
        shap_values = explainer.shap_values(X_test_tfidf)
        feature_names = tfidf.get_feature_names_out()

        plt.figure(figsize=(12, 6))
        shap.summary_plot(
            shap_values, X_test_tfidf,
            feature_names=feature_names,
            max_display=20,
            show=False,
        )
        plt.title(f"SHAP Summary — Modelo de {task_name.capitalize()}", fontweight="bold")
        plt.tight_layout()
        path = MODEL_DIR / f"shap_{task_name}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        LOG.info("SHAP guardado: %s", path)
    except ImportError:
        LOG.warning("SHAP no instalado. Instala con: pip install shap --break-system-packages")


def _shap_rf(pipeline, X_test, task_name: str):
    """Genera análisis SHAP para el modelo de Random Forest."""
    try:
        import shap
        tfidf = pipeline.named_steps["tfidf"]
        clf   = pipeline.named_steps["clf"]
        X_test_tfidf = tfidf.transform(X_test)
        # TreeExplainer es más eficiente para RandomForest
        explainer = shap.TreeExplainer(clf)
        # Limitar muestra para velocidad
        sample_size = min(50, X_test_tfidf.shape[0])
        X_sample = X_test_tfidf[:sample_size]
        # Convertir a denso float32 — requerido por shap con matrices sparse de scipy
        if hasattr(X_sample, "toarray"):
            X_sample = X_sample.toarray().astype(np.float32)
        shap_values = explainer.shap_values(X_sample)
        feature_names = tfidf.get_feature_names_out()

        plt.figure(figsize=(12, 6))
        shap.summary_plot(
            shap_values[0] if isinstance(shap_values, list) else shap_values,
            X_sample, feature_names=feature_names, max_display=20, show=False,
        )
        plt.title(f"SHAP Summary — Modelo de {task_name.capitalize()}", fontweight="bold")
        plt.tight_layout()
        path = MODEL_DIR / f"shap_{task_name}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        LOG.info("SHAP guardado: %s", path)
    except ImportError:
        LOG.warning("SHAP no instalado. Instala con: pip install shap --break-system-packages")


# ─── Punto de entrada ─────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Entrena clasificadores sobre el corpus SPIJ.")
    p.add_argument("--task", choices=["vigencia", "jerarquia", "ambos"], default="ambos")
    p.add_argument("--shap", action="store_true", help="Generar análisis SHAP (más lento)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    session = get_session()
    try:
        df = load_dataset(session)
        if df.empty:
            print("❌ No hay datos. Ejecuta primero el scraper:\n  python main.py scrape --limit 200")
        else:
            if args.task in ("vigencia", "ambos"):
                train_vigencia(df, run_shap=args.shap)
            if args.task in ("jerarquia", "ambos"):
                train_jerarquia(df, run_shap=args.shap)
    finally:
        session.close()

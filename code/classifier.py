"""Modelos supervisados para clasificación de normas SPIJ.

Entrena y evalúa dos modelos:

    1. VIGENCIA (binario, a nivel de ARTÍCULO):  Regresión Logística sobre TF-IDF
       label ∈ {1=Vigente, 0=Derogado}

       El dataset proviene de build_vigencia_dataset.py: cada artículo de los
       códigos de SPIJ se etiqueta con las notas OFICIALES de derogación
       "(*) Artículo/Numeral derogado por ...". Esto reemplaza la antigua
       heurística a nivel de norma (que producía un corpus 99.3% vigente con
       etiquetas ruidosas y F1=0 para la clase Derogada).

       Se entrenan dos variantes:
         · PRINCIPAL: sobre el texto publicado por SPIJ (incluye la nota de
           vigencia). Es el clasificador operativo: filtra disposiciones
           derogadas de los resultados de búsqueda.
         · ABLACIÓN: sobre el cuerpo del artículo SIN la nota de vigencia, para
           cuantificar de forma transparente cuánta señal proviene de la
           anotación oficial vs. del contenido.

    2. JERARQUÍA (multiclase, a nivel de NORMA):  Random Forest sobre TF-IDF
       level ∈ {1=Constitución, 2=Ley/rango de ley, 3=D.Supremo,
                4=Resolución, 5=Otros}

También genera análisis SHAP para explicar las predicciones.

Uso:
    python classifier.py --task vigencia
    python classifier.py --task jerarquia
    python classifier.py --task ambos --shap
"""

import argparse
import logging
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from sqlalchemy import text as sqltext
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
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


# ─── Utilidades ──────────────────────────────────────────────────────────────

def _strip_html(text: str) -> str:
    if not text:
        return ""
    if "<" in text and ">" in text:
        return BeautifulSoup(text, "html.parser").get_text(separator=" ", strip=True)
    return text


# ─── Carga de datos ──────────────────────────────────────────────────────────

def load_vigencia_dataset(session) -> pd.DataFrame:
    """Carga el dataset de vigencia a nivel de artículo (article_samples)."""
    try:
        rows = session.execute(sqltext(
            "SELECT code_name, article_num, text, raw_text, label FROM article_samples"
        )).fetchall()
    except Exception as e:
        LOG.error("No se pudo leer article_samples: %s. Ejecuta build_vigencia_dataset.py primero.", e)
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["code_name", "article_num", "text", "raw_text", "label"])
    LOG.info("Dataset de vigencia (artículos): %d filas.", len(df))
    return df


def load_norm_dataset(session) -> pd.DataFrame:
    """Carga el dataset de normas para el clasificador de jerarquía."""
    norms = session.query(Norm).all()
    rows = []
    for norm in norms:
        latest = (
            session.query(NormVersion)
            .filter_by(norm_id=norm.id)
            .order_by(NormVersion.version_date.desc())
            .first()
        )
        raw = _strip_html(latest.raw_text if latest else "")
        # Nota: se EXCLUYE el campo `type` de las features. La etiqueta `level`
        # se deriva de `type` mediante reglas (categorize_norm), de modo que
        # incluirlo produciría fuga circular. El modelo debe inferir el nivel a
        # partir del contenido textual (título + cuerpo del documento).
        combined_text = " ".join(filter(None, [
            norm.title or "", raw[:2500],
        ]))
        rows.append({
            "norm_id": norm.id,
            "type": norm.type or "",
            "title": norm.title or "",
            "text": combined_text,
            "level": int(norm.level) if norm.level else 5,
        })
    df = pd.DataFrame(rows)
    LOG.info("Dataset de normas (jerarquía): %d filas.", len(df))
    return df


# ─── Modelo 1: Vigencia (nivel artículo) ─────────────────────────────────────

def _build_logreg_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=20_000, ngram_range=(1, 2),
                                  sublinear_tf=True, min_df=2)),
        ("clf", LogisticRegression(C=1.0, class_weight="balanced",
                                   max_iter=1000, random_state=RANDOM_STATE)),
    ])


def _cv_and_test_f1(X, y, scoring="f1"):
    """Devuelve (cv_mean, cv_std, f1_derogado_test) para una ablación rápida."""
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.15, stratify=y, random_state=RANDOM_STATE)
    pipe = _build_logreg_pipeline()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(pipe, Xtr, ytr, cv=cv, scoring=scoring)
    pipe.fit(Xtr, ytr)
    f1_derog = f1_score(yte, pipe.predict(Xte), pos_label=0)
    return scores.mean(), scores.std(), f1_derog


def train_vigencia(df: pd.DataFrame, run_shap: bool = False):
    """Entrena Regresión Logística (TF-IDF) para predecir vigencia por artículo."""
    if df.empty or len(df) < 20:
        LOG.error("Datos insuficientes para vigencia. Ejecuta build_vigencia_dataset.py.")
        return

    y = df["label"].astype(int)
    LOG.info("Distribución de clases — Vigencia: %s", dict(y.value_counts()))

    # --- Modelo PRINCIPAL: texto publicado por SPIJ (con nota de vigencia) ---
    X = df["raw_text"].fillna("")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=RANDOM_STATE)

    pipeline = _build_logreg_pipeline()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1")
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    report = classification_report(y_test, y_pred,
                                   target_names=["Derogado (0)", "Vigente (1)"], digits=4)

    # --- ABLACIÓN: cuerpo del artículo SIN la nota de vigencia ---
    abl_mean, abl_std, abl_f1d = _cv_and_test_f1(df["text"].fillna(""), y, scoring="f1")

    print("\n" + "=" * 60)
    print("  MODELO DE VIGENCIA (nivel artículo) — Reg. Logística (TF-IDF)")
    print("=" * 60)
    print(f"  Dataset: {len(df)} artículos | Derogados={int((y==0).sum())} Vigentes={int((y==1).sum())}")
    print(f"  CV F1 (5-fold, texto publicado): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print("\n" + report)
    print(f"  [Ablación] texto SIN nota de vigencia — CV F1: {abl_mean:.4f} ± {abl_std:.4f} | "
          f"F1(Derogado) test: {abl_f1d:.4f}")

    (MODEL_DIR / "vigencia_report.txt").write_text(
        f"MODELO DE VIGENCIA — nivel articulo (Regresion Logistica TF-IDF)\n"
        f"Dataset: {len(df)} articulos | Derogados={int((y==0).sum())} Vigentes={int((y==1).sum())}\n\n"
        f"CV F1 (5-fold, texto publicado por SPIJ): {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}\n\n"
        f"{report}\n"
        f"[Ablacion] texto SIN la nota oficial de vigencia:\n"
        f"  CV F1 (5-fold): {abl_mean:.4f} +/- {abl_std:.4f}\n"
        f"  F1(Derogado) en test: {abl_f1d:.4f}\n"
    )

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Derogado", "Vigente"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Matriz de Confusión — Vigencia (artículo)", fontweight="bold")
    plt.tight_layout()
    fig.savefig(MODEL_DIR / "vigencia_confusion_matrix.png", dpi=150)
    plt.close(fig)

    with open(MODEL_DIR / "vigencia_model.pkl", "wb") as f:
        pickle.dump(pipeline, f)
    LOG.info("Modelo de vigencia guardado.")

    if run_shap:
        _shap_logreg(pipeline, X_test, "vigencia")
    return pipeline


# ─── Modelo 2: Jerarquía (nivel norma) ───────────────────────────────────────

def train_jerarquia(df: pd.DataFrame, run_shap: bool = False):
    """Entrena Random Forest (TF-IDF) para predecir el nivel jerárquico."""
    df_j = df.dropna(subset=["level"]).copy()
    df_j = df_j[df_j["level"].between(1, 5)]
    class_counts = df_j["level"].value_counts()
    valid_classes = sorted(class_counts[class_counts >= 5].index)
    df_j = df_j[df_j["level"].isin(valid_classes)]
    if len(df_j) < 20:
        LOG.error("Datos insuficientes para jerarquía.")
        return

    X, y = df_j["text"], df_j["level"].astype(int)
    LOG.info("Distribución de clases — Jerarquía: %s", dict(y.value_counts().sort_index()))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=RANDOM_STATE)

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=15_000, ngram_range=(1, 2),
                                  sublinear_tf=True, min_df=2)),
        ("clf", RandomForestClassifier(n_estimators=200, max_depth=None,
                                       class_weight="balanced_subsample",
                                       random_state=RANDOM_STATE, n_jobs=-1)),
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1_macro")
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    level_names = {1: "Constitución", 2: "Ley/Cód.", 3: "D.Supremo", 4: "Resolución", 5: "Otros"}
    target_names = [level_names.get(l, str(l)) for l in valid_classes]

    report = classification_report(y_test, y_pred, target_names=target_names, digits=4)
    print("\n" + "=" * 60)
    print("  MODELO DE JERARQUÍA (nivel norma) — Random Forest (TF-IDF)")
    print("=" * 60)
    print(f"  Dataset: {len(df_j)} normas | clases={dict(y.value_counts().sort_index())}")
    print(f"  CV F1-macro (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print("\n" + report)

    (MODEL_DIR / "jerarquia_report.txt").write_text(
        f"MODELO DE JERARQUIA — nivel norma (Random Forest TF-IDF)\n"
        f"Dataset: {len(df_j)} normas | clases={dict(y.value_counts().sort_index())}\n\n"
        f"CV F1-macro (5-fold): {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}\n\n{report}"
    )

    cm = confusion_matrix(y_test, y_pred, labels=valid_classes)
    disp = ConfusionMatrixDisplay(cm, display_labels=target_names)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Greens", colorbar=False, xticks_rotation=45)
    ax.set_title("Matriz de Confusión — Jerarquía", fontweight="bold")
    plt.tight_layout()
    fig.savefig(MODEL_DIR / "jerarquia_confusion_matrix.png", dpi=150)
    plt.close(fig)

    tfidf = pipeline.named_steps["tfidf"]
    rf = pipeline.named_steps["clf"]
    feature_names = tfidf.get_feature_names_out()
    importances = rf.feature_importances_
    top_idx = np.argsort(importances)[-20:][::-1]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(feature_names[top_idx][::-1], importances[top_idx][::-1], color="#1F4E79")
    ax.set_xlabel("Importancia (Gini)")
    ax.set_title("Top 20 features — Random Forest Jerarquía", fontweight="bold")
    plt.tight_layout()
    fig.savefig(MODEL_DIR / "jerarquia_feature_importance.png", dpi=150)
    plt.close(fig)

    with open(MODEL_DIR / "jerarquia_model.pkl", "wb") as f:
        pickle.dump(pipeline, f)
    LOG.info("Modelo de jerarquía guardado.")

    if run_shap:
        _shap_rf(pipeline, X_test, "jerarquia")
    return pipeline


# ─── SHAP ────────────────────────────────────────────────────────────────────

def _shap_logreg(pipeline, X_test, task_name: str):
    try:
        import shap
        tfidf = pipeline.named_steps["tfidf"]
        clf = pipeline.named_steps["clf"]
        X_test_tfidf = tfidf.transform(X_test)
        explainer = shap.LinearExplainer(clf, X_test_tfidf, feature_perturbation="interventional")
        shap_values = explainer.shap_values(X_test_tfidf)
        feature_names = tfidf.get_feature_names_out()
        plt.figure(figsize=(12, 6))
        shap.summary_plot(shap_values, X_test_tfidf, feature_names=feature_names,
                          max_display=20, show=False)
        plt.title(f"SHAP — Modelo de {task_name.capitalize()}", fontweight="bold")
        plt.tight_layout()
        plt.savefig(MODEL_DIR / f"shap_{task_name}.png", dpi=150, bbox_inches="tight")
        plt.close()
        LOG.info("SHAP guardado: shap_%s.png", task_name)
    except Exception as e:
        LOG.warning("SHAP falló (%s): %s", task_name, e)


def _shap_rf(pipeline, X_test, task_name: str):
    try:
        import shap
        tfidf = pipeline.named_steps["tfidf"]
        clf = pipeline.named_steps["clf"]
        X_test_tfidf = tfidf.transform(X_test)
        explainer = shap.TreeExplainer(clf)
        n = min(50, X_test_tfidf.shape[0])
        X_sample = X_test_tfidf[:n]
        if hasattr(X_sample, "toarray"):
            X_sample = X_sample.toarray().astype(np.float32)
        shap_values = explainer.shap_values(X_sample)
        feature_names = tfidf.get_feature_names_out()
        vals = shap_values[0] if isinstance(shap_values, list) else shap_values
        if hasattr(vals, "ndim") and vals.ndim == 3:
            vals = vals[:, :, 0]
        plt.figure(figsize=(12, 6))
        shap.summary_plot(vals, X_sample, feature_names=feature_names, max_display=20, show=False)
        plt.title(f"SHAP — Modelo de {task_name.capitalize()}", fontweight="bold")
        plt.tight_layout()
        plt.savefig(MODEL_DIR / f"shap_{task_name}.png", dpi=150, bbox_inches="tight")
        plt.close()
        LOG.info("SHAP guardado: shap_%s.png", task_name)
    except Exception as e:
        LOG.warning("SHAP falló (%s): %s", task_name, e)


def parse_args():
    p = argparse.ArgumentParser(description="Entrena clasificadores sobre el corpus SPIJ.")
    p.add_argument("--task", choices=["vigencia", "jerarquia", "ambos"], default="ambos")
    p.add_argument("--shap", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    session = get_session()
    try:
        if args.task in ("vigencia", "ambos"):
            train_vigencia(load_vigencia_dataset(session), run_shap=args.shap)
        if args.task in ("jerarquia", "ambos"):
            train_jerarquia(load_norm_dataset(session), run_shap=args.shap)
    finally:
        session.close()

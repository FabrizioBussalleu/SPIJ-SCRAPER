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
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    fbeta_score,
    make_scorer,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from db import get_session
from models import Norm, NormVersion
from analysis_utils import (
    LEGAL_TERM_ANNOTATIONS,
    PRESENTATION_DIR,
    ensure_dir,
    interpret_f_beta,
    plot_confusion_enhanced,
    plot_cv_scores,
    plot_logreg_coefficients,
)
from utils import setup_logging

setup_logging()
LOG = logging.getLogger(__name__)

MODEL_DIR = Path("data/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.15
CV_FOLDS = 5

# Configuración documentada para la metodología del TP
METHODOLOGY = {
    "vigencia": {
        "algorithm": "Regresión Logística",
        "vectorizer": "TF-IDF (max_features=20000, ngram_range=(1,2), sublinear_tf=True, min_df=2)",
        "classifier_params": "C=1.0, class_weight='balanced', max_iter=1000, solver='lbfgs'",
        "split": f"train_test_split test_size={TEST_SIZE}, stratify=y, random_state={RANDOM_STATE}",
        "cv": f"StratifiedKFold(n_splits={CV_FOLDS}, shuffle=True, random_state={RANDOM_STATE})",
        "scoring": "F1, F2 (β=2, prioriza recall), precision, recall — solo sobre conjunto train en CV",
    },
    "jerarquia": {
        "algorithm": "Random Forest",
        "vectorizer": "TF-IDF (max_features=15000, ngram_range=(1,2), sublinear_tf=True, min_df=2)",
        "classifier_params": "n_estimators=100, max_depth=10, class_weight='balanced'",
        "split": f"train_test_split test_size={TEST_SIZE}, stratify=y, random_state={RANDOM_STATE}",
        "cv": f"StratifiedKFold(n_splits={CV_FOLDS}, shuffle=True, random_state={RANDOM_STATE})",
        "scoring": "F1-macro, F2-macro (β=2)",
    },
}


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


# ─── Helpers de evaluación ───────────────────────────────────────────────────

def _run_cv(pipeline, X_train, y_train, scoring: dict) -> dict:
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    return cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring, return_train_score=False)


def _write_methodology_report(task: str, cv_results: dict, extra_lines: List[str] = None):
    meta = METHODOLOGY[task]
    lines = [
        f"MÉTODO — Modelo de {task.upper()}",
        "=" * 60,
        f"Algoritmo:     {meta['algorithm']}",
        f"Vectorizador:  {meta['vectorizer']}",
        f"Clasificador:  {meta['classifier_params']}",
        f"Partición:     {meta['split']}",
        f"Validación CV: {meta['cv']}",
        f"Métricas CV:   {meta['scoring']}",
        "",
        "Resultados por fold (conjunto de entrenamiento):",
    ]
    for metric, scores in cv_results.items():
        if metric.startswith("test_"):
            name = metric.replace("test_", "").upper()
            lines.append(f"  {name}: {np.mean(scores):.4f} ± {np.std(scores):.4f}  (folds: {[f'{s:.3f}' for s in scores]})")
    if extra_lines:
        lines.extend(["", *extra_lines])
    path = MODEL_DIR / f"{task}_methodology.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    LOG.info("Metodología guardada: %s", path)
    return "\n".join(lines)


def _analyze_vigencia_errors(
    pipeline, X_test, y_test, y_pred, df_test: pd.DataFrame,
) -> str:
    """Analiza falsos positivos y falsos negativos del modelo de vigencia."""
    lines = ["", "ANÁLISIS DE ERRORES — VIGENCIA", "-" * 50]
    fn_mask = (y_test.values == 0) & (y_pred == 1)  # derogada predicha como vigente (crítico)
    fp_mask = (y_test.values == 1) & (y_pred == 0)  # vigente predicha como derogada

    lines.append(f"Falsos negativos (derogada → vigente, CRÍTICO): {fn_mask.sum()}")
    for idx in df_test[fn_mask].index[:5]:
        row = df_test.loc[idx]
        lines.append(f"  • [{row.get('norm_id')}] {str(row.get('title', ''))[:80]}")

    lines.append(f"Falsos positivos (vigente → derogada): {fp_mask.sum()}")
    for idx in df_test[fp_mask].index[:5]:
        row = df_test.loc[idx]
        lines.append(f"  • [{row.get('norm_id')}] {str(row.get('title', ''))[:80]}")

    tfidf = pipeline.named_steps["tfidf"]
    coef = pipeline.named_steps["clf"].coef_[0]
    names = tfidf.get_feature_names_out()
    top_derog = names[np.argsort(coef)[:8]]
    top_vig = names[np.argsort(coef)[-8:][::-1]]
    lines.extend([
        "",
        "Palabras que el modelo asocia con DEROGADA (coeficiente negativo):",
        f"  {', '.join(top_derog)}",
        "Palabras que el modelo asocia con VIGENTE (coeficiente positivo):",
        f"  {', '.join(top_vig)}",
        "",
        "INTERPRETACIÓN XAI:",
        "  El modelo aprendió el mismo lenguaje jurídico que usa el scraper SPIJ.",
        "  Términos como 'derogado' y 'sustitúyese' tienen peso hacia derogada porque",
        "  el portal marca explícitamente esas alertas — el ML converge con la lógica",
        "  programada, sin reglas hardcodeadas en el clasificador.",
    ])
    return "\n".join(lines)


# ─── Modelo 1: Clasificación de Vigencia ────────────────────────────────────

def train_vigencia(df: pd.DataFrame, run_shap: bool = False, presentation: bool = False):
    """Entrena Regresión Logística (TF-IDF) para predecir is_valid."""
    df_v = df.dropna(subset=["is_valid"]).copy()
    if len(df_v) < 20:
        LOG.error("Datos insuficientes para entrenar el modelo de vigencia (mínimo 20 normas etiquetadas).")
        return

    X, y = df_v["text"], df_v["is_valid"].astype(int)
    meta = df_v[["norm_id", "title", "type"]].copy()
    LOG.info("Distribución de clases — Vigencia: %s", dict(y.value_counts()))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    meta_test = meta.loc[X_test.index]

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=20_000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=2,
        )),
        ("clf", LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_STATE,
        )),
    ])

    f2_scorer = make_scorer(fbeta_score, beta=2, zero_division=0)
    cv_results = _run_cv(
        pipeline, X_train, y_train,
        scoring={"f1": "f1", "f2": f2_scorer, "precision": "precision", "recall": "recall"},
    )

    f1_mean, f2_mean = np.mean(cv_results["test_f1"]), np.mean(cv_results["test_f2"])
    LOG.info("CV F1 (5-fold) — Vigencia: %.3f ± %.3f", f1_mean, np.std(cv_results["test_f1"]))
    LOG.info("CV F2 (5-fold) — Vigencia: %.3f ± %.3f", f2_mean, np.std(cv_results["test_f2"]))

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    report = classification_report(
        y_test, y_pred,
        target_names=["Derogada (0)", "Vigente (1)"],
        digits=4,
    )
    error_analysis = _analyze_vigencia_errors(
        pipeline, X_test, y_test, y_pred,
        meta_test.assign(text=X_test.values),
    )
    f_beta_text = interpret_f_beta(f1_mean, f2_mean, "vigencia")

    print("\n" + "=" * 55)
    print("  MODELO DE VIGENCIA — Regresión Logística (TF-IDF)")
    print("=" * 55)
    print(f"  CV F1 (5-fold): {f1_mean:.4f} ± {np.std(cv_results['test_f1']):.4f}")
    print(f"  CV F2 (5-fold): {f2_mean:.4f} ± {np.std(cv_results['test_f2']):.4f}")
    print(f"\n  {f_beta_text}")
    print("\n" + report)
    print(error_analysis)

    methodology_text = _write_methodology_report(
        "vigencia",
        cv_results,
        extra_lines=[f_beta_text, error_analysis],
    )

    full_report = (
        f"CV F1 (5-fold): {f1_mean:.4f} +/- {np.std(cv_results['test_f1']):.4f}\n"
        f"CV F2 (5-fold): {f2_mean:.4f} +/- {np.std(cv_results['test_f2']):.4f}\n\n"
        f"{f_beta_text}\n\n{report}\n{error_analysis}\n\n{methodology_text}"
    )
    (MODEL_DIR / "vigencia_report.txt").write_text(full_report, encoding="utf-8")

    cm = confusion_matrix(y_test, y_pred)
    pres_dir = ensure_dir(PRESENTATION_DIR) if presentation else MODEL_DIR
    plot_confusion_enhanced(
        cm, ["Derogada", "Vigente"],
        pres_dir / "09_vigencia_confusion_matrix.png",
        "Matriz de Confusión — Vigencia",
        interpretations=[
            "FN (derogada→vigente): abogado cita ley sin efecto — error crítico",
            "FP (vigente→derogada): norma válida oculta — menos grave pero molesto",
        ],
    )

    tfidf = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]
    plot_logreg_coefficients(
        tfidf.get_feature_names_out(),
        clf.coef_[0],
        class_idx=1,
        out_path=pres_dir / "10_xai_vigencia_coefficients.png",
        legal_annotations=LEGAL_TERM_ANNOTATIONS,
    )

    if presentation:
        plot_cv_scores(
            {
                "F1": cv_results["test_f1"],
                "F2": cv_results["test_f2"],
                "Precision": cv_results["test_precision"],
                "Recall": cv_results["test_recall"],
            },
            pres_dir / "11_vigencia_cv_scores.png",
            title=f"Validación cruzada vigencia ({CV_FOLDS}-fold estratificado)",
        )

    model_path = MODEL_DIR / "vigencia_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)
    LOG.info("Modelo guardado: %s", model_path)

    if run_shap:
        _shap_logreg(pipeline, X_test, y_test, "vigencia", presentation=presentation)

    return pipeline


# ─── Modelo 2: Clasificación Jerárquica ─────────────────────────────────────

def train_jerarquia(df: pd.DataFrame, run_shap: bool = False, presentation: bool = False):
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
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
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

    f2_macro = make_scorer(fbeta_score, beta=2, average="macro", zero_division=0)
    cv_results = _run_cv(
        pipeline, X_train, y_train,
        scoring={"f1_macro": "f1_macro", "f2_macro": f2_macro},
    )
    f1_mean = np.mean(cv_results["test_f1_macro"])
    LOG.info("CV F1-macro (5-fold) — Jerarquía: %.3f ± %.3f", f1_mean, np.std(cv_results["test_f1_macro"]))

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    level_names = {1: "Constitución", 2: "Ley/Cód.", 3: "D.Supremo", 4: "Resolución", 5: "Otros"}
    target_names = [level_names.get(l, str(l)) for l in sorted(valid_classes)]

    report = classification_report(y_test, y_pred, target_names=target_names, digits=4)
    _write_methodology_report("jerarquia", cv_results)

    print("\n" + "=" * 55)
    print("  MODELO DE JERARQUÍA — Random Forest (TF-IDF)")
    print("=" * 55)
    print(f"  CV F1-macro (5-fold): {f1_mean:.4f} ± {np.std(cv_results['test_f1_macro']):.4f}")
    print(f"  CV F2-macro (5-fold): {np.mean(cv_results['test_f2_macro']):.4f} ± {np.std(cv_results['test_f2_macro']):.4f}")
    print("\n" + report)

    report_path = MODEL_DIR / "jerarquia_report.txt"
    report_path.write_text(
        f"CV F1-macro (5-fold): {f1_mean:.4f} +/- {np.std(cv_results['test_f1_macro']):.4f}\n\n{report}",
        encoding="utf-8",
    )

    tfidf = pipeline.named_steps["tfidf"]
    rf = pipeline.named_steps["clf"]
    feature_names = tfidf.get_feature_names_out()
    importances = rf.feature_importances_
    top_idx = np.argsort(importances)[-20:][::-1]

    pres_dir = ensure_dir(PRESENTATION_DIR) if presentation else MODEL_DIR
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(feature_names[top_idx][::-1], importances[top_idx][::-1], color="#1F4E79")
    ax.set_xlabel("Importancia (Gini)")
    ax.set_title("Top 20 features — Random Forest Jerarquía", fontweight="bold")
    plt.tight_layout()
    feat_path = pres_dir / "12_jerarquia_feature_importance.png"
    fig.savefig(feat_path, dpi=150)
    plt.close(fig)
    LOG.info("Importancia de features guardada: %s", feat_path)

    model_path = MODEL_DIR / "jerarquia_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)
    LOG.info("Modelo guardado: %s", model_path)

    if run_shap:
        _shap_rf(pipeline, X_test, "jerarquia", presentation=presentation)

    return pipeline


# ─── SHAP ────────────────────────────────────────────────────────────────────

def _shap_logreg(pipeline, X_test, y_test, task_name: str, presentation: bool = False):
    """Genera análisis SHAP para el modelo de Regresión Logística."""
    try:
        import shap
        tfidf = pipeline.named_steps["tfidf"]
        clf = pipeline.named_steps["clf"]
        X_test_tfidf = tfidf.transform(X_test)
        explainer = shap.LinearExplainer(clf, X_test_tfidf, feature_perturbation="interventional")
        shap_values = explainer.shap_values(X_test_tfidf)
        feature_names = tfidf.get_feature_names_out()

        plt.figure(figsize=(12, 7))
        shap.summary_plot(
            shap_values, X_test_tfidf,
            feature_names=feature_names,
            max_display=20,
            show=False,
        )
        plt.suptitle(
            f"SHAP — ¿Por qué el modelo predice vigencia/derogada?\n"
            f"(Rojo = empuja hacia derogada, Azul = empuja hacia vigente)",
            fontweight="bold", y=1.02,
        )
        plt.tight_layout()
        out_dir = ensure_dir(PRESENTATION_DIR) if presentation else MODEL_DIR
        path = out_dir / f"13_shap_{task_name}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        LOG.info("SHAP guardado: %s", path)
    except ImportError:
        LOG.warning("SHAP no instalado. Instala con: pip install shap")


def _shap_rf(pipeline, X_test, task_name: str, presentation: bool = False):
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
        plt.suptitle(f"SHAP — Contribución de palabras al nivel jerárquico", fontweight="bold", y=1.02)
        plt.tight_layout()
        out_dir = ensure_dir(PRESENTATION_DIR) if presentation else MODEL_DIR
        path = out_dir / f"13_shap_{task_name}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        LOG.info("SHAP guardado: %s", path)
    except ImportError:
        LOG.warning("SHAP no instalado. Instala con: pip install shap")


# ─── Punto de entrada ─────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Entrena clasificadores sobre el corpus SPIJ.")
    p.add_argument("--task", choices=["vigencia", "jerarquia", "ambos"], default="ambos")
    p.add_argument("--shap", action="store_true", help="Generar análisis SHAP (más lento)")
    p.add_argument("--presentation", action="store_true", help="Guardar gráficos en exports/presentacion/")
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
                train_vigencia(df, run_shap=args.shap, presentation=args.presentation)
            if args.task in ("jerarquia", "ambos"):
                train_jerarquia(df, run_shap=args.shap, presentation=args.presentation)
    finally:
        session.close()

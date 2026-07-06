"""Análisis Exploratorio de Datos (EDA) para el corpus de normas del SPIJ.

Genera visualizaciones desde la base de datos local y las guarda en exports/eda/.

Uso:
    python eda_analysis.py

Requiere: pandas, matplotlib, seaborn, wordcloud
"""

import logging
import os
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud

from sqlalchemy import text as sqltext
from db import get_session
from models import Norm, NormCitation
from utils import setup_logging

setup_logging()
LOG = logging.getLogger(__name__)

OUT_DIR = Path("exports/eda")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PALETTE = ["#1F4E79", "#2E75B6", "#9DC3E6", "#BDD7EE", "#DEEAF1"]
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})


# ─── Carga de datos ──────────────────────────────────────────────────────────

def load_dataframe() -> pd.DataFrame:
    """Carga todas las normas de la BD en un DataFrame de pandas."""
    session = get_session()
    try:
        norms = session.query(Norm).all()
        rows = [n.to_dict() for n in norms]
        if not rows:
            LOG.warning("La base de datos está vacía. Ejecuta primero: python main.py scrape --limit 100")
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["year"] = df["date"].dt.year
        LOG.info("Cargadas %d normas de la base de datos.", len(df))
        return df
    finally:
        session.close()


def load_citations() -> pd.DataFrame:
    """Carga todas las citas inter-normativas."""
    session = get_session()
    try:
        cits = session.query(NormCitation).all()
        return pd.DataFrame([
            {"citation_type": c.citation_type, "target_text": c.target_text}
            for c in cits
        ])
    finally:
        session.close()


# ─── Gráfico 1: Distribución por tipo de norma ───────────────────────────────

def plot_norms_by_type(df: pd.DataFrame):
    counts = df["type"].fillna("Sin tipo").value_counts().head(12)
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(counts.index[::-1], counts.values[::-1], color=PALETTE[0])
    ax.bar_label(bars, padding=4, fontsize=10)
    ax.set_xlabel("Cantidad de normas")
    ax.set_title("Distribución de normas por tipo", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    path = OUT_DIR / "01_norms_by_type.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    LOG.info("Guardado: %s", path)


# ─── Gráfico 2: Serie temporal de publicaciones ──────────────────────────────

def plot_temporal_distribution(df: pd.DataFrame):
    yearly = df["year"].dropna().astype(int)
    if yearly.empty:
        LOG.warning("Sin datos de fecha para gráfico temporal.")
        return
    counts = yearly.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.fill_between(counts.index, counts.values, alpha=0.3, color=PALETTE[1])
    ax.plot(counts.index, counts.values, color=PALETTE[0], linewidth=2)
    ax.set_xlabel("Año de publicación")
    ax.set_ylabel("Normas publicadas")
    ax.set_title("Evolución temporal de publicaciones normativas", fontsize=14, fontweight="bold", pad=15)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
    plt.tight_layout()
    path = OUT_DIR / "02_temporal_distribution.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    LOG.info("Guardado: %s", path)


# ─── Gráfico 3: Nube de palabras de títulos ──────────────────────────────────

STOPWORDS_ES = {
    "de", "del", "la", "el", "los", "las", "que", "en", "y", "a", "por",
    "para", "con", "se", "al", "su", "sus", "una", "un", "sobre", "o",
    "número", "nro", "no", "mediante", "decreto", "resolución", "ley",
    "norma", "aprueba", "establece",
}

def plot_wordcloud(df: pd.DataFrame):
    text = " ".join(df["title"].dropna().astype(str).tolist())
    wc = WordCloud(
        width=1200, height=600,
        background_color="white",
        stopwords=STOPWORDS_ES,
        colormap="Blues",
        max_words=80,
        collocations=False,
    ).generate(text)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("Nube de palabras — títulos de normas SPIJ", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    path = OUT_DIR / "03_wordcloud_titles.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    LOG.info("Guardado: %s", path)


# ─── Gráfico 4: Distribución por nivel jerárquico y vigencia ─────────────────

def plot_level_validity_heatmap(df: pd.DataFrame):
    """Gráfico 4 (revisado): vigencia REAL a nivel de artículo.

    La vigencia a nivel de norma ya no es informativa (el corpus es 100%
    vigente por diseño del muestreo). La señal real de derogación se modela a
    nivel de artículo con las notas oficiales (*) de SPIJ. Este gráfico muestra,
    por código, cuántos artículos están vigentes vs. derogados en el dataset
    balanceado (article_samples) que alimenta el clasificador de vigencia.
    """
    session = get_session()
    try:
        rows = session.execute(sqltext(
            "SELECT code_name, label, COUNT(*) c FROM article_samples GROUP BY code_name, label"
        )).fetchall()
    except Exception as e:
        LOG.warning("No se pudo leer article_samples para el gráfico 4: %s", e)
        return
    finally:
        session.close()
    if not rows:
        LOG.warning("article_samples vacío; omitiendo gráfico 4.")
        return
    dfa = pd.DataFrame(rows, columns=["code", "label", "count"])
    dfa["estado"] = dfa["label"].map({1: "Vigente", 0: "Derogado"})
    # abreviar nombres de códigos
    dfa["code"] = dfa["code"].str.replace("CODIGO", "C.", regex=False).str.title().str[:26]
    pivot = dfa.pivot_table(index="code", columns="estado", values="count",
                            aggfunc="sum", fill_value=0)
    pivot["tot"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("tot", ascending=True).drop(columns="tot")
    pivot = pivot[[c for c in ["Derogado", "Vigente"] if c in pivot.columns]]
    fig, ax = plt.subplots(figsize=(9, 6))
    pivot.plot(kind="barh", stacked=True, ax=ax,
               color={"Derogado": "#C0392B", "Vigente": "#1F4E79"})
    ax.set_xlabel("N.º de artículos")
    ax.set_ylabel("")
    ax.set_title("Vigencia por artículo según notas oficiales de SPIJ\n(dataset del clasificador de vigencia)",
                 fontsize=12, fontweight="bold", pad=12)
    ax.legend(title="Estado")
    plt.tight_layout()
    path = OUT_DIR / "04_heatmap_level_validity.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    LOG.info("Guardado: %s", path)


# ─── Gráfico 5: Distribución de tipos de citas ───────────────────────────────

def plot_citation_types(df_cit: pd.DataFrame):
    if df_cit.empty:
        LOG.warning("Sin citas registradas para graficar.")
        return
    counts = df_cit["citation_type"].fillna("refers").value_counts()
    labels_map = {
        "refers": "Referencias\n(refers)",
        "modifies": "Modificaciones\n(modifies)",
        "repeals": "Derogaciones\n(repeals)",
        "repealed_by": "Derogada por\n(repealed_by)",
    }
    labels = [labels_map.get(k, k) for k in counts.index]
    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, texts, autotexts = ax.pie(
        counts.values, labels=labels, autopct="%1.1f%%",
        colors=PALETTE[:len(counts)], startangle=140,
        wedgeprops={"linewidth": 1, "edgecolor": "white"},
    )
    for at in autotexts:
        at.set_fontsize(11)
    ax.set_title("Distribución de citas inter-normativas por tipo", fontsize=13, fontweight="bold", pad=20)
    plt.tight_layout()
    path = OUT_DIR / "05_citation_types.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    LOG.info("Guardado: %s", path)


# ─── Reporte estadístico en consola ──────────────────────────────────────────

def print_summary(df: pd.DataFrame, df_cit: pd.DataFrame):
    print("\n" + "=" * 55)
    print("  RESUMEN ESTADÍSTICO DEL CORPUS SPIJ")
    print("=" * 55)
    print(f"  Total de normas:          {len(df):>8,}")
    print(f"  Tipos únicos de norma:    {df['type'].nunique():>8,}")
    print(f"  Con fecha de publicación: {df['date'].notna().sum():>8,}")
    if df['date'].notna().any():
        print(f"  Rango temporal:           {int(df['year'].min())} – {int(df['year'].max())}")
    print(f"  Normas vigentes:          {(df['is_valid'] == True).sum():>8,}")
    print(f"  Normas derogadas:         {(df['is_valid'] == False).sum():>8,}")
    print(f"  Vigencia desconocida:     {df['is_valid'].isna().sum():>8,}")
    print(f"  Citas registradas:        {len(df_cit):>8,}")
    print(f"  Nulos en 'title':         {df['title'].isna().sum():>8,}")
    print("=" * 55 + "\n")


# ─── Punto de entrada ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = load_dataframe()
    if df.empty:
        print("No hay datos. Ejecuta primero el scraper:\n  python main.py scrape --limit 200")
    else:
        df_cit = load_citations()
        print_summary(df, df_cit)
        print("Generando visualizaciones...")
        plot_norms_by_type(df)
        plot_temporal_distribution(df)
        plot_wordcloud(df)
        plot_level_validity_heatmap(df)
        plot_citation_types(df_cit)
        print(f"Listo. Visualizaciones guardadas en: {OUT_DIR.resolve()}/")

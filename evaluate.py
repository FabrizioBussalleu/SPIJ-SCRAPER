"""Framework de evaluación del motor de búsqueda semántica.

Compara el sistema semántico (embeddings + FAISS) contra la línea base
léxica (BM25) usando métricas estándar de recuperación de información:
    - Precision@k
    - Recall@k
    - Mean Reciprocal Rank (MRR@k)
    - Normalized Discounted Cumulative Gain (NDCG@k)

El dataset de evaluación se define en QUERIES: una lista de consultas
jurídicas en lenguaje natural con sus normas relevantes esperadas
(identificadas por palabras clave del título o tipo+número).

Uso:
    python evaluate.py
    python evaluate.py --k 5
    python evaluate.py --k 10 --output exports/evaluation_report.txt
"""

import argparse
import logging
import math
from pathlib import Path
from typing import Dict, List

import numpy as np

from semantic_search import SemanticSearchEngine
from utils import setup_logging

setup_logging()
LOG = logging.getLogger(__name__)

# ─── Dataset de evaluación ───────────────────────────────────────────────────
# Formato: cada query tiene una lista de "relevant_keywords" — fragmentos
# de texto que DEBEN aparecer en el título de una norma relevante.
# Esto permite evaluar sin anotar IDs de norma específicos.

QUERIES = [
    {
        "query": "despido de trabajadora embarazada gestante gravidez",
        "relevant_keywords": ["protección", "maternidad", "madre trabajadora", "gestante", "embarazo"],
    },
    {
        "query": "contrato de trabajo a plazo fijo temporal",
        "relevant_keywords": ["contrato de trabajo", "plazo fijo", "temporal", "laboral"],
    },
    {
        "query": "protección del medio ambiente contaminación ambiental",
        "relevant_keywords": ["medio ambiente", "ambiental", "contaminación", "ecología"],
    },
    {
        "query": "impuesto a la renta personas naturales",
        "relevant_keywords": ["impuesto", "renta", "tributo", "tributario"],
    },
    {
        "query": "licitación pública contrataciones del estado adquisiciones",
        "relevant_keywords": ["contratación", "licitación", "adquisición", "estado", "compras públicas"],
    },
    {
        "query": "violencia contra la mujer feminicidio",
        "relevant_keywords": ["violencia", "mujer", "feminicidio", "género"],
    },
    {
        "query": "sistema nacional de salud seguro médico",
        "relevant_keywords": ["salud", "seguro", "médico", "sanitario", "SIS"],
    },
    {
        "query": "educación básica regular colegios escuelas",
        "relevant_keywords": ["educación", "escuela", "colegio", "básica regular"],
    },
    {
        "query": "inversión privada concesiones obras de infraestructura",
        "relevant_keywords": ["inversión", "concesión", "infraestructura", "privado"],
    },
    {
        "query": "procedimiento administrativo general tramite entidad pública",
        "relevant_keywords": ["procedimiento administrativo", "trámite", "entidad pública", "administración"],
    },
    {
        "query": "pension jubilacion sistema previsional AFP ONP",
        "relevant_keywords": ["pensión", "jubilación", "AFP", "ONP", "previsional", "retiro"],
    },
    {
        "query": "propiedad intelectual derechos de autor patentes marcas",
        "relevant_keywords": ["propiedad intelectual", "derechos de autor", "patente", "marca"],
    },
]


# ─── Métricas de recuperación ────────────────────────────────────────────────

def _is_relevant(result: Dict, relevant_keywords: List[str]) -> bool:
    """Determina si un resultado es relevante según las palabras clave."""
    title = (result.get("title") or "").lower()
    type_ = (result.get("type") or "").lower()
    text  = f"{title} {type_}"
    return any(kw.lower() in text for kw in relevant_keywords)


def precision_at_k(results: List[Dict], relevant_keywords: List[str], k: int) -> float:
    top_k = results[:k]
    if not top_k:
        return 0.0
    hits = sum(_is_relevant(r, relevant_keywords) for r in top_k)
    return hits / k


def recall_at_k(results: List[Dict], relevant_keywords: List[str], k: int) -> float:
    """Recall aproximado: fracción de keywords encontradas en top-k."""
    top_k = results[:k]
    if not relevant_keywords:
        return 0.0
    found_keywords = set()
    for r in top_k:
        title = (r.get("title") or "").lower()
        for kw in relevant_keywords:
            if kw.lower() in title:
                found_keywords.add(kw)
    return len(found_keywords) / len(relevant_keywords)


def reciprocal_rank(results: List[Dict], relevant_keywords: List[str], k: int) -> float:
    for i, r in enumerate(results[:k], 1):
        if _is_relevant(r, relevant_keywords):
            return 1.0 / i
    return 0.0


def ndcg_at_k(results: List[Dict], relevant_keywords: List[str], k: int) -> float:
    gains = [
        1.0 if _is_relevant(r, relevant_keywords) else 0.0
        for r in results[:k]
    ]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    # IDCG: todos los relevantes al inicio
    n_relevant = min(sum(gains), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(int(n_relevant)))
    return dcg / idcg if idcg > 0 else 0.0


# ─── Evaluación completa ─────────────────────────────────────────────────────

def evaluate(k: int = 5, output: str = None):
    """Evalúa ambos sistemas (semántico y BM25) sobre el dataset de consultas."""
    try:
        engine = SemanticSearchEngine()
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return

    metrics_semantic = {"precision": [], "recall": [], "mrr": [], "ndcg": []}
    metrics_bm25     = {"precision": [], "recall": [], "mrr": [], "ndcg": []}

    print(f"\n{'='*65}")
    print(f"  EVALUACIÓN DEL MOTOR DE BÚSQUEDA SEMÁNTICA  |  k = {k}")
    print(f"{'='*65}\n")

    for i, item in enumerate(QUERIES, 1):
        query    = item["query"]
        keywords = item["relevant_keywords"]

        res_sem  = engine.search(query, k=k, only_valid=True)
        res_bm25 = engine.search_bm25(query, k=k, only_valid=True)

        for name, res, metrics in [
            ("Semántico", res_sem,  metrics_semantic),
            ("BM25",      res_bm25, metrics_bm25),
        ]:
            p = precision_at_k(res, keywords, k)
            r = recall_at_k(res, keywords, k)
            m = reciprocal_rank(res, keywords, k)
            n = ndcg_at_k(res, keywords, k)
            metrics["precision"].append(p)
            metrics["recall"].append(r)
            metrics["mrr"].append(m)
            metrics["ndcg"].append(n)

        hit_sem  = "✅" if any(_is_relevant(r, keywords) for r in res_sem[:k])  else "❌"
        hit_bm25 = "✅" if any(_is_relevant(r, keywords) for r in res_bm25[:k]) else "❌"
        print(f"  [{i:02d}] {query[:55]:<55}")
        print(f"        Semántico P@{k}={precision_at_k(res_sem, keywords, k):.2f} {hit_sem}  |  BM25 P@{k}={precision_at_k(res_bm25, keywords, k):.2f} {hit_bm25}")

    # Promedios
    def avg(lst):
        return np.mean(lst) if lst else 0.0

    lines = []
    lines.append(f"\n{'='*65}")
    lines.append(f"  RESULTADOS AGREGADOS  (n={len(QUERIES)} consultas, k={k})")
    lines.append(f"{'='*65}")
    lines.append(f"  {'Métrica':<20} {'Semántico':>12} {'BM25':>12} {'Mejora':>10}")
    lines.append(f"  {'-'*55}")
    for metric_key, label in [
        ("precision", f"Precision@{k}"),
        ("recall",    f"Recall@{k}"),
        ("mrr",       f"MRR@{k}"),
        ("ndcg",      f"NDCG@{k}"),
    ]:
        sem_val  = avg(metrics_semantic[metric_key])
        bm25_val = avg(metrics_bm25[metric_key])
        mejora   = f"+{(sem_val - bm25_val):.3f}" if sem_val >= bm25_val else f"{(sem_val - bm25_val):.3f}"
        lines.append(f"  {label:<20} {sem_val:>12.4f} {bm25_val:>12.4f} {mejora:>10}")
    lines.append(f"{'='*65}\n")

    report = "\n".join(lines)
    print(report)

    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(report)
        print(f"Reporte guardado en: {output}")


# ─── Punto de entrada ─────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Evalúa el motor de búsqueda semántica SPIJ.")
    p.add_argument("--k", type=int, default=5, help="Top-k resultados a evaluar (default: 5)")
    p.add_argument("--output", type=str, default=None, help="Ruta para guardar el reporte de evaluación")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(k=args.k, output=args.output)

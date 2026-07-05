"""Framework de evaluación del motor de búsqueda semántica.

Compara el sistema semántico (embeddings + FAISS) contra la línea base
léxica (BM25) con métricas estándar, interpretación contextual para
abogados, análisis de falsos positivos/negativos y gráficos para presentación.

Uso:
    python evaluate.py
    python evaluate.py --k 5 --charts
    python evaluate.py --k 10 --output exports/evaluation_report.txt --charts
"""

import argparse
import logging
import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from analysis_utils import (
    PRESENTATION_DIR,
    ensure_dir,
    interpret_mrr,
    interpret_ndcg,
    interpret_precision,
    interpret_recall,
    plot_error_summary,
    plot_metrics_comparison,
    plot_per_query_heatmap,
)
from semantic_search import SemanticSearchEngine
from utils import setup_logging

setup_logging()
LOG = logging.getLogger(__name__)

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

TOKEN_RE = re.compile(r"[a-záéíóúüñ]+", re.I)
STOPWORDS = {
    "de", "del", "la", "el", "los", "las", "que", "en", "y", "a", "por", "para",
    "con", "se", "al", "su", "sus", "una", "un", "sobre", "o", "nro", "numero",
}


def _is_relevant(result: Dict, relevant_keywords: List[str]) -> bool:
    title = (result.get("title") or "").lower()
    type_ = (result.get("type") or "").lower()
    text = f"{title} {type_}"
    return any(kw.lower() in text for kw in relevant_keywords)


def precision_at_k(results: List[Dict], relevant_keywords: List[str], k: int) -> float:
    top_k = results[:k]
    if not top_k:
        return 0.0
    hits = sum(_is_relevant(r, relevant_keywords) for r in top_k)
    return hits / k


def recall_at_k(results: List[Dict], relevant_keywords: List[str], k: int) -> float:
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
    gains = [1.0 if _is_relevant(r, relevant_keywords) else 0.0 for r in results[:k]]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    n_relevant = min(sum(gains), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(int(n_relevant)))
    return dcg / idcg if idcg > 0 else 0.0


def _tokenize_title(title: str) -> List[str]:
    return [t.lower() for t in TOKEN_RE.findall(title or "") if t.lower() not in STOPWORDS and len(t) > 2]


def analyze_query_errors(
    results: List[Dict],
    relevant_keywords: List[str],
    k: int,
) -> Tuple[List[Dict], List[str]]:
    """Retorna falsos positivos (títulos irrelevantes) y keywords no recuperadas (FN)."""
    top_k = results[:k]
    false_positives = [r for r in top_k if not _is_relevant(r, relevant_keywords)]
    found = set()
    for r in top_k:
        title = (r.get("title") or "").lower()
        for kw in relevant_keywords:
            if kw.lower() in title:
                found.add(kw)
    false_negative_keywords = [kw for kw in relevant_keywords if kw not in found]
    return false_positives, false_negative_keywords


def _build_interpretation_section(k: int, sem_avg: Dict[str, float], bm25_avg: Dict[str, float]) -> List[str]:
    lines = [
        "",
        "=" * 70,
        "  INTERPRETACIÓN PARA EL CONTEXTO LEGAL",
        "=" * 70,
        "",
        interpret_precision(bm25_avg["precision"], k),
        "",
        interpret_precision(sem_avg["precision"], k).replace("Precisión", "  → Semántico — Precisión"),
        "",
        interpret_recall(sem_avg["recall"], k),
        "",
        interpret_recall(bm25_avg["recall"], k).replace("Recall", "  → BM25 — Recall"),
        "",
        interpret_mrr(bm25_avg["mrr"], k),
        "",
        interpret_mrr(sem_avg["mrr"], k).replace("MRR", "  → Semántico — MRR"),
        "",
        interpret_ndcg(sem_avg["ndcg"], bm25_avg["ndcg"], k),
        "",
        "─" * 70,
        "  CONCLUSIÓN PARA LA PRESENTACIÓN:",
        "  • BM25 gana en precisión/MRR porque los títulos legales usan formulación",
        "    formal y exacta (ej. 'Ley de Protección de la Madre Trabajadora').",
        "  • El modelo semántico gana en recall: recupera más conceptos cuando el",
        "    abogado usa lenguaje coloquial distinto al título oficial.",
        "  • NDCG penaliza resultados relevantes en posiciones bajas: un abogado",
        "    rara vez revisa más de 3–5 títulos antes de decidir.",
        "",
    ]
    return lines


def evaluate(k: int = 5, output: str = None, charts: bool = False):
    try:
        engine = SemanticSearchEngine()
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return None

    metrics_semantic = {"precision": [], "recall": [], "mrr": [], "ndcg": []}
    metrics_bm25 = {"precision": [], "recall": [], "mrr": [], "ndcg": []}
    per_query_rows = []
    fp_word_counter: Counter = Counter()
    fn_word_counter: Counter = Counter()
    error_details: List[str] = []

    print(f"\n{'=' * 65}")
    print(f"  EVALUACIÓN DEL MOTOR DE BÚSQUEDA  |  k = {k}")
    print(f"{'=' * 65}\n")

    for i, item in enumerate(QUERIES, 1):
        query = item["query"]
        keywords = item["relevant_keywords"]

        res_sem = engine.search(query, k=k, only_valid=True)
        res_bm25 = engine.search_bm25(query, k=k, only_valid=True)

        row_sem, row_bm25 = {}, {}
        for name, res, metrics in [
            ("Semántico", res_sem, metrics_semantic),
            ("BM25", res_bm25, metrics_bm25),
        ]:
            p = precision_at_k(res, keywords, k)
            r = recall_at_k(res, keywords, k)
            m = reciprocal_rank(res, keywords, k)
            n = ndcg_at_k(res, keywords, k)
            metrics["precision"].append(p)
            metrics["recall"].append(r)
            metrics["mrr"].append(m)
            metrics["ndcg"].append(n)
            if name == "Semántico":
                row_sem = {"p": p, "r": r, "m": m, "n": n}
            else:
                row_bm25 = {"p": p, "r": r, "m": m, "n": n}

        # Análisis de errores (ambos motores)
        for engine_name, res in [("Semántico", res_sem), ("BM25", res_bm25)]:
            fps, fn_kws = analyze_query_errors(res, keywords, k)
            for fp in fps:
                fp_word_counter.update(_tokenize_title(fp.get("title", "")))
            fn_word_counter.update(fn_kws)
            if fps or fn_kws:
                error_details.append(f"\n  Consulta [{i:02d}]: {query[:50]}… ({engine_name})")
                if fps:
                    error_details.append(f"    Falsos positivos ({len(fps)}):")
                    for fp in fps[:3]:
                        error_details.append(f"      • {fp.get('title', '')[:70]}")
                if fn_kws:
                    error_details.append(f"    Keywords NO recuperadas (FN): {', '.join(fn_kws)}")

        per_query_rows.append({"query": query, "sem": row_sem, "bm25": row_bm25})

        hit_sem = "✅" if any(_is_relevant(r, keywords) for r in res_sem[:k]) else "❌"
        hit_bm25 = "✅" if any(_is_relevant(r, keywords) for r in res_bm25[:k]) else "❌"
        print(f"  [{i:02d}] {query[:55]:<55}")
        print(f"        Sem P@{k}={row_sem['p']:.2f} {hit_sem}  |  BM25 P@{k}={row_bm25['p']:.2f} {hit_bm25}")

    def avg(lst):
        return np.mean(lst) if lst else 0.0

    sem_avg = {key: avg(metrics_semantic[key]) for key in metrics_semantic}
    bm25_avg = {key: avg(metrics_bm25[key]) for key in metrics_bm25}

    lines = []
    lines.append(f"\n{'=' * 65}")
    lines.append(f"  RESULTADOS AGREGADOS  (n={len(QUERIES)} consultas, k={k})")
    lines.append(f"{'=' * 65}")
    lines.append(f"  {'Métrica':<20} {'Semántico':>12} {'BM25':>12} {'Δ':>10}")
    lines.append(f"  {'-' * 55}")
    for metric_key, label in [
        ("precision", f"Precision@{k}"),
        ("recall", f"Recall@{k}"),
        ("mrr", f"MRR@{k}"),
        ("ndcg", f"NDCG@{k}"),
    ]:
        sem_val = sem_avg[metric_key]
        bm25_val = bm25_avg[metric_key]
        delta = sem_val - bm25_val
        sign = "+" if delta >= 0 else ""
        lines.append(f"  {label:<20} {sem_val:>12.4f} {bm25_val:>12.4f} {sign}{delta:>9.3f}")
    lines.append(f"{'=' * 65}")

    lines.extend(_build_interpretation_section(k, sem_avg, bm25_avg))

    lines.append("=" * 70)
    lines.append("  ANÁLISIS DE ERRORES (Falsos positivos y negativos)")
    lines.append("=" * 70)
    lines.extend(error_details)
    lines.append("")
    lines.append("  Palabras frecuentes en títulos de FALSOS POSITIVOS:")
    for word, cnt in fp_word_counter.most_common(10):
        lines.append(f"    • {word}: {cnt}")
    lines.append("")
    lines.append("  Keywords más omitidas (FALSOS NEGATIVOS):")
    for word, cnt in fn_word_counter.most_common(10):
        lines.append(f"    • {word}: {cnt} consultas sin recuperar")
    lines.append("")
    lines.append("  ¿Palabras que confunden? Términos genéricos ('ley', 'decreto', 'aprueba')")
    lines.append("  aparecen en títulos tanto relevantes como irrelevantes, reduciendo precisión.")
    lines.append("")

    report = "\n".join(lines)
    print(report)

    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(report, encoding="utf-8")
        print(f"Reporte guardado en: {output}")

    if charts:
        out_dir = ensure_dir(PRESENTATION_DIR)
        plot_metrics_comparison(
            sem_avg, bm25_avg, k,
            out_dir / f"06_metrics_comparison_k{k}.png",
        )
        query_labels = [r["query"] for r in per_query_rows]
        sem_matrix = np.array([[r["sem"]["p"], r["sem"]["r"], r["sem"]["m"], r["sem"]["n"]] for r in per_query_rows])
        bm25_matrix = np.array([[r["bm25"]["p"], r["bm25"]["r"], r["bm25"]["m"], r["bm25"]["n"]] for r in per_query_rows])
        plot_per_query_heatmap(
            query_labels, sem_matrix, bm25_matrix,
            ["Prec", "Rec", "MRR", "NDCG"],
            out_dir / f"07_per_query_heatmap_k{k}.png",
        )
        plot_error_summary(
            dict(fp_word_counter), dict(fn_word_counter),
            out_dir / f"08_error_analysis_k{k}.png",
        )
        print(f"Gráficos guardados en: {out_dir.resolve()}/")

    return {"semantic": sem_avg, "bm25": bm25_avg, "k": k}


def parse_args():
    p = argparse.ArgumentParser(description="Evalúa el motor de búsqueda semántica SPIJ.")
    p.add_argument("--k", type=int, default=5, help="Top-k resultados a evaluar (default: 5)")
    p.add_argument("--output", type=str, default=None, help="Ruta para guardar el reporte")
    p.add_argument("--charts", action="store_true", help="Generar gráficos en exports/presentacion/")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(k=args.k, output=args.output, charts=args.charts)

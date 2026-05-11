"""Motor de búsqueda semántica sobre el corpus de normas SPIJ.

Carga el índice FAISS construido por embeddings_indexer.py y permite
consultas en lenguaje natural, devolviendo las normas más relevantes
ordenadas por similitud semántica.

Uso como script interactivo:
    python semantic_search.py

Uso como módulo:
    from semantic_search import SemanticSearchEngine
    engine = SemanticSearchEngine()
    results = engine.search("despido de trabajadora embarazada", k=5)
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

from db import get_session
from models import Norm
from utils import setup_logging

setup_logging()
LOG = logging.getLogger(__name__)

INDEX_FILE = Path("data/index/norms_index.faiss")
META_FILE  = Path("data/index/norms_meta.pkl")
MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"


class SemanticSearchEngine:
    """Motor de búsqueda semántica densa sobre el corpus SPIJ.

    Implementa dos estrategias de recuperación:
        - Semántica (dense):  embeddings + FAISS (método principal)
        - Léxica (sparse):    BM25 clásico (baseline de comparación)

    Ambas pueden combinarse con re-ranking por nivel jerárquico y
    filtrado por is_valid para excluir normas derogadas.
    """

    def __init__(self):
        if not INDEX_FILE.exists() or not META_FILE.exists():
            raise FileNotFoundError(
                "Índice no encontrado. Ejecuta primero:\n"
                "  python embeddings_indexer.py"
            )
        LOG.info("Cargando índice FAISS desde %s...", INDEX_FILE)
        self._index = faiss.read_index(str(INDEX_FILE))

        with open(META_FILE, "rb") as f:
            self._meta: List[Dict] = pickle.load(f)

        LOG.info("Cargando modelo de embeddings: %s", MODEL_NAME)
        self._model = SentenceTransformer(MODEL_NAME)
        LOG.info("Motor listo. Normas indexadas: %d", self._index.ntotal)

    # ── Búsqueda semántica (dense retrieval) ─────────────────────────────────

    def search(
        self,
        query: str,
        k: int = 5,
        only_valid: bool = True,
        level_filter: Optional[int] = None,
    ) -> List[Dict]:
        """Recupera las k normas más relevantes para la consulta dada.

        Args:
            query:        Consulta en lenguaje natural (español).
            k:            Número de resultados a devolver.
            only_valid:   Si True, filtra normas derogadas.
            level_filter: Si se indica (1-5), filtra por nivel jerárquico.

        Returns:
            Lista de dicts con keys: norm_id, title, type, number, url,
            level, is_valid, score (similitud coseno ∈ [0, 1]).
        """
        # Encodear la consulta con el mismo modelo
        q_vec = self._model.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        ).astype(np.float32)

        # Recuperar más resultados de los necesarios para poder filtrar
        search_k = min(k * 10, self._index.ntotal)
        scores, indices = self._index.search(q_vec, search_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            meta = self._meta[idx]
            if only_valid and not meta.get("is_valid", True):
                continue
            if level_filter is not None and meta.get("level") != level_filter:
                continue
            results.append({**meta, "score": float(score)})
            if len(results) >= k:
                break

        return results

    # ── Búsqueda BM25 (sparse / baseline) ────────────────────────────────────

    def search_bm25(
        self,
        query: str,
        k: int = 5,
        only_valid: bool = True,
    ) -> List[Dict]:
        """Recupera normas usando BM25 (baseline léxico).

        Tokeniza los títulos y construye el índice BM25 en memoria.
        Útil para comparar contra el método semántico.
        """
        # Tokenización simple por espacios (sin spaCy para evitar dependencia)
        corpus_tokens = [
            (m.get("title") or "").lower().split()
            for m in self._meta
        ]
        bm25 = BM25Okapi(corpus_tokens)
        query_tokens = query.lower().split()
        raw_scores = bm25.get_scores(query_tokens)

        # Ordenar por score descendente
        ranked_indices = np.argsort(raw_scores)[::-1]

        results = []
        for idx in ranked_indices:
            if raw_scores[idx] <= 0:
                break
            meta = self._meta[idx]
            if only_valid and not meta.get("is_valid", True):
                continue
            results.append({**meta, "score": float(raw_scores[idx])})
            if len(results) >= k:
                break

        return results

    # ── Utilidades de presentación ────────────────────────────────────────────

    @staticmethod
    def format_results(results: List[Dict], method: str = "Semántico") -> str:
        lines = [f"\n{'─'*60}", f"  Resultados ({method}) — {len(results)} normas", f"{'─'*60}"]
        for i, r in enumerate(results, 1):
            valid_tag = "✅" if r.get("is_valid") else "⚠️ DEROGADA"
            lines.append(
                f"\n  [{i}] {valid_tag} {r.get('type', '')} {r.get('number', '')} | Score: {r['score']:.4f}"
            )
            lines.append(f"      Título: {(r.get('title') or '')[:90]}")
            lines.append(f"      URL:    {r.get('url', '')}")
        return "\n".join(lines)


# ─── Modo interactivo ─────────────────────────────────────────────────────────

def _interactive_session():
    """Sesión interactiva de búsqueda desde la línea de comandos."""
    try:
        engine = SemanticSearchEngine()
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return

    print("\n🔍 Motor de Búsqueda Semántica de Normas SPIJ")
    print("   Escribe tu consulta jurídica en lenguaje natural.")
    print("   Comandos: 'bm25:<consulta>' para modo léxico | 'q' para salir\n")

    while True:
        try:
            query = input("Consulta > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nSaliendo.")
            break

        if not query or query.lower() == "q":
            break

        k = 5
        if query.startswith("bm25:"):
            query_text = query[5:].strip()
            results = engine.search_bm25(query_text, k=k)
            print(engine.format_results(results, method="BM25 (léxico)"))
        else:
            results = engine.search(query, k=k)
            print(engine.format_results(results, method="Semántico (embeddings)"))
            # También mostrar BM25 para comparación
            results_bm25 = engine.search_bm25(query, k=k)
            print(engine.format_results(results_bm25, method="BM25 (baseline)"))


if __name__ == "__main__":
    _interactive_session()

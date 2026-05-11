"""Pipeline de generación de embeddings e indexación vectorial con FAISS.

Lee las normas de la base de datos, genera embeddings con sentence-transformers
y construye un índice FAISS para búsqueda semántica eficiente.

Uso:
    python embeddings_indexer.py --limit 0         # indexar todo
    python embeddings_indexer.py --limit 500       # indexar las 500 más recientes
    python embeddings_indexer.py --level-filter 2  # solo leyes (nivel 2)
    python embeddings_indexer.py --rebuild         # reconstruir índice desde cero

Archivos generados en data/index/:
    norms_index.faiss   — índice FAISS de vectores normalizados
    norms_meta.pkl      — mapeo index_position -> (norm_id, title, type, url, is_valid)
"""

import argparse
import logging
import pickle
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from bs4 import BeautifulSoup

from db import get_session
from models import Norm, NormVersion
from utils import setup_logging

setup_logging()
LOG = logging.getLogger(__name__)

# ─── Configuración ────────────────────────────────────────────────────────────

MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"
INDEX_DIR  = Path("data/index")
INDEX_FILE = INDEX_DIR / "norms_index.faiss"
META_FILE  = INDEX_DIR / "norms_meta.pkl"
BATCH_SIZE = 32   # Número de textos a encodear por batch (ajustar según RAM/GPU)
DIM        = 768  # Dimensión del modelo paraphrase-multilingual-mpnet-base-v2


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _strip_html(text: str) -> str:
    """Elimina etiquetas HTML y normaliza espacios."""
    if not text:
        return ""
    if "<" in text and ">" in text:
        text = BeautifulSoup(text, "html.parser").get_text(separator=" ", strip=True)
    return " ".join(text.split())


def _build_text_for_embedding(norm: Norm, raw_text: str = "") -> str:
    """Construye el texto de entrada para el modelo de embeddings.

    Combina tipo + número + título + primeros 1000 chars del texto,
    dando prioridad a los metadatos estructurados (más señal semántica).
    """
    parts = []
    if norm.type:
        parts.append(norm.type)
    if norm.number:
        parts.append(f"N° {norm.number}")
    if norm.title:
        parts.append(norm.title)
    if raw_text:
        clean = _strip_html(raw_text)[:1000]
        parts.append(clean)
    return ". ".join(parts)


def load_norms_for_indexing(
    session,
    limit: int = 0,
    level_filter: int = None,
    only_valid: bool = True,
) -> List[Tuple[Norm, str]]:
    """Carga normas y su texto completo más reciente desde norm_versions."""
    q = session.query(Norm)
    if only_valid:
        q = q.filter(Norm.is_valid == 1)
    if level_filter is not None:
        q = q.filter(Norm.level == level_filter)
    q = q.order_by(Norm.updated_at.desc())
    if limit:
        q = q.limit(limit)
    norms = q.all()

    LOG.info("Cargando texto completo para %d normas...", len(norms))
    result = []
    for norm in norms:
        # Obtener la versión de texto más reciente
        latest_version = (
            session.query(NormVersion)
            .filter_by(norm_id=norm.id)
            .order_by(NormVersion.version_date.desc())
            .first()
        )
        raw_text = latest_version.raw_text if latest_version else ""
        result.append((norm, raw_text or ""))
    return result


# ─── Indexación ──────────────────────────────────────────────────────────────

def build_index(
    limit: int = 0,
    level_filter: int = None,
    only_valid: bool = True,
    rebuild: bool = False,
) -> None:
    """Construye (o reconstruye) el índice FAISS y el archivo de metadatos."""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    if INDEX_FILE.exists() and not rebuild:
        LOG.info("El índice ya existe en %s. Usa --rebuild para reconstruirlo.", INDEX_FILE)
        return

    LOG.info("Cargando modelo de embeddings: %s", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)

    session = get_session()
    try:
        pairs = load_norms_for_indexing(session, limit=limit, level_filter=level_filter, only_valid=only_valid)
        if not pairs:
            LOG.error("No se encontraron normas. ¿Ejecutaste el scraper primero?")
            return

        texts = [_build_text_for_embedding(norm, raw) for norm, raw in pairs]
        meta  = [
            {
                "norm_id":  norm.id,
                "title":    norm.title,
                "type":     norm.type,
                "number":   norm.number,
                "url":      norm.url,
                "level":    norm.level,
                "is_valid": bool(norm.is_valid),
                "status":   norm.status,
            }
            for norm, _ in pairs
        ]

        LOG.info("Generando embeddings para %d normas (batch_size=%d)...", len(texts), BATCH_SIZE)
        embeddings = model.encode(
            texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=True,
            normalize_embeddings=True,  # normalizar para usar producto interno = coseno
            convert_to_numpy=True,
        )
        embeddings = embeddings.astype(np.float32)

        # Índice FAISS: producto interior (coseno con vectores normalizados)
        index = faiss.IndexFlatIP(DIM)
        index.add(embeddings)

        faiss.write_index(index, str(INDEX_FILE))
        with open(META_FILE, "wb") as f:
            pickle.dump(meta, f)

        LOG.info("Índice guardado: %s (%d vectores)", INDEX_FILE, index.ntotal)
        LOG.info("Metadatos guardados: %s", META_FILE)
        print(f"\n✅ Índice construido con {index.ntotal} normas.")
        print(f"   Archivo FAISS: {INDEX_FILE}")
        print(f"   Metadatos:     {META_FILE}")

    finally:
        session.close()


# ─── Punto de entrada ─────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Construye el índice FAISS de normas SPIJ.")
    p.add_argument("--limit", type=int, default=0, help="Número máximo de normas (0 = todas)")
    p.add_argument("--level-filter", type=int, default=None, help="Filtrar por nivel jerárquico (1-5)")
    p.add_argument("--no-valid-filter", action="store_true", help="Incluir normas derogadas en el índice")
    p.add_argument("--rebuild", action="store_true", help="Reconstruir el índice aunque ya exista")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_index(
        limit=args.limit,
        level_filter=args.level_filter,
        only_valid=not args.no_valid_filter,
        rebuild=args.rebuild,
    )

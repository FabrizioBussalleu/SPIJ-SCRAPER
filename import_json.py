"""Importa normas desde un export JSON al SQLite local.

Restaura metadatos del corpus cuando no se puede re-ejecutar el scraper.
Genera norm_versions con texto minimo (tipo + titulo) para permitir
TF-IDF, EDA e indexacion basica. Para texto legal completo, ejecutar:
    python main.py scrape --limit N

Uso:
    python import_json.py
    python import_json.py --file ai-training-dataset/normas_export.json
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

from categorizer import categorize_norm
from db import get_session, init_db, upsert_norm_with_version
from utils import setup_logging

setup_logging()
LOG = logging.getLogger(__name__)

DEFAULT_FILE = Path("ai-training-dataset/normas_export.json")


def _minimal_raw_text(row: dict) -> str:
    """Texto minimo cuando el export no incluye raw_text HTML."""
    parts = [row.get("type") or "", row.get("number") or "", row.get("title") or ""]
    msg = row.get("validation_message") or row.get("status") or ""
    if msg:
        parts.append(str(msg))
    return " ".join(p for p in parts if p).strip()


def import_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"No se encontro: {path}")

    init_db()
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("El JSON debe ser una lista de normas")

    session = get_session()
    added = updated = unchanged = 0
    try:
        for row in rows:
            if not row.get("url"):
                continue
            item = {
                "type": row.get("type"),
                "number": row.get("number"),
                "title": row.get("title"),
                "date": row.get("date"),
                "source": row.get("source"),
                "status": row.get("status"),
                "url": row["url"],
                "text_hash": row.get("text_hash") or row.get("last_hash"),
                "level": row.get("level") or categorize_norm(row),
                "is_valid": row.get("is_valid", True),
            }
            raw = _minimal_raw_text(row)
            outcome = upsert_norm_with_version(item, raw, session=session)
            if outcome == "added":
                added += 1
            elif outcome == "updated":
                updated += 1
            else:
                unchanged += 1
    finally:
        session.close()

    stats = {"total": len(rows), "added": added, "updated": updated, "unchanged": unchanged}
    LOG.info("Importacion completada: %s", stats)
    return stats


def main():
    p = argparse.ArgumentParser(description="Importa normas desde JSON exportado.")
    p.add_argument("--file", type=str, default=str(DEFAULT_FILE))
    args = p.parse_args()
    stats = import_json(Path(args.file))
    print(f"Importadas: {stats['added']} nuevas | {stats['updated']} actualizadas | {stats['unchanged']} sin cambios")


if __name__ == "__main__":
    main()

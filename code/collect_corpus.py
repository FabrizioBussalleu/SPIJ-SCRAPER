"""Recolección reproducible del corpus de normas SPIJ para el Trabajo Final.

Autentica contra las APIs internas del SPIJ, muestrea normas de varias
agrupaciones y ventanas temporales (para lograr variedad de tipos y años),
descarga el detalle completo (textoCompleto) de cada norma, asigna el nivel
de la Pirámide de Kelsen y persiste todo en SQLite (norms + norm_versions +
norm_articles + norm_citations).

A diferencia del scraper original, NO usa la validación de vigencia basada en
la página web Angular (que devuelve solo el shell JS y producía falsos
positivos). La vigencia se modela por separado y a nivel de artículo con las
notas oficiales (*) de SPIJ; ver build_vigencia_dataset.py.

Uso:
    python collect_corpus.py --target 900
"""
import argparse
import logging
import time

import requests
from bs4 import BeautifulSoup

from categorizer import categorize_norm
from db import init_db, get_session, upsert_norm_with_version
from models import Norm, NormArticle, NormCitation
from parser import extract_articles, extract_citations
from scraper.spij_scraper import SpijScraper
from utils import setup_logging

setup_logging()
LOG = logging.getLogger(__name__)

# Combinaciones (agrupación, orden) para lograr variedad de tipos y años.
# orden '1' = más recientes primero; '2' = más antiguas primero.
SAMPLING_PLAN = [
    ("CONSTITUCION  POLITICA, LEYES  ORGANICAS Y CODIGOS", "1"),
    ("CONSTITUCION  POLITICA, LEYES  ORGANICAS Y CODIGOS", "2"),
    ("NORMAS ADMINISTRATIVAS DE CARACTER PARTICULAR", "1"),
    ("NORMAS ADMINISTRATIVAS DE CARACTER PARTICULAR", "2"),
    ("LEGISLACIÓN EMITIDA POR ENTIDADES VINCULADAS A LA ADMINISTRACIÓN DE JUSTICIA", "1"),
    ("LEGISLACION SUPRANACIONAL", "1"),
    ("LEGISLACIÓN EMITIDA POR GOBIERNOS LOCALES Y REGIONALES", "1"),
]

# Los códigos son gigantes (varios MB) y sesgarían el corpus de EDA/búsqueda;
# se procesan aparte para el dataset de vigencia. Los saltamos aquí por tamaño.
MAX_TEXT_CHARS = 400_000


def build_payload(scraper, agrupacion, desde, hasta, orden):
    return {
        "filtros": {
            "buscarHistorico": False,
            "busquedaSugerida": False,
            "numeroDispositivoLegal": " ",
            "dispositivoLegal": [],
            "tomo": {"id": "", "nombre": ""},
            "materia": {"id": "", "nombre": ""},
            "agrupacion": [agrupacion],
            "sector": [],
            "subSector": {"id": "", "nombre": ""},
            "orden": orden,
        },
        "facetsSeleccionadas": {"fechaPublicacionGap": {"numero": 10, "unidad": "YEAR"}},
        "tipoNorma": "NR",
        "textoBusqueda": None,
        "textoSumilla": None,
        "desde": desde,
        "hasta": hasta,
    }


def main(target: int, rate_limit: float, per_combo: int):
    init_db()
    scraper = SpijScraper(rate_limit=0)  # autenticación + sesiones
    session = get_session()

    seen_ids = set()
    # Resumabilidad: precargar ids ya presentes en la BD
    for (url,) in session.query(Norm.url).all():
        seen_ids.add(url.rstrip('/').split('/')[-1])
    LOG.info('Precargados %s ids existentes', len(seen_ids))
    added = 0
    processed = 0
    page = 25

    for agrupacion, orden in SAMPLING_PLAN:
        if added >= target:
            break
        combo_added = 0
        offset = 0
        LOG.info("Muestreando agrupación=%s orden=%s", agrupacion[:40], orden)
        while combo_added < per_combo and added < target:
            payload = build_payload(scraper, agrupacion, offset, offset + page, orden)
            try:
                data = scraper._search(payload)
            except Exception as e:
                LOG.warning("Búsqueda falló offset=%s: %s", offset, e)
                break
            docs = data.get("resultados", []) or []
            if not docs:
                break
            for doc in docs:
                if added >= target or combo_added >= per_combo:
                    break
                nid = doc.get("id")
                if not nid or nid in seen_ids:
                    continue
                seen_ids.add(nid)
                try:
                    detail = scraper._fetch_detail(nid)
                except Exception as e:
                    LOG.warning("Detalle falló %s: %s", nid, e)
                    continue
                raw_html = detail.get("textoCompleto", "") or ""
                if not raw_html or len(raw_html) > MAX_TEXT_CHARS:
                    # vacío (sin texto) o código gigante -> fuera del corpus general
                    continue
                title_html = detail.get("titulo") or doc.get("sumilla") or ""
                title = scraper._strip_html(title_html) or scraper._strip_html(doc.get("sumilla"))
                if title == "<Campo no disponible>" or not title:
                    title = scraper._strip_html(doc.get("sumilla")) or (doc.get("codigoNorma") or nid)
                item = {
                    "type": doc.get("dispositivoLegal") or doc.get("palabra"),
                    "number": doc.get("codigoNorma"),
                    "title": title,
                    "date": detail.get("fechaPublicacion") or doc.get("fechaPublicacion") or None,
                    "source": detail.get("sector") or doc.get("sector"),
                    "status": "VIGENTE",
                    "is_valid": True,
                    "url": scraper.WEB_DETAIL_URL.format(norm_id=nid),
                    "text_hash": scraper._hash_text(raw_html),
                }
                item["level"] = categorize_norm(item)
                try:
                    result = upsert_norm_with_version(item, raw_html, session=session)
                except Exception as e:
                    LOG.warning("DB falló %s: %s", nid, e)
                    session.rollback()
                    continue
                processed += 1
                if result == "added":
                    added += 1
                    combo_added += 1
                    # Parsear artículos y citas
                    norm = session.query(Norm).filter_by(url=item["url"]).first()
                    if norm:
                        _persist_structure(session, norm, raw_html)
                if added % 50 == 0 and result == "added":
                    LOG.info("Progreso: %s normas añadidas (%s procesadas)", added, processed)
                if rate_limit:
                    time.sleep(rate_limit)
            offset += page
    session.close()
    LOG.info("LISTO. Normas nuevas añadidas: %s (procesadas: %s)", added, processed)


def _persist_structure(session, norm, raw_html):
    """Extrae y guarda artículos y citas de la norma."""
    try:
        articles = extract_articles(raw_html)
        for a in articles[:500]:
            session.add(NormArticle(norm_id=norm.id, number=a["number"],
                                    content=a["content"][:8000], order=a["order"]))
        citations = extract_citations(raw_html)
        for c in citations[:300]:
            session.add(NormCitation(source_norm_id=norm.id,
                                     target_text=c.get("target_text", "")[:256],
                                     citation_type=c.get("type") or c.get("citation_type")))
        session.commit()
    except Exception as e:
        LOG.warning("Estructura falló norm=%s: %s", norm.id, e)
        session.rollback()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=900)
    ap.add_argument("--rate-limit", type=float, default=0.0)
    ap.add_argument("--per-combo", type=int, default=200)
    args = ap.parse_args()
    main(args.target, args.rate_limit, args.per_combo)

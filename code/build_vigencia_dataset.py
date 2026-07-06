"""Construye un dataset de vigencia a NIVEL DE ARTÍCULO usando las notas
oficiales de derogación de SPIJ como ground-truth.

Motivación: el enfoque original etiquetaba la vigencia a nivel de norma con una
heurística que revisaba la página web (Angular, sin render) y confundía "norma
que deroga a otras" con "norma derogada". El resultado era un corpus 99.3%
vigente con etiquetas ruidosas: F1 de la clase Derogada = 0.

SPIJ, en cambio, anota CADA disposición derogada con una nota oficial dentro del
texto, del tipo:
    (*) Artículo derogado por la Séptima Disposición Final de la Ley Nº 26497 ...
    (*) Numeral derogado por el Literal a) ... del Decreto Legislativo N° 1384 ...

Recorremos los códigos (Civil, Penal, Procesal, etc.), partimos su texto en
artículos y etiquetamos cada artículo como Derogado (0) o Vigente (1) según
tenga o no una nota oficial de derogación. Esto produce etiquetas reales,
confiables y balanceables, con ambas clases bien representadas.

Salida: tabla `article_samples` en la BD + CSV en data/vigencia_articulos.csv
"""
import argparse
import logging
import re

import requests
from bs4 import BeautifulSoup
from sqlalchemy import Column, Integer, String, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
import os

from utils import setup_logging

setup_logging()
LOG = logging.getLogger(__name__)

BASE = declarative_base()


class ArticleSample(BASE):
    __tablename__ = "article_samples"
    id = Column(Integer, primary_key=True)
    code_id = Column(String(32))
    code_name = Column(String(256))
    article_num = Column(String(64))
    text = Column(Text)          # texto del artículo SIN la nota de vigencia
    raw_text = Column(Text)      # texto completo tal como lo publica SPIJ
    label = Column(Integer)      # 1=Vigente, 0=Derogado


AUTH = "https://spijwsii.minjus.gob.pe/spij-ext-back/authenticate"
DETAIL = "https://spijwsii.minjus.gob.pe/spij-ext-back/api/detallenorma/{}"
MAESTROS = "https://spijwsii.minjus.gob.pe/spij-ext-back/api/maestros"

ART_SPLIT = re.compile(r'Art[ií]culo\s+(\d+[A-Za-z\-]*)\s*[\.\-º°]')
# Nota oficial de derogación (aplica a artículo, numeral, inciso, literal, etc.)
DEROG_NOTE = re.compile(
    r'\(\s*\*\s*\)\s*(?:\(\s*\*\s*\)\s*)?'
    r'(?:Art[ií]culo|Numeral|Inciso|Literal|Ap[ae]rtado|P[aá]rrafo|Disposici[oó]n|Cap[ií]tulo|T[ií]tulo|Secci[oó]n)?'
    r'[^.]{0,60}?\b(?:derogad|d[eé]jes[ee]\s+sin\s+efecto|dejad[oa]\s+sin\s+efecto|dejar\s+sin\s+efecto|abrogad)',
    re.IGNORECASE)
# Para separar la nota del cuerpo (y evitar fuga trivial en el texto 'limpio')
NOTE_STRIP = re.compile(r'\(\s*\*\s*\).*$', re.DOTALL)


def get_session():
    url = os.getenv("DB_URL", "sqlite:///data/norms.db")
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    eng = create_engine(url, connect_args=connect_args)
    BASE.metadata.create_all(eng)
    return sessionmaker(bind=eng)()


def main(max_vigente_ratio: float):
    back = requests.post(AUTH, json={"usuario": "spijext", "clave": "password", "tipo": 0}, timeout=30).json()["value"]
    bh = {"Authorization": f"Bearer {back}"}
    codigos = [(c["id"], c["nombre"]) for c in requests.get(MAESTROS, headers=bh, timeout=30).json()[0]["constitucion"] if c["id"] != "NINGUNO"]
    LOG.info("Procesando %s códigos", len(codigos))

    session = get_session()
    session.query(ArticleSample).delete()
    session.commit()

    derog, vig = [], []
    for cid, cname in codigos:
        try:
            det = requests.get(DETAIL.format(cid), headers=bh, timeout=180).json()
        except Exception as e:
            LOG.warning("Detalle %s falló: %s", cid, e)
            continue
        txt = BeautifulSoup(det.get("textoCompleto", "") or "", "html.parser").get_text(" ", strip=True)
        idxs = [(m.start(), m.group(1)) for m in ART_SPLIT.finditer(txt)]
        nd = 0
        for i, (pos, num) in enumerate(idxs):
            end = idxs[i + 1][0] if i + 1 < len(idxs) else len(txt)
            block = txt[pos:end].strip()
            if len(block) < 40:
                continue
            is_derog = bool(DEROG_NOTE.search(block))
            clean = NOTE_STRIP.sub("", block).strip()  # cuerpo sin la nota (*)
            if len(clean) < 30:
                clean = block[:300]
            rec = dict(code_id=cid, code_name=cname[:256], article_num=num,
                       text=clean[:6000], raw_text=block[:8000],
                       label=0 if is_derog else 1)
            (derog if is_derog else vig).append(rec)
            if is_derog:
                nd += 1
        LOG.info("  %-42s arts=%4d derogados=%d", cname[:42], len(idxs), nd)

    # Balancear: limitar vigentes a max_vigente_ratio * derogados
    import random
    random.seed(42)
    random.shuffle(vig)
    keep_vig = vig[: int(len(derog) * max_vigente_ratio)]
    samples = derog + keep_vig
    random.shuffle(samples)
    for r in samples:
        session.add(ArticleSample(**r))
    session.commit()

    # CSV export
    import csv
    out = "data/vigencia_articulos.csv"
    try:
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["code_id", "code_name", "article_num", "label", "text"])
            for r in samples:
                w.writerow([r["code_id"], r["code_name"], r["article_num"], r["label"], r["text"]])
    except Exception as e:
        LOG.warning("CSV no escrito: %s", e)

    LOG.info("LISTO. Derogados=%d Vigentes(muestreados)=%d Total=%d",
             len(derog), len(keep_vig), len(samples))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-vigente-ratio", type=float, default=1.5,
                    help="vigentes = ratio * derogados (para balancear)")
    args = ap.parse_args()
    main(args.max_vigente_ratio)

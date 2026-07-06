"""Módulo de conexión a la base de datos usando SQLAlchemy.

Por defecto usa SQLite local, pero puede configurarse vía DB_URL en .env.
"""

import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from models import Base, Norm, NormVersion
from sqlalchemy.exc import SQLAlchemyError
from datetime import datetime
import json
from dotenv import load_dotenv
import logging

load_dotenv()

# Default DB URL (SQLite local file). Can be overridden by setting `DB_URL` in `.env`.
DB_URL = os.getenv('DB_URL', 'sqlite:///data/norms.db')

# Create engine differently depending on the scheme. Only import psycopg2 when needed.
if DB_URL.startswith('postgresql://'):
    try:
        import psycopg2  # noqa: F401 - runtime dependency for PostgreSQL
    except ImportError as e:
        raise ModuleNotFoundError("psycopg2 is required for PostgreSQL DB_URL. Install 'psycopg2-binary' or set DB_URL to a SQLite URL.") from e

    # Use a pooled engine for PostgreSQL
    engine = create_engine(DB_URL, pool_size=10, max_overflow=20, pool_pre_ping=True)

    # Test PostgreSQL connection
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    except Exception as e:
        raise ConnectionError("Failed to connect to PostgreSQL database. Check DB_URL.") from e
else:
    # Fallback to SQLite or other DBs supported by SQLAlchemy without requiring psycopg2
    connect_args = {"check_same_thread": False} if DB_URL.startswith('sqlite:') else {}
    engine = create_engine(DB_URL, connect_args=connect_args)

SessionLocal = sessionmaker(bind=engine)

LOG = logging.getLogger(__name__)

_NORM_FIELD_LIMITS = {
    'type': 128,
    'number': 128,
    'title': 1024,
    'source': 256,
    'status': 128,
    'url': 2048,
    'text_hash': 128,
}


def _truncate_field(value, limit, field_name):
    if value is None or not isinstance(value, str):
        return value
    if len(value) <= limit:
        return value
    LOG.warning("Truncando campo %s a %s caracteres (tenía %s)", field_name, limit, len(value))
    return value[:limit]


def _sanitize_norm_data(norm_data: dict) -> dict:
    sanitized = dict(norm_data)
    for field, limit in _NORM_FIELD_LIMITS.items():
        if field in sanitized:
            sanitized[field] = _truncate_field(sanitized.get(field), limit, field)
    return sanitized

def init_db():
    """Crear todas las tablas en la base de datos."""
    Base.metadata.create_all(engine)

def get_session():
    return SessionLocal()

def upsert_norm_with_version(norm_data: dict, raw_text: str, session=None):
    """
    Inserta o actualiza una norma y crea una versión si el hash cambió.
    norm_data: dict con metadatos y text_hash
    raw_text: texto completo de la norma
    session: sesión SQLAlchemy (opcional)
    """
    close_session = False
    if session is None:
        session = get_session()
        close_session = True
    norm_data = _sanitize_norm_data(norm_data)
    try:
        norm = session.query(Norm).filter_by(url=norm_data['url']).first()
        now = datetime.utcnow()
        if not norm:
            norm = Norm.from_dict(norm_data)
            norm.last_hash = norm_data['text_hash']
            norm.last_updated_at = now
            session.add(norm)
            session.flush()  # Para obtener el id
            version = NormVersion(
                norm_id=norm.id,
                version_date=now,
                text_hash=norm_data['text_hash'],
                raw_text=raw_text,
                metadata_json=json.dumps(norm_data, ensure_ascii=False)
            )
            session.add(version)
            session.commit()
            return 'added'
        else:
            # Si el hash cambió, crear nueva versión y actualizar campos
            if norm.last_hash != norm_data['text_hash']:
                norm.type = norm_data.get('type')
                norm.number = norm_data.get('number')
                norm.title = norm_data.get('title')
                date_value = norm_data.get('date')
                if isinstance(date_value, str):
                    try:
                        date_value = datetime.fromisoformat(date_value).date()
                    except ValueError:
                        date_value = None
                norm.date = date_value
                norm.source = norm_data.get('source')
                norm.status = norm_data.get('status')
                norm.text_hash = norm_data.get('text_hash')
                norm.level = norm_data.get('level')
                norm.last_hash = norm_data['text_hash']
                norm.last_updated_at = now
                version = NormVersion(
                    norm_id=norm.id,
                    version_date=now,
                    text_hash=norm_data['text_hash'],
                    raw_text=raw_text,
                    metadata_json=json.dumps(norm_data, ensure_ascii=False)
                )
                session.add(version)
                session.commit()
                return 'updated'
            else:
                return 'unchanged'
    except SQLAlchemyError as e:
        session.rollback()
        raise e
    finally:
        if close_session:
            session.close()

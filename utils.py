"""Funciones utilitarias: logging, parsing y helpers.
"""
import logging
import os
import sys
from dotenv import load_dotenv


def _ensure_utf8_console():
    """Evita UnicodeEncodeError en consola Windows (cp1252) con emojis del CLI."""
    if sys.platform == "win32":
        for stream in (sys.stdout, sys.stderr):
            if hasattr(stream, "reconfigure"):
                try:
                    stream.reconfigure(encoding="utf-8")
                except Exception:
                    pass


def setup_logging(level: str = None):
    """Configura logging básico usando variable de entorno LOG_LEVEL."""
    _ensure_utf8_console()
    load_dotenv()
    lvl = level or os.getenv('LOG_LEVEL', 'INFO')
    numeric = getattr(logging, lvl.upper(), logging.INFO)
    logging.basicConfig(level=numeric, format='%(asctime)s %(levelname)s %(name)s: %(message)s')

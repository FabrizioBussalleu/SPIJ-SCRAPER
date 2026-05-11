"""Funciones utilitarias: logging, parsing y helpers.
"""
import logging
import os
from dotenv import load_dotenv


def setup_logging(level: str = None):
    """Configura logging básico usando variable de entorno LOG_LEVEL."""
    load_dotenv()
    lvl = level or os.getenv('LOG_LEVEL', 'INFO')
    numeric = getattr(logging, lvl.upper(), logging.INFO)
    logging.basicConfig(level=numeric, format='%(asctime)s %(levelname)s %(name)s: %(message)s')

"""Migración para agregar campos de validación de vigencia a la tabla norms.

Ejecutar este script una sola vez para actualizar la estructura de la BD.
"""
import logging
from sqlalchemy import text
from db import engine

LOG = logging.getLogger(__name__)

def migrate_add_validation_fields():
    """Agregar campos de validación a la tabla norms."""
    
    # Lista de columnas a agregar
    columns_to_add = [
        "ALTER TABLE norms ADD COLUMN is_valid INTEGER DEFAULT 1;",
        "ALTER TABLE norms ADD COLUMN validation_message TEXT;",
        "ALTER TABLE norms ADD COLUMN validation_date TIMESTAMP;"
    ]
    
    with engine.connect() as conn:
        for sql in columns_to_add:
            try:
                # Usar transacción separada para cada comando
                with conn.begin():
                    conn.execute(text(sql))
                    LOG.info(f"✅ Ejecutado: {sql}")
            except Exception as e:
                # Si la columna ya existe, continuar
                if "already exists" in str(e).lower() or "duplicate column" in str(e).lower() or "ya existe la columna" in str(e).lower():
                    LOG.info(f"✅ Columna ya existe, omitiendo: {sql}")
                else:
                    LOG.error(f"❌ Error ejecutando {sql}: {e}")
                    raise
    
    LOG.info("Migración completada exitosamente.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    migrate_add_validation_fields()
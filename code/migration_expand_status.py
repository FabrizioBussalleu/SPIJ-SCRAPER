"""Migración para aumentar el tamaño del campo status."""
import logging
from sqlalchemy import text
from db import engine

LOG = logging.getLogger(__name__)

def migrate_expand_status_field():
    """Aumentar el tamaño del campo status de 128 a 512 caracteres."""
    
    sql = "ALTER TABLE norms ALTER COLUMN status TYPE VARCHAR(512);"
    
    with engine.connect() as conn:
        try:
            conn.execute(text(sql))
            conn.commit()
            LOG.info("✅ Campo status expandido a 512 caracteres")
        except Exception as e:
            LOG.error(f"❌ Error expandiendo campo status: {e}")
            raise
    
    LOG.info("Migración de status completada exitosamente.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    migrate_expand_status_field()
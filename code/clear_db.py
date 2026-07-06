from db import get_session, engine
from models import Base
from sqlalchemy import text

def clear_db():
    session = get_session()
    try:
        # Drop all tables and recreate
        Base.metadata.drop_all(engine)
        Base.metadata.create_all(engine)
        print("Base de datos limpiada y recreada.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        session.close()

if __name__ == "__main__":
    clear_db()

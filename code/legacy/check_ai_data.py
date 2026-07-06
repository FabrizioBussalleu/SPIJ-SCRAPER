from db import get_session
from models import Norm, NormArticle, NormCitation

def check_data():
    session = get_session()
    try:
        norms = session.query(Norm).count()
        articles = session.query(NormArticle).count()
        citations = session.query(NormCitation).count()
        
        print(f"Normas: {norms}")
        print(f"Artículos extraídos: {articles}")
        print(f"Citas extraídas: {citations}")
        
        if norms > 0:
            print("\n--- DEBUG RAW TEXT ---")
            norm = session.query(Norm).first()
            print(f"Norma: {norm.type} {norm.number}")
            print(f"Raw Text Preview (500 chars):")
            print(norm.versions[0].raw_text[:500] if norm.versions else "No version text")
            print("----------------------")

        if articles > 0:

            print("\nEjemplo de artículo:")
            art = session.query(NormArticle).first()
            print(f"Norma ID {art.norm_id} - Art. {art.number}: {art.content[:50]}...")
            
        if citations > 0:
            print("\nEjemplo de cita:")
            cit = session.query(NormCitation).first()
            print(f"Norma ID {cit.source_norm_id} cita a: {cit.target_text} ({cit.citation_type})")
            
    finally:
        session.close()

if __name__ == "__main__":
    check_data()

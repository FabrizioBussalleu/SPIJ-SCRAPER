"""Script para obtener IDs reales de normas y probar validación."""
import logging
from db import get_session
from models import Norm
from scraper.spij_scraper import SpijScraper

logging.basicConfig(level=logging.INFO)

def get_real_ids_and_test():
    session = get_session()
    try:
        # Obtener algunas normas reales de la BD
        norms = session.query(Norm).limit(3).all()
        
        print("🔍 Probando validación con IDs reales...")
        
        scraper = SpijScraper(rate_limit=2.0)
        
        for norm in norms:
            print(f"\n📄 Probando: {norm.type} {norm.number}")
            print(f"   URL: {norm.url}")
            
            try:
                # Extraer ID de la URL
                norm_id = norm.url.split('/')[-1] if norm.url else None
                if not norm_id:
                    print("   ❌ No se pudo extraer ID de la URL")
                    continue
                    
                print(f"   ID extraído: {norm_id}")
                
                # Obtener detalle
                detail = scraper._fetch_detail(norm_id)
                print(f"   ✅ Detalle obtenido: {detail.get('titulo', 'Sin título')[:100]}")
                
                # Validar vigencia
                is_valid, status_message = scraper._validate_norm_status(norm_id, detail)
                status_icon = "✅" if is_valid else "⚠️"
                print(f"   {status_icon} Estado: {status_message}")
                
                # Mostrar un poco del contenido para verificar
                content_preview = detail.get('textoCompleto', '')[:200]
                if content_preview:
                    print(f"   📝 Vista previa: {content_preview}...")
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
    
    finally:
        session.close()

if __name__ == "__main__":
    get_real_ids_and_test()
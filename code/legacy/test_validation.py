"""Script de prueba para validación de vigencia de normas SPIJ.

Ejemplo de uso de la nueva funcionalidad de validación.
"""
import logging
from scraper.spij_scraper import SpijScraper

def test_validation():
    """Probar la validación con algunas normas específicas."""
    
    logging.basicConfig(level=logging.INFO)
    
    # Inicializar scraper
    scraper = SpijScraper(rate_limit=2.0)
    
    print("🔍 Probando validación de normas SPIJ...\n")
    
    # IDs de ejemplo (reemplazar con IDs reales del SPIJ)
    test_norm_ids = [
        "58019",  # Ejemplo - reemplazar con ID real
        "123456",  # Ejemplo - reemplazar con ID real
    ]
    
    for norm_id in test_norm_ids:
        try:
            print(f"📄 Probando norma ID: {norm_id}")
            
            # Obtener detalle
            detail = scraper._fetch_detail(norm_id)
            title = detail.get('titulo', 'Sin título')[:100]
            print(f"   Título: {title}")
            
            # Validar vigencia
            is_valid, status_message = scraper._validate_norm_status(norm_id, detail)
            
            status_icon = "✅" if is_valid else "⚠️"
            print(f"   {status_icon} Estado: {status_message}")
            print()
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            print()

if __name__ == "__main__":
    test_validation()
from scraper.spij_scraper import SpijScraper
import logging

# Configurar logging para ver resultados
logging.basicConfig(level=logging.INFO, format='%(message)s')

def test_targeted():
    scraper = SpijScraper(rate_limit=1.0)
    
    # IDs de ejemplo (Necesitamos IDs reales para probar)
    # Como no tengo la lista exacta de IDs "falsos positivos" que salieron antes,
    # usaré la lógica de validación directa sobre IDs conocidos si los tuviera.
    # Pero mejor, haré que el script busque normas y reporte su estado.
    
    print("--- INICIANDO TEST DIRIGIDO ---")
    print("Buscando normas para verificar la nueva lógica de validación...")
    
    # Normas que sabemos que dieron problemas antes (Falsos Positivos)
    # H1423545 fue la que reportaste
    # H1412121 es una norma DEROGADA real proporcionada por el usuario
    # H1370891 es una norma DEJADA SIN EFECTO
    falsos_positivos = ['H1423545', 'H1412121', 'H1370891'] 
    
    print("\n--- GRUPO MIXTO: FALSOS POSITIVOS Y DEROGADAS ---")
    for norm_id in falsos_positivos:
        try:
            detail = scraper._fetch_detail(norm_id)
            is_valid, msg = scraper._validate_norm_status(norm_id, detail)
            status = "✅ VIGENTE" if is_valid else f"❌ NO VIGENTE"
            print(f"Norma {norm_id}: {status} ({msg})")
        except Exception as e:
            print(f"Norma {norm_id}: Error al consultar ({e})")

    # Para encontrar Derogadas reales, tendríamos que buscar IDs específicos.
    # Como no tengo IDs a mano, confiaré en que el usuario valide los falsos positivos primero,
    # que es lo más crítico ahora.
    
    print("\n--- Fin del Test ---")

if __name__ == "__main__":
    test_targeted()

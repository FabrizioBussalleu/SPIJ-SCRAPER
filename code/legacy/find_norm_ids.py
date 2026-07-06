from scraper.spij_scraper import SpijScraper
import logging
import re

def find_ids():
    scraper = SpijScraper(rate_limit=1.0)
    
    targets = [
        ("RESOLUCION MINISTERIAL", "608-2025/MINSA"),
        ("RESOLUCION DIRECTORAL", "000011-2025-INS/CETS"),
        ("DECRETO SUPREMO", "090-2025-PCM"),
        ("RESOLUCION MINISTERIAL", "000097-2025-MC"),
        ("RESOLUCION DIRECTORAL", "008-2025-MTC/18")
    ]
    
    print("--- BUSCANDO IDs DE NORMAS ---")
    
    with open('found_ids.txt', 'w', encoding='utf-8') as f:
        for tipo, numero in targets:
            msg = f"\nBuscando: {tipo} N° {numero}"
            print(msg)
            f.write(msg + "\n")
            
            agrupacion_config = scraper._AGRUPACION_CONFIG['1']
            filtros = agrupacion_config['agrupaciones']
            
            payload = scraper._build_payload(
                agrupaciones=filtros,
                tipo_norma='NR'
            )
            
            # Try to extract just the numeric part
            match = re.search(r'(\d+)', numero)
            if match:
                num_only = match.group(1)
                payload['filtros']['numeroDispositivoLegal'] = num_only
                print(f"  (Buscando por número simplificado: {num_only})")
            else:
                payload['filtros']['numeroDispositivoLegal'] = numero
            
            try:
                # Increase limit to find the correct one among many matches
                payload['desde'] = 0
                payload['hasta'] = 100
                
                data = scraper._search(payload)
                results = data.get('resultados', [])
                
                found_match = False
                if results:
                    print(f"  (Encontrados {len(results)} candidatos, filtrando...)")
                    for res in results:
                        # Check if the full number string is in the title or sumilla
                        # Normalize text for comparison
                        title = (res.get('sumilla') or res.get('titulo') or '').upper()
                        full_num = numero.upper()
                        
                        # Also check the 'codigoNorma' field if it exists?
                        # The results usually have 'dispositivoLegal' or 'palabra' which might be the type
                        # and 'codigoNorma' which is the number.
                        
                        # Let's try to match the number string in the title
                        if full_num in title:
                            res_str = f"✅ MATCH: ID: {res.get('id')} | Titulo: {title[:100]}..."
                            print(res_str)
                            f.write(res_str + "\n")
                            found_match = True
                            break # Stop after finding the best match? Or keep looking?
                    
                    if not found_match:
                        print("❌ No se encontró coincidencia exacta en los resultados.")
                        f.write("❌ No se encontró coincidencia exacta en los resultados.\n")
                else:
                    print("❌ No encontrado (0 resultados).")
                    f.write("❌ No encontrado (0 resultados).\n")
                    
            except Exception as e:
                print(f"Error buscando {numero}: {e}")
                f.write(f"Error buscando {numero}: {e}\n")

if __name__ == "__main__":
    find_ids()

from scraper.spij_scraper import SpijScraper
import logging

def debug_norm_content():
    scraper = SpijScraper(rate_limit=1.0)
    norm_id = 'H1412121'
    
    print(f"--- INSPECCIONANDO CONTENIDO DE {norm_id} ---")
    try:
        detail = scraper._fetch_detail(norm_id)
        raw_text = detail.get('textoCompleto', '')
        
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(raw_text, 'html.parser')
        clean_text = soup.get_text(separator=' ', strip=True)
        
        print(f"Longitud del texto limpio: {len(clean_text)}")
        print("--- PRIMEROS 1000 CARACTERES LIMPIOS ---")
        print(clean_text[:1000])
        print("--- FIN ---")
        
        if '(*)' in clean_text:
            print("\n✅ Se encontró '(*)' en el texto limpio.")
        else:
            print("\n❌ NO se encontró '(*)' en el texto limpio.")
            
        with open('debug_output.txt', 'w', encoding='utf-8') as f:
            f.write(clean_text)
            
        if 'DEROGADO' in clean_text:
            print("✅ Se encontró 'DEROGADO'. Ver debug_output.txt")
            idx = clean_text.find('DEROGADO')
            print(f"CONTEXTO (ver archivo para más): {clean_text[max(0, idx-50):idx+50]}")
        else:
            print("❌ NO se encontró 'DEROGADO' en el texto limpio.")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    debug_norm_content()

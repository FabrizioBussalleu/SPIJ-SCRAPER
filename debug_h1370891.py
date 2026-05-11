from scraper.spij_scraper import SpijScraper
import logging

def debug_norm():
    scraper = SpijScraper(rate_limit=1.0)
    norm_id = "H1370891"
    
    with open('debug_result.txt', 'w', encoding='utf-8') as f:
        f.write(f"--- DEBUGGING {norm_id} ---\n")
        try:
            detail = scraper._fetch_detail(norm_id)
            if not detail:
                f.write("❌ No detail returned\n")
                return

            f.write("✅ Detail fetched\n")
            is_valid, msg = scraper._validate_norm_status(norm_id, detail)
            f.write(f"Status: {is_valid}\n")
            f.write(f"Msg: {msg}\n")
            
            # Print snippet of text to verify content
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(detail.get('textoCompleto', ''), 'html.parser')
            clean = soup.get_text(separator=' ', strip=True)
            f.write(f"Clean text start: {clean[:500]}\n")
            
        except Exception as e:
            f.write(f"❌ Error: {e}\n")
            import traceback
            traceback.print_exc(file=f)

if __name__ == "__main__":
    debug_norm()

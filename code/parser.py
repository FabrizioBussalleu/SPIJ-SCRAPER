"""Módulo de parsing para extraer estructura de textos legales."""
import re
from typing import List, Dict, Tuple
from bs4 import BeautifulSoup

# Regex para detectar artículos: "Artículo 1.-", "Artículo 12°", "Art. 5", etc.
# Se busca ser flexible con el formato.
ARTICLE_PATTERN = re.compile(
    r'(?:^|\n)\s*(?:Artículo|Art\.|Art)\s+(\d+(?:[\-º°]?[A-Za-z0-9]*)?)\s*[\.\-\°)]',
    re.IGNORECASE | re.MULTILINE
)

# Regex para detectar citas: "Ley N° 12345", "D.L. 123", "Decreto Legislativo 555"
CITATION_PATTERNS = [
    (r'(?:Ley|Ley N[°º])\s+(\d+)', 'Ley'),
    (r'(?:Decreto Legislativo|D\.L\.|DL)\s*[Nn]?[°º]?\s*(\d+)', 'Decreto Legislativo'),
    (r'(?:Decreto de Urgencia|D\.U\.|DU)\s*[Nn]?[°º]?\s*(\d+[-\d]*)', 'Decreto de Urgencia'),
    (r'(?:Decreto Supremo|D\.S\.|DS)\s*[Nn]?[°º]?\s*(\d+[-\d]+-[A-Z]+)', 'Decreto Supremo'),
    (r'(?:Resolución Ministerial|R\.M\.|RM)\s*[Nn]?[°º]?\s*(\d+[-\d]+-[A-Z]+)', 'Resolución Ministerial'),
    (r'(?:Código Penal)', 'Código'),
    (r'(?:Código Civil)', 'Código'),
    (r'(?:Constitución Política)', 'Constitución'),
]

def _clean_text(text: str) -> str:
    """Limpia HTML y normaliza espacios."""
    if not text:
        return ""
    # Si parece HTML, limpiarlo
    if '<' in text and '>' in text:
        soup = BeautifulSoup(text, 'html.parser')
        text = soup.get_text(separator='\n', strip=True)
    return text

def extract_articles(text: str) -> List[Dict]:
    """Divide el texto completo en una lista de artículos.
    
    Retorna:
        Lista de dicts: [{'number': '1', 'content': '...', 'order': 1}, ...]
    """
    text = _clean_text(text)
    if not text:
        return []

    matches = list(ARTICLE_PATTERN.finditer(text))

    if not matches:
        # Si no detecta artículos, retorna todo como un solo bloque (o nada)
        return []

    articles = []
    for i, match in enumerate(matches):
        start = match.start()
        # El final de este artículo es el inicio del siguiente, o el final del texto
        end = matches[i+1].start() if i + 1 < len(matches) else len(text)
        
        article_num = match.group(1)
        # El contenido incluye el encabezado del artículo para contexto
        content = text[start:end].strip()
        
        articles.append({
            'number': article_num,
            'content': content,
            'order': i + 1
        })
    
    return articles

def extract_citations(text: str) -> List[Dict]:
    """Extrae referencias a otras normas mencionadas en el texto.
    
    Retorna:
        Lista de dicts: [{'target_text': 'Ley 12345', 'type': 'refers'}, ...]
    """
    text = _clean_text(text)
    if not text:
        return []

    citations = []
    seen = set()

    for pattern, label in CITATION_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            full_match = match.group(0).strip()
            # Normalizar espacios
            clean_match = re.sub(r'\s+', ' ', full_match)
            
            if clean_match in seen:
                continue
            
            seen.add(clean_match)
            
            # Análisis de contexto simple para determinar el tipo de cita
            start_idx = match.start()
            # Mirar 50 caracteres antes de la cita
            context_before = text[max(0, start_idx-50):start_idx].lower()
            
            cit_type = 'refers'
            if any(w in context_before for w in ['deroga', 'derógase', 'déjase sin efecto', 'sustituye']):
                cit_type = 'repeals'  # Esta norma deroga a la citada (o viceversa, depende del fraseo, pero marca relación fuerte)
            elif any(w in context_before for w in ['modifica', 'modifíquese', 'incorpóra']):
                cit_type = 'modifies'
            elif any(w in context_before for w in ['derogada por', 'sustituida por']):
                cit_type = 'repealed_by' # La norma actual es derogada por la citada
            
            citations.append({
                'target_text': clean_match,
                'type': cit_type,
                'source_type': label
            })
            
    return citations

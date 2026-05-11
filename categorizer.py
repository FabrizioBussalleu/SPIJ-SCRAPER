"""Lógica para clasificar normas en niveles 1-5 (Pirámide de Kelsen detallada)."""
from typing import Dict

# Definición de niveles jerárquicos
# 1: Constitución
# 2: Leyes y normas con rango de ley
# 3: Decretos Supremos (Reglamentarios)
# 4: Resoluciones (Ministeriales, Directorales, etc.)
# 5: Otros / Sin clasificar

LEVEL_KEYWORDS = {
    1: ['constitución política', 'reforma constitucional'],
    2: [
        'ley orgánica', 'ley ordinaria', 'ley n°', 'ley número', 
        'decreto legislativo', 'decreto de urgencia', 'código', 
        'reglamento del congreso', 'resolución legislativa'
    ],
    3: ['decreto supremo'],
    4: [
        'resolución ministerial', 'resolución viceministerial', 
        'resolución directoral', 'resolución jefatural', 
        'resolución de superintendencia', 'resolución administrativa',
        'ordenanza regional', 'ordenanza municipal', 'acuerdo de consejo'
    ],
}

def categorize_norm(item: Dict) -> int:
    """Asignar un nivel (1-5) a una norma basada en sus metadatos.

    Parámetros:
        item: dict que contiene al menos 'type' y 'title'

    Retorna:
        nivel entero entre 1 y 5. Por defecto 5 si no se detecta otra.
    """
    # Normalizar texto de búsqueda
    type_str = str(item.get('type') or '').lower()
    title_str = str(item.get('title') or '').lower()
    full_text = f"{type_str} {title_str}"

    # Prioridad: Buscar coincidencia exacta en 'type' primero (más preciso)
    for level, keywords in LEVEL_KEYWORDS.items():
        for kw in keywords:
            if kw in type_str:
                return level

    # Fallback: Buscar en el título si el tipo es ambiguo
    for level, keywords in LEVEL_KEYWORDS.items():
        for kw in keywords:
            if kw in full_text:
                return level

    return 5

"""Lógica para clasificar normas en niveles 1-5 (Pirámide de Kelsen detallada)."""
import unicodedata
from typing import Dict

# Definición de niveles jerárquicos
# 1: Constitución
# 2: Leyes y normas con rango de ley (Ley, Decreto Legislativo, Decreto de
#    Urgencia, Código, Resolución Legislativa, Decreto Ley)
# 3: Decretos Supremos (Reglamentarios)
# 4: Resoluciones y normas administrativas (Ministeriales, Directorales,
#    Jefaturales, Supremas, ordenanzas, acuerdos, directivas, etc.)
# 5: Otros / Sin clasificar

# Se mapea principalmente el campo `type` (dispositivoLegal de SPIJ), que llega
# limpio y en mayúsculas: "LEY", "DECRETO SUPREMO", "RESOLUCION MINISTERIAL"...
# El orden importa: los patrones más específicos se evalúan primero.
LEVEL_TYPE_KEYWORDS = {
    1: ['constitucion', 'reforma constitucional'],
    3: ['decreto supremo'],  # antes que 'decreto' genérico
    2: [
        'ley organica', 'ley ordinaria', 'decreto legislativo',
        'decreto de urgencia', 'decreto ley', 'codigo',
        'resolucion legislativa', 'reglamento del congreso', 'ley',
    ],
    4: [
        'resolucion', 'ordenanza', 'acuerdo de concejo', 'acuerdo de consejo',
        'acuerdo de directorio', 'directiva', 'decreto de alcaldia',
        'decreto regional', 'decreto de consejo', 'circular', 'oficio',
        'edicto', 'acuerdo',
    ],
}


def _norm(text: str) -> str:
    """Minúsculas sin acentos, para comparar de forma robusta."""
    text = str(text or '').lower()
    return ''.join(c for c in unicodedata.normalize('NFD', text)
                   if unicodedata.category(c) != 'Mn')


def categorize_norm(item: Dict) -> int:
    """Asignar un nivel (1-5) a una norma basada en sus metadatos.

    Parámetros:
        item: dict que contiene al menos 'type' y 'title'

    Retorna:
        nivel entero entre 1 y 5. Por defecto 5 si no se detecta otra.
    """
    type_str = _norm(item.get('type'))
    title_str = _norm(item.get('title'))

    # Prioridad 1: coincidencia en el campo 'type' (más preciso y limpio).
    for level in (1, 3, 2, 4):
        for kw in LEVEL_TYPE_KEYWORDS[level]:
            if kw in type_str:
                return level

    # Prioridad 2: fallback al título si el tipo es ambiguo o vacío.
    full_text = f"{type_str} {title_str}"
    for level in (1, 3, 2, 4):
        for kw in LEVEL_TYPE_KEYWORDS[level]:
            if kw in full_text:
                return level

    return 5

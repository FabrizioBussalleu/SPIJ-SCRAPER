# SPIJ – Modelo de Búsqueda Semántica de Leyes (Trabajo Final 1ACC0219)

Modelo de Búsqueda Sistemática de Leyes en Función a un Caso — SPIJ Scraper



## Objetivo
Construir un sistema de recuperación semántica basado en embeddings vectoriales que comprenda el significado de una consulta en lenguaje natural y devuelva las normas peruanas más relevantes del Sistema Peruano de Información Jurídica (SPIJ), filtradas por jerarquía y vigencia.

## Integrantes
* Aaron Alvaro Felices Vallejos - U202315164
* Fabrizio Bussalleu Salcedo - U202315655
* Diego Alexander Huaman Sirio - u20211f983

## Dataset
Los datos provienen de los endpoints internos del Sistema Peruano de Información Jurídica (SPIJ) del Ministerio de Justicia. El corpus consta de 1,035 normas vigentes y derogadas extraídas mediante un scraper propio. Los datos son semi-estructurados e incluyen campos como el texto completo en HTML, tipo de norma, estado de vigencia, nivel jerárquico y relaciones de citas inter-normativas.

**✨ Nueva funcionalidad:** Validación automática de vigencia de normas antes del scraping.

## Requisitos
- Python 3.11+

## Instalación

1. Crear y activar un entorno virtual.
2. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```
3. Copiar `.env.example` a `.env` y ajustar variables si es necesario.

## Configuración inicial

### Primera vez:
```bash
# Crear tablas de la base de datos
python main.py init-db

# Ejecutar migraciones (agregar campos de validación)
python main.py migrate-db
```

## Uso básico

### Scraping normal (sin validación)
```bash
python main.py scrape --limit 10
```

### Scraping con validación completa
```bash
# Validar todas las normas y marcar estado
python main.py scrape --validate-all --limit 50

# Solo procesar normas vigentes (descarta las inválidas)
python main.py scrape --only-valid --limit 100
```

### Validar normas existentes en la BD
```bash
# Validar normas ya guardadas que no han sido validadas
python main.py validate-existing --limit 50
```

### Exportar resultados
```bash
python main.py export --format csv
python main.py export --format json
```

## Funcionalidades de validación

El scraper ahora puede detectar automáticamente si una norma sigue vigente:

- ✅ **Validación automática**: Busca alertas en las páginas de detalle de SPIJ
- ⚠️ **Detección de derogaciones**: Identifica términos como "derogado", "sin efecto", etc.
- 📊 **Campos nuevos en BD**:
  - `is_valid`: 1=vigente, 0=no vigente, NULL=desconocido
  - `validation_message`: Mensaje descriptivo del estado
  - `validation_date`: Cuándo se validó por última vez

## Comandos disponibles

| Comando | Descripción |
|---------|-------------|
| `init-db` | Crear tablas de la base de datos |
| `migrate-db` | Ejecutar migraciones pendientes |
| `scrape` | Ejecutar scraping con opciones de validación |
| `validate-existing` | Validar normas ya guardadas |
| `export` | Exportar resultados a CSV/JSON |
| `list-versions` | Ver historial de versiones |

## Notas importante

- **Rate limiting**: Se recomienda usar `--rate-limit 2.0` o superior para validación
- **Robots.txt**: El scraper respeta las políticas del sitio
- **Performance**: La validación es más lenta pero más precisa

## Conclusiones y Trabajo Futuro

* **Infraestructura Implementada:** Se construyó y ejecutó exitosamente la infraestructura técnica de un motor de búsqueda semántica para leyes peruanas sobre el corpus real del SPIJ. El sistema consta de cinco módulos construidos en Python operando en producción.
* **Extracción de Datos:** El scraper modular logró autenticarse en las APIs del SPIJ y recolectar un total de 1,035 normas junto con 1,839 citas inter-normativas en aproximadamente 22 minutos de ejecución. Adicionalmente, el parser implementado permite segmentar textos en artículos y clasificar cuatro tipos de relación en las citas.
* **Rendimiento de los Modelos de Búsqueda:** Se evaluó un motor semántico (paraphrase-multilingual-mpnet-base-v2 con FAISS) contra un baseline léxico (BM25) utilizando 12 consultas curadas. El modelo BM25 alcanzó un Precision@5 de 0.550 y un MRR@5 de 0.778. Por su parte, el modelo semántico superó al léxico en Recall@5 obteniendo 0.325 (una mejora de 0.032), demostrando su capacidad de capturar la sinonimia terminológica que el modelo léxico pierde.
* **Clasificación y Explicabilidad:** El proyecto deja disponibles clasificadores supervisados (Regresión Logística y Random Forest sobre TF-IDF) con sus respectivos análisis de explicabilidad basados en SHAP.
* **Limitaciones:** El conjunto de datos extraído presenta un desbalance pronunciado con 1,028 normas vigentes frente a solo 7 derogadas. Asimismo, el corpus actual de 1,035 normas no representa el universo completo del SPIJ, el cual asciende a más de 25,000 normas.
* **Trabajo Futuro:** Los siguientes pasos del proyecto incluyen ampliar el scraping para incorporar normas derogadas explícitamente. También se proyecta desarrollar una interfaz web para el buscador e integrar un sistema de re-ranking por nivel jerárquico para priorizar leyes frente a resoluciones.

## Licencia

Este proyecto está bajo la Licencia MIT - mira el archivo [LICENSE.md](LICENSE.md) para más detalles.

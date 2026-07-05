# SPIJ Scraper

Scraper modular para extraer metadatos de normas del portal SPIJ (Ministerio de Justicia del Perú).

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

## Notas importantes

- **Rate limiting**: Se recomienda usar `--rate-limit 2.0` o superior para validación
- **Robots.txt**: El scraper respeta las políticas del sitio
- **Performance**: La validación es más lento pero más precisa

## Presentación del TP (retroalimentación del profesor)

### Setup local (primera vez)

```powershell
cd D:\2026-01\APLIDATASCIENCE\SPIJ-SCRAPER

# Usar Python 3 (no el python 2.7 del PATH)
py -3 -m pip install -r requirements.txt
py -3 setup_local.py              # import JSON + indice + graficos
py -3 setup_local.py --scrape 200 # o scrapear del SPIJ en vivo
```

Para el corpus completo (~1035 normas, ~22 min):
```powershell
py -3 main.py scrape --validate-all --rate-limit 1.5
py -3 embeddings_indexer.py --rebuild
py -3 presentation_report.py --k 5 --shap
```

Para generar **todos los gráficos y la guía de diapositivas numeradas**:

```bash
# Generación completa (EDA + búsqueda + clasificadores + XAI)
python presentation_report.py --k 5 --shap

# Solo búsqueda y EDA (sin re-entrenar modelos)
python presentation_report.py --skip-train --k 5
```

**Salidas:**
| Ruta | Contenido |
|------|-----------|
| `exports/presentacion/` | Gráficos numerados 01–13 para slides |
| `exports/presentacion/evaluation_report.txt` | Métricas + interpretación legal + errores FP/FN |
| `docs/GUIA_PRESENTACION.md` | Guía de 14 diapositivas con qué decir en cada una |
| `data/models/vigencia_methodology.txt` | Parámetros CV, TF-IDF, LogReg documentados |

**Scripts individuales:**
```bash
python eda_analysis.py
python evaluate.py --k 5 --charts --output exports/presentacion/evaluation_report.txt
python classifier.py --task vigencia --presentation --shap
```

### Mejoras incluidas (rama `feature/mejoras-tp`)
- Interpretación contextual de Precision, Recall, MRR, NDCG para abogados
- Análisis de falsos positivos y falsos negativos en búsqueda
- F2 (F-beta) además de F1 — prioriza recall en vigencia (falso negativo crítico)
- Metodología CV documentada (StratifiedKFold 5-fold, parámetros completos)
- Gráficos XAI: coeficientes LogReg + SHAP con anotaciones jurídicas
- Guía de presentación con diapositivas numeradas

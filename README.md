# SPIJ – Modelo de Búsqueda Semántica de Leyes (Trabajo Final 1ACC0219)

## Objetivo
Construir un sistema de Ciencia de Datos sobre el corpus legal del **SPIJ** (Sistema
Peruano de Información Jurídica) que: (1) recupere las normas más relevantes para un
caso descrito en lenguaje natural mediante **búsqueda semántica** (embeddings + FAISS)
comparada con un baseline **BM25**; (2) clasifique el **nivel jerárquico** de una norma
(Pirámide de Kelsen); y (3) prediga la **vigencia** de una disposición (vigente/derogada)
usando las notas oficiales de derogación del SPIJ como verdad de referencia.

## Integrantes
- Aaron Alvaro Felices Vallejos – U202315164
- Fabrizio Bussalleu Salcedo – U202315655
- Diego Alexander Huaman Sirio – U20211F983

Profesor: Carlos Fernando Montoya Cubas · Ciclo 2026-01

## Dataset
- **Corpus de normas** (`ai-training-dataset/corpus_normas.json`, BD `data/norms_corpus.db`):
  1 091 normas del SPIJ (1861–2026), 40 tipos, 5 817 citas inter-normativas. Generado con
  `collect_corpus.py` desde las APIs internas del SPIJ.
- **Dataset de vigencia por artículo** (`ai-training-dataset/vigencia_articulos.csv`):
  1 075 artículos de los códigos del SPIJ etiquetados con las notas oficiales
  `(*) Artículo derogado por…` → 430 derogados / 645 vigentes. Generado con
  `build_vigencia_dataset.py`.

## Estructura del repositorio
```
code/    -> código fuente (Python): scraper, EDA, clasificadores, búsqueda
data/    -> base de datos (norms_corpus.db), índice FAISS, modelos y figuras
ai-training-dataset/ -> exports del corpus y del dataset de vigencia (JSON/CSV)
exports/ -> visualizaciones EDA y reporte de evaluación
```

## Pipeline reproducible
```bash
pip install -r requirements.txt
python code/collect_corpus.py --target 900        # 1) corpus de normas -> SQLite
python code/build_vigencia_dataset.py             # 2) dataset de vigencia por artículo
python code/eda_analysis.py                        # 3) EDA -> exports/eda/
python code/classifier.py --task ambos --shap      # 4) clasificadores + SHAP -> data/models/
python code/embeddings_indexer.py --rebuild        # 5) índice FAISS (MiniLM multilingüe)
python code/evaluate.py --k 5                       # 6) métricas Semántico vs BM25
```

## Resultados (ejecución real, ver informe)
- **Recuperación (k=5, 12 consultas):** Semántico vs BM25 — Precision@5 0.550/0.583,
  Recall@5 0.394/0.365, MRR@5 0.767/0.725, NDCG@5 0.780/0.748. El semántico gana en
  recall, MRR y NDCG.
- **Vigencia (artículo):** CV F1 = 0.973; test F1 Derogado = 0.911, Vigente = 0.945.
  Ablación sin la nota oficial: F1(Derogado) = 0.694.
- **Jerarquía (norma):** CV F1-macro = 0.824 ± 0.071.

## Conclusiones
El motor semántico supera al baseline BM25 en la calidad del ranking (Recall@5, MRR@5,
NDCG@5), capturando sinonimia terminológica. La reformulación de la vigencia a nivel de
artículo con las notas oficiales del SPIJ corrigió el problema del corpus 99.3 % vigente
de la entrega parcial (donde F1 de la clase derogada era 0), logrando un clasificador
balanceado con F1 alto en ambas clases. SHAP confirma que ambos modelos se apoyan en
señales textuales correctas e interpretables.

## Licencia
MIT License. Uso académico. Los datos pertenecen al Ministerio de Justicia del Perú (SPIJ).

---

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
- **Performance**: La validación es más lenta pero más precisa

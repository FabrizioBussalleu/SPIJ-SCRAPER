# Guía de Presentación — Búsqueda de Leyes SPIJ

> Generado: 2026-07-05 15:00
> Dataset de evaluación: 12 consultas jurídicas | k=5

---

## Estructura de diapositivas (numeradas)

### Diapositiva 1 — Portada
**Título:** Sistema de Búsqueda Semántica de Normas Legales (SPIJ Perú)
**Subtítulo:** Comparación BM25 vs embeddings + clasificación de vigencia con XAI

---

### Diapositiva 2 — Problema y contexto
**Qué decir:** Un abogado busca leyes por concepto, pero los títulos son formales.
SPIJ tiene miles de normas; la búsqueda exacta falla con vocabulario coloquial.

**No mostrar solo:** "Usamos NLP y ML."
**Mostrar:** Ejemplo real — consulta *"despido trabajadora embarazada"* vs título formal de la ley.

---

### Diapositiva 3 — Corpus y metodología de datos
**Gráfico:** `exports/presentacion/01_eda_norms_by_type.png`
**Gráfico:** `exports/presentacion/02_eda_temporal.png`

**Qué decir:** Describir scraping, validación de vigencia en portal, campos en BD.
Mencionar rate limiting y etiquetado automático (`is_valid`).

---

### Diapositiva 4 — EDA: vigencia y jerarquía normativa
**Gráfico:** `exports/presentacion/04_eda_level_validity.png`
**Gráfico:** `exports/presentacion/05_eda_citation_types.png`

**Qué decir:** La pirámide de Kelsen importa — una resolución no puede contradecir una ley.
El heatmap muestra cuántas normas vigentes/derogadas hay por nivel.

---

### Diapositiva 5 — Arquitectura del sistema
**Diagrama sugerido (draw.io / PowerPoint):**
```
Scraper SPIJ → BD PostgreSQL → Embeddings (mpnet) → FAISS
                              → BM25 baseline
                              → Clasificador vigencia (TF-IDF + LogReg)
```

---

### Diapositiva 6 — Metodología de modelado (DETALLADA)
**Archivo de referencia:** `data/models/vigencia_methodology.txt`

**Qué decir (obligatorio según retroalimentación del profesor):**
| Parámetro | Valor | Justificación |
|-----------|-------|---------------|
| Partición | 85% train / 15% test, estratificada | Preserva proporción vigente/derogada |
| CV | StratifiedKFold, 5 folds, shuffle, seed=42 | Estimación robusta sin fugas |
| TF-IDF | max_features=20000, ngrams (1,2), min_df=2 | Captura bigramas jurídicos ("sin efecto") |
| LogReg | C=1.0, class_weight=balanced | Compensa desbalance (pocas derogadas) |
| Métricas | F1, **F2** (β=2) | F2 penaliza más falsos negativos: citar ley derogada es crítico |

**Gráfico:** `exports/presentacion/11_vigencia_cv_scores.png`

---

### Diapositiva 7 — Resultados de búsqueda: desvelar los números
**Gráfico:** `exports/presentacion/06_metrics_comparison_k5.png`

**Interpretación (NO leer solo la tabla):**

| Métrica | BM25 | Semántico | Para el abogado significa… |
|---------|------|-----------|----------------------------|
| Precision@5 | ~0.55 | ~0.48 | BM25: más resultados pertinentes en pantalla |
| Recall@5 | ~0.29 | ~0.33 | Semántico: encuentra más conceptos con vocabulario distinto |
| MRR@5 | ~0.78 | ~0.63 | BM25: la ley correcta aparece antes |
| NDCG@5 | ~0.76 | ~0.68 | BM25 ordena mejor el ranking |

**Frase clave:** *"Un NDCG de 0.76 (BM25) vs 0.68 (semántico) implica que BM25 coloca
la norma pertinente más arriba — el abogado la ve en 1–2 clics vs 3–4."*

---

### Diapositiva 8 — Análisis por consulta
**Gráfico:** `exports/presentacion/07_per_query_heatmap_k5.png`

**Qué decir:** Mostrar 1–2 consultas donde semántico gana (recall alto) y 1–2 donde BM25 gana.
Demostrar que conocen los datos, no repiten texto genérico.

---

### Diapositiva 9 — Análisis de errores (FP / FN)
**Gráfico:** `exports/presentacion/08_error_analysis_k5.png`
**Reporte:** `exports/presentacion/evaluation_report.txt`

**Qué decir:**
- **Falsos positivos:** títulos con palabras genéricas ("decreto", "aprueba") que no responden a la consulta.
- **Falsos negativos:** keywords del dataset no aparecen en top-5 — posible norma relevante oculta.
- Pregunta del profesor: *"¿Hay palabras que se repiten en FP y TP?"* → Sí, términos genéricos del lenguaje legal formal.

---

### Diapositiva 10 — Modelo de vigencia: resultados
**Gráfico:** `exports/presentacion/09_vigencia_confusion_matrix.png`

**Qué decir:** Interpretar cada celda:
- FN (derogada→vigente): **error crítico** — abogado cita norma sin efecto.
- FP (vigente→derogada): molesto pero menos grave.

Usar F2 además de F1: prioriza no perder derogadas.

---

### Diapositiva 11 — Explicabilidad (XAI) — DIAPOSITIVA CLAVE ⭐
**Gráfico:** `exports/presentacion/10_xai_vigencia_coefficients.png`
**Gráfico:** `exports/presentacion/13_shap_vigencia.png` *(si se generó con --shap)*

**Qué decir (gastar más tiempo aquí):**
1. **Qué** palabras pesan: "derogado", "sustitúyese", "sin efecto"
2. **Por qué:** son las mismas alertas que detecta el scraper en SPIJ
3. **Cómo:** el modelo aprendió lenguaje jurídico **sin reglas hardcodeadas**
4. **Hallazgo fascinante (profesor):** convergencia ML ↔ lógica del scraper

**No decir:** "SHAP muestra importancia de features."
**Decir:** "Cuando el texto contiene 'derogado', el modelo resta 2.3 puntos de probabilidad
de vigencia — igual que la alerta roja del portal."

---

### Diapositiva 12 — Jerarquía normativa (Random Forest)
**Gráfico:** `exports/presentacion/12_jerarquia_feature_importance.png`

**Qué decir:** Palabras como "decreto supremo", "resolución ministerial" predicen el nivel Kelsen.

---

### Diapositiva 13 — Limitaciones y trabajo futuro
- Dataset de evaluación pequeño (12 consultas) — ampliar con anotación manual por abogados
- Recall aproximado por keywords — ideal: IDs de norma anotados
- Híbrido BM25 + semántico (re-ranking) como siguiente paso

---

### Diapositiva 14 — Conclusiones
1. BM25 superior en precisión/MRR por títulos formales del derecho peruano
2. Semántico superior en recall para consultas en lenguaje natural
3. Modelo de vigencia aprende lenguaje jurídico (XAI valida el scraper)
4. Métricas interpretadas en contexto legal, no solo números

---

## Archivos generados en esta ejecución

- `01_eda_norms_by_type.png`
- `02_eda_temporal.png`
- `03_eda_wordcloud.png`
- `04_eda_level_validity.png`
- `05_eda_citation_types.png`
- `06_metrics_comparison_k5.png`
- `07_per_query_heatmap_k5.png`
- `08_error_analysis_k5.png`
- `09_vigencia_confusion_matrix.png`
- `10_xai_vigencia_coefficients.png`
- `11_vigencia_cv_scores.png`
- `12_jerarquia_feature_importance.png`
- `13_shap_vigencia.png`

## Comandos para regenerar

```bash
python presentation_report.py --k 5 --shap
python presentation_report.py --skip-train --k 5
python evaluate.py --k 5 --charts --output exports/presentacion/evaluation_report.txt
python classifier.py --task vigencia --presentation --shap
python eda_analysis.py
```

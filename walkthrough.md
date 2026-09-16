# Walkthrough — Pipeline de Churn Día +1 (D1) para Etermax

## 1. Contexto del Proyecto

**Objetivo:** construir un pipeline de preparación de datos para un modelo predictivo de churn Día +1 (D1) en videojuegos móviles, utilizando exclusivamente datos internos del dataset.

**Stack:** Polars (lazy evaluation) → LightGBM

| Dato | Valor |
|---|---|
| Dataset original | `Dataset.DS - 2026 (1) (2).csv` |
| Filas originales | 20,000 |
| Filas finales (post-limpieza) | 17,692 |
| Columnas originales | 14 |
| Columnas finales | 30 |
| Target | `target_churn_indicator` (binario 0/1) |

### Columnas originales del CSV

| Columna | Tipo original | Descripción |
|---|---|---|
| `user_id` | string | Identificador único del usuario |
| `install_time` | string | Timestamp de instalación (UTC) |
| `platform` | string | Android / iOS |
| `country_region` | string | Provincia / región |
| `city` | string | Ciudad (con nulos) |
| `gender` | string | male / female |
| `min_age_range` | int | Límite inferior del rango etario |
| `max_age_range` | int | Límite superior del rango etario |
| `event_1` … `event_5` | int | Conteo de 5 tipos de eventos del usuario |
| `target_churn_indicator` | int | 1 = churneó al día siguiente, 0 = retuvo |

---

## 2. Scripts Creados

### [`churn_feature_pipeline.py`](./src/churn_feature_pipeline.py)

Pipeline modular de feature engineering con 4 funciones encadenadas via `.pipe()` en lazy evaluation.

### [`eda_churn_bivariado.py`](./src/eda_churn_bivariado.py)

Script de EDA bivariado que importa el pipeline, genera tablas agregadas y 7 gráficos ejecutivos en PNG.

### [`eda_churn_utc_raw.py`](./src/eda_churn_utc_raw.py)

Script auxiliar exploratorio generado para evidenciar el borde temporal truncado en los datos crudos (sin filtros ni conversión UTC-3). Exporta un gráfico de diagnóstico: `churn_by_install_date_UTC.png`.

---

## 3. Fase 1 — Limpieza y Normalización Geográfica

> Función: [`fase_1_geo()`](./src/churn_feature_pipeline.py#L26-L46)

### Acciones realizadas

1. **Normalización CABA:** si `country_region == "Buenos Aires F.D."`, se fuerza `city = "Buenos Aires"`. Esto unifica la representación de la Ciudad Autónoma de Buenos Aires, evitando variantes inconsistentes.

2. **Imputación de nulos en `city`:** los registros donde `city` seguía siendo `null` se rellenaron con el valor de `country_region` del mismo registro. Resultado: **0 nulos restantes**.

3. **Cast a Categorical:** ambas columnas (`city`, `country_region`) se convirtieron a tipo `Categorical` para que LightGBM las procese nativamente sin necesidad de one-hot encoding.

> [!IMPORTANT]
> Se conservaron ambas columnas geográficas deliberadamente — `country_region` captura la región macro y `city` el detalle local, pudiendo aportar señal complementaria al modelo.

---

## 4. Fase 2 — Manejo de Fechas y Feature Engineering Temporal

> Función: [`fase_2_temporal()`](./src/churn_feature_pipeline.py#L52-L127)

### Registros eliminados

> [!WARNING]
> Se identificó un problema de borde temporal: los registros del **30 de junio de 2018 (UTC)** tenían una ventana de retención Día +1 (D1) truncada, lo que inflaba artificialmente su tasa de churn.

| Fecha (UTC) | Filas | Churn rate | Problema |
|---|---|---|---|
| 2018-06-30 | 2,308 | 100.0% | Ventana temporal truncada al final del dataset |
| **Total eliminado** | **2,308** | — | Filtrados en el pipeline |

**Corrección aplicada (v2):** el flujo se reordenó para:
1. Parsear `install_time` a datetime **manteniéndolo en UTC**
2. Filtrar registros donde `install_time.dt.date() == 2018-06-30` (en UTC, antes del offset)
3. Recién después aplicar el offset de `-3h` para convertir a hora local ART

Esto evita que registros del 1° de julio UTC (que al restar 3h caen en la noche del 30 de junio local) sean eliminados incorrectamente. Esos registros **sí** tienen ventana Día +1 (D1) completa.

### Variables nuevas creadas

| Variable | Tipo | Descripción |
|---|---|---|
| `install_hour` | int | Hora de instalación (0-23, hora local ART) |
| `install_date` | date | Fecha de instalación (hora local ART) |
| `install_dow` | int | Día de la semana ISO (1=Lunes … 7=Domingo) |
| `time_of_day` | Categorical | Franja horaria: Madrugada (0-5), Mañana (6-11), Tarde (12-18), Noche (19-23) |
| `is_weekend_day0` | Int8 (0/1) | 1 si la instalación fue en Sábado o Domingo |
| `is_weekend_day1` | Int8 (0/1) | 1 si el día siguiente a la instalación cae en Sábado o Domingo |

---

## 5. Fase 3 — Demografía y Comportamiento (Eventos)

> Función: [`fase_3_demo_eventos()`](./src/churn_feature_pipeline.py#L133-L173)

### Variables creadas

| Variable | Tipo | Descripción |
|---|---|---|
| `age_segment` | Categorical | Concatenación `min_age_range` + `_` + `max_age_range` (ej. `"18_20"`) |
| `total_events` | int | Suma horizontal de `event_1` a `event_5` |
| `ratio_event_1` … `ratio_event_5` | float | Proporción de cada evento sobre el total (denominador + 1e-5 para evitar div/0) |
| `has_done_event_1` … `has_done_event_5` | Int8 (0/1) | Flag binario: 1 si el usuario realizó al menos una vez ese evento |

### Columnas eliminadas

| Columna | Razón |
|---|---|
| `min_age_range` | Reemplazada por `age_segment` |
| `max_age_range` | Reemplazada por `age_segment` |

---

## 6. Casteo final de Categoricals

> Función: [`castear_categoricals()`](./src/churn_feature_pipeline.py#L179-L188)

Columnas casteadas a `Categorical` para procesamiento nativo por LightGBM:

- `platform` (2 valores: Android, iOS)
- `gender` (2 valores: male, female)
- `age_segment` (ya era categorical desde Fase 3)
- `city` (ya era categorical desde Fase 1)
- `country_region` (ya era categorical desde Fase 1)
- `time_of_day` (ya era categorical desde Fase 2)

---

## 7. Schema final del DataFrame (30 columnas)

```
user_id                      String
install_time                 Datetime[μs]
platform                     Categorical
country_region               Categorical
city                         Categorical
gender                       Categorical
event_1 … event_5            Int64
target_churn_indicator       Int64
install_hour                 Int8
install_date                 Date
install_dow                  Int8
time_of_day                  Categorical
is_weekend_day0              Int8
is_weekend_day1              Int8
age_segment                  Categorical
total_events                 Int64
ratio_event_1 … ratio_5      Float64
has_done_event_1 … has_5     Int8
```

---

## 8. EDA Bivariado — Análisis y Gráficos

El script [`eda_churn_bivariado.py`](./src/eda_churn_bivariado.py) analiza la tasa de churn segmentada por 7 variables clave. Cada gráfico muestra el porcentaje de churn y el volumen `(n=X)` sobre cada barra para detectar problemas de volumetría.

### 8.1 Churn por Fecha de Instalación (hora local ART)

![Tasa de Churn Día +1 (D1) por Fecha de Instalación](./plots/churn_by_install_date.png)

- **Rango de churn estable:** 46.3% – 51.5% a lo largo de los 8 días.
- El 30-Jun local (n=541) son registros del 1° Jul UTC madrugada que tienen ventana completa.
- El 7-Jul marca el pico (51.5%) — posible efecto de último día del dataset.
- Las fechas están expresadas en **hora local ART (UTC-3)**, no en UTC.

### 8.2 Churn por Weekend Day+1

![Tasa de Churn Día +1 (D1) — ¿El día +1 es fin de semana?](./plots/churn_by_weekend_day1.png)

- **Después de la limpieza del borde:** la diferencia se redujo de 17.8pp a solo **2.1pp** (49.6% vs 47.5%).
- El efecto "weekend" que parecía fuerte era en gran parte un artefacto de los registros truncados del 30-Jun.

> [!NOTE]
> Esto demuestra la importancia de haber corregido el orden del filtro temporal. Sin la corrección, esta variable habría sobreajustado el modelo.

### 8.3 Churn por Franja Horaria

![Tasa de Churn Día +1 (D1) por Franja Horaria](./plots/churn_by_time_of_day.png)

- Diferencias mínimas entre franjas: 47.1% (Noche) a 48.9% (Mañana).
- **Baja capacidad discriminativa** para churn Día +1 (D1) por sí sola.
- La volumetría es razonable en las 4 franjas (1,758 a 6,724 usuarios).

### 8.4 Churn por Segmento de Edad

![Tasa de Churn Día +1 (D1) por Segmento de Edad](./plots/churn_by_age_segment.png)

- Solo **2 segmentos son estadísticamente confiables:** `13_17` (n=6,534, churn 50.7%) y `18_20` (n=11,133, churn 46.5%).
- Los segmentos con tasas extremas (100%, 83.3%) tienen n ≤ 12 — **ruido estadístico, no actionable**.

> [!TIP]
> Para el modelo, considerar agrupar los segmentos minoritarios (`18_18`, `21_20`, `17_17`, `20_20`, `21_17`, `13_13`) en una categoría "Otro" para evitar overfitting en grupos con n < 30.

### 8.5 Churn por Plataforma

![Tasa de Churn Día +1 (D1) por Plataforma](./plots/churn_by_platform.png)

- **Android** (n=17,205): 48.3% churn vs **iOS** (n=487): 40.0%. Diferencia de 8.3pp.
- iOS representa solo el 2.8% del volumen total — la señal es interesante pero la muestra es reducida.

### 8.6 Churn por Volumen de Interacción ⭐

![Tasa de Churn Día +1 (D1) por Volumen de Interacción](./plots/churn_by_total_events.png)

- **El predictor más fuerte encontrado en el EDA.** Relación monotónica inversa perfecta:
  - `0-10 eventos` → 72.8% churn (n=4,641)
  - `11-50` → 50.8% (n=8,249)
  - `51-150` → 21.4% (n=4,002)
  - `+150` → 10.1% (n=800)
- Todos los bins tienen volumetría sólida. La variable `total_events` y sus derivados (`ratio_event_X`) serán features clave para LightGBM.

### 8.7 Churn por Descubrimiento de Eventos

![Tasa de Churn Día +1 (D1) — Descubrimiento de Eventos](./plots/churn_by_events_discovery.png)

- **Event 3 es el diferenciador estrella:** quien lo realizó tiene solo 20.7% de churn vs 57.6% si no lo hizo (Δ=36.9pp). Ambos grupos con volumetría sólida (n=4,554 y n=13,138).
- **Events 1 y 2** también discriminan bien: ~71% churn sin interacción vs ~46% con interacción.
- **Event 4:** n=31 en "No interactuó" → estadísticamente no confiable.
- **Event 5:** casi no discrimina (Δ=2pp).

> [!IMPORTANT]
> El EDA señala a `has_done_event_3` y `total_events` como métricas fuertes en el análisis univariado. No obstante, el modelo multivariado posterior (ver gráfico SHAP) revelará que el **Evento 4** es el verdadero predictor dominante, demostrando que el Evento 3 es solo una puerta de entrada al hábito.

---

## 9. Entrenamiento del Modelo de Churn Día +1 (D1) (LightGBM)

El script [`src/train_churn_model.py`](./src/train_churn_model.py) orquesta el entrenamiento de un clasificador binario.

### 9.1 Preparación de Datos y Split
- **Dataset:** 17,692 filas generadas por el pipeline.
- **Variables excluidas:** `user_id`, `install_date`, `install_time`.
- **Split Estratificado:** 80% Train, 20% Test (fijando `random_state=42` para reproducibilidad).

### 9.2 Optimización de Hiperparámetros (Optuna)
Se ejecutaron **50 trials** utilizando un `StratifiedKFold` (n=5) sobre el set de Entrenamiento, buscando maximizar el **PR-AUC (Average Precision)**.
Los mejores hiperparámetros hallados fueron:
- `learning_rate`: 0.036
- `num_leaves`: 89
- `max_depth`: 7
- `min_child_samples`: 86
- `colsample_bytree`: 0.656

### 9.3 Evaluación en Test
El modelo final (entrenado sobre el 100% del set de Train) logró los siguientes resultados sobre el set de Test puro:
- **PR-AUC:** 0.7503
- **ROC-AUC:** 0.7838

### 9.4 Explicabilidad (SHAP)

![SHAP Summary Plot](./plots/shap_summary.png)

El gráfico SHAP confirma los hallazgos del EDA multivariado:
- Las variables conductuales (`event_4`, `event_1`, `ratio_event_5`, `total_events`) dominan por completo el top de features predictivos.
- Las variables categóricas de geografía (`city`, `country_region`) también muestran peso, confirmando la utilidad de la Fase 1 del pipeline.
- Las variables temporales puros (`install_hour`, `install_dow`) tienen impacto, pero secundario frente al comportamiento in-app.

---

## 10. Archivos generados

| Archivo | Descripción |
|---|---|
| [`churn_feature_pipeline.py`](./src/churn_feature_pipeline.py) | Pipeline modular de feature engineering (Polars lazy) |
| [`eda_churn_bivariado.py`](./src/eda_churn_bivariado.py) | Script de EDA bivariado con tablas + 7 visualizaciones |
| [`churn_by_install_date.png`](./plots/churn_by_install_date.png) | Gráfico: churn por fecha de instalación (hora local ART) |
| [`churn_by_weekend_day1.png`](./plots/churn_by_weekend_day1.png) | Gráfico: churn por weekend day+1 |
| [`churn_by_time_of_day.png`](./plots/churn_by_time_of_day.png) | Gráfico: churn por franja horaria |
| [`churn_by_age_segment.png`](./plots/churn_by_age_segment.png) | Gráfico: churn por segmento etario |
| [`churn_by_platform.png`](./plots/churn_by_platform.png) | Gráfico: churn por plataforma |
| [`churn_by_total_events.png`](./plots/churn_by_total_events.png) | Gráfico: churn por volumen de interacción |
| [`churn_by_events_discovery.png`](./plots/churn_by_events_discovery.png) | Gráfico: churn por descubrimiento de eventos (barras agrupadas) |
| [`train_churn_model.py`](./src/train_churn_model.py) | Script de entrenamiento LightGBM + Optuna + SHAP |
| [`shap_summary.png`](./plots/shap_summary.png) | Gráfico SHAP de explicabilidad del modelo |

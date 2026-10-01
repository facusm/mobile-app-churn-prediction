"""
Pipeline de preparación de datos para modelo predictivo de Churn (Día 1)
=========================================================================
Industria: Videojuegos móviles  |  Motor: Polars (lazy evaluation)
Target: LightGBM-ready DataFrame con features derivados exclusivamente del dataset interno.

Autor : Pipeline de Procesamiento de Datos
Fecha : 2026-09-10
"""

from pathlib import Path
import polars as pl

# ─────────────────────────────────────────────────────────────────────────────
# Configuración
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "dataset_raw.csv"
UTC_OFFSET_HOURS = -3          # Ajuste UTC → hora local (ART)
EPSILON = 1e-5                 # Suavizado para evitar div/0 en ratios
EVENT_COLS = [f"event_{i}" for i in range(1, 6)]


# ═════════════════════════════════════════════════════════════════════════════
# Fase 1 — Limpieza y Normalización Geográfica
# ═════════════════════════════════════════════════════════════════════════════
def fase_1_geo(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    1. Normalización CABA: si country_region == 'Buenos Aires F.D.'
       → city = 'Buenos Aires'.
    2. Imputación de nulos: si city sigue siendo null → usar country_region.
    3. Castear city y country_region a categorical.
    """
    return lf.with_columns(
        # Paso 1: forzar city = "Buenos Aires" cuando country_region es "Buenos Aires F.D."
        pl.when(pl.col("country_region") == "Buenos Aires F.D.")
        .then(pl.lit("Buenos Aires"))
        .otherwise(pl.col("city"))
        .alias("city"),
    ).with_columns(
        # Paso 2: si city sigue null, rellenar con country_region
        pl.col("city").fill_null(pl.col("country_region")),
    ).with_columns(
        # Paso 3: castear a categorical
        pl.col("city").cast(pl.Categorical),
        pl.col("country_region").cast(pl.Categorical),
    )


# ═════════════════════════════════════════════════════════════════════════════
# Fase 2 — Manejo de Fechas y Feature Engineering Temporal
# ═════════════════════════════════════════════════════════════════════════════
def fase_2_temporal(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    1. Parsear install_time a datetime UTC.
    2. Filtrar registros de borde (2018-06-30 UTC) con ventana de retención
       truncada para evitar sesgo en el target de churn D1.
    3. Aplicar offset -3h → hora local (ART).
    4. Extraer install_hour, install_date, install_dow (1=Lun, 7=Dom).
    5. Crear time_of_day categórico (Madrugada / Mañana / Tarde / Noche).
    6. is_weekend_day0  → 1 si Sáb/Dom, 0 si no.
    7. is_weekend_day1  → 1 si el día siguiente cae en Sáb/Dom.
    """
    # Paso 1a: parseo a datetime (mantener en UTC)
    lf = lf.with_columns(
        pl.col("install_time")
        .str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S")
        .alias("install_time"),
    )

    # Paso 1b: eliminar artefacto de borde — registros del 30-Jun UTC
    #           cuya ventana de retención D1 está truncada
    lf = lf.filter(
        pl.col("install_time").dt.date() != pl.lit("2018-06-30").str.strptime(pl.Date, "%Y-%m-%d")
    )

    # Paso 1c: aplicar ajuste horario UTC → ART (UTC-3)
    lf = lf.with_columns(
        pl.col("install_time")
        .dt.offset_by(f"{UTC_OFFSET_HOURS}h")
        .alias("install_time"),
    )

    # Paso 2: extracción de componentes temporales
    lf = lf.with_columns(
        pl.col("install_time").dt.hour().alias("install_hour"),
        pl.col("install_time").dt.date().alias("install_date"),
        # Polars weekday(): 1=Lun … 7=Dom  (ISO)
        pl.col("install_time").dt.weekday().alias("install_dow"),
    )

    # Paso 3: franja horaria categórica
    lf = lf.with_columns(
        pl.when(pl.col("install_hour").is_between(0, 5, closed="both"))
        .then(pl.lit("Madrugada"))
        .when(pl.col("install_hour").is_between(6, 11, closed="both"))
        .then(pl.lit("Mañana"))
        .when(pl.col("install_hour").is_between(12, 18, closed="both"))
        .then(pl.lit("Tarde"))
        .otherwise(pl.lit("Noche"))
        .cast(pl.Categorical)
        .alias("time_of_day"),
    )

    # Paso 4: fin de semana día 0
    lf = lf.with_columns(
        pl.when(pl.col("install_dow").is_in([6, 7]))
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
        .cast(pl.Int8)
        .alias("is_weekend_day0"),
    )

    # Paso 5: fin de semana día +1
    lf = lf.with_columns(
        (
            (pl.col("install_date").cast(pl.Datetime) + pl.duration(days=1))
            .dt.weekday()
        ).alias("_next_dow"),
    ).with_columns(
        pl.when(pl.col("_next_dow").is_in([6, 7]))
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
        .cast(pl.Int8)
        .alias("is_weekend_day1"),
    ).drop("_next_dow")

    return lf


# ═════════════════════════════════════════════════════════════════════════════
# Fase 3 — Demografía y Comportamiento (Eventos)
# ═════════════════════════════════════════════════════════════════════════════
def fase_3_demo_eventos(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    1. age_segment: concatenar min_age_range + '_' + max_age_range → categorical.
       Eliminar columnas originales.
    2. total_events: suma horizontal de event_1 … event_5.
    3. ratio_event_X: event_X / (total_events + ε).
    4. has_done_event_X: 1 si event_X > 0, sino 0.
    """
    # Paso 1: segmento de edad
    lf = lf.with_columns(
        (
            pl.col("min_age_range").cast(pl.Utf8)
            + pl.lit("_")
            + pl.col("max_age_range").cast(pl.Utf8)
        )
        .cast(pl.Categorical)
        .alias("age_segment"),
    ).drop("min_age_range", "max_age_range")

    # Paso 2: total de eventos
    lf = lf.with_columns(
        pl.sum_horizontal(EVENT_COLS).alias("total_events"),
    )

    # Paso 3: ratios de afinidad
    lf = lf.with_columns(
        [
            (pl.col(c) / (pl.col("total_events") + EPSILON)).alias(f"ratio_{c}")
            for c in EVENT_COLS
        ]
    )

    # Paso 4: flags de descubrimiento
    lf = lf.with_columns(
        [
            (pl.col(c) > 0).cast(pl.Int8).alias(f"has_done_{c}")
            for c in EVENT_COLS
        ]
    )

    return lf


# ═════════════════════════════════════════════════════════════════════════════
# Fase Final — Casteo de categoricals restantes
# ═════════════════════════════════════════════════════════════════════════════
def castear_categoricals(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Castear columnas string de baja cardinalidad a categorical
    para que LightGBM las procese nativamente.
    """
    return lf.with_columns(
        pl.col("platform").cast(pl.Categorical),
        pl.col("gender").cast(pl.Categorical),
        # age_segment ya es categorical desde Fase 3
    )


# ═════════════════════════════════════════════════════════════════════════════
# Orquestador principal
# ═════════════════════════════════════════════════════════════════════════════
def build_churn_features(path: Path = DATA_PATH) -> pl.DataFrame:
    """
    Lee el CSV, ejecuta las 3 fases de feature engineering en modo lazy
    y materializa un DataFrame listo para LightGBM.
    """
    lf = (
        pl.scan_csv(path, infer_schema_length=5000)
        .pipe(fase_1_geo)
        .pipe(fase_2_temporal)
        .pipe(fase_3_demo_eventos)
        .pipe(castear_categoricals)
    )

    df = lf.collect()
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Ejecución & diagnóstico rápido
# ─────────────────────────────────────────────────────────────────────────────
def main():
    df = build_churn_features()

    print("=" * 72)
    print("PIPELINE COMPLETADO — Resumen del DataFrame")
    print("=" * 72)
    print(f"  Filas   : {df.shape[0]:,}")
    print(f"  Columnas: {df.shape[1]}")
    print()

    print("─── Schema ───")
    for name, dtype in zip(df.columns, df.dtypes):
        print(f"  {name:<28s} {str(dtype)}")
    print()

    print("─── Primeras 5 filas ───")
    print(df.head(5))
    print()

    print("─── Estadísticas descriptivas (numéricas) ───")
    print(df.describe())
    print()

    # Validaciones rápidas
    print("─── Validaciones ───")
    caba_check = df.filter(pl.col("country_region") == "Buenos Aires F.D.")
    if caba_check.height > 0:
        ciudades_caba = caba_check.select("city").unique()
        print(f"  CABA normalization: {ciudades_caba['city'].to_list()}")
    else:
        print("  CABA normalization: no rows with 'Buenos Aires F.D.' found (OK or verify data)")

    nulls_city = df.select(pl.col("city").is_null().sum()).item()
    print(f"  city nulls restantes: {nulls_city}")

    nulls_region = df.select(pl.col("country_region").is_null().sum()).item()
    print(f"  country_region nulls restantes: {nulls_region}")

    print(f"  total_events rango: [{df['total_events'].min()}, {df['total_events'].max()}]")
    print(f"  age_segment únicos: {df['age_segment'].n_unique()}")
    print(f"  time_of_day únicos: {df['time_of_day'].unique().to_list()}")
    print(f"  is_weekend_day0 distribución:\n{df['is_weekend_day0'].value_counts().sort('is_weekend_day0')}")
    print(f"  is_weekend_day1 distribución:\n{df['is_weekend_day1'].value_counts().sort('is_weekend_day1')}")
    print()
    print("✅ DataFrame listo para LightGBM.")

if __name__ == "__main__":
    main()

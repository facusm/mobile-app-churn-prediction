# Pipeline Automatizado de Predicción de Churn de Usuarios en Apps Móviles

Modelo predictivo de **churn Día +1 (D1)** para videojuegos móviles y aplicaciones.  
Pipeline de feature engineering con [Polars](https://pola.rs/) (lazy evaluation) orientado a entrenamiento con [LightGBM](https://lightgbm.readthedocs.io/).

---

## 🚀 Arquitectura y Tecnologías Destacadas

Este proyecto funciona como una solución integral MLOps para la predicción de fuga de usuarios, empleando:
- **Polars**: Procesamiento de datos ultrarrápido mediante evaluación perezosa (lazy evaluation), superando ampliamente los tiempos de pandas tradicionales.
- **LightGBM**: Modelo de boosting de gradiente altamente eficiente, ideal para datasets tabulares de gran escala y distribuciones desbalanceadas.
- **Optuna**: Optimización bayesiana automática de hiperparámetros, maximizando el PR-AUC mediante una búsqueda inteligente y eficiente.
- **SHAP (SHapley Additive exPlanations)**: Explicabilidad avanzada del modelo, permitiendo interpretar el impacto de cada variable en las predicciones (Feature Importance).
- **Docker**: Contenerización completa del entorno para asegurar la reproducibilidad exacta en cualquier infraestructura.

---

## 📦 Requisitos

- Python ≥ 3.10
- Dependencias:

```bash
pip install polars lightgbm matplotlib numpy scikit-learn optuna shap
```

---

## 📂 Estructura del proyecto

```text
proyecto/
├── data/
│   └── dataset_raw.csv              # Dataset crudo (20,000 registros)
├── src/
│   ├── churn_feature_pipeline.py    # Pipeline de preparación de datos
│   ├── eda_churn_bivariado.py       # EDA bivariado (tablas + 7 gráficos)
│   └── train_churn_model.py         # Entrenamiento LightGBM (Optuna + SHAP)
├── plots/                           # Carpeta generada con las visualizaciones (.png)
├── main.py                          # Orquestador del pipeline completo
├── requirements.txt                 # Dependencias del proyecto
├── Dockerfile                       # Configuración para empaquetado Docker
├── walkthrough.md                   # Documentación detallada del proceso
└── README.md                        # Este archivo
```

---

## ⚙️ Ejecución

### Opción 1: Docker (Recomendado)

El proyecto está completamente empaquetado en un contenedor Docker, garantizando una ejecución estable en cualquier ambiente corporativo.

```bash
# 1. Construir la imagen
docker build -t churn-pipeline .

# 2. Ejecutar el contenedor (los resultados se mostrarán por consola)
docker run --rm churn-pipeline
```

### Opción 2: Entorno Local

También es posible ejecutar localmente el orquestador principal que automatiza de forma secuencial: feature engineering, EDA y entrenamiento.

```bash
# Ejecutar todo el flujo
python -X utf8 main.py
```

Los gráficos y explicaciones visuales se generarán automáticamente en el directorio `plots/`.

**Métricas Finales (Validation Set):**
- **PR-AUC:** 0.7503
- **ROC-AUC:** 0.7838

---

## 📊 Insights y Datos Clave

| Métrica | Valor |
|---|---|
| Filas originales | 20,000 |
| Filas eliminadas (filtro de borde temporal) | 2,308 |
| Filas procesadas | 17,692 |
| Features generados | 30 |
| Tasa de churn global | ~48% |

### Análisis Predictivo Principal

A partir de las herramientas de explicabilidad multivariada (SHAP), se descubrieron los verdaderos drivers de retención:

| Driver | Impacto Observado |
|---|---|
| **Interacción Temprana** | Relación monotónica inversa: 72.8% de churn en usuarios pasivos (0-10 eventos) → 10.1% de retención en perfiles altamente activos (+150 eventos). |
| **Descubrimiento de Hitos** | El EDA univariado expone variables de descubrimiento temprano con grandes caídas en la tasa de churn (ej. Δ=36.9pp). |
| **Efectos Multivariados** | SHAP revela que eventos específicos actúan como anclas reales de retención (ej. Evento 4), mientras que otros operan como meros activadores del flujo de usuario (Evento 3). |

---

## 📖 Documentación

Se incluye un análisis técnico detallado en [`walkthrough.md`](walkthrough.md), donde se expone paso a paso la racionalidad detrás de cada fase del pipeline, decisiones de preprocesamiento, estrategias de modelado e interpretación de gráficos exploratorios.

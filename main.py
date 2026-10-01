import sys
from pathlib import Path

# Configurar el path para que los módulos en src puedan encontrarse mutuamente
PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "src"))

# Importamos las funciones principales (ahora desde src)
from src import churn_feature_pipeline
from src import eda_churn_bivariado
from src import train_churn_model

def main():
    print("="*60)
    print("🚀 INICIANDO PIPELINE DE CHURN D1")
    print("="*60)
    
    print("\n▶ FASE 1: Feature Engineering (Polars)")
    print("-" * 40)
    churn_feature_pipeline.main()
    
    print("\n▶ FASE 2: Análisis Exploratorio Bivariado (EDA)")
    print("-" * 40)
    eda_churn_bivariado.main()
    
    print("\n▶ FASE 3: Entrenamiento del Modelo (LightGBM)")
    print("-" * 40)
    train_churn_model.main()
    
    print("\n" + "="*60)
    print("✅ PIPELINE FINALIZADO CORRECTAMENTE")
    print("   Resultados y gráficos generados en la carpeta /plots")
    print("="*60)

if __name__ == "__main__":
    main()

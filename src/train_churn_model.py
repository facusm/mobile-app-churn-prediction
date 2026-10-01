"""
Script de Entrenamiento del Modelo de Churn D1
=========================================================
Entrenamiento de un modelo LightGBM optimizado con Optuna.
Incluye validación cruzada, métricas de evaluación (PR-AUC) y explicabilidad con SHAP.
"""

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import polars as pl
import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import average_precision_score, roc_auc_score
import shap
import matplotlib.pyplot as plt

from churn_feature_pipeline import build_churn_features, DATA_PATH

# Semilla fija para reproducibilidad estricta
RANDOM_STATE = 42

def main():
    print("1. Cargando datos desde el pipeline...")
    df_pl = build_churn_features(DATA_PATH)
    
    # ── Paso 1: Preparación ──
    # Eliminar user_id, install_date (e install_time por ser datetime)
    df_pl = df_pl.drop(["user_id", "install_date", "install_time"])
    
    # Convertir a Pandas (LightGBM y sklearn trabajan mejor con df de pandas)
    df = df_pl.to_pandas()
    
    X = df.drop(columns=["target_churn_indicator"])
    y = df["target_churn_indicator"]
    
    print("2. Split Train/Validation (80/20) estratificado...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    
    # ── Paso 2: Optuna con CV ──
    print("3. Configurando Optuna (50 trials) para maximizar PR-AUC...")
    
    def objective(trial):
        params = {
            "random_state": RANDOM_STATE,
            "n_estimators": 1000,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 100),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "min_child_samples": trial.suggest_int("min_child_samples", 30, 100),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9),
            "verbose": -1,
            "n_jobs": -1
        }
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        pr_aucs = []
        
        for train_idx, val_idx in cv.split(X_train, y_train):
            X_fold_train, y_fold_train = X_train.iloc[train_idx], y_train.iloc[train_idx]
            X_fold_val, y_fold_val = X_train.iloc[val_idx], y_train.iloc[val_idx]
            
            model = lgb.LGBMClassifier(**params)
            callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)]
            
            model.fit(
                X_fold_train, y_fold_train,
                eval_set=[(X_fold_val, y_fold_val)],
                callbacks=callbacks
            )
            
            # Evaluar PR-AUC en la partición de validación
            y_pred_proba = model.predict_proba(X_fold_val)[:, 1]
            pr_auc = average_precision_score(y_fold_val, y_pred_proba)
            pr_aucs.append(pr_auc)
            
        return np.mean(pr_aucs)
    
    # Suprimir logs detallados de optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=RANDOM_STATE))
    
    # Ejecutar optimización
    study.optimize(objective, n_trials=50, show_progress_bar=True)
    
    best_params = study.best_params
    print("\n4. Mejores parámetros encontrados (Optuna):")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
        
    # ── Paso 3: Entrenamiento Final ──
    print("\n5. Entrenando modelo final con los mejores parámetros sobre todo el Train set...")
    final_params = {
        **best_params, 
        "random_state": RANDOM_STATE, 
        "n_estimators": 500, # Un número razonable fijo o podríamos usar el iter promedio
        "verbose": -1, 
        "n_jobs": -1
    }
    
    final_model = lgb.LGBMClassifier(**final_params)
    final_model.fit(X_train, y_train)
    
    print("\n6. Evaluación sobre el set de Validación...")
    y_test_pred_proba = final_model.predict_proba(X_test)[:, 1]
    y_test_pred = final_model.predict(X_test)
    
    pr_auc_test = average_precision_score(y_test, y_test_pred_proba)
    roc_auc_test = roc_auc_score(y_test, y_test_pred_proba)
    
    print(f"  PR-AUC:  {pr_auc_test:.4f}")
    print(f"  ROC-AUC: {roc_auc_test:.4f}")
    
    # ── Paso 4: Explicabilidad (SHAP) ──
    print("\n7. Explicabilidad con SHAP...")
    # TreeExplainer optimizado para modelos basados en árboles
    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(X_test)
    
    # Para LGBM Classifier binario, shap_values puede ser una lista de matrices (una por clase).
    # Queremos la clase positiva (1).
    if isinstance(shap_values, list):
        shap_values_to_plot = shap_values[1]
    else:
        shap_values_to_plot = shap_values
        
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values_to_plot, X_test, show=False)
    
    out_png = PROJECT_DIR / "plots" / "shap_summary.png"
    # bbox_inches="tight" ajusta los márgenes para que los nombres no queden cortados
    plt.savefig(out_png, bbox_inches="tight", dpi=200)
    plt.close()
    
    print(f"[OK] Análisis SHAP (Summary Plot) guardado en: {out_png}")

if __name__ == "__main__":
    main()

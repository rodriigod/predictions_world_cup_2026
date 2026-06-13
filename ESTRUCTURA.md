# 📊 Estructura Completa del Proyecto ML

## 🎯 Visión General

Plantilla profesional de Machine Learning organizada por **fases de datos** y **tipos de código**.

```
template_ml_projects/
│
├── 📁 files/                          ← DATOS (fases de procesamiento)
│   ├── f0_raw/                        # Fase 0: Datos sin procesar
│   ├── f1_input/                      # Fase 1: Datos limpios y validados
│   ├── f2_intermedia/                 # Fase 2: Datos con features engineered
│   └── f3_output/                     # Fase 3: Predicciones y resultados
│
├── 📁 notebooks/                      ← ANÁLISIS INTERACTIVO
│   ├── 01_data_exploration.ipynb      # Exploración y estadísticas
│   ├── 02_data_cleaning.ipynb         # Limpieza de datos
│   ├── 03_feature_engineering.ipynb   # Creación de features
│   ├── 04_eda.ipynb                   # EDA profundo
│   ├── 05_model_training.ipynb        # Entrenamiento de modelos
│   ├── 06_model_evaluation.ipynb      # Evaluación y métricas
│   └── 07_results_visualization.ipynb # Visualización de resultados
│
├── 📁 src/                            ← CÓDIGO REUTILIZABLE
│   ├── data/                          # Manipulación de datos
│   │   ├── __init__.py
│   │   ├── loader.py                  # Cargar datos (CSV, Excel, Parquet)
│   │   ├── preprocessor.py            # Limpieza y transformación
│   │   └── features.py                # Feature engineering
│   │
│   ├── models/                        # Modelos ML
│   │   ├── __init__.py
│   │   ├── baseline.py                # Modelos simples (RF, LR, etc)
│   │   └── advanced.py                # Modelos complejos (GB, SVM, etc)
│   │
│   └── utils/                         # Utilidades
│       ├── __init__.py
│       ├── metrics.py                 # Métricas de evaluación
│       └── visualization.py           # Gráficos y visualizaciones
│
├── 📁 scripts/                        ← SCRIPTS EJECUTABLES
│   ├── config.py                      # Configuración global
│   ├── preprocess.py                  # Pipeline de preprocesamiento
│   ├── train.py                       # Entrenar modelo
│   ├── predict.py                     # Hacer predicciones
│   └── evaluate.py                    # Evaluar modelo
│
├── 📁 models/                         ← MODELOS GUARDADOS
│   ├── model_baseline.pkl
│   ├── model_v2.pkl
│   └── scalers/                       # Scalers y transformadores
│
├── 📁 results/                        ← RESULTADOS
│   ├── metrics/                       # Archivos de métricas
│   ├── plots/                         # Gráficos generados
│   └── reports/                       # Reportes
│
├── 📁 config/                         ← CONFIGURACIÓN
│   ├── config.yaml                    # Configuración principal
│   └── config_local.yaml              # Configuración local
│
├── 📁 tests/                          ← TESTS UNITARIOS
│   ├── __init__.py
│   ├── test_data.py
│   ├── test_models.py
│   └── test_utils.py
│
├── 📝 README.md                       # Documentación del proyecto
├── 🚀 QUICK_START.md                  # Guía rápida
├── 📋 requirements.txt                # Dependencias Python
├── 📦 setup.py                        # Configuración del paquete
├── 🔐 .env.example                    # Variables de entorno (ejemplo)
└── 📌 .gitignore                      # Archivos a ignorar en git


## 🔄 Flujo de Datos

```
f0_raw (datos originales)
    ↓ [Limpieza, validación]
f1_input (datos procesados)
    ↓ [Feature engineering, escalado]
f2_intermedia (features engineered)
    ↓ [Split train/test, entrenamiento]
Modelos entrenados
    ↓ [Evaluación, predicción]
f3_output (predicciones y resultados)
```

## 📊 Módulos Principales

### `src/data/` - Manipulación de Datos
- **loader.py**: Carga datos de CSV, Excel, Parquet
- **preprocessor.py**: Manejo de NaN, duplicados, outliers, escalado, encoding
- **features.py**: Creación de features polinomiales, interacciones, ratios

### `src/models/` - Modelado ML
- **baseline.py**: Random Forest, Decision Trees, Logistic Regression
- **advanced.py**: Gradient Boosting, SVM, KNN

### `src/utils/` - Utilidades
- **metrics.py**: Accuracy, Precision, Recall, F1, ROC-AUC, MSE, RMSE, MAE, R²
- **visualization.py**: Confusion matrices, ROC curves, feature importance, distributions

## 🚀 Flujo de Trabajo Típico

1. **Exploración** → `notebooks/01_data_exploration.ipynb`
2. **Limpieza** → `notebooks/02_data_cleaning.ipynb`
3. **Features** → `notebooks/03_feature_engineering.ipynb`
4. **EDA** → `notebooks/04_eda.ipynb`
5. **Entrenamiento** → `notebooks/05_model_training.ipynb`
6. **Evaluación** → `notebooks/06_model_evaluation.ipynb`
7. **Resultados** → `notebooks/07_results_visualization.ipynb`

## 💻 Ejecución por Línea de Comando

```bash
# Preprocesar datos
python scripts/preprocess.py

# Entrenar modelo
python scripts/train.py

# Hacer predicciones
python scripts/predict.py --input files/f1_input/data.csv --output files/f3_output/predictions.csv

# Evaluar modelo
python scripts/evaluate.py
```

## ⚙️ Configuración

Edita `config/config.yaml`:
- Rutas de datos
- Parámetros del modelo
- Hiperparámetros de entrenamiento
- Semillas aleatorias para reproducibilidad
- Configuración de logging

## 📦 Dependencias Incluidas

```
Core Data Science
├─ numpy, pandas, scipy

Machine Learning
├─ scikit-learn, xgboost, lightgbm

Deep Learning (opcional)
├─ tensorflow, torch

Visualization
├─ matplotlib, seaborn, plotly

Development
├─ jupyter, pytest, black, flake8
```

## ✅ Mejores Prácticas Implementadas

- ✓ **Modularidad**: Código separado en módulos reutilizables
- ✓ **Configuración**: Uso de config.yaml en lugar de hardcoding
- ✓ **Reproducibilidad**: Seeds fijas, documentación clara
- ✓ **Pipeline**: Flujo claro de datos de raw → output
- ✓ **Testing**: Ejemplos de tests unitarios
- ✓ **Documentación**: Notebooks con explicaciones paso a paso
- ✓ **Versionado**: .gitignore configurado para datos/modelos grandes

## 📚 Para Extensiones

### NLP
- Añade tokenizadores en `src/data/`
- Usa transformers para embeddings
- Crea notebook específico para NLP

### Deep Learning
- Crea `src/models/neural_networks.py`
- Usa TensorFlow/PyTorch
- Añade notebook de redes neuronales

### Series de Tiempo
- Funciones de lagging en `src/data/features.py`
- Usa Prophet o statsmodels
- Crea notebook de predicción temporal

## 🎓 Recursos Recomendados

- Scikit-learn: https://scikit-learn.org/
- Pandas: https://pandas.pydata.org/
- MLOps: https://ml-ops.systems/
- Jupyter: https://jupyter.org/

---
**Plantilla versión**: 1.0.0 | **Actualizado**: Junio 2026

"""
Guía rápida para usar la plantilla
"""

# 📋 QUICK START GUIDE

## 1. Configurar el entorno
```bash
cd template_ml_projects
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

pip install -r requirements.txt
```

## 2. Estructura de carpetas
- **files/**: Datos en diferentes fases
  - f0_raw: Datos originales
  - f1_input: Datos limpios
  - f2_intermedia: Datos con features
  - f3_output: Resultados finales

- **notebooks/**: Jupyter notebooks para análisis paso a paso
- **src/**: Código reutilizable (data, models, utils)
- **scripts/**: Scripts ejecutables
- **models/**: Modelos guardados
- **results/**: Resultados y visualizaciones

## 3. Flujo de trabajo típico

### Paso 1: Exploración
```bash
jupyter lab
# Abre: notebooks/01_data_exploration.ipynb
```

### Paso 2: Limpieza
- Ejecuta: notebooks/02_data_cleaning.ipynb
- O ejecuta: python scripts/preprocess.py

### Paso 3: Feature Engineering
- Ejecuta: notebooks/03_feature_engineering.ipynb

### Paso 4: EDA profundo
- Ejecuta: notebooks/04_eda.ipynb

### Paso 5: Entrenamiento
- Ejecuta: notebooks/05_model_training.ipynb
- O ejecuta: python scripts/train.py

### Paso 6: Evaluación
- Ejecuta: notebooks/06_model_evaluation.ipynb

### Paso 7: Visualización de resultados
- Ejecuta: notebooks/07_results_visualization.ipynb

## 4. Scripts útiles

### Preprocesar datos
```bash
python scripts/preprocess.py
```

### Entrenar modelo
```bash
python scripts/train.py
```

### Hacer predicciones
```bash
python scripts/predict.py --input files/f1_input/data.csv --output files/f3_output/predictions.csv
```

### Evaluar modelo
```bash
python scripts/evaluate.py
```

## 5. Configuración

Edita `config/config.yaml` para personalizar:
- Rutas de datos
- Parámetros del modelo
- Hiperparámetros
- Métricas de evaluación

## 6. Dependencias principales

- **Data**: pandas, numpy
- **ML**: scikit-learn, xgboost, lightgbm
- **Deep Learning**: tensorflow, torch (opcional)
- **Visualization**: matplotlib, seaborn, plotly
- **Notebooks**: jupyter, jupyterlab

## 7. Mejores prácticas

✅ Mantén código modular
✅ Documenta funciones importantes
✅ Usa configuración en config.yaml
✅ Escribe tests para funciones críticas
✅ Guarda modelos en models/
✅ Versionea tus datos

## 8. Extensiones comunes

### Para Deep Learning:
- Agrega notebooks para redes neuronales
- Usa tensorflow o pytorch en src/models

### Para NLP:
- Agrega tokenizadores en src/data
- Usa transformers para embeddings

### Para Series de Tiempo:
- Crea funciones de lagging en src/data/features.py
- Usa Prophet o statsmodels

## 9. Recursos útiles

- Scikit-learn: https://scikit-learn.org/
- Pandas: https://pandas.pydata.org/
- Jupyter: https://jupyter.org/
- MLOps best practices: https://ml-ops.systems/

---

¡Listo para empezar! 🚀

# 🤖 ML Project Template

Plantilla **profesional y completa** para proyectos de Machine Learning, organizada por **fases de datos** (f0→f3) y **tipos de código** (notebooks, src, scripts).

## ⚡ Inicio Rápido

```bash
# 1. Configurar entorno
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Iniciar notebooks
jupyter lab

# 3. Seguir notebooks en orden (01 → 07)
```

## 📊 Estructura Organizada por Fases

### **Fases de Datos** (`files/`)
- **f0_raw** ← Datos originales sin procesar
- **f1_input** ← Datos limpios y validados
- **f2_intermedia** ← Datos con features engineered
- **f3_output** ← Predicciones y resultados finales

### **Análisis Interactivo** (`notebooks/`)
1. 📊 Exploración de datos
2. 🧹 Limpieza de datos
3. ⚙️ Feature engineering
4. 📈 EDA profundo
5. 🎯 Entrenamiento de modelos
6. ✅ Evaluación del modelo
7. 📉 Visualización de resultados

### **Código Reutilizable** (`src/`)
- **data/**: Cargar, limpiar, transformar datos
- **models/**: Modelos baseline y advanced
- **utils/**: Métricas, visualizaciones

### **Scripts Ejecutables** (`scripts/`)
- Preprocesamiento automático
- Entrenamiento del modelo
- Predicciones batch
- Evaluación

## 🚀 Features de la Plantilla

✅ **Modular**: Código separado en componentes reutilizables  
✅ **Configurable**: config.yaml centralizado  
✅ **Reproducible**: Seeds fijas, documentación clara  
✅ **Pipeline**: Flujo claro de datos (raw → output)  
✅ **Tested**: Ejemplos de tests unitarios  
✅ **Documentado**: 7 notebooks con explicaciones paso a paso  
✅ **Profesional**: Estructura de industria  

## 📋 Contenido de la Plantilla

```
notebooks/          ← 7 Jupyter notebooks listos para usar
src/
├── data/           ← DataLoader, DataPreprocessor, FeatureEngineer
├── models/         ← Baseline y Advanced models
└── utils/          ← Métricas, visualizaciones

scripts/            ← preprocess.py, train.py, predict.py, evaluate.py
config/
├── config.yaml     ← Todos los parámetros
└── .env.example    ← Variables de entorno

requirements.txt    ← 50+ librerías científicas pre-seleccionadas
setup.py            ← Para instalar como paquete
tests/              ← Tests unitarios de ejemplo
```

## 🎯 Modelos Incluidos

### Baseline (Rápidos)
- Random Forest
- Decision Tree
- Logistic Regression
- Linear Regression

### Advanced (Complejos)
- Gradient Boosting
- Support Vector Machines
- K-Nearest Neighbors
- XGBoost, LightGBM

## 📊 Métricas Implementadas

**Clasificación**: Accuracy, Precision, Recall, F1, ROC-AUC  
**Regresión**: MSE, RMSE, MAE, R²

## 📖 Documentación Completa

- `README.md` - Este archivo
- `QUICK_START.md` - Guía rápida
- `ESTRUCTURA.md` - Descripción detallada
- Docstrings en cada módulo

## 🔧 Instalación de Dependencias

```bash
pip install -r requirements.txt
```

Incluye:
- pandas, numpy, scipy (Data Science)
- scikit-learn, xgboost, lightgbm (ML)
- tensorflow, torch (Deep Learning - opcional)
- matplotlib, seaborn, plotly (Visualización)
- jupyter, pytest, black (Desarrollo)

## 💡 Ejemplo de Uso

### 1. Exploración
```python
from src.data import DataLoader

loader = DataLoader('./files/f0_raw')
df = loader.load_csv('data.csv')
```

### 2. Limpieza
```python
from src.data import DataPreprocessor

preprocessor = DataPreprocessor()
df = preprocessor.handle_missing_values(df, method='mean')
df = preprocessor.remove_outliers(df, columns=['feature1', 'feature2'])
```

### 3. Features
```python
from src.data import FeatureEngineer

engineer = FeatureEngineer()
df = engineer.create_polynomial_features(df, columns=['x'], degree=2)
df = engineer.create_interaction_features(df, columns=['x', 'y'])
```

### 4. Entrenamiento
```python
from src.models import BaselineModel

model = BaselineModel.random_forest_classifier(
    X_train, y_train, X_test,
    n_estimators=100,
    max_depth=10
)
```

### 5. Evaluación
```python
from src.utils import ModelMetrics, Visualizer

metrics = ModelMetrics.classification_metrics(y_test, predictions)
Visualizer.plot_confusion_matrix(y_test, predictions)
```

## 🎓 Estructura por Tipo

| Tipo | Ubicación | Uso |
|------|-----------|-----|
| **Datos** | `files/f0-f3/` | Almacenar datos en fases |
| **Análisis** | `notebooks/` | Exploración interactiva |
| **Código** | `src/` | Lógica reutilizable |
| **Scripts** | `scripts/` | Automatización |
| **Modelos** | `models/` | Persistencia de modelos |
| **Resultados** | `results/` | Salidas y reportes |

## 🔄 Flujo Típico

1. **Cargar datos** → f0_raw
2. **Limpiar** → f1_input
3. **Features** → f2_intermedia
4. **Entrenar** → models/
5. **Predecir** → f3_output
6. **Evaluar** → results/

## 🎨 Personalización

La plantilla es **flexible y extensible**:

### Para Deep Learning
- Añade `src/models/neural_networks.py`
- Crea notebooks con TensorFlow/PyTorch

### Para NLP
- Extiende `src/data/features.py` con tokenizadores
- Usa transformers para embeddings

### Para Series de Tiempo
- Añade funciones de lagging
- Usa Prophet o statsmodels

## 📚 Mejores Prácticas

✅ Mantén datos en `files/`, código en `src/`  
✅ Usa `config.yaml` en lugar de hardcoding  
✅ Documenta funciones importantes  
✅ Escribe tests para código crítico  
✅ Guarda modelos con versiones  
✅ Usa virtual environments  

## 🔗 Recursos

- [Scikit-learn Docs](https://scikit-learn.org/)
- [Pandas Docs](https://pandas.pydata.org/)
- [Jupyter](https://jupyter.org/)
- [ML Ops Guide](https://ml-ops.systems/)

## 📝 Licencia

MIT - Libre para usar en tus proyectos

---

**¿Listo para empezar?** 👉 Lee [QUICK_START.md](QUICK_START.md) o [ESTRUCTURA.md](ESTRUCTURA.md)

**Versión**: 1.0.0 | **Última actualización**: Junio 2026
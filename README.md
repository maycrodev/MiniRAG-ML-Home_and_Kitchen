# Proyecto Integrador Fase 2 — Machine Learning

**Universidad Catolica Boliviana | Gestion 2026-I**

| | |
|---|---|
| Materia | Machine Learning |
| Grupo | 7 |
| Integrantes | Juan Jose Cordeiro, Alan Flores, Christian Coronel, Leonardo Delgado, Antonio Yujra |
| Docente | Ovidio Roger Paton |
| Metodologia | CRISP-DM |

---

## Descripcion

Mini proyecto de clasificacion de sentimiento sobre resenas de Amazon (categoria *Home and Kitchen*). Se aplica el ciclo completo CRISP-DM: EDA, preparacion de datos, tres modelos supervisados, benchmark AutoML (FLAML) y clustering no supervisado (K-Means).

---

## Dataset

- **Fuente:** Amazon Reviews 2023 — `Home_and_Kitchen`
- **Formato:** JSONL comprimido (`.jsonl.gz`, ~7 GB)
- **Submuestreo:** 20 000 resenas por calificacion (1–5 estrellas) = **100 000 total**
- **Variable objetivo:**
  - `0` = Negativo (1–2 estrellas)
  - `1` = Neutro (3 estrellas)
  - `2` = Positivo (4–5 estrellas)

> El archivo `Home_and_Kitchen.jsonl.gz` no se incluye en el repositorio por su tamano (~7 GB). Descargarlo manualmente desde la fuente oficial antes de ejecutar el notebook.

---

## Estructura del repositorio

```
MiniRAG-ML-Home_and_Kitchen/
├── requirements.txt
├── data/
│   ├── Home_and_Kitchen.jsonl.gz   <- descargar manualmente (no incluido)
│   ├── subset_raw.parquet          <- generado por el notebook
│   └── subset_split.parquet        <- generado por el notebook
├── notebooks/
│   └── Fase2_ML_Grupo7.ipynb       <- notebook principal
├── models/
│   ├── tfidf.pkl
│   ├── svd_lsa.pkl
│   ├── kmeans.pkl
│   ├── lr_model.pkl
│   ├── lgbm_model.pkl
│   └── flaml_automl.pkl
├── reports/
│   └── figuras/                    <- graficos generados
└── mlruns/                         <- experimentos MLflow (no incluido en git)
```

---

## Instalacion

```bash
# 1. Clonar el repositorio
git clone <url-del-repo>
cd MiniRAG-ML-Home_and_Kitchen

# 2. Crear entorno virtual (recomendado)
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
.venv\Scripts\activate           # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Descargar datos de NLTK (stopwords)
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

---

## Uso

1. Colocar `Home_and_Kitchen.jsonl.gz` dentro de `data/`. (Descargar desde: https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Home_and_Kitchen.jsonl.gz)
2. Abrir y ejecutar `notebooks/Fase2_ML_Grupo7.ipynb` de arriba hacia abajo.
3. El notebook genera los archivos `.parquet`, entrena todos los modelos y guarda los artefactos en `models/` y los experimentos en `mlruns/`.

Para visualizar los experimentos con MLflow:

```bash
mlflow ui --backend-store-uri mlruns/
# Abrir http://localhost:5000
```

---

## Modelos y resultados

Particion estratificada 70 / 15 / 15 con `SEED = 42`. TF-IDF ajustado unicamente sobre `X_train`.

| Modelo | Accuracy | F1-Macro | ROC-AUC (OvR) |
|---|---|---|---|
| Logistic Regression | 0.7185 | 0.6334 | 0.8514 |
| LightGBM | 0.7124 | 0.6317 | 0.8452 |
| FLAML AutoML | 0.6957 | 0.6181 | 0.8293 |
| Random Forest | 0.6895 | 0.5216 | 0.8299 |

**Mejor modelo:** Logistic Regression (F1-Macro y ROC-AUC mas altos).

---

## Clustering

K-Means sobre representacion LSA (SVD truncado) del corpus. Metricas evaluadas: coeficiente de silueta y Davies-Bouldin. Los clusters se analizan en relacion con la distribucion de sentimientos.

---

## Stack tecnico

| Libreria | Uso |
|---|---|
| pandas / numpy | Manipulacion de datos |
| scikit-learn | TF-IDF, SVD, LR, RF, K-Means, metricas |
| lightgbm | Gradient boosting |
| flaml | AutoML benchmark |
| nltk | Preprocesamiento NLP |
| mlflow | Seguimiento de experimentos |
| matplotlib / seaborn | Visualizacion |

---

## Notas

- `SEED = 42` en todos los procesos aleatorios.
- `rf_model.pkl` no se incluye en el repositorio por su tamano (excede el limite de GitHub).
- Los archivos `.parquet` en `data/` tampoco se incluyen; se regeneran ejecutando el notebook.

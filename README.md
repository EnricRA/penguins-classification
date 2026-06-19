# Penguin Species Classification

Proyecto de clasificación para el conjunto de datos de pingüinos del Archipiélago Palmer utilizando cuatro modelos: regresión logística, SVM, árboles de decisión y KNN.

## Estructura del proyecto

```
penguins-classification/
├── client/
│   └── client.py              # HTTP client (2 peticiones por modelo)
├── data/
│   └── penguins_size.csv      # Kaggle dataset
├── models/                    # Serialized models (generados despues de entrenar)
├── services/
│   ├── logistic_regression_app.py  # Servicio Flask en el puerto 5001
│   ├── svm_app.py                  # Servicio Flask en el puerto 5002
│   ├── decision_tree_app.py        # Servicio Flask en el puerto 5003
│   └── knn_app.py                  # Servicio Flask en el puerto 5004
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   └── train_models.py
└── environment.yml
```

## Setup

```bash
conda env create -f environment.yml
conda activate penguins-classification
```

## Entrenar los modelos

En la raiz del proyecto:

```bash
python -m src.train_models
```

Esto hara:
- Cargar el dataset `data/penguins_size.csv` y eliminar los registros con valores faltantes.
- Realizar un Train/Test Split del 80% para entrenamiento y 20% para prueba.
- Codificar `species` utilizando `LabelEncoder`
- Aplicar One-Hot Encoding a las variables categóricas mediante `DictVectorizer`
- Escalar las variables numéricas con `StandardScaler` (ajustado solo para el conjunto de entrenamiento)
- Entrenar y serializar los cuatro modelos en el directorio `models/`

## Ejecutar los servicios Flask

Abre cuatro terminales (con el enviorament de conda) y  ejecuta:

```bash
python -m services.logistic_regression_app
python -m services.svm_app
python -m services.decision_tree_app
python -m services.knn_app
```

| Model                 | Port |
|-----------------------|------|
| Logistic regression   | 5001 |
| SVM                   | 5002 |
| Decision tree         | 5003 |
| KNN                   | 5004 |

## Ejecuta el cliente

Con los cuatro servicios ya ejecutandose en diferentes terminales:

```bash
python client/client.py
```

## Ejemplo de solicitud de predicción

```bash
curl -X POST http://127.0.0.1:5001/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"island\":\"Biscoe\",\"culmen_length_mm\":50.0,\"culmen_depth_mm\":15.0,\"flipper_length_mm\":220,\"body_mass_g\":5400,\"sex\":\"FEMALE\"}"
```

## Exportar la configuracion en enviorament

```bash
conda env export > environment.yml
```
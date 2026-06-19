# Penguin Species Classification

Classifier project for the Palmer Archipelago penguins dataset using four models:
logistic regression, SVM, decision trees, and KNN.

## Project structure

```
penguins-classification/
├── client/
│   └── client.py              # HTTP client (2+ requests per model)
├── data/
│   └── penguins_size.csv      # Kaggle dataset
├── models/                    # Serialized models (generated after training)
├── services/
│   ├── logistic_regression_app.py  # Flask service on port 5001
│   ├── svm_app.py                  # Flask service on port 5002
│   ├── decision_tree_app.py        # Flask service on port 5003
│   └── knn_app.py                  # Flask service on port 5004
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

## Train models

From the project root:

```bash
python -m src.train_models
```

This will:
- Load `data/penguins_size.csv` and drop rows with missing values
- Split 80% train / 20% test (stratified)
- Encode `species` with `LabelEncoder`
- One-hot encode categorical features with `DictVectorizer`
- Standard-scale numeric features with `StandardScaler` (fit on train only)
- Train and serialize all four models into `models/`

## Run Flask services

Open four terminals (with the conda environment active) and run:

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

## Run the client

With all four services running:

```bash
python client/client.py
```

## Example prediction request

```bash
curl -X POST http://127.0.0.1:5001/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"island\":\"Biscoe\",\"culmen_length_mm\":50.0,\"culmen_depth_mm\":15.0,\"flipper_length_mm\":220,\"body_mass_g\":5400,\"sex\":\"FEMALE\"}"
```

## Export conda environment

After installing all packages:

```bash
conda env export > environment.yml
```

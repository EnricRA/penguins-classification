from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.data_loader import TARGET_COLUMN, load_penguins_dataset, records_from_dataframe
from src.preprocessing import PenguinPreprocessor

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
RANDOM_STATE = 42
TEST_SIZE = 0.2

MODEL_CONFIGS = {
    "logistic_regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "svm": SVC(kernel="rbf", random_state=RANDOM_STATE),
    "decision_tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "knn": KNeighborsClassifier(n_neighbors=5),
}


def train_and_save_models() -> None:
    # Carga datos y prepara registros para entrenamiento
    df = load_penguins_dataset()
    feature_records = records_from_dataframe(df)

    # Codifica las etiquetas de especie en valores numéricos
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(df[TARGET_COLUMN])

    # Separa datos en entrenamiento y prueba
    train_records, test_records, y_train, y_test = train_test_split(
        feature_records,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    # Ajusta el preprocesador solo con los datos de entrenamiento
    preprocessor = PenguinPreprocessor()
    x_train = preprocessor.fit_transform(train_records)
    x_test = preprocessor.transform(test_records)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for model_name, estimator in MODEL_CONFIGS.items():
        estimator.fit(x_train, y_train)
        train_accuracy = estimator.score(x_train, y_train)
        test_accuracy = estimator.score(x_test, y_test)

        artifact_path = MODELS_DIR / f"{model_name}.joblib"
        joblib.dump(
            {
                "model": estimator,
                "preprocessor": preprocessor,
                "label_encoder": label_encoder,
            },
            artifact_path,
        )

        print(f"{model_name}: train={train_accuracy:.3f}, test={test_accuracy:.3f}")
        print(f"  saved to {artifact_path}")


if __name__ == "__main__":
    train_and_save_models()

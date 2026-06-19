from pathlib import Path

import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

REQUIRED_FIELDS = [
    "island",
    "culmen_length_mm",
    "culmen_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
    "sex",
]


def load_artifact(model_name: str) -> dict:
    artifact_path = MODELS_DIR / f"{model_name}.joblib"
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {artifact_path}. Run src/train_models.py first."
        )
    return joblib.load(artifact_path)


def predict_species(artifact: dict, payload: dict) -> dict:
    # Valida que el payload contenga todos los campos requeridos
    missing_fields = [field for field in REQUIRED_FIELDS if field not in payload]
    if missing_fields:
        raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")

    record = {
        "island": payload["island"],
        "culmen_length_mm": float(payload["culmen_length_mm"]),
        "culmen_depth_mm": float(payload["culmen_depth_mm"]),
        "flipper_length_mm": float(payload["flipper_length_mm"]),
        "body_mass_g": float(payload["body_mass_g"]),
        "sex": payload["sex"],
    }

    # Prepara la entrada y genera la predicción
    features = artifact["preprocessor"].transform([record])
    prediction_code = int(artifact["model"].predict(features)[0])
    species = artifact["label_encoder"].inverse_transform([prediction_code])[0]

    probabilities = None
    model = artifact["model"]
    if hasattr(model, "predict_proba"):
        class_probabilities = model.predict_proba(features)[0]
        probabilities = {
            species_name: float(probability)
            for species_name, probability in zip(
                artifact["label_encoder"].classes_, class_probabilities
            )
        }

    return {
        "species": species,
        "species_code": prediction_code,
        "probabilities": probabilities,
    }

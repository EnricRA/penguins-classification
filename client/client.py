import json
import sys
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# URLs de los servicios de predicción desplegados para cada modelo
SERVICES = {
    "logistic_regression": "http://127.0.0.1:5001/predict",
    "svm": "http://127.0.0.1:5002/predict",
    "decision_tree": "http://127.0.0.1:5003/predict",
    "knn": "http://127.0.0.1:5004/predict",
}

SAMPLE_PENGUINS = [
    {
        "label": "Adelie sample",
        "payload": {
            "island": "Torgersen",
            "culmen_length_mm": 39.1,
            "culmen_depth_mm": 18.7,
            "flipper_length_mm": 181,
            "body_mass_g": 3750,
            "sex": "MALE",
        },
    },
    {
        "label": "Gentoo sample",
        "payload": {
            "island": "Biscoe",
            "culmen_length_mm": 50.0,
            "culmen_depth_mm": 15.0,
            "flipper_length_mm": 220,
            "body_mass_g": 5400,
            "sex": "FEMALE",
        },
    },
    {
        "label": "Chinstrap sample",
        "payload": {
            "island": "Dream",
            "culmen_length_mm": 46.5,
            "culmen_depth_mm": 17.9,
            "flipper_length_mm": 192,
            "body_mass_g": 3500,
            "sex": "MALE",
        },
    },
    {
        "label": "Large Gentoo sample",
        "payload": {
            "island": "Biscoe",
            "culmen_length_mm": 52.0,
            "culmen_depth_mm": 14.5,
            "flipper_length_mm": 230,
            "body_mass_g": 5700,
            "sex": "MALE",
        },
    },
]


def send_prediction(model_name: str, url: str, sample: dict) -> None:
    print(f"\n--- {model_name} | {sample['label']} ---")
    response = requests.post(url, json=sample["payload"], timeout=10)
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))


def main() -> None:
    print("Penguin classification client")
    print("Sending at least two requests to each deployed model...\n")

    for model_name, url in SERVICES.items():
        for sample in SAMPLE_PENGUINS[:2]:
            send_prediction(model_name, url, sample)

        extra_sample = SAMPLE_PENGUINS[2 if model_name in {"svm", "knn"} else 3]
        send_prediction(model_name, url, extra_sample)


if __name__ == "__main__":
    main()

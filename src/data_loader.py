from pathlib import Path

import pandas as pd

FEATURE_COLUMNS = [
    "island",
    "culmen_length_mm",
    "culmen_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
    "sex",
]
TARGET_COLUMN = "species"
NUMERIC_COLUMNS = [
    "culmen_length_mm",
    "culmen_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
]
CATEGORICAL_COLUMNS = ["island", "sex"]


def load_penguins_dataset(csv_path: Path | str | None = None) -> pd.DataFrame:
    if csv_path is None:
        csv_path = Path(__file__).resolve().parent.parent / "data" / "penguins_size.csv"
    csv_path = Path(csv_path)

    df = pd.read_csv(csv_path)
    df = df.replace("NA", pd.NA).dropna().reset_index(drop=True)

    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column])

    return df


def records_from_dataframe(df: pd.DataFrame) -> list[dict]:
    return df[FEATURE_COLUMNS].to_dict(orient="records")

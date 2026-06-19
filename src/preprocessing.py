from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler


class PenguinPreprocessor:
    """Codifica categorías y escala variables numéricas para los modelos."""

    def __init__(self) -> None:
        # Vectorizador para convertir diccionarios en matrices numéricas
        self.vectorizer = DictVectorizer(sparse=False)
        # Escalador para normalizar las columnas numéricas
        self.scaler = StandardScaler()
        self.numeric_indices_: list[int] | None = None

    def _numeric_feature_indices(self, feature_names: list[str]) -> list[int]:
        # Identifica índices de características numéricas tras one-hot encoding
        return [index for index, name in enumerate(feature_names) if "=" not in name]

    def fit(self, records: list[dict]) -> "PenguinPreprocessor":
        # Ajusta el vectorizador y el escalador usando los datos de entrenamiento
        features = self.vectorizer.fit_transform(records)
        feature_names = self.vectorizer.get_feature_names_out().tolist()
        self.numeric_indices_ = self._numeric_feature_indices(feature_names)
        features[:, self.numeric_indices_] = self.scaler.fit_transform(
            features[:, self.numeric_indices_]
        )
        return self

    def transform(self, records: list[dict]):
        if self.numeric_indices_ is None:
            raise RuntimeError("Preprocessor must be fitted before calling transform.")

        # Transforma nuevos registros usando el preprocesador ya ajustado
        features = self.vectorizer.transform(records)
        features[:, self.numeric_indices_] = self.scaler.transform(
            features[:, self.numeric_indices_]
        )
        return features

    def fit_transform(self, records: list[dict]):
        self.fit(records)
        return self.transform(records)

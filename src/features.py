"""Feature engineering, воспроизводящий логику notebooks/binary-classification.ipynb.

Все преобразования оформлены как чистые функции уровня модуля,
чтобы sklearn Pipeline с FunctionTransformer корректно сериализовался
(используется и при обучении, и в inference-сервисе).
"""

import numpy as np
import pandas as pd

# Порядковая шкала образования из ноутбука
EDUCATION_CATEGORIES = ["unknown", "primary", "secondary", "tertiary"]
_EDUCATION_MAP = {v: i for i, v in enumerate(EDUCATION_CATEGORIES)}

# Итоговые колонки после feature engineering
NUMERIC_FEATURES = [
    "age", "balance", "day", "duration", "campaign", "pdays", "previous",
    "education_ordinal", "balance_positive", "avg_duration_per_campaign",
    "pdays_negative", "prev_to_campaign_ratio", "balance_log",
    "duration_log", "campaign_log",
]
CATEGORICAL_FEATURES = [
    "job", "marital", "default", "housing", "loan", "contact", "month",
    "marital_housing",
]

TARGET = "y"
DROP_COLUMNS = ["poutcome"]  # ~90% unknown — удалён в ноутбуке


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Полный feature engineering: вход — сырой датасет, выход — X для модели."""
    features = df.copy()

    # Удаляем служебные/неинформативные колонки
    features.drop(columns=[c for c in [TARGET, "id", *DROP_COLUMNS] if c in features],
                  inplace=True)

    # education -> порядковый признак (unknown < primary < secondary < tertiary)
    features["education_ordinal"] = (
        features["education"].map(_EDUCATION_MAP).fillna(0).astype(np.int8)
    )
    features.drop(columns=["education"], inplace=True)

    # Ручные признаки (feature_constuction из ноутбука)
    features["balance_positive"] = (features["balance"] > 0).astype(int)
    features["avg_duration_per_campaign"] = features["duration"] / (features["campaign"] + 1)
    features["pdays_negative"] = (features["pdays"] == -1).astype(int)
    features["prev_to_campaign_ratio"] = features["previous"] / (features["campaign"] + 1)
    features["marital_housing"] = features["marital"].astype(str) + "_" + features["housing"].astype(str)
    features["balance_log"] = np.sign(features["balance"]) * np.log1p(np.abs(features["balance"]))
    features["duration_log"] = np.log1p(features["duration"])
    features["campaign_log"] = np.log1p(features["campaign"])

    # Фиксированный порядок колонок для ColumnTransformer
    return features[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

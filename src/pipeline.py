"""Сборка sklearn Pipeline: feature engineering -> препроцессинг -> модель."""

import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, RobustScaler

from src.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, build_features


def build_pipeline(**model_params) -> Pipeline:
    preprocessor = ColumnTransformer(
        [
            ("num", RobustScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )

    return Pipeline(
        [
            ("features", FunctionTransformer(build_features, validate=False)),
            ("preprocess", preprocessor),
            ("model", xgb.XGBClassifier(**model_params)),
        ]
    )

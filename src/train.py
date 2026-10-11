"""Обучение модели бинарной классификации вклада (XGBoost + Optuna) с MLflow.

Запуск:
    python -m src.train --data data/data.csv --n-trials 20 --alias champion

MLflow-трекинг локальный (./mlruns + mlflow.db):
- родительский run содержит финальную модель, метрики и артефакты
  (drift baseline, отчёт по Optuna);
- каждый Optuna-trial логируется как nested run с параметрами и ROC-AUC;
- модель регистрируется в Model Registry и помечается alias
  (`champion` по умолчанию, `challenger` для кандидатов).
"""

import argparse
import json
import tempfile
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.drift import build_baseline, save_baseline
from src.pipeline import build_pipeline

MODEL_NAME = "bank-deposit-classifier"
RANDOM_STATE = 42


def split_data(features: pd.DataFrame, target: pd.Series):
    """60/20/20 стратифицированное разбиение как в ноутбуке."""
    X_train, X_val_test, y_train, y_val_test = train_test_split(
        features, target, random_state=RANDOM_STATE, test_size=0.4, stratify=target
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_val_test, y_val_test, random_state=RANDOM_STATE, test_size=0.5,
        stratify=y_val_test,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def evaluate(model, X, y, prefix: str) -> dict:
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        f"{prefix}_roc_auc": roc_auc_score(y, proba),
        f"{prefix}_pr_auc": average_precision_score(y, proba),
        f"{prefix}_accuracy": accuracy_score(y, pred),
        f"{prefix}_precision": precision_score(y, pred, zero_division=0),
        f"{prefix}_recall": recall_score(y, pred, zero_division=0),
        f"{prefix}_f1": f1_score(y, pred, zero_division=0),
    }


def set_alias(model_name: str, version: str, alias: str):
    mlflow.tracking.MlflowClient().set_registered_model_alias(model_name, alias, version)
    print(f"Alias '{alias}' -> {model_name} v{version}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/data.csv", help="Путь к CSV с обучающими данными")
    parser.add_argument("--n-trials", type=int, default=20, help="Число trials Optuna")
    parser.add_argument("--sample", type=int, default=150_000,
                        help="Макс. размер трейна для ускорения подбора (0 — всё)")
    parser.add_argument("--experiment", default="bank-deposit", help="Имя MLflow experiment")
    parser.add_argument("--no-register", action="store_true",
                        help="Не регистрировать модель в Model Registry")
    parser.add_argument("--alias", default="champion",
                        choices=["champion", "challenger", "none"],
                        help="Alias для зарегистрированной версии")
    args = parser.parse_args()

    mlflow.set_experiment(args.experiment)

    df = pd.read_csv(args.data)
    print(f"Загружено {len(df)} строк из {args.data}")

    features = df.drop(columns=["y", "id"], errors="ignore")
    target = df["y"]
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(features, target)

    if args.sample and len(X_train) > args.sample:
        idx = np.random.RandomState(RANDOM_STATE).choice(len(X_train), args.sample, replace=False)
        X_train = X_train.iloc[idx]
        y_train = y_train.iloc[idx]
        print(f"Трейн урезан до {len(X_train)} строк для подбора гиперпараметров")

    scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())

    with mlflow.start_run(run_name="xgb-optuna") as parent_run:
        def objective(trial: optuna.Trial) -> float:
            params = {
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 3e-1, log=True),
                "max_depth": trial.suggest_int("max_depth", 2, 12),
                "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
                "gamma": trial.suggest_float("gamma", 0, 5),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
                "reg_alpha": trial.suggest_float("reg_alpha", 0, 10),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-5, 10.0, log=True),
                "scale_pos_weight": scale_pos_weight,
                "n_jobs": -1,
            }
            # Каждый trial — отдельный nested run в MLflow
            with mlflow.start_run(run_name=f"trial-{trial.number}", nested=True):
                mlflow.log_params(params)
                model = build_pipeline(**params)
                model.fit(X_train, y_train)
                auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
                mlflow.log_metric("val_roc_auc", auc)
                mlflow.log_metric("trial_number", trial.number)
            return auc

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)
        print(f"Лучший ROC-AUC на val: {study.best_value:.4f}")
        print(f"Лучшие параметры: {study.best_params}")

        # Финальная модель на train+val, оценка на отложенном test
        best_params = {**study.best_params, "scale_pos_weight": scale_pos_weight, "n_jobs": -1}
        mlflow.log_params(best_params)
        mlflow.log_param("n_trials", args.n_trials)
        mlflow.log_param("train_size", len(X_train))

        X_full = pd.concat([X_train, X_val])
        y_full = pd.concat([y_train, y_val])
        final_model = build_pipeline(**best_params)
        final_model.fit(X_full, y_full)

        metrics = evaluate(final_model, X_test, y_test, "test")
        metrics["val_roc_auc"] = study.best_value
        mlflow.log_metrics(metrics)
        print("Метрики на test:", {k: round(v, 4) for k, v in metrics.items()})

        with tempfile.TemporaryDirectory() as tmp:
            # Baseline для мониторинга дрейфа — артефакт рядом с моделью
            baseline_path = Path(tmp) / "drift_baseline.json"
            save_baseline(build_baseline(df.drop(columns=["id"], errors="ignore")), baseline_path)
            mlflow.log_artifact(str(baseline_path))

            # Сводка Optuna-стади
            trials_path = Path(tmp) / "optuna_trials.csv"
            study.trials_dataframe().to_csv(trials_path, index=False)
            mlflow.log_artifact(str(trials_path))

        signature = mlflow.models.infer_signature(
            X_test.head(5), final_model.predict_proba(X_test.head(5))
        )
        model_info = mlflow.sklearn.log_model(
            final_model, name="model", signature=signature,
            registered_model_name=None if args.no_register else MODEL_NAME,
        )
        print(f"Модель залогирована: {model_info.model_uri}")

        if not args.no_register and args.alias != "none":
            # В mlflow>=3 версия доступна через model_info.registered_model_version
            version = getattr(model_info, "registered_model_version", None)
            if version:
                set_alias(MODEL_NAME, version, args.alias)

    print("Готово.")


if __name__ == "__main__":
    main()

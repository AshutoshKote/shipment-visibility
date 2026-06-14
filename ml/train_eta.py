"""Train the ETA prediction model and track the full lifecycle in MLflow.

- loads the generated training dataset
- trains a gradient-boosted regressor (sklearn HistGradientBoosting --
  LightGBM-class model, no native deps to fight on macOS)
- logs params, metrics (MAE / RMSE / R2) and a permutation-importance
  artifact to MLflow
- registers the model and promotes it to the 'production' alias, so the
  serving layer (Day 4) can load models:/eta-predictor@production

Run:
    python ml/train_eta.py
View:
    mlflow ui --backend-store-uri sqlite:///mlflow.db
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

import mlflow
import mlflow.sklearn
from mlflow.client import MlflowClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

NUMERIC = ["hour", "dow", "dist_remaining_km", "progress_pct",
           "speed_kmh", "avg_speed_kmh", "dist_to_route_km", "is_stopped"]
TARGET = "minutes_to_arrival"
MODEL_NAME = "eta-predictor"


def prepare(df):
    """One-hot the route, assemble the feature matrix; return X, y, columns."""
    X = pd.get_dummies(df[NUMERIC + ["route_id"]], columns=["route_id"])
    y = df[TARGET]
    return X, y, list(X.columns)


def main():
    df = pd.read_parquet(config.TRAINING_DATA_PATH)
    X, y, feature_cols = prepare(df)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    params = dict(max_iter=400, learning_rate=0.08, max_depth=8,
                  l2_regularization=0.0, random_state=42)

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment("eta-prediction")

    with mlflow.start_run() as run:
        model = HistGradientBoostingRegressor(**params)
        model.fit(X_tr, y_tr)

        pred = model.predict(X_te)
        mae = mean_absolute_error(y_te, pred)
        rmse = float(np.sqrt(mean_squared_error(y_te, pred)))
        r2 = r2_score(y_te, pred)

        mlflow.log_params(params)
        mlflow.log_param("n_features", len(feature_cols))
        mlflow.log_param("n_rows", len(df))
        mlflow.log_metrics({"mae_min": mae, "rmse_min": rmse, "r2": r2})

        # permutation importance on a sample (kept small for speed)
        sample = min(2000, len(X_te))
        imp = permutation_importance(model, X_te.iloc[:sample], y_te.iloc[:sample],
                                     n_repeats=5, random_state=42)
        importances = sorted(
            zip(feature_cols, imp.importances_mean), key=lambda t: t[1], reverse=True
        )
        mlflow.log_text(
            "\n".join(f"{n}: {v:.3f}" for n, v in importances),
            "permutation_importance.txt",
        )

        # persist the feature column order so serving builds rows identically
        os.makedirs(os.path.dirname(config.FEATURE_COLUMNS_PATH), exist_ok=True)
        with open(config.FEATURE_COLUMNS_PATH, "w") as f:
            json.dump(feature_cols, f, indent=2)
        mlflow.log_artifact(config.FEATURE_COLUMNS_PATH)

        # register + promote
        mlflow.sklearn.log_model(
            model, name="model",
            registered_model_name=MODEL_NAME,
            input_example=X_te.iloc[:2],
        )
        client = MlflowClient()
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        latest = max(int(v.version) for v in versions)
        client.set_registered_model_alias(MODEL_NAME, "production", latest)

        print("\n=== ETA model trained ===")
        print(f"  MAE  : {mae:.2f} min")
        print(f"  RMSE : {rmse:.2f} min")
        print(f"  R2   : {r2:.3f}")
        print(f"  registered: {MODEL_NAME} v{latest} -> alias 'production'")
        print(f"  run_id: {run.info.run_id}")
        print("\n  top features:")
        for n, v in importances[:6]:
            print(f"    {n:<22} {v:.3f}")
        print("\n  view: mlflow ui --backend-store-uri sqlite:///mlflow.db")


if __name__ == "__main__":
    main()

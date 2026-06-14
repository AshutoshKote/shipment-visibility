"""Champion / challenger retraining.

Trains a fresh challenger model, evaluates it against the current production
champion on the SAME held-out split, and promotes the challenger to the
'production' alias only if it beats the champion. Otherwise the champion
stays live. Both are logged to MLflow.

Run:
    python ml/retrain.py
"""

import os
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.client import MlflowClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from ml.train_eta import prepare, MODEL_NAME

IMPROVE_MARGIN = 0.01   # challenger must be >=1% better on MAE to be promoted


def champion_mae(client, X_te, y_te):
    """Evaluate the current production champion on the held-out set."""
    try:
        model = mlflow.sklearn.load_model(config.MODEL_URI)
        return float(mean_absolute_error(y_te, model.predict(X_te)))
    except Exception as e:
        print(f"[retrain] no champion to compare ({e}); challenger wins by default.")
        return None


def main():
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment("eta-prediction")
    client = MlflowClient()

    df = pd.read_parquet(config.TRAINING_DATA_PATH)
    X, y, _ = prepare(df)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=7)

    champ_mae = champion_mae(client, X_te, y_te)

    # train challenger (slightly deeper / more iterations)
    params = dict(max_iter=600, learning_rate=0.06, max_depth=10, random_state=7)
    with mlflow.start_run(run_name="challenger") as run:
        challenger = HistGradientBoostingRegressor(**params)
        challenger.fit(X_tr, y_tr)
        chal_mae = float(mean_absolute_error(y_te, challenger.predict(X_te)))
        mlflow.log_params(params)
        mlflow.log_metric("mae_min", chal_mae)

        print(f"\n[retrain] champion MAE : {champ_mae if champ_mae is None else round(champ_mae,3)}")
        print(f"[retrain] challenger MAE: {chal_mae:.3f}")

        promote = champ_mae is None or chal_mae < champ_mae * (1 - IMPROVE_MARGIN)
        mlflow.sklearn.log_model(challenger, name="model",
                                 registered_model_name=MODEL_NAME,
                                 input_example=X_te.iloc[:2])
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        latest = max(int(v.version) for v in versions)

        if promote:
            client.set_registered_model_alias(MODEL_NAME, "production", latest)
            print(f"[retrain] PROMOTED challenger -> v{latest} is now 'production'")
        else:
            print(f"[retrain] challenger did NOT beat champion by {IMPROVE_MARGIN:.0%}; "
                  f"champion stays live. (challenger logged as v{latest}, not promoted)")


if __name__ == "__main__":
    main()

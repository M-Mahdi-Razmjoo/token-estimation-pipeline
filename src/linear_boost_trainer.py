import os
import gc
import pickle
import string
from typing import List, Dict, Any

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import config
from data_loader import DataLoader
from utils import get_environment_info

from feature_engineering import get_code_markup_count, URL_RE, EMAIL_RE


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _compute_required_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute ONLY the features used by NONLINEAR_SELECTED_FEATURES
    (so this stays fast and consistent with your existing selected set).
    """
    df = df.copy()
    content = df[config.CONTENT_COLUMN].astype(str).fillna("")

    df["char_count"] = content.str.len()
    words = content.str.split()
    df["word_count"] = words.str.len()

    safe_char = df["char_count"] + 1e-6
    safe_word = df["word_count"] + 1e-6

    df["avg_word_length"] = (df["char_count"] / safe_word).fillna(0.0)
    df["max_word_length"] = words.apply(lambda ws: max((len(w) for w in ws), default=0))

    df["code_markup_count"] = content.apply(get_code_markup_count)
    df["url_email_count"] = content.str.count(URL_RE) + content.str.count(EMAIL_RE)

    df["punctuation_proportion"] = content.apply(
        lambda x: sum(1 for ch in x if ch in string.punctuation)
    ) / safe_char

    return df


def _make_adaboost_regressor(params: Dict[str, Any]) -> AdaBoostRegressor:
    """
    Build AdaBoostRegressor with a *linear* base estimator (SGDRegressor).
    Backward-compatible with sklearn versions that still use base_estimator.
    """
    base_params = params.get("base_estimator_params", {})
    ada_params = params.get("adaboost_params", {})

    base_est = SGDRegressor(**base_params)

    try:
        model = AdaBoostRegressor(estimator=base_est, **ada_params)
    except TypeError:
        model = AdaBoostRegressor(base_estimator=base_est, **ada_params)

    return model


def train_linear_boost_model_pipeline(target_column: str, language_scope: str, tokenizer_name: str):
    """
    AdaBoost "linear boosting" (AdaBoostRegressor) on ENRICHED_INPUT_FILES.
    Base learner is linear (SGDRegressor).
    """
    output_dir = os.path.join(config.RESULTS_PATH, tokenizer_name, language_scope)
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, "linear_boost_results.txt")
    model_path = os.path.join(output_dir, "linear_boost_model.pkl")

    selected_features: List[str] = getattr(
        config,
        "LINEAR_BOOST_SELECTED_FEATURES",
        config.NONLINEAR_SELECTED_FEATURES
    )

    params: Dict[str, Any] = getattr(config, "LINEAR_BOOST_PARAMS", {
        "base_estimator_params": {
            "loss": "squared_error",
            "penalty": "l2",
            "alpha": 1e-4,
            "max_iter": 2000,
            "tol": 1e-3,
            "random_state": 42,
        },
        "adaboost_params": {
            "n_estimators": 50,
            "learning_rate": 0.5,
            "loss": "linear",
            "random_state": 42,
        }
    })

    loader = DataLoader(config.ENRICHED_INPUT_FILES)
    cols = [config.CONTENT_COLUMN, config.LANGUAGE_COLUMN, target_column]
    chunks = [c for c in loader.load_data_chunks(columns=cols)]
    df = pd.concat(chunks, ignore_index=True).dropna(subset=[target_column])

    if language_scope == "english":
        df = df[df[config.LANGUAGE_COLUMN] == "English"].copy()

    df = df.reset_index(drop=True)

    df = _compute_required_features(df)
    df = df.dropna(subset=selected_features + [target_column]).reset_index(drop=True)

    X = df[selected_features].to_numpy(dtype=np.float32)
    y = df[target_column].to_numpy(dtype=np.float32)

    cv = KFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=42)

    fold_rows = []
    print(f"starting {config.CV_FOLDS}-Fold CV for linear_boost (AdaBoost) on ENRICHED data...")
    for fold, (tr, te) in enumerate(cv.split(X), start=1):
        print(f" processing Fold {fold}/{config.CV_FOLDS}")

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr])
        X_te = scaler.transform(X[te])

        model = _make_adaboost_regressor(params)
        model.fit(X_tr, y[tr])

        pred = model.predict(X_te)

        fold_rows.append({
            "Fold": fold,
            "R2": float(r2_score(y[te], pred)),
            "MAE": float(mean_absolute_error(y[te], pred)),
            "RMSE": _rmse(y[te], pred),
        })

        del X_tr, X_te, pred
        gc.collect()

    fold_df = pd.DataFrame(fold_rows)
    summary = {
        "r2_mean": float(fold_df["R2"].mean()),
        "r2_std": float(fold_df["R2"].std(ddof=1)),
        "mae_mean": float(fold_df["MAE"].mean()),
        "mae_std": float(fold_df["MAE"].std(ddof=1)),
        "rmse_mean": float(fold_df["RMSE"].mean()),
        "rmse_std": float(fold_df["RMSE"].std(ddof=1)),
    }

    final_scaler = StandardScaler().fit(X)
    X_all = final_scaler.transform(X)

    final_model = _make_adaboost_regressor(params)
    final_model.fit(X_all, y)

    artifact = {
        "model_type": "linear_boost_adaboost",
        "tokenizer": tokenizer_name,
        "scope": language_scope,
        "target_column": target_column,
        "features": selected_features,
        "params": params,
        "x_scaler": final_scaler,
        "model": final_model,
    }
    with open(model_path, "wb") as f:
        pickle.dump(artifact, f)

    # -----------------------
    # Report
    # -----------------------
    with open(results_path, "w", encoding="utf-8") as f:
        f.write(get_environment_info())
        f.write("\n============================================================\n")
        f.write(f"LINEAR BOOST (AdaBoostRegressor + linear base) | tokenizer={tokenizer_name} | scope={language_scope}\n")
        f.write("============================================================\n\n")

        f.write("data source\n")
        f.write("------------------------------------------------------------\n")
        f.write(f"ENRICHED_INPUT_FILES: {len(config.ENRICHED_INPUT_FILES)} files\n")
        f.write(f"rows used: {len(df)}\n\n")

        f.write("features\n")
        f.write("------------------------------------------------------------\n")
        f.write(", ".join(selected_features) + "\n\n")

        f.write("hyperparameters\n")
        f.write("------------------------------------------------------------\n")
        f.write("base_estimator_params:\n")
        for k, v in params.get("base_estimator_params", {}).items():
            f.write(f"  {k}: {v}\n")
        f.write("adaboost_params:\n")
        for k, v in params.get("adaboost_params", {}).items():
            f.write(f"  {k}: {v}\n")
        f.write("\n")

        f.write(f"{config.CV_FOLDS}-fold CV metrics\n")
        f.write("------------------------------------------------------------\n")
        f.write(fold_df.to_string(index=False, formatters={
            "R2": "{:,.10f}".format,
            "MAE": "{:,.10f}".format,
            "RMSE": "{:,.10f}".format,
        }))
        f.write("\n\nsummary statistics:\n")
        f.write(f"  - R2:   {summary['r2_mean']:.10f} (std: {summary['r2_std']:.10f})\n")
        f.write(f"  - MAE:  {summary['mae_mean']:.10f} (std: {summary['mae_std']:.10f})\n")
        f.write(f"  - RMSE: {summary['rmse_mean']:.10f} (std: {summary['rmse_std']:.10f})\n\n")

        f.write("notes\n")
        f.write("------------------------------------------------------------\n")
        f.write("- This is an ensemble of linear weak learners (AdaBoost.R2).\n")
        f.write("- There is no single (a,b) closed-form like plain linear regression.\n\n")

        f.write(f"saved model: {model_path}\n")
        f.write(f"saved report: {results_path}\n")

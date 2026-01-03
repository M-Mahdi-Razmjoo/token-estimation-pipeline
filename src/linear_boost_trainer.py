import os
import pickle
import string
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import config
from data_loader import DataLoader
from utils import get_environment_info
from feature_engineering import get_code_markup_count, URL_RE, EMAIL_RE


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def _compute_linear_boost_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    content = df[config.CONTENT_COLUMN].astype(str).fillna('')

    df['char_count'] = content.str.len()
    words = content.str.split()
    df['word_count'] = words.str.len()

    safe_char_count = df['char_count'] + 1e-6
    safe_word_count = df['word_count'] + 1e-6

    df['avg_word_length'] = (df['char_count'] / safe_word_count).fillna(0.0)
    df['max_word_length'] = words.apply(lambda ws: max((len(w) for w in ws), default=0))

    df['code_markup_count'] = content.apply(get_code_markup_count)
    df['url_email_count'] = content.str.count(URL_RE) + content.str.count(EMAIL_RE)

    df['punctuation_proportion'] = content.apply(
        lambda x: sum(1 for ch in x if ch in string.punctuation)
    ) / safe_char_count

    return df


def _extract_gblinear_weights(model, feature_names: List[str]) -> Tuple[np.ndarray, float]:
    booster = model.get_booster()
    dump = booster.get_dump(with_stats=False)[0].splitlines()

    bias = None
    weights: Dict[int, float] = {}
    mode = None

    for line in dump:
        line = line.strip()
        if not line:
            continue
        if line.startswith("bias:"):
            mode = "bias"
            continue
        if line.startswith("weight:"):
            mode = "weight"
            continue

        if mode == "bias":
            try:
                bias = float(line)
            except ValueError:
                pass
            mode = None
            continue

        if mode == "weight":
            if ":" in line:
                a, b = line.split(":", 1)
                try:
                    idx = int(a)
                    val = float(b)
                    weights[idx] = val
                except ValueError:
                    continue

    if bias is None:
        bias = 0.0

    w = np.array([weights.get(i, 0.0) for i in range(len(feature_names))], dtype=float)
    return w, float(bias)


def _unscale_linear_model(w_scaled: np.ndarray, bias_scaled: float, scaler: StandardScaler) -> Tuple[np.ndarray, float]:
    mean = scaler.mean_
    scale = scaler.scale_
    a = w_scaled / scale
    b = bias_scaled - float(np.sum(w_scaled * mean / scale))
    return a, b


def train_linear_boost_model_pipeline(target_column: str, language_scope: str, tokenizer_name: str, selected_features: List[str] = None,):
    try:
        import xgboost as xgb
    except ImportError as e:
        raise ImportError("pip install xgboost") from e

    output_dir = os.path.join(config.RESULTS_PATH, tokenizer_name, language_scope)
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, "linear_boost_results.txt")
    model_path = os.path.join(output_dir, "linear_boost_model.pkl")

    if selected_features is None:
        selected_features = getattr(config, "LINEAR_BOOST_SELECTED_FEATURES", config.NONLINEAR_SELECTED_FEATURES)

    loader = DataLoader(config.ENRICHED_INPUT_FILES)
    columns_to_load = [config.CONTENT_COLUMN, config.LANGUAGE_COLUMN, target_column]
    all_chunks = [chunk for chunk in loader.load_data_chunks(columns=columns_to_load)]
    df = pd.concat(all_chunks, ignore_index=True).dropna(subset=[target_column])

    if language_scope == "english":
        df = df[df[config.LANGUAGE_COLUMN] == "English"].copy()
    df = df.reset_index(drop=True)

    df = _compute_linear_boost_features(df)

    df = df.dropna(subset=selected_features + [target_column]).reset_index(drop=True)

    X = df[selected_features].to_numpy(dtype=np.float32)
    y = df[target_column].to_numpy(dtype=np.float32)

    params = getattr(config, "LINEAR_BOOST_PARAMS", {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
    })

    kf = KFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=42)

    fold_rows = []
    for fold, (tr, te) in enumerate(kf.split(X), start=1):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr])
        X_te = scaler.transform(X[te])

        model = xgb.XGBRegressor(
            booster="gblinear",
            objective="reg:squarederror",
            n_jobs=-1,
            random_state=42,
            **params,
        )
        model.fit(X_tr, y[tr])

        pred = model.predict(X_te)
        fold_rows.append({
            "fold": fold,
            "mae": float(mean_absolute_error(y[te], pred)),
            "rmse": _rmse(y[te], pred),
            "r2": float(r2_score(y[te], pred)),
        })

    fold_df = pd.DataFrame(fold_rows)
    summary = {
        "mae_mean": float(fold_df["mae"].mean()),
        "mae_std": float(fold_df["mae"].std(ddof=1)),
        "rmse_mean": float(fold_df["rmse"].mean()),
        "rmse_std": float(fold_df["rmse"].std(ddof=1)),
        "r2_mean": float(fold_df["r2"].mean()),
        "r2_std": float(fold_df["r2"].std(ddof=1)),
    }

    final_scaler = StandardScaler().fit(X)
    X_all = final_scaler.transform(X)

    final_model = xgb.XGBRegressor(
        booster="gblinear",
        objective="reg:squarederror",
        n_jobs=-1,
        random_state=42,
        **params,
    )
    final_model.fit(X_all, y)

    w_scaled, bias_scaled = _extract_gblinear_weights(final_model, selected_features)
    a_unscaled, b_unscaled = _unscale_linear_model(w_scaled, bias_scaled, final_scaler)
    
    artifact = {
        "model_type": "linear_boost",
        "tokenizer": tokenizer_name,
        "scope": language_scope,
        "target_column": target_column,
        "features": selected_features,
        "params": params,
        "scaler": final_scaler,
        "model": final_model,
    }
    with open(model_path, "wb") as f:
        pickle.dump(artifact, f)

    with open(results_path, "w", encoding="utf-8") as f:
        f.write(get_environment_info())
        f.write("\n============================================================\n")
        f.write(f"LINEAR BOOSTING (XGBoost gblinear) | tokenizer={tokenizer_name} | scope={language_scope}\n")
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
        for k, v in params.items():
            f.write(f"{k}: {v}\n")
        f.write("\n")

        f.write(f"{config.CV_FOLDS}-fold CV metrics\n")
        f.write("------------------------------------------------------------\n")
        f.write(f"MAE:  {summary['mae_mean']:.6f}  (std: {summary['mae_std']:.6f})\n")
        f.write(f"RMSE: {summary['rmse_mean']:.6f} (std: {summary['rmse_std']:.6f})\n")
        f.write(f"R2:   {summary['r2_mean']:.6f}   (std: {summary['r2_std']:.6f})\n\n")

        f.write("final model (trained on all data)\n")
        f.write("------------------------------------------------------------\n")
        f.write("y = b + sum_i a_i * feature_i   (a_i in ORIGINAL feature scale)\n")
        f.write(f"b (intercept): {b_unscaled:.10f}\n")
        for name, coef in zip(selected_features, a_unscaled):
            f.write(f"{name}: {coef:.10f}\n")
        f.write("\n")

        f.write(f"saved model: {model_path}\n")

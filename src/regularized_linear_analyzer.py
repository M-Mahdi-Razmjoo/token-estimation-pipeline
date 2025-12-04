import os
import gc
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    mean_absolute_percentage_error,
    max_error,
    d2_absolute_error_score
)
import config
from utils import get_environment_info

def load_full_dataset_from_features(file_paths, features, target_col, scope):
    try:
        cols_to_load = list(set(features + [target_col, config.LANGUAGE_COLUMN]))
        list_of_dfs = [pd.read_parquet(f, columns=cols_to_load) for f in file_paths]
        df = pd.concat(list_of_dfs, ignore_index=True)
    except Exception as e:
        print(f"Error loading parquet files: {e}.")
        return None
    if scope == 'english':
        df = df[df[config.LANGUAGE_COLUMN] == 'English'].copy()
    df = df.dropna(subset=features + [target_col])
    if df.empty:
        return None
    return df.reset_index(drop=True)

def run_batch_regularized_benchmark():
    output_dir = "results/batch_linear_benchmark"
    os.makedirs(output_dir, exist_ok=True)
    output_log_file = os.path.join(output_dir, "batch_linear_models_report.txt")
    scopes_to_run = ['all', 'english']
    features_to_use = [
        'char_count', 'word_count', 'social_media_count', 'code_markup_count', 
        'max_word_length', 'language_count', 'whitespace_ratio', 'url_email_count', 
        'min_word_length', 'avg_word_length', 'alnum_special_ratio', 'stopword_proportion'
    ]
    models_config = [
        {'name': 'Ridge (alpha=1.0)',     'class': Ridge,      'params': {'alpha': 1.0, 'random_state': 42}},
        {'name': 'Ridge (alpha=10.0)',    'class': Ridge,      'params': {'alpha': 10.0, 'random_state': 42}},
        {'name': 'Lasso (alpha=0.01)',    'class': Lasso,      'params': {'alpha': 0.01, 'max_iter': 4000, 'tol': 1e-3, 'random_state': 42}},
        {'name': 'Lasso (alpha=0.1)',     'class': Lasso,      'params': {'alpha': 0.1, 'max_iter': 4000, 'tol': 1e-3, 'random_state': 42}},
        {'name': 'ElasticNet (a=0.01)',   'class': ElasticNet, 'params': {'alpha': 0.01, 'l1_ratio': 0.5, 'max_iter': 4000, 'tol': 1e-3, 'random_state': 42}},
    ]
    feature_files = [
        os.path.join(config.FEATURES_PATH, f"features_{os.path.basename(fname)}") 
        for fname in config.ENRICHED_INPUT_FILES
    ]
    with open(output_log_file, "w", encoding="utf-8") as f:
        def log(msg):
            print(msg)
            f.write(msg + "\n")
            f.flush()
        log(get_environment_info())
        log("-"*50)
        log("BATCH REGULARIZED LINEAR MODELS BENCHMARK (10-Fold CV)")
        log("-"*50)

        for scope in scopes_to_run:
            log(f"\n\n{'-'*80}\nLANGUAGE SCOPE: {scope.upper()}\n{'-'*80}")
            for tokenizer_name, target_col in config.TOKENIZER_COLUMNS.items():
                log(f"\nTarget: {tokenizer_name.upper()}")
                log("Loading full dataset from feature files")
                full_df = load_full_dataset_from_features(feature_files, features_to_use, target_col, scope)
                if full_df is None:
                    log("No data found for this configuration. Skipping.")
                    continue
                log(f"Full Dataset Loaded. Shape: {full_df.shape}")
                X_full = full_df[features_to_use].values
                y_full = full_df[target_col].values
                for model_conf in models_config:
                    model_name = model_conf['name']
                    log(f"\nEvaluating Model: {model_name}")
                    cv_strategy = KFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=42)
                    fold_scores = {
                        'r2': [], 'mae': [], 'mse': [], 'median_ae': [],
                        'mape': [], 'max_error': [], 'd2_abs': []
                    }
                    for fold, (train_idx, test_idx) in enumerate(cv_strategy.split(X_full)):
                        X_train, X_test = X_full[train_idx], X_full[test_idx]
                        y_train, y_test = y_full[train_idx], y_full[test_idx]
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_test_scaled = scaler.transform(X_test)
                        try:
                            model = model_conf['class'](**model_conf['params'])
                            model.fit(X_train_scaled, y_train)
                            y_pred = model.predict(X_test_scaled)
                            fold_scores['r2'].append(r2_score(y_test, y_pred))
                            fold_scores['mae'].append(mean_absolute_error(y_test, y_pred))
                            fold_scores['mse'].append(mean_squared_error(y_test, y_pred))
                            fold_scores['median_ae'].append(median_absolute_error(y_test, y_pred))
                            fold_scores['mape'].append(mean_absolute_percentage_error(y_test, y_pred))
                            fold_scores['max_error'].append(max_error(y_test, y_pred))
                            fold_scores['d2_abs'].append(d2_absolute_error_score(y_test, y_pred))
                        except Exception as e:
                            log(f"Fold {fold+1} failed: {e}")
                            for key in fold_scores: fold_scores[key].append(np.nan)
                    log(f"FINAL RESULTS for {model_name}:")
                    display_name_map = {
                        'r2': 'R2 Score', 'mae': 'Mean Absolute Error', 'mse': 'Mean Squared Error',
                        'median_ae': 'Median Absolute Error', 'mape': 'Mean Abs % Error',
                        'max_error': 'Max Error', 'd2_abs': 'D2 Absolute Score'
                    }
                    for metric_name, values in fold_scores.items():
                        if values and not all(np.isnan(values)):
                            mean_val = np.nanmean(values)
                            std_val = np.nanstd(values)
                            display_name = display_name_map.get(metric_name, metric_name.upper())
                            log(f"- {display_name:<22}: Mean={mean_val:<12.6f} (Std={std_val:.6f})")
                del full_df, X_full, y_full
                gc.collect()

    print(f"\n\n{'-'*50}\nBATCH BENCHMARK COMPLETE!\nDetailed report saved to: {output_log_file}\n{'-'*50}")

if __name__ == "__main__":
    run_batch_regularized_benchmark()
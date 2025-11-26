import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import Pipeline
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
from data_loader import DataLoader
from utils import get_environment_info

def run_univariate_polynomial_cv(feature_name: str, degree: int, target_column: str, tokenizer_name: str, language_scope: str = 'all'):
    feature_label = "Character Count" if feature_name == 'char_count' else "Word Count"
    print("\n"+"-"*50)
    print(f"Starting Run: Feature='{feature_label}', Degree={degree}, Tokenizer='{tokenizer_name}', Scope='{language_scope.upper()}'")
    print("-"*50)

    metrics = {
        'r2': [], 'mae': [], 'mse': [], 'median_ae': [],
        'mape': [], 'max_error': [], 'd2_abs': []
    }

    for i in range(len(config.ENRICHED_INPUT_FILES)):
        test_file = [config.ENRICHED_INPUT_FILES[i]]
        train_files = config.ENRICHED_INPUT_FILES[:i] + config.ENRICHED_INPUT_FILES[i+1:]
        
        pipeline = Pipeline([
            ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
            ('scaler', StandardScaler()),
            ('model', SGDRegressor(loss='huber', max_iter=20000, tol=1e-5, learning_rate='invscaling', eta0=0.01, random_state=42))
        ])

        print(f"  - Fold {i+1}/{len(config.ENRICHED_INPUT_FILES)}: Training on {len(train_files)} files...")
        train_loader = DataLoader(train_files)
        is_first_chunk = True
        for chunk in tqdm(train_loader.load_data_chunks(), desc=f"    Training"):
            df = chunk.copy()
            if language_scope.lower() == 'english':
                df = df[df.get('language') == 'English']

            df = df.dropna(subset=[config.CONTENT_COLUMN, target_column])
            if df.empty:
                continue
            
            if feature_name == 'char_count':
                df[feature_name] = df[config.CONTENT_COLUMN].astype(str).str.len()
            else:
                df[feature_name] = df[config.CONTENT_COLUMN].astype(str).str.count(r'\s+') + 1

            X_train_chunk = df[[feature_name]]
            y_train_chunk = df[target_column]

            if is_first_chunk:
                X_poly = pipeline.named_steps['poly'].fit_transform(X_train_chunk)
                X_scaled = pipeline.named_steps['scaler'].fit_transform(X_poly)
                pipeline.named_steps['model'].partial_fit(X_scaled, y_train_chunk)
                is_first_chunk = False
            else:
                X_poly = pipeline.named_steps['poly'].transform(X_train_chunk)
                X_scaled = pipeline.named_steps['scaler'].transform(X_poly)
                pipeline.named_steps['model'].partial_fit(X_scaled, y_train_chunk)

        print(f"  - Fold {i+1}: Evaluating on {os.path.basename(test_file[0])}...")
        test_loader = DataLoader(test_file)
        y_true_all, y_pred_all = [], []

        for chunk in tqdm(test_loader.load_data_chunks(), desc=f"    Evaluating"):
            df = chunk.copy()
            if language_scope.lower() == 'english':
                df = df[df.get('language') == 'English']

            df = df.dropna(subset=[config.CONTENT_COLUMN, target_column])
            if df.empty:
                continue

            if feature_name == 'char_count':
                df[feature_name] = df[config.CONTENT_COLUMN].astype(str).str.len()
            else:
                df[feature_name] = df[config.CONTENT_COLUMN].astype(str).str.count(r'\s+') + 1
            
            X_test_chunk = df[[feature_name]]
            y_true_chunk = df[target_column]
            
            y_pred_chunk = pipeline.predict(X_test_chunk)

            y_true_all.extend(y_true_chunk.values)
            y_pred_all.extend(y_pred_chunk)

        if not y_true_all:
            print(f"    - Warning: No test data found for Fold {i+1}. Skipping.")
            continue
            
        y_true, y_pred = np.array(y_true_all), np.array(y_pred_all)
        metrics['r2'].append(r2_score(y_true, y_pred))
        metrics['mae'].append(mean_absolute_error(y_true, y_pred))
        metrics['mse'].append(mean_squared_error(y_true, y_pred))
        metrics['median_ae'].append(median_absolute_error(y_true, y_pred))
        metrics['mape'].append(mean_absolute_percentage_error(y_true, y_pred))
        metrics['max_error'].append(max_error(y_true, y_pred))
        metrics['d2_abs'].append(d2_absolute_error_score(y_true, y_pred))
        
        print(f"    - Fold {i+1} Done. R2: {metrics['r2'][-1]:.6f}, MAE: {metrics['mae'][-1]:.4f}")

    print("\n" + "-"*80)
    display_name_map = {
        'r2': 'R2 Score', 'mae': 'Mean Absolute Error', 'mse': 'Mean Squared Error',
        'median_ae': 'Median Absolute Error', 'mape': 'Mean Abs % Error',
        'max_error': 'Max Error', 'd2_abs': 'D2 Absolute Score'
    }
    for m_name, values in metrics.items():
        if values:
            avg, std = np.mean(values), np.std(values)
            display_name = display_name_map.get(m_name, m_name)
            print(f"  - {display_name:<22}: Mean={avg:<12.6f} (Std={std:.6f})")

def main():
    parser = argparse.ArgumentParser(description="run univariate polynomial regression analysis.")
    parser.add_argument('--tokenizer', choices=config.TOKENIZER_COLUMNS.keys(), help="run for a specific tokenizer.")
    args = parser.parse_args()

    print(get_environment_info())

    feature_types = ['char_count', 'word_count']
    polynomial_degrees = [2, 3]
    
    if args.tokenizer:
        tokenizers_to_run = {args.tokenizer: config.TOKENIZER_COLUMNS[args.tokenizer]}
    else:
        tokenizers_to_run = config.TOKENIZER_COLUMNS

    language_scopes = ['all', 'english']
    
    total = len(feature_types) * len(polynomial_degrees) * len(tokenizers_to_run) * len(language_scopes)
    counter = 0

    for feat in feature_types:
        for degree in polynomial_degrees:
            for tokenizer_name, target_col in tokenizers_to_run.items():
                for scope in language_scopes:
                    counter += 1
                    run_univariate_polynomial_cv(
                        feature_name=feat,
                        degree=degree,
                        target_column=target_col,
                        tokenizer_name=tokenizer_name,
                        language_scope=scope
                    )

if __name__ == "__main__":
    main()
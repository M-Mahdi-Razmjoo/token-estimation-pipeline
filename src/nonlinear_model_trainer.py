import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
import config
from utils import get_environment_info

def train_nonlinear_model_pipeline(target_column: str, model_type: str, tokenizer_name: str, language_scope: str, selected_features: list):
    output_dir = os.path.join(config.RESULTS_PATH, tokenizer_name, language_scope)
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, f"{model_type}_results.txt")

    feature_files = [os.path.join(config.FEATURES_PATH, f"features_{fname}") for fname in config.FILE_NAMES]
    df = pd.concat([pd.read_parquet(f) for f in feature_files], ignore_index=True)
    
    if language_scope == 'english':
        df = df[df[config.LANGUAGE_COLUMN] == 'English'].copy()

    df = df.dropna(subset=[target_column] + selected_features)
    df = df.reset_index(drop=True)
    
    X_full = df[selected_features]
    y_full = df[target_column]

    model_map = {
        'mlp': (MLPRegressor(random_state=42, max_iter=500, early_stopping=True, verbose=True), config.MLP_PARAM_GRID),
        'rf': (RandomForestRegressor(random_state=42, verbose=1), config.RF_PARAM_GRID),
        'et': (ExtraTreesRegressor(random_state=42, verbose=1), config.ET_PARAM_GRID)
    }
    model_template, param_grid = model_map[model_type]
    pipeline = Pipeline([('scaler', StandardScaler()), ('model', model_template)])
    cv_strategy = KFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=42)
    scoring_metrics = {
        'r2': 'r2',
        'mae': 'neg_mean_absolute_error',
        'mse': 'neg_mean_squared_error'
    }

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv_strategy,
        scoring=scoring_metrics,
        refit='r2',
        n_jobs=-1,
        verbose=2,
        return_train_score=False
    )
    
    grid_search.fit(X_full, y_full)
    
    with open(output_file_path, 'w', encoding='utf-8') as f:
        f.write(get_environment_info())
        f.write(f"results for model '{model_type.upper()}' for tokenizer '{tokenizer_name}' and scope '{language_scope}'\n")
        f.write("-" * 60 + "\n")
        f.write("features used in the model\n")
        f.write(", ".join(selected_features) + "\n\n")

        f.write(f"full GridSearchCV results ({config.CV_FOLDS}-Fold Cross-Validation on raw data)\n")
        f.write("=" * 60 + "\n\n")
        
        results_df = pd.DataFrame(grid_search.cv_results_).sort_values(by='rank_test_r2')

        for index, row in results_df.iterrows():
            f.write(f"rank: {row['rank_test_r2']}\n")
            f.write(f"  parameters: {row['params']}\n")
            f.write("  ----------------------------------------\n")
            
            mean_r2 = row['mean_test_r2']
            std_r2 = row['std_test_r2']
            mean_mae = -row['mean_test_mae']
            std_mae = row['std_test_mae']
            mean_mse = -row['mean_test_mse']
            std_mse = row['std_test_mse']
            
            f.write(f"    - R2:  {mean_r2:.6f} (std: {std_r2:.6f})\n")
            f.write(f"    - MAE: {mean_mae:.4f} (std: {std_mae:.4f})\n")
            f.write(f"    - MSE: {mean_mse:.4f} (std: {std_mse:.4f})\n")
            f.write("\n  per fold :\n")
            r2_folds = [row[f'split{i}_test_r2'] for i in range(config.CV_FOLDS)]
            mae_folds = [-row[f'split{i}_test_mae'] for i in range(config.CV_FOLDS)]
            mse_folds = [-row[f'split{i}_test_mse'] for i in range(config.CV_FOLDS)]
            f.write("    fold |      R2      |     MAE      |      MSE\n")
            f.write("    -----|--------------|--------------|--------------\n")
            for i in range(config.CV_FOLDS):
                f.write(f"    {i+1:02d}   |  {r2_folds[i]:.6f}  |  {mae_folds[i]:10.4f}  |  {mse_folds[i]:12.4f}\n")

            f.write("\n" + "=" * 60 + "\n\n")

        f.write("best model (based on Mean R2):\n")
        f.write(f"  best parameters: {grid_search.best_params_}\n")
        f.write(f"  best Mean R2: {grid_search.best_score_:.6f}\n")

    print(f"final results report for {model_type.upper()} saved successfully to {output_file_path}")
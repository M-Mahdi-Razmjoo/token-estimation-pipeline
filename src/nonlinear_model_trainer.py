import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, cross_validate, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
import config

def train_nonlinear_model_pipeline(target_column: str, model_type: str, tokenizer_name: str, language_scope: str, selected_features: list):
    output_dir = os.path.join(config.RESULTS_PATH, tokenizer_name, language_scope)
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, f"{model_type}_results.txt")

    feature_files = [os.path.join(config.FEATURES_PATH, f"features_{fname}") for fname in config.FILE_NAMES]
    all_chunks = [pd.read_parquet(f) for f in feature_files]
    df = pd.concat(all_chunks, ignore_index=True)
    
    if language_scope == 'english':
        df = df[df[config.LANGUAGE_COLUMN] == 'English'].copy()

    df = df.dropna(subset=[target_column] + selected_features)
    X = df[selected_features]
    y = df[target_column]

    model_map = {
        'mlp': (MLPRegressor(random_state=42, max_iter=500, early_stopping=True, verbose=True), config.MLP_PARAM_GRID),
        'rf': (RandomForestRegressor(random_state=42, verbose=1), config.RF_PARAM_GRID),
        'et': (ExtraTreesRegressor(random_state=42, verbose=1), config.ET_PARAM_GRID)
    }
    model, param_grid = model_map[model_type]
    
    pipeline = Pipeline([('scaler', StandardScaler()), ('model', model)])
    scoring_metrics = {
        'r2': 'r2',
        'mae': 'neg_mean_absolute_error',
        'mse': 'neg_mean_squared_error'
    }
    grid_search = GridSearchCV(
        pipeline, 
        param_grid, 
        cv=5, 
        scoring=scoring_metrics, 
        refit='r2',
        n_jobs=-1, 
        verbose=2
    )
    grid_search.fit(X, y)

    best_model_pipeline = grid_search.best_estimator_
    final_scoring_metrics = ['r2', 'neg_mean_absolute_error', 'neg_mean_squared_error']
    scores = cross_validate(best_model_pipeline, X, y, cv=cv, scoring=final_scoring_metrics, n_jobs=-1)

    avg_r2 = np.mean(scores['test_r2'])
    avg_mae = -np.mean(scores['test_neg_mean_absolute_error'])
    avg_mse = -np.mean(scores['test_neg_mean_squared_error'])

    with open(output_file_path, 'w', encoding='utf-8') as f:
        f.write(f"results for model '{model_type.upper()}' for tokenizer '{tokenizer_name}' and scope '{language_scope}'\n")
        f.write("-"*60)
        f.write("features used in the model\n")
        f.write(", ".join(selected_features) + "\n")
        f.write("hyperparameter search results (GridSearchCV)\n")
        f.write(f"best found parameters (based on R2):\n{grid_search.best_params_}\n")
        f.write(f"best R2 score during search: {grid_search.best_score_:.6f}\n")
        f.write("full hyperparameter search results (sorted by best R2 score)\n")
        f.write("-" *60)
        
        cv_results_df = pd.DataFrame(grid_search.cv_results_)
        cv_results_df['mean_test_mae'] = -cv_results_df['mean_test_mae']
        cv_results_df['mean_test_mse'] = -cv_results_df['mean_test_mse']
        relevant_columns = [
            'rank_test_r2', 
            'mean_test_r2', 
            'mean_test_mae', 
            'mean_test_mse', 
            'params'
        ]
        sorted_results_df = cv_results_df[relevant_columns].sort_values(by='rank_test_r2')
        sorted_results_df = sorted_results_df.rename(columns={
            'rank_test_r2': 'Rank',
            'mean_test_r2': 'Mean R2',
            'mean_test_mae': 'Mean MAE',
            'mean_test_mse': 'Mean MSE',
            'params': 'Parameters'
        })
        f.write(sorted_results_df.to_string(index=False))
        f.write("\n")
        f.write(f"final evaluation of the best model with {config.CV_FOLDS}-Fold Cross-Validation\n")
        f.write(f"Average R2: {avg_r2:.6f}\n")
        f.write(f"Average MAE: {avg_mae:.4f}\n")
        f.write(f"Average MSE: {avg_mse:.4f}\n")

    print(f"final results report for {model_type.upper()} saved successfully to {output_file_path}")
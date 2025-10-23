import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.base import clone
import gc
import config
from data_loader import get_filtered_indices
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
    
    X_full = df[selected_features]
    y_full = df[target_column]

    model_map = {
        'mlp': (MLPRegressor(random_state=42, max_iter=500, early_stopping=True, verbose=True), config.MLP_PARAM_GRID),
        'rf': (RandomForestRegressor(random_state=42, verbose=1), config.RF_PARAM_GRID),
        'et': (ExtraTreesRegressor(random_state=42, verbose=1), config.ET_PARAM_GRID)
    }
    model_template, param_grid = model_map[model_type]
    
    pipeline = Pipeline([('scaler', StandardScaler()), ('model', model_template)])
    
    print("starting GridSearchCV")
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=2)
    grid_search.fit(X_full, y_full)
    
    best_model_pipeline = grid_search.best_estimator_
    
    cv = KFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=42)
    r2_scores, mae_scores, mse_scores = [], [], []

    print(f"\nstarting {config.CV_FOLDS}-Fold CV for final evaluation")
    for fold, (train_idx_initial, test_idx) in enumerate(cv.split(df)):
        print(f"  - processing Fold {fold+1}/{config.CV_FOLDS}")
        
        train_idx_filtered = get_filtered_indices(df, train_idx_initial, target_column)
        X_train = df.loc[train_idx_filtered, selected_features]
        y_train = df.loc[train_idx_filtered, target_column]
        X_test = df.loc[test_idx, selected_features]
        y_test = df.loc[test_idx, target_column]
        model = clone(best_model_pipeline)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2_scores.append(r2_score(y_test, y_pred))
        mae_scores.append(mean_absolute_error(y_test, y_pred))
        mse_scores.append(mean_squared_error(y_test, y_pred))
        del X_train, y_train, X_test, y_test
        gc.collect()

    avg_r2 = np.mean(r2_scores)
    avg_mae = np.mean(mae_scores)
    avg_mse = np.mean(mse_scores)

    with open(output_file_path, 'w', encoding='utf-8') as f:
        f.write(get_environment_info())
        f.write(f"results for model '{model_type.upper()}' for tokenizer '{tokenizer_name}' and scope '{language_scope}'\n")
        f.write("-" * 60 + "\n")
        f.write("features used in the model\n")
        f.write(", ".join(selected_features) + "\n\n")
        f.write("hyperparameter search results (GridSearchCV on full data)\n")
        f.write(f"best found parameters:\n{grid_search.best_params_}\n")
        f.write(f"best R2 score during search: {grid_search.best_score_:.6f}\n\n")
        f.write(f"final evaluation of the best model with {config.CV_FOLDS}-Fold Cross-Validation\n")
        f.write(f"Average R2: {avg_r2:.17f}\n")
        f.write(f"Average MAE: {avg_mae:.17f}\n")
        f.write(f"Average MSE: {avg_mse:.17f}\n")

    print(f"final results report for {model_type.upper()} saved successfully to {output_file_path}")
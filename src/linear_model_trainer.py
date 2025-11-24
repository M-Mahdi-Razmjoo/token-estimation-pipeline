import os
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
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
from sklearn.base import clone
import gc
import config
from data_loader import DataLoader
from utils import get_environment_info

def train_linear_model_pipeline(target_column: str, language_scope: str, tokenizer_name: str):
    output_dir = os.path.join(config.RESULTS_PATH, tokenizer_name, language_scope)
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, "linear_results.txt")

    loader = DataLoader(config.INPUT_FILES)
    columns_to_load = [config.CONTENT_COLUMN, config.LANGUAGE_COLUMN] + list(config.TOKENIZER_COLUMNS.values())
    all_chunks = [chunk for chunk in loader.load_data_chunks(columns=columns_to_load)]
    full_df = pd.concat(all_chunks, ignore_index=True).dropna(subset=[target_column])

    if language_scope == 'english':
        full_df = full_df[full_df[config.LANGUAGE_COLUMN] == 'English'].copy()
    full_df = full_df.reset_index(drop=True)
    
    full_df['word_count'] = full_df[config.CONTENT_COLUMN].astype(str).str.count(r'\s+') + 1
    
    pipeline_template = TransformedTargetRegressor(
        regressor=Pipeline([
            ('scaler', StandardScaler()), 
            ('model', SGDRegressor(loss='huber', random_state=42))
        ]),
        transformer=StandardScaler()
    )

    cv = KFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=42)    
    scores = {
        'r2': [], 'mae': [], 'mse': [], 'median_ae': [],
        'mape': [], 'max_error': [], 'd2_abs': []
    }

    print(f"starting {config.CV_FOLDS}-Fold CV for linear model on raw data...")
    for fold, (train_idx, test_idx) in enumerate(cv.split(full_df)):
        print(f" processing Fold {fold+1}/{config.CV_FOLDS}")
        X_train = full_df.loc[train_idx, ['word_count']]
        y_train = full_df.loc[train_idx, target_column]
        
        X_test = full_df.loc[test_idx, ['word_count']]
        y_test = full_df.loc[test_idx, target_column]

        model = clone(pipeline_template)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        scores['r2'].append(r2_score(y_test, y_pred))
        scores['mae'].append(mean_absolute_error(y_test, y_pred))
        scores['mse'].append(mean_squared_error(y_test, y_pred))
        scores['median_ae'].append(median_absolute_error(y_test, y_pred))
        scores['mape'].append(mean_absolute_percentage_error(y_test, y_pred))
        scores['max_error'].append(max_error(y_test, y_pred))
        scores['d2_abs'].append(d2_absolute_error_score(y_test, y_pred))

        del X_train, y_train, X_test, y_test
        gc.collect()
    
    summary_stats = {key: (np.mean(value), np.std(value)) for key, value in scores.items()}

    X_final_train = full_df[['word_count']]
    y_final_train = full_df[target_column]
    
    final_model = clone(pipeline_template)
    final_model.fit(X_final_train, y_final_train)

    final_y_scaler = final_model.transformer_
    final_pipeline = final_model.regressor_
    final_x_scaler = final_pipeline.named_steps['scaler']
    final_model_reg = final_pipeline.named_steps['model']
    
    scaled_a = final_model_reg.coef_[0]
    scaled_b = final_model_reg.intercept_[0]
    x_mean, x_std = final_x_scaler.mean_[0], final_x_scaler.scale_[0]
    y_mean, y_std = final_y_scaler.mean_[0], final_y_scaler.scale_[0]

    final_a = scaled_a * (y_std / x_std)
    final_b = y_mean - final_a * x_mean + (scaled_b * y_std)

    with open(output_file_path, 'w', encoding='utf-8') as f:
        f.write(get_environment_info())
        f.write(f"linear model results for tokenizer '{tokenizer_name}' and scope '{language_scope}'\n")
        f.write("-" * 60 + "\n")
        f.write(f"evaluation details ({config.CV_FOLDS}-Fold cross-validation on raw data)\n")
        f.write("-" * 60 + "\n")
        
        f.write("Per-Fold Scores:\n")
        fold_df = pd.DataFrame({
            'Fold': [i + 1 for i in range(config.CV_FOLDS)],
            'R2': scores['r2'], 'MAE': scores['mae'], 'MSE': scores['mse'],
            'MedianAE': scores['median_ae'], 'MAPE': scores['mape'],
            'MaxError': scores['max_error'], 'D2Abs': scores['d2_abs']
        })
        f.write(fold_df.to_string(index=False, formatters={
            'R2': '{:,.17f}'.format, 'MAE': '{:,.17f}'.format, 'MSE': '{:,.17f}'.format,
            'MedianAE': '{:,.17f}'.format, 'MAPE': '{:,.17f}'.format,
            'MaxError': '{:,.17f}'.format, 'D2Abs': '{:,.17f}'.format
        }))
        
        f.write("\n\nsummary statistics:\n")
        f.write(f"  - R2:               {summary_stats['r2'][0]:.17f} (std: {summary_stats['r2'][1]:.17f})\n")
        f.write(f"  - MAE:              {summary_stats['mae'][0]:.17f} (std: {summary_stats['mae'][1]:.17f})\n")
        f.write(f"  - MSE:              {summary_stats['mse'][0]:.17f} (std: {summary_stats['mse'][1]:.17f})\n")
        f.write(f"  - Median Abs Error: {summary_stats['median_ae'][0]:.17f} (std: {summary_stats['median_ae'][1]:.17f})\n")
        f.write(f"  - MAPE:             {summary_stats['mape'][0]:.17f} (std: {summary_stats['mape'][1]:.17f})\n")
        f.write(f"  - Max Error:        {summary_stats['max_error'][0]:.17f} (std: {summary_stats['max_error'][1]:.17f})\n")
        f.write(f"  - D2 Abs Score:     {summary_stats['d2_abs'][0]:.17f} (std: {summary_stats['d2_abs'][1]:.17f})\n\n")

        f.write("final model parameters (trained on all raw data)\n")
        f.write("------------------------------------------------------------\n")
        f.write("unscaled parameters (y = a * word_count + b):\n")
        f.write(f"  Coefficient (a): {final_a:.17f}\n")
        f.write(f"  Intercept (b): {final_b:.17f}\n\n")

        f.write("scaling & scaled parameters (for reproducibility):\n")
        f.write(f"  Input Scaler (X=word_count) Mean:             {x_mean:.17f}\n")
        f.write(f"  Input Scaler (X=word_count) Scale (Std Dev):  {x_std:.17f}\n")
        f.write(f"  Output Scaler (y=target) Mean:            {y_mean:.17f}\n")
        f.write(f"  Output Scaler (y=target) Scale (Std Dev): {y_std:.17f}\n")
        f.write(f"  Model Coefficient on Scaled Data (scaled_a):  {scaled_a:.17f}\n")
        f.write(f"  Model Intercept on Scaled Data (scaled_b):  {scaled_b:.17f}\n")
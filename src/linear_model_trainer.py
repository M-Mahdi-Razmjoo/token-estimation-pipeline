import os
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import cross_validate, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
import config
from data_loader import DataLoader

def train_linear_model_pipeline(target_column: str, language_scope: str, tokenizer_name: str):
    output_dir = os.path.join(config.RESULTS_PATH, tokenizer_name, language_scope)
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, "linear_results.txt")
    loader = DataLoader(config.FILTERED_INPUT_FILES)
    columns_to_load = [config.CONTENT_COLUMN, config.LANGUAGE_COLUMN, target_column]
    all_chunks = [chunk for chunk in loader.load_data_chunks(columns=columns_to_load)]
    full_df = pd.concat(all_chunks, ignore_index=True).dropna(subset=[target_column])

    if language_scope == 'english':
        full_df = full_df[full_df[config.LANGUAGE_COLUMN] == 'English'].copy()

    full_df['char_count'] = full_df[config.CONTENT_COLUMN].astype(str).str.len()
    
    X = full_df[['char_count']]
    y = full_df[target_column]
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()), 
        ('model', SGDRegressor(loss='huber', random_state=42))
    ])
    model_with_scaled_target = TransformedTargetRegressor(regressor=pipeline, transformer=StandardScaler())

    cv = KFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=42)
    scoring_metrics = ['r2', 'neg_mean_absolute_error', 'neg_mean_squared_error']
    scores = cross_validate(model_with_scaled_target, X, y, cv=cv, scoring=scoring_metrics, n_jobs=-1)
    
    avg_r2 = np.mean(scores['test_r2'])
    avg_mae = -np.mean(scores['test_neg_mean_absolute_error'])
    avg_mse = -np.mean(scores['test_neg_mean_squared_error'])
    
    model_with_scaled_target.fit(X, y)
    final_y_scaler = model_with_scaled_target.transformer_
    final_pipeline = model_with_scaled_target.regressor_
    final_x_scaler = final_pipeline.named_steps['scaler']
    final_model = final_pipeline.named_steps['model']
    
    scaled_a = final_model.coef_[0]
    scaled_b = final_model.intercept_[0]
    x_mean, x_std = final_x_scaler.mean_[0], final_x_scaler.scale_[0]
    y_mean, y_std = final_y_scaler.mean_[0], final_y_scaler.scale_[0]

    final_a = scaled_a * (y_std / x_std)
    final_b = y_mean - final_a * x_mean + (scaled_b * y_std)
    
    with open(output_file_path, 'w', encoding='utf-8') as f:
        f.write(f"linear model results for tokenizer '{tokenizer_name}' and scope '{language_scope}'\n")
        f.write("-"*60 +"\n")
        f.write(f"{config.CV_FOLDS}-Fold Cross-Validation results\n")
        f.write(f"Average R2: {avg_r2:.6f}\n")
        f.write(f"Average MAE: {avg_mae:.4f}\n")
        f.write(f"Average MSE: {avg_mse:.4f}\n")
        f.write("final model parameters (y = a * char_count + b)\n")
        f.write(f"coefficient a: {final_a:.12f}\n")
        f.write(f"intercept b: {final_b:.12f}\n")

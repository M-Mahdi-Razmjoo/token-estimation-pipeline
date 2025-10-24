import os
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.base import clone
import gc
import config
from data_loader import DataLoader, get_filtered_indices
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
    
    full_df['char_count'] = full_df[config.CONTENT_COLUMN].astype(str).str.len()
    
    pipeline_template = TransformedTargetRegressor(
        regressor=Pipeline([
            ('scaler', StandardScaler()), 
            ('model', SGDRegressor(loss='huber', random_state=42))
        ]),
        transformer=StandardScaler()
    )

    cv = KFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=42)
    r2_scores, mae_scores, mse_scores = [], [], []

    print(f"starting {config.CV_FOLDS}-fold CV for linear model")
    for fold, (train_idx_initial, test_idx) in enumerate(cv.split(full_df)):
        print(f"  - processing fold {fold+1}/{config.CV_FOLDS}")
        
        train_idx_filtered = get_filtered_indices(full_df, train_idx_initial, target_column)

        X_train = full_df.loc[train_idx_filtered, ['char_count']]
        y_train = full_df.loc[train_idx_filtered, target_column]
        
        X_test = full_df.loc[test_idx, ['char_count']]
        y_test = full_df.loc[test_idx, target_column]

        model = clone(pipeline_template)
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

    final_train_indices = get_filtered_indices(full_df, full_df.index, target_column)
    X_final_train = full_df.loc[final_train_indices, ['char_count']]
    y_final_train = full_df.loc[final_train_indices, target_column]
    
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
        f.write("-"*60 +"\n")
        f.write(f"{config.CV_FOLDS}-fold cross-validation results\n")
        f.write(f"Average R2: {avg_r2:.17f}\n")
        f.write(f"Average MAE: {avg_mae:.17f}\n")
        f.write(f"Average MSE: {avg_mse:.17f}\n\n")
        f.write("------------------------------------------------------------\n")
        f.write(f"  Coefficient (a): {final_a:.17f}\n")
        f.write(f"  Intercept (b): {final_b:.17f}\n\n")
        f.write("scaled Parameters:\n")
        f.write(f"  Input Scaler Mean:             {x_mean:.17f}\n")
        f.write(f"  Input Scaler Scale:  {x_std:.17f}\n")
        f.write(f"  Output Scaler Mean:            {y_mean:.17f}\n")
        f.write(f"  Output Scaler Scale: {y_std:.17f}\n")
        f.write(f"  Model Coefficient on Scaled Data (scaled_a):  {scaled_a:.17f}\n")
        f.write(f"  Model Intercept on Scaled Data (scaled_b):  {scaled_b:.17f}\n")
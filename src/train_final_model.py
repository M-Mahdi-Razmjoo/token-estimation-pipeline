import os
import argparse
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.compose import TransformedTargetRegressor
import pickle
import json

try:
    from pymilo import Export, Import
    PYMILO_AVAILABLE = True
    print("Pymilo library found.")
except ImportError:
    PYMILO_AVAILABLE = False
    print("Pymilo library not found.")

import config
from data_loader import DataLoader
from final_config import BEST_HYPERPARAMS

def main():
    parser = argparse.ArgumentParser(description="Train and save a final model with the best hyperparameters.")
    parser.add_argument('--tokenizer', required=True, choices=config.TOKENIZER_COLUMNS.keys())
    parser.add_argument('--model-type', required=True, choices=['linear', 'mlp', 'rf', 'et'])
    parser.add_argument('--scope', required=True, choices=config.LANGUAGE_SCOPES)
    args = parser.parse_args()

    print(f"--- Training Final Model ---")
    print(f"Model Type: {args.model_type.upper()}, Tokenizer: {args.tokenizer}, Scope: {args.scope}")
    target_column = config.TOKENIZER_COLUMNS[args.tokenizer]
    is_linear = args.model_type == 'linear'

    if is_linear:
        print("Linear model selected. Loading data from ENRICHED files ('data_enriched/')...")
        loader = DataLoader(config.ENRICHED_INPUT_FILES)
        columns_to_load = [config.CONTENT_COLUMN, config.LANGUAGE_COLUMN] + list(config.TOKENIZER_COLUMNS.values())
        df = pd.concat(loader.load_data_chunks(columns=columns_to_load), ignore_index=True)
    else:
        print("Non-linear model selected. Loading data from FEATURE files ('features/')...")
        feature_files = [
            os.path.join(config.FEATURES_PATH, f"features_{os.path.basename(fname)}") 
            for fname in config.ENRICHED_INPUT_FILES
        ]
        df = pd.concat([pd.read_parquet(f) for f in feature_files], ignore_index=True)

    if args.scope == 'english':
        df = df[df[config.LANGUAGE_COLUMN] == 'English'].copy()
    df = df.dropna(subset=[target_column])
    df = df.reset_index(drop=True)

    print(f"Loaded {len(df)} rows for final training.")

    try:
        best_params = BEST_HYPERPARAMS[args.model_type][args.tokenizer][args.scope]
        print(f"Using best hyperparameters: {best_params}")
    except KeyError:
        print(f"ERROR: Best hyperparameters not found in final_config.py for {args.model_type}/{args.tokenizer}/{args.scope}")
        exit(1)

    if is_linear:
        df['char_count'] = df[config.CONTENT_COLUMN].astype(str).str.len()
        X = df[['char_count']]
        y = df[target_column]
        
        final_model_pipeline = TransformedTargetRegressor(
            regressor=Pipeline([
                ('scaler', StandardScaler()), 
                ('model', SGDRegressor(loss='huber', random_state=42))
            ]),
            transformer=StandardScaler()
        )
    else:
        X = df[config.NONLINEAR_SELECTED_FEATURES]
        y = df[target_column]
        
        model_map = {
            'mlp': MLPRegressor(random_state=42, max_iter=500, early_stopping=False, verbose=True),
            'rf': RandomForestRegressor(random_state=42, verbose=1),
            'et': ExtraTreesRegressor(random_state=42, verbose=1)
        }
        model_template = model_map[args.model_type]
        
        model_params = {key.replace('model__', ''): value for key, value in best_params.items()}
        model_template.set_params(**model_params)
        
        final_model_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model_template)
        ])

    print("\nTraining final model on the entire dataset...")
    final_model_pipeline.fit(X, y)
    print("Training complete.")

    output_dir = os.path.join("final_models", args.tokenizer, args.scope)
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = f"{args.model_type}_model"
    pymilo_path = os.path.join(output_dir, f"{base_name}.pymilo.json")
    pickle_path = os.path.join(output_dir, f"{base_name}.pickle.pkl")

    if PYMILO_AVAILABLE:
        try:
            print(f"\nAttempting to export model using Pymilo to: {pymilo_path}")
            exporter = Export(final_model_pipeline)
            exporter.save(pymilo_path)
            print("[SUCCESS] Pymilo export successful!")
        except Exception as e:
            print(f"[WARNING] Pymilo export failed. Error: {e}")
    else:
        print("\nSkipping Pymilo export: library not available.")

    try:
        print(f"\nAttempting to export model using standard pickle to: {pickle_path}")
        with open(pickle_path, 'wb') as f:
            pickle.dump(final_model_pipeline, f)
        print("[SUCCESS] Standard pickle export successful!")
    except Exception as e:
        print(f"[WARNING] Standard pickle export failed. Error: {e}")

if __name__ == "__main__":
    main()
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
try:
    from pymilo.pymilo_obj import Pymilo
    PYMILO_AVAILABLE = True
except ImportError:
    PYMILO_AVAILABLE = False
    print("Pymilo library not found. falling back to standard pickle for model serialization.")
import config
from data_loader import DataLoader
from final_config import BEST_HYPERPARAMS

def main():
    parser = argparse.ArgumentParser(description="train and save a final model with the best hyperparameters.")
    parser.add_argument('--tokenizer', required=True, choices=config.TOKENIZER_COLUMNS.keys())
    parser.add_argument('--model-type', required=True, choices=['linear', 'mlp', 'rf', 'et'])
    parser.add_argument('--scope', required=True, choices=config.LANGUAGE_SCOPES)
    args = parser.parse_args()
    print(f"training final model")
    print(f"Model Type: {args.model_type.upper()}, Tokenizer: {args.tokenizer}, Scope: {args.scope}")
    target_column = config.TOKENIZER_COLUMNS[args.tokenizer]
    is_linear = args.model_type == 'linear'

    if is_linear:
        loader = DataLoader(config.ENRICHED_INPUT_FILES)
        columns_to_load = [config.CONTENT_COLUMN, config.LANGUAGE_COLUMN] + list(config.TOKENIZER_COLUMNS.values())
        df = pd.concat(loader.load_data_chunks(columns=columns_to_load), ignore_index=True)
    else:
        feature_files = [
            os.path.join(config.FEATURES_PATH, f"features_{os.path.basename(fname)}") 
            for fname in config.ENRICHED_INPUT_FILES
        ]
        df = pd.concat([pd.read_parquet(f) for f in feature_files], ignore_index=True)

    if args.scope == 'english':
        df = df[df[config.LANGUAGE_COLUMN] == 'English'].copy()
    df = df.dropna(subset=[target_column])
    df = df.reset_index(drop=True)

    try:
        best_params = BEST_HYPERPARAMS[args.model_type][args.tokenizer][args.scope]
        print(f"using hyperparameters: {best_params}")
    except KeyError:
        print(f"hyperparameters not found in final_config.py for {args.model_type}/{args.tokenizer}/{args.scope}")
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

    final_model_pipeline.fit(X, y)
    print("training complete.")

    output_dir = os.path.join("final_models", args.tokenizer, args.scope)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{args.model_type}_model.pkl")
    if PYMILO_AVAILABLE:
        try:
            print(f"attempting to export model using Pymilo to: {output_path}")
            pymilo_exporter = Pymilo(final_model_pipeline)
            pymilo_exporter.export(output_path)
            print("Pymilo export successful")
        except Exception as e:
            print(f"pymilo export failed. error: {e}")
            print("falling back to standard pickle serialization.")
            try:
                with open(output_path, 'wb') as f:
                    pickle.dump(final_model_pipeline, f)
                print(f"standard pickle export successful to: {output_path}")
            except Exception as pickle_e:
                print(f"standard pickle export also failed. error: {pickle_e}")
                print("model could not be saved.")
    else:
        try:
            print(f"exporting model using standard pickle to: {output_path}")
            with open(output_path, 'wb') as f:
                pickle.dump(final_model_pipeline, f)
            print("standard pickle export successful")
        except Exception as pickle_e:
            print(f"standard pickle export failed. error: {pickle_e}")
            print("model could not be saved.")

if __name__ == "__main__":
    main()
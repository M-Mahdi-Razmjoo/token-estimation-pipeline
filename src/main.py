import argparse
import config
from linear_model_trainer import train_linear_model_pipeline
from nonlinear_model_trainer import train_nonlinear_model_pipeline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer', required=True, choices=config.TOKENIZER_COLUMNS.keys())
    parser.add_argument('--model-type', required=True, choices=['linear', 'mlp', 'rf', 'et'])
    parser.add_argument('--scope', required=True, choices=config.LANGUAGE_SCOPES)
    args = parser.parse_args()
    target_column = config.TOKENIZER_COLUMNS[args.tokenizer]
    if args.model_type == 'linear':
        train_linear_model_pipeline(
            target_column=target_column, 
            language_scope=args.scope,
            tokenizer_name=args.tokenizer
        )
    else:
        selected_features = config.NONLINEAR_SELECTED_FEATURES
        train_nonlinear_model_pipeline(
            target_column=target_column, 
            model_type=args.model_type,
            tokenizer_name=args.tokenizer,
            language_scope=args.scope,
            selected_features=selected_features
        )

if __name__ == "__main__":
    main()

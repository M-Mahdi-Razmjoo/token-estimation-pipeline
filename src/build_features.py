import os
import pandas as pd
from tqdm import tqdm
import config
from data_loader import DataLoader
from feature_engineering import extract_all_features

def main():
    os.makedirs(config.FEATURES_PATH, exist_ok=True)
    columns_to_load = [config.CONTENT_COLUMN, config.LANGUAGE_COLUMN] + list(config.TOKENIZER_COLUMNS.values())
    for input_file in tqdm(config.INPUT_FILES, desc="processing input files"):
        base_filename = os.path.basename(input_file)
        output_filename = f"features_{base_filename}"
        output_path = os.path.join(config.FEATURES_PATH, output_filename)
        loader = DataLoader([input_file])
        df = pd.concat(loader.load_data_chunks(columns=columns_to_load), ignore_index=True)
        featured_df = extract_all_features(df)
        featured_df.to_parquet(output_path, index=False)

if __name__ == "__main__":
    main()
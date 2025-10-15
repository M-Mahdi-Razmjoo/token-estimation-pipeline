import os
import pandas as pd
from tqdm import tqdm
import config
from data_loader import DataLoader

def main():
    FILTER_TARGET_COLUMN = config.TOKENIZER_COLUMNS['cl100k'] 

    os.makedirs(config.FILTERED_PATH, exist_ok=True)
    full_loader = DataLoader(config.INPUT_FILES)
    full_loader._calculate_iqr_bounds(iqr_multiplier=1.5, target_column=FILTER_TARGET_COLUMN)

    for i, input_file in enumerate(tqdm(config.INPUT_FILES, desc="filtering and saving files")):
        output_file = config.FILTERED_INPUT_FILES[i]
        loader = DataLoader([input_file])
        loader.bounds = full_loader.bounds
        filtered_chunks = list(loader.load_preprocessed_data(
            iqr_multiplier=1.5,
            target_column=FILTER_TARGET_COLUMN
        ))
        if not filtered_chunks:
            print(f"warning: no data left for {os.path.basename(input_file)} after filtering. creating empty file.")
            pd.DataFrame().to_parquet(output_file, index=False)
            continue
        filtered_df = pd.concat(filtered_chunks, ignore_index=True)
        filtered_df.to_parquet(output_file, index=False)
        print(f"successfully saved filtered data to {output_file}")

if __name__ == "__main__":
    main()
# In src/data_loader.py

import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm
import numpy as np
import os
import config

def get_filtered_indices(df: pd.DataFrame, indices_to_consider, target_column: str, iqr_multiplier: float = 1.5):
    if len(indices_to_consider) == 0:
        return np.array([])
    subset_df = df.loc[indices_to_consider]
    word_counts = subset_df[config.CONTENT_COLUMN].astype(str).str.count(r'\s+') + 1
    valid_mask = word_counts > 0
    if not valid_mask.any():
        return np.array([])
    ratios = subset_df.loc[valid_mask, target_column] / word_counts[valid_mask]
    Q1 = ratios.quantile(0.25)
    Q3 = ratios.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - (iqr_multiplier * IQR)
    upper_bound = Q3 + (iqr_multiplier * IQR)
    final_mask = ratios.between(lower_bound, upper_bound)
    return subset_df.index[valid_mask][final_mask]

class DataLoader:
    def __init__(self, file_paths: list):
        self.file_paths = file_paths

    def load_data_chunks(self, columns: list = None, chunk_size: int = 8192):
        for file_path in self.file_paths:
            parquet_file = pq.ParquetFile(file_path)
            desc = os.path.basename(file_path)
            for batch in tqdm(parquet_file.iter_batches(batch_size=chunk_size, columns=columns), desc=f"loading raw from {desc}"):
                yield batch.to_pandas()
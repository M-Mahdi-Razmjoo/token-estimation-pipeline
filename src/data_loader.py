import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm
import numpy as np
import os

class DataLoader:
    def __init__(self, file_paths: list):
        self.file_paths = file_paths
        self.bounds = {}

    def _calculate_iqr_bounds(self, iqr_multiplier: float, language: str = 'all', target_column: str = 'tiktoken_r50k_base_len'):
        bounds_key = (language, iqr_multiplier, target_column)
        if bounds_key in self.bounds:
            return
        
        all_ratios = []
        for file_path in self.file_paths:
            parquet_file = pq.ParquetFile(file_path)
            desc = os.path.basename(file_path)
            
            columns_to_load = ["content", target_column]
            if language != 'all':
                columns_to_load.append("language")

            for batch in tqdm(parquet_file.iter_batches(batch_size=8192, columns=columns_to_load), desc=f"calculating ratios for '{language}' on {desc}"):
                chunk = batch.to_pandas()
                
                target_chunk = chunk
                if language != 'all':
                    target_chunk = chunk[chunk['language'] == language]

                if target_chunk.empty: continue
                target_chunk = target_chunk.reset_index(drop=True)

                word_counts = target_chunk['content'].astype(str).str.split().str.len()
                valid_mask = word_counts > 0
                if valid_mask.any():
                    ratios = target_chunk.loc[valid_mask, target_column] / word_counts[valid_mask]
                    all_ratios.extend(ratios.dropna().tolist())
        
        if not all_ratios:
            self.bounds[bounds_key] = (-np.inf, np.inf)
            return

        ratios_series = pd.Series(all_ratios)
        Q1 = ratios_series.quantile(0.25)
        Q3 = ratios_series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - (iqr_multiplier * IQR)
        upper_bound = Q3 + (iqr_multiplier * IQR)
        self.bounds[bounds_key] = (lower_bound, upper_bound)

    def load_preprocessed_data(self, iqr_multiplier: float = 1.5, language: str = 'all', target_column: str = 'tiktoken_r50k_base_len'):
        bounds_key = (language, iqr_multiplier, target_column)
        if bounds_key not in self.bounds:
            self._calculate_iqr_bounds(iqr_multiplier, language, target_column)
        lower_bound, upper_bound = self.bounds[bounds_key]

        columns_to_load = list(set(["content", target_column, "language"]))

        for chunk in self.load_data_chunks(columns=columns_to_load):
            target_chunk = chunk
            if language != 'all':
                target_chunk = chunk[chunk['language'] == language].copy()

            if target_chunk.empty:
                yield target_chunk
                continue
            
            target_chunk = target_chunk.reset_index(drop=True)
            word_counts = target_chunk['content'].astype(str).str.split().str.len()
            
            ratios = pd.Series(index=target_chunk.index, dtype=float)
            valid_mask = word_counts > 0
            if valid_mask.any():
                ratios.loc[valid_mask] = target_chunk.loc[valid_mask, target_column] / word_counts[valid_mask]
            
            mask = ratios.between(lower_bound, upper_bound)
            yield target_chunk[mask]

    def load_data_chunks(self, columns: list = None, chunk_size: int = 8192):
        for file_path in self.file_paths:
            parquet_file = pq.ParquetFile(file_path)
            desc = os.path.basename(file_path)
            for batch in tqdm(parquet_file.iter_batches(batch_size=chunk_size, columns=columns), desc=f"loading raw from {desc}"):
                yield batch.to_pandas()
import pyarrow.parquet as pq
from tqdm import tqdm
import os

class DataLoader:
    def __init__(self, file_paths: list):
        self.file_paths = file_paths

    def load_data_chunks(self, columns: list = None, chunk_size: int = 8192):
        for file_path in self.file_paths:
            parquet_file = pq.ParquetFile(file_path)
            desc = os.path.basename(file_path)
            for batch in tqdm(parquet_file.iter_batches(batch_size=chunk_size, columns=columns), desc=f"loading from {desc}"):
                yield batch.to_pandas()

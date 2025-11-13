import os
import pandas as pd
from tqdm import tqdm
import pyarrow.parquet as pq
from transformers import AutoTokenizer
import config

DEEPSEEK_MODEL = "deepseek-ai/DeepSeek-R1"
QWEN_MODEL = "Qwen/QwQ-32B"
LLAMA_MODEL = "meta-llama/Meta-Llama-3.1-8B"

try:
    deepseek_tokenizer = AutoTokenizer.from_pretrained(DEEPSEEK_MODEL, trust_remote_code=True)
    qwen_tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL, trust_remote_code=True)
    print("DeepSeek and Qwen tokenizers loaded successfully.")
except Exception as e:
    print(f"error loading tokenizers: {e}")
    exit(1)

llama_tokenizer = None
try:
    llama_tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL, trust_remote_code=True)
    print("Meta-Llama-3.1-8B tokenizer loaded successfully.")
except Exception as e:
    print("could not load the Meta-Llama-3.1-8B tokenizer!")
    print(f"  Error: {e}")
    print("this is expected if you do not have access to the model on Hugging Face.")
    print("the 'llama3_1_8b' column will be filled with 0.")

def count_tokens_generic(text, tokenizer):
    if not isinstance(text, str) or not text:
        return 0
    return len(tokenizer.encode(text, add_special_tokens=False))

def main():
    os.makedirs(config.ENRICHED_PATH, exist_ok=True)
    file_map = zip(config.INPUT_FILES, config.ENRICHED_INPUT_FILES)
    for input_file, output_file in tqdm(file_map, total=len(config.INPUT_FILES), desc="enriching dataset files"):
        print(f"\nprocessing {os.path.basename(input_file)}...")
        parquet_file = pq.ParquetFile(input_file)
        processed_chunks = []
        for batch in tqdm(parquet_file.iter_batches(batch_size=4096), desc=f"  - Processing chunks"):
            chunk = batch.to_pandas()
            chunk[config.TOKENIZER_COLUMNS['deepseek_r1']] = chunk[config.CONTENT_COLUMN].apply(
                lambda x: count_tokens_generic(x, deepseek_tokenizer)
            )
            chunk[config.TOKENIZER_COLUMNS['qwen_qwq']] = chunk[config.CONTENT_COLUMN].apply(
                lambda x: count_tokens_generic(x, qwen_tokenizer)
            )
            chunk[config.TOKENIZER_COLUMNS['llama3_1_8b']] = chunk[config.CONTENT_COLUMN].apply(
                lambda x: count_tokens_generic(x, llama_tokenizer)
            )
            processed_chunks.append(chunk)
        if processed_chunks:
            full_enriched_df = pd.concat(processed_chunks, ignore_index=True)
            full_enriched_df.to_parquet(output_file, index=False)

if __name__ == "__main__":
    main()
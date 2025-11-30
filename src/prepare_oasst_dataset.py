import os
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
import config

OASST_DATASET_NAME = "OpenAssistant/oasst1"
OUTPUT_DIR = "oasst_dataset"
OUTPUT_FILENAME = "oasst_prompts.parquet"
TARGET_ROLE = "prompter"
TEXT_COLUMN = "text"

tokenizers = {}
tokenizer_errors = []

try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("'transformers' library not found.")

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    print("'tiktoken' library not found.")

HF_MODEL_MAP = {
    'deepseek_r1': "deepseek-ai/DeepSeek-R1",
    'qwen_qwq': "Qwen/QwQ-32B",
    'llama3_1_8b': "meta-llama/Meta-Llama-3.1-8B"
}
TIKTOKEN_ENCODING_MAP = {
    'r50k': "r50k_base",
    'cl100k': "cl100k_base",
    'o200k': "o200k_base"
}

for name, col_name in config.TOKENIZER_COLUMNS.items():
    if name in HF_MODEL_MAP and TRANSFORMERS_AVAILABLE:
        try:
            tokenizers[col_name] = AutoTokenizer.from_pretrained(HF_MODEL_MAP[name], trust_remote_code=True)
            print(f"successfully loaded tokenizer for '{name}'")
        except Exception as e:
            tokenizer_errors.append(name)
            print(f"could not load HF tokenizer for '{name}'. error: {e}")
    elif name in TIKTOKEN_ENCODING_MAP and TIKTOKEN_AVAILABLE:
        try:
            tokenizers[col_name] = tiktoken.get_encoding(TIKTOKEN_ENCODING_MAP[name])
            print(f"successfully loaded tokenizer for '{name}'")
        except Exception as e:
            tokenizer_errors.append(name)
            print(f"could not load tiktoken for '{name}'. error: {e}")

print("-" * 50)
if tokenizer_errors:
    print(f"the following tokenizers could not be loaded and will be skipped: {', '.join(tokenizer_errors)}")
print("-" * 50)

def count_tokens_generic(text, tokenizer):
    if not isinstance(text, str) or not text or tokenizer is None:
        return 0
    if hasattr(tokenizer, 'encode'):
        return len(tokenizer.encode(text))
    elif hasattr(tokenizer, 'encode_plus'):
        return len(tokenizer.encode(text, add_special_tokens=False))
    return 0

def add_token_counts(batch):
    for col_name, tokenizer in tokenizers.items():
        batch[col_name] = [count_tokens_generic(text, tokenizer) for text in batch[TEXT_COLUMN]]
    return batch

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)

    print(f"downloading dataset '{OASST_DATASET_NAME}'.")
    oasst_dataset = load_dataset(OASST_DATASET_NAME, split='train')
    print(f"initial dataset size: {len(oasst_dataset)} rows")

    print(f"filtering dataset for role '{TARGET_ROLE}'.")
    prompter_dataset = oasst_dataset.filter(lambda example: example['role'] == TARGET_ROLE, num_proc=4)
    print(f"size after filtering: {len(prompter_dataset)} rows")

    enriched_dataset = prompter_dataset.map(
        add_token_counts, 
        batched=True, 
        batch_size=1024,
        desc="calculating token counts"
    )
    
    enriched_dataset.to_parquet(output_path)
    
    print("\n" + "-"*50)
    print(f"final dataset saved to: {output_path}")
    print("-"*50)

if __name__ == "__main__":
    main()
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

MODEL_TYPE="linear_boost"
TOKENIZERS=(r50k cl100k o200k deepseek_r1 qwen_qwq llama3_1_8b)
SCOPES=(all english)

for tok in "${TOKENIZERS[@]}"; do
  for scope in "${SCOPES[@]}"; do
    echo "============================================================"
    echo "Running ${MODEL_TYPE} | tokenizer=${tok} | scope=${scope}"
    echo "============================================================"
    python src/main.py --tokenizer "$tok" --model-type "$MODEL_TYPE" --scope "$scope"
    echo
  done
done

#!/bin/bash
set -e
PROJECT_ROOT=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )/../.." &> /dev/null && pwd )
cd "$PROJECT_ROOT"
TOKENIZER="qwen_qwq"
SCOPE="english"
MODEL_TYPE="et"
echo "start: ${MODEL_TYPE^^} model for ${TOKENIZER^^} / ${SCOPE^^}"
python src/main.py --tokenizer "$TOKENIZER" --model-type "$MODEL_TYPE" --scope "$SCOPE"
echo "end: ${MODEL_TYPE^^} model for ${TOKENIZER^^} / ${SCOPE^^}"
echo "----------------------------------------"
#!/bin/bash
set -e
PROJECT_ROOT=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )/../.." &> /dev/null && pwd )
cd "$PROJECT_ROOT"
TOKENIZER="r50k"
SCOPE="all"
MODEL_TYPE="rf"
echo "----------------------------------------"
echo "Model: ${MODEL_TYPE^^}, Tokenizer: ${TOKENIZER^^}, Scope: ${SCOPE^^}"
python src/train_final_model.py --tokenizer "$TOKENIZER" --model-type "$MODEL_TYPE" --scope "$SCOPE"
echo "----------------------------------------"
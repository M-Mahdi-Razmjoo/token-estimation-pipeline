#!/bin/bash
set -e
PROJECT_ROOT="../.."
cd "$PROJECT_ROOT"
TOKENIZER="r50k"
SCOPE="all"
MODEL_TYPE="linear"
echo "start: ${MODEL_TYPE^^} model for ${TOKENIZER^^} / ${SCOPE^^}"
python src/main.py --tokenizer "$TOKENIZER" --model-type "$MODEL_TYPE" --scope "$SCOPE"
echo "end: ${MODEL_TYPE^^} model for ${TOKENIZER^^} / ${SCOPE^^}"
echo "----------------------------------------"
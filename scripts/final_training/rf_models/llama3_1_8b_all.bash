#!/bin/bash
set -e
PROJECT_ROOT=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )/../.." &> /dev/null && pwd )
cd "$PROJECT_ROOT"
TOKENIZER="llama3_1_8b"
SCOPE="all"
MODEL_TYPE="rf"
OUTPUT_DIR="final_models/$TOKENIZER/$SCOPE"
mkdir -p "$OUTPUT_DIR"
LOG_FILE="$OUTPUT_DIR/${MODEL_TYPE}_final_training_log.txt"
echo "start final training: ${MODEL_TYPE^^} model for ${TOKENIZER^^} / ${SCOPE^^}"
python src/train_final_model.py --tokenizer "$TOKENIZER" --model-type "$MODEL_TYPE" --scope "$SCOPE" > "$LOG_FILE" 2>&1
echo "end final training: ${MODEL_TYPE^^} model for ${TOKENIZER^^} / ${SCOPE^^}"
echo "final model and log file saved in: $OUTPUT_DIR"
echo "----------------------------------------"
#!/bin/bash
set -e
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
echo "this will train and save the final version of all models sequentially."
echo

FINAL_TRAINING_DIR="$SCRIPT_DIR/final_training"

echo "step 1: Training Final Linear Models"
for script in "$FINAL_TRAINING_DIR"/linear_models/*.bash; do
    echo "running: $script"
    "$script"
done
echo

echo "step 2: Training Final MLP Models"
for script in "$FINAL_TRAINING_DIR"/mlp_models/*.bash; do
    echo "running: $script"
    "$script"
done
echo

echo "step 3: Training Final Random Forest Models"
for script in "$FINAL_TRAINING_DIR"/rf_models/*.bash; do
    echo "running: $script"
    "$script"
done
echo

echo "step 4: Training Final Extra Trees Models"
for script in "$FINAL_TRAINING_DIR"/et_models/*.bash; do
    echo "running: $script"
    "$script"
done
echo

echo "all final models trained and saved successfully"
echo "you can find the output models in the 'final_models/' directory."
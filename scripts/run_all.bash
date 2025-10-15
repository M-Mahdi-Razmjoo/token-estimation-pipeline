#!/bin/bash

set -e
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

echo "step 1: preprocessing raw data files"
"$SCRIPT_DIR/preprocess_data.bash"
echo

echo "step 2: building feature files"
"$SCRIPT_DIR/build_features.bash"
echo

echo "step 3: training linear models"
for script in "$SCRIPT_DIR"/linear_models/*.bash; do
    echo "running: $script"
    "$script"
done
echo

echo "step 4: training mlp models"
for script in "$SCRIPT_DIR"/mlp_models/*.bash; do
    echo "running: $script"
    "$script"
done
echo

echo "step 5: training random forest models"
for script in "$SCRIPT_DIR"/rf_models/*.bash; do
    echo "running: $script"
    "$script"
done
echo

echo "step 6: training extra trees models"
for script in "$SCRIPT_DIR"/et_models/*.bash; do
    echo "running: $script"
    "$script"
done
echo

echo "pipeline completed successfully."

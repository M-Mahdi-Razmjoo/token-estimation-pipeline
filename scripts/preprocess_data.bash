#!/bin/bash
set -e
PROJECT_ROOT=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )/.." &> /dev/null && pwd )
cd "$PROJECT_ROOT"
python src/preprocess_and_filter.py
echo "data preprocessing and filtering step completed successfully."
echo "----------------------------------------"
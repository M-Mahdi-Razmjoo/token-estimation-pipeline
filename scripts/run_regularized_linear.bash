#!/bin/bash
set -e
PROJECT_ROOT=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )/.." &> /dev/null && pwd )
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/.venv/Scripts/activate"
python src/regularized_linear_analyzer.py
echo "Batch Benchmark Completed Successfully"
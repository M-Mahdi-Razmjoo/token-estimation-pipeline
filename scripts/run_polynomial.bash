#!/bin/bash
set -e
PROJECT_ROOT=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )/.." &> /dev/null && pwd )
cd "$PROJECT_ROOT"
OUTPUT_DIR="results/polynomial_analysis"
mkdir -p "$OUTPUT_DIR"
LOG_FILE="$OUTPUT_DIR/full_polynomial_run_log.txt"
python src/polynomial_trainer.py > "$LOG_FILE" 2>&1
echo "--- Polynomial Analysis Completed Successfully ---"
echo "Log file is available at: $LOG_FILE"
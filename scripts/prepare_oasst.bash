#!/bin/bash
set -e
PROJECT_ROOT=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )/.." &> /dev/null && pwd )
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/.venv/Scripts/activate"
python src/prepare_oasst_dataset.py
echo "OASST1 dataset preparation completed successfully"
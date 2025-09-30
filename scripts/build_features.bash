#!/bin/bash
set -e
PROJECT_ROOT=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )/.." &> /dev/null && pwd )
cd "$PROJECT_ROOT"
python src/build_features.py
echo "feature file building step completed successfully."
echo "----------------------------------------"
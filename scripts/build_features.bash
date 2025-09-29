#!/bin/bash
set -e
PROJECT_ROOT=".."
cd "$PROJECT_ROOT"
python src/build_features.py
echo "feature file building step completed successfully."
echo "----------------------------------------"
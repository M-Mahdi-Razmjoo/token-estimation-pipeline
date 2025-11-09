#!/bin/bash
set -e
PROJECT_ROOT=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )/.." &> /dev/null && pwd )
cd "$PROJECT_ROOT"
echo "starting data enrichment step"
python src/enrich_dataset.py
echo "data enrichment step completed successfully."
echo "----------------------------------------"
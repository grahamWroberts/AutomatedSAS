#!/bin/bash
python3 src/run_model.py  --datadir data --configdir configs --resultsdir results --evaluate_file ./data/experimental_curves.csv --hierarchy_file hierarchical_structure.txt --uncertainty True

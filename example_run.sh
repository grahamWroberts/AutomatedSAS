#!/bin/bash
python3 src/run_model.py  --datadir data --configdir configs --resultsdir results --evaluate_file ./data/experimental_curves.csv --extrapolation True

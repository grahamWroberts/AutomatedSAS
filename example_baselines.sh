#!/bin/bash
python3 src/baseline.py --datadir data --test True
python3 src/baseline.py --datadir data --test True --extrapolation True
python3 src/baseline.py --datadir data --test True --classifier knn
python3 src/baseline.py --datadir data --test True --classifier random-forest --n_est 1000
python3 src/baseline.py --datadir data 

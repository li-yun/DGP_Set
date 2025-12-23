# Data-Driven Uncertainty Set via DBSCAN, Gaussian Mixture Models and PCA. (DGP)

## Overview
This project aims to provide a python toolbox to implement the DGP algorithm developed in "Li, Y., Yorke-Smith, N., & Keviczky, T. (2024). Machine learning enabled uncertainty set for data-driven robust optimization. Journal of Process Control, 144, 103339.".

## Files
the source files of the python package are:
- src
- tests
- setup.py
### supplymentary materials for the JPC paper is: Appendix_JPC.pdf
### the tutorial files for using the "ddset" package is: quickstart.ipynb

## Features
- automatic construction for data-driven uncertainty sets based on the DGP algorithm
- supports both training and evaluation
- visualization is possible for 2D uncertainty sets
- reproducible experiments

## Requirements
- Python >= 3.8
All required Python packages are specified in 'setup.py'

Install dependencies by:
```bash
python -m pip install .


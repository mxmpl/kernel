# Kaggle competition

Maxime Poli, Clarine Vongpaseut

## Installation

The `environment.yml` file contains the dependencies: `numpy`, `scipy`, `joblib`, `cvxpot` and `tqdm` are explicitely used here, and `scikit-learn` in use in `main.py` to do the cross-validation.
You can use `pip` to install this repository as a Python package:

```bash
conda create -f environment.yml
conda activate kernel
pip install -e .
```

## Results

You can run the `main.py` file to reproduce our results.
The latest submission was with 15 cross-validation; in the script given there is only 3 for the sake of speed.

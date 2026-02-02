import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import importlib
plots = importlib.import_module("2-Plots")
save_fig = plots.save_fig
settings = plots.settings

settings()

housing = pd.read_csv(Path("data/train.csv"))

housing = housing.drop("median_house_value",axis=1)
housing_labels = housing['median_house_value'].copy()


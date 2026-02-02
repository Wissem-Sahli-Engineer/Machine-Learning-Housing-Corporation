import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import importlib
plots = importlib.import_module("2-Plots")
save_fig = plots.save_fig
settings = plots.settings

settings()

housing = pd.read_csv(Path("datasets/housing/housing.csv"))

housing.hist(bins= 50 , figsize=(12,8))
plt.show()
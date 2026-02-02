import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import importlib
plots = importlib.import_module("2-Plots")
save_fig = plots.save_fig
plots.settings()


housing = pd.read_csv(Path("datasets/housing/housing.csv"))

print(housing.info())

housing['income_cat'] = pd.cut(housing['median_income'],
                               bins=[0.,1.5,3.0,4.5,6.,np.inf],
                               labels=[1,2,3,4,5])       

a = housing['income_cat'].value_counts().sort_index()

a.plot.bar(rot=0,grid=True)

plt.xlabel('income category')
plt.ylabel('count')
plt.show() 
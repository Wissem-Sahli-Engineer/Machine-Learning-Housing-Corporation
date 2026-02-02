import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt



housing = pd.read_csv(Path("datasets/housing/housing.csv"))

print(housing.info())

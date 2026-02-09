import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import importlib
import pprint

# Import custom plot settings if available
try:
    plots = importlib.import_module("2-Plots")
    save_fig = plots.save_fig
    settings = plots.settings
    settings()
except (ImportError, ModuleNotFoundError):
    print("Warning: 2-Plots module not found. Using default matplotlib settings.")

# Load data
housing = pd.read_csv(Path("data/train.csv"))

# Separate labels
housing_labels = housing['median_house_value'].copy()
housing = housing.drop("median_house_value", axis=1)

'''
1-Missing values in numerical features will be imputed by replacing them
with the median, as most ML algorithms don’t expect missing values. In
categorical features, missing values will be replaced by the most
frequent category.

2-The categorical feature will be one-hot encoded, as most ML algorithms
only accept numerical inputs.

3-A few ratio features will be computed and added: bedrooms_ratio,
rooms_per_house, and people_per_house. Hopefully these will better
correlate with the median house value, and thereby help the ML models.

4-A few cluster similarity features will also be added. These will likely be
more useful to the model than latitude and longitude.

5-Features with a long tail will be replaced by their logarithm, as most
models prefer features with roughly uniform or Gaussian distributions.

6-All numerical features will be standardized, as most ML algorithms
prefer when all features have roughly the same scale
'''

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import rbf_kernel

# Define custom ClusterSimilarity transformer
class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state
        
    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(self.n_clusters, n_init=10, random_state=self.random_state)
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self  # always return self!
        
    def transform(self, X):
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)
        
    def get_feature_names_out(self, names=None):
        return [f"Cluster {i} similarity" for i in range(self.n_clusters)]

# Define functions for ratio transformation
def column_ratio(X):
    return X[:, [0]] / X[:, [1]]

def ratio_name(function_transformer, feature_names_in):
    return ["ratio"]  # feature names out

# Define specialized pipelines
def ratio_pipeline():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(column_ratio, feature_names_out=ratio_name),
        StandardScaler())

log_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    FunctionTransformer(np.log, feature_names_out="one-to-one"),
    StandardScaler())

cat_pipeline = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore"))

cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1., random_state=42)

default_num_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler())

# Combine everything into a single preprocessing ColumnTransformer
preprocessing = ColumnTransformer([
        ("bedrooms", ratio_pipeline(), ["total_bedrooms", "total_rooms"]),
        ("rooms_per_house", ratio_pipeline(), ["total_rooms", "households"]),
        ("people_per_house", ratio_pipeline(), ["population", "households"]),
        ("log", log_pipeline, ["total_bedrooms", "total_rooms", "population",
                               "households", "median_income"]),
        ("geo", cluster_simil, ["latitude", "longitude"]),
        ("cat", cat_pipeline, make_column_selector(dtype_include=object)),
    ],
    remainder=default_num_pipeline)  # one column remaining: housing_median_age

# Apply the preprocessing
housing_prepared = preprocessing.fit_transform(housing)

# Create DataFrame
housing_cleaned = pd.DataFrame(
    housing_prepared, 
    columns=preprocessing.get_feature_names_out()
)

# --- THE MISSING PIECES ---

# 1. Handle Labels (y)
# Ensure indices match if you shuffled or dropped rows earlier (outside this script)
housing_labels = housing_labels.reset_index(drop=True) 
housing_cleaned["median_house_value"] = housing_labels

# 2. Save
output_path = Path("data/housing_cleaned.csv")
housing_cleaned.to_csv(output_path, index=False)

print(f"Complete dataset (X and y) saved to {output_path}")

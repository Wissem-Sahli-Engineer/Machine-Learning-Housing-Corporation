import pandas as pd
import numpy as np
from pathlib import Path

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
        return self
        
    def transform(self, X):
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)
        
    def get_feature_names_out(self, names=None):
        return [f"Cluster {i} similarity" for i in range(self.n_clusters)]

def column_ratio(X):
    return X[:, [0]] / X[:, [1]]

def ratio_name(function_transformer, feature_names_in):
    return ["ratio"]

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

preprocessing = ColumnTransformer([
        ("bedrooms", ratio_pipeline(), ["total_bedrooms", "total_rooms"]),
        ("rooms_per_house", ratio_pipeline(), ["total_rooms", "households"]),
        ("people_per_house", ratio_pipeline(), ["population", "households"]),
        ("log", log_pipeline, ["total_bedrooms", "total_rooms", "population",
                               "households", "median_income"]),
        ("geo", cluster_simil, ["latitude", "longitude"]),
        ("cat", cat_pipeline, make_column_selector(dtype_include=object)),
    ],
    remainder=default_num_pipeline)

# Load RAW data (not pre-cleaned)
housing = pd.read_csv(Path("data/train.csv"))
test_set = pd.read_csv(Path("data/test.csv"))

housing_labels = housing['median_house_value'].copy()
housing = housing.drop("median_house_value", axis=1)

# ============================================================
# MODELS WITH FULL PIPELINE (preprocessing + model)
# ============================================================

try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:
    from sklearn.metrics import mean_squared_error
    def root_mean_squared_error(y_true, y_pred):
        return mean_squared_error(y_true, y_pred, squared=False)

# 1. Linear Regression
from sklearn.linear_model import LinearRegression

lin_reg = make_pipeline(preprocessing, LinearRegression())
lin_reg.fit(housing, housing_labels)

housing_predictions = lin_reg.predict(housing)
rmse = root_mean_squared_error(housing_labels, housing_predictions)
print(f"RMSE Linear Regression: {rmse}")

# 2. Decision Tree
from sklearn.tree import DecisionTreeRegressor

tree_reg = make_pipeline(preprocessing, DecisionTreeRegressor(random_state=42))
tree_reg.fit(housing, housing_labels)

housing_predictions = tree_reg.predict(housing)
rmse = root_mean_squared_error(housing_labels, housing_predictions)
print(f"RMSE Decision Tree: {rmse}")

# 3. Random Forest with Cross-Validation
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

forest_reg = make_pipeline(preprocessing, RandomForestRegressor(random_state=42))
forest_rmses = -cross_val_score(forest_reg, housing, housing_labels,
                                scoring="neg_root_mean_squared_error", cv=10)

print("\nRandom Forest Regressor (Cross-Validation)")
print(pd.Series(forest_rmses).describe())

# ============================================================
# HYPERPARAMETER TUNING with full pipeline
# ============================================================
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

# Create a named pipeline so we can access components
full_pipeline = Pipeline([
    ("preprocessing", preprocessing),
    ("random_forest", RandomForestRegressor(random_state=42)),
])

# Parameter names use the pipeline step names
param_distribs = {
    'preprocessing__geo__n_clusters': randint(low=3, high=50),
    'random_forest__max_features': randint(low=2, high=20)
}

rnd_search = RandomizedSearchCV(
    full_pipeline,
    param_distributions=param_distribs,
    n_iter=10, cv=3,
    scoring='neg_root_mean_squared_error',
    random_state=42
)

print("\nRunning Randomized Search (this may take a few minutes)...")
rnd_search.fit(housing, housing_labels)

# ============================================================
# ANALYZE BEST MODEL
# ============================================================
final_model = rnd_search.best_estimator_

# Access feature importances from the random_forest step
feature_importances = final_model["random_forest"].feature_importances_

# Access feature names from the preprocessing step
print("\nFeature Importances (sorted):")
sorted_importances = sorted(
    zip(feature_importances, final_model["preprocessing"].get_feature_names_out()),
    reverse=True
)
for importance, name in sorted_importances:
    print(f"  {name}: {importance:.4f}")

print(f"\nBest Parameters: {rnd_search.best_params_}")

# ============================================================
# EVALUATE ON TEST SET
# ============================================================
X_test = test_set.drop("median_house_value", axis=1)
y_test = test_set["median_house_value"].copy()

# The final_model includes preprocessing, so we can predict on raw data
final_predictions = final_model.predict(X_test)
final_rmse = root_mean_squared_error(y_test, final_predictions)

print(f"\nFinal Test RMSE: {final_rmse}")

from scipy import stats
import numpy as np

confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2
rmse_confidence_interval = np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,
        loc=squared_errors.mean(),
        scale=stats.sem(squared_errors)))
print(f"\n95% Confidence Interval: {rmse_confidence_interval}")
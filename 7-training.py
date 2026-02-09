import pandas as pd
import numpy as np
from pathlib import Path

housing = pd.read_csv(Path("data/housing_cleaned.csv"))
test_set = pd.read_csv(Path("data/test.csv"))

housing_labels = housing['median_house_value'].copy()
housing = housing.drop("median_house_value", axis=1)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error

lin_reg = LinearRegression()
lin_reg.fit(housing, housing_labels)

housing_predictions = lin_reg.predict(housing)

rmse = root_mean_squared_error(housing_labels, housing_predictions)

print(f"RMSE Linear Regression: {rmse}")


from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(housing, housing_labels)

housing_predictions = tree_reg.predict(housing)

rmse = root_mean_squared_error(housing_labels, housing_predictions)

print(f"RMSE Decision Tree: {rmse}")

'''
from sklearn.model_selection import cross_val_score

tree_rmses = -cross_val_score(tree_reg, housing, housing_labels,
scoring="neg_root_mean_squared_error", cv=10,
                             n_jobs=-1)

print("Decision Tree Regressor")
print( pd.Series(tree_rmses).describe() )
'''

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

forest_reg = RandomForestRegressor(random_state=42, n_jobs=-1)
forest_rmses = -cross_val_score(forest_reg, housing, housing_labels,
                                scoring="neg_root_mean_squared_error", cv=10,
                                n_jobs=-1)

print("Random Forest Regressor")
print( pd.Series(forest_rmses).describe() )

'''
from sklearn.svm import SVR

svm_reg = SVR(kernel="rbf", C=1000, epsilon=0.1)
svm_rmses = -cross_val_score(svm_reg, housing, housing_labels,
                             scoring="neg_root_mean_squared_error", cv=10,
                             n_jobs=-1)

print("Support Vector Machine Regressor")
print( pd.Series(svm_rmses).describe() )

from sklearn.ensemble import GradientBoostingRegressor

gbr = GradientBoostingRegressor(random_state=42)
gbr.fit(housing, housing_labels)

gbr_rmses = -cross_val_score(gbr, housing, housing_labels,
                             scoring="neg_root_mean_squared_error", cv=10,
                             n_jobs=-1)

print("Gradient Boosting Regressor")
print( pd.Series(gbr_rmses).describe() )
'''

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

# Note: We are tuning only the Random Forest hyperparameters here 
# because the data is already preprocessed.
param_distribs = {
    'n_estimators': randint(low=50, high=500),
    'max_features': randint(low=2, high=20)
}

rnd_search = RandomizedSearchCV(forest_reg, 
                                param_distributions=param_distribs, 
                                n_iter=10, cv=3,
                                scoring='neg_root_mean_squared_error', 
                                random_state=42)

rnd_search.fit(housing, housing_labels)

final_model = rnd_search.best_estimator_ 
feature_importances = final_model.feature_importances_
print("Feature Importances:")
print(feature_importances.round(2))
        
print(sorted(zip(feature_importances, 
             final_model["preprocessing"].get_feature_names_out()),
             reverse=True))


X_test = test_set.drop("median_house_value", axis=1)
y_test = test_set["median_house_value"].copy()

final_predictions = final_model.predict(X_test)
final_rmse = root_mean_squared_error(y_test, final_predictions)

print(final_rmse)
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import importlib
import pprint
plots = importlib.import_module("2-Plots")
save_fig = plots.save_fig
settings = plots.settings

settings()

housing = pd.read_csv(Path("data/train.csv"))

housing_labels = housing['median_house_value'].copy()
housing = housing.drop("median_house_value",axis=1)

stats_dict = {}
for column in housing.select_dtypes(include= [np.number]).columns:
    stats_dict[column] = {
        "skew": float(housing[column].skew().round(3)),
        "kurtosis": float(housing[column].kurt().round(3))
    }

pprint.pprint(stats_dict)

for column in stats_dict:
    if stats_dict[column]["skew"] > 1 or stats_dict[column]["skew"] < -1:
        print(f"{column} is skewed")
    if stats_dict[column]["kurtosis"] > 3 or stats_dict[column]["kurtosis"] < -3:
        print(f"{column} has heavy tails")

housing.select_dtypes(include= [np.number]).hist(figsize=(12, 8))
plt.show()

"""
# 1- handling missing values
from sklearn.impute import SimpleImputer 

imputer = SimpleImputer(strategy="median")
housing_num = housing.select_dtypes(include= [np.number])
imputer.fit(housing_num)

''' print(imputer.statistics_) '''

X = imputer.transform(housing_num)

housing_tr = pd.DataFrame(X, columns=housing_num.columns,
                          index=housing_num.index)

''' print(housing_tr.info()) '''


housing_cat = housing[["ocean_proximity"]]

'''
from sklearn.preprocessing import OrdinalEncoder 

# handle categorical data => categorical to numerical data

ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
'''

# 2- handle categorical data => one hot encoding 

from sklearn.preprocessing import OneHotEncoder

cat_encoder = OneHotEncoder()
housing_cat_encoded = cat_encoder.fit_transform(housing_cat)
housing_cat_encoded = pd.DataFrame ( housing_cat_encoded.toarray(),
                                    columns = cat_encoder.categories_[0],
                                    index = housing_cat.index
                                    )

housing = pd.concat([housing_tr.drop("ocean_proximity", axis=1), housing_cat_encoded], axis=1)

# 3- handling outliers

from sklearn.ensemble import IsolationForest

isolation_forest = IsolationForest(random_state=42)
outlier_pred = isolation_forest.fit_predict(housing)

housing = housing.iloc[outlier_pred == 1]
housing_labels = housing_labels.iloc[outlier_pred == 1]

# 4- Feature Scaling
from sklearn.preprocessing import StandardScaler

std_scaler = StandardScaler()
# Transform and immediately convert back to DataFrame
housing_scaled = pd.DataFrame(
    std_scaler.fit_transform(housing),
    columns=housing.columns,
    index=housing.index
)

# Handling heavy tails and skewness
stats_dict = {}
for column in housing_scaled.columns:
    stats_dict[column] = {
        "skew": housing_scaled[column].skew(),
        "kurtosis": housing_scaled[column].kurt()
    }

# View the results nicely
import pprint
pprint.pprint(stats_dict)

"""
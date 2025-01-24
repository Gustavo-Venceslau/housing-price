from dataset import housing
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from transformers.CombinedAtributesAdder import CombinedAttributesAdder

housing["income_cat"] = pd.cut(housing["median_income"],
                              bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5]
                              )

# Divide the training set into training and validation sets
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

# Remove the income_cat column
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

# Copy the training set
housing = strat_train_set.copy()

# Creating a clean training set for numeric columns and labels for prediction
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

# Deleting non-numeric column to add values to the nulls
imputer = SimpleImputer(strategy="median")
housing_num = housing.drop("ocean_proximity", axis=1)

from sklearn.cluster import KMeans

# Creating a pipeline for the transformations we must do
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
	('kmeans', KMeans())

])

housing_predictions = num_pipeline.fit_predict(housing_num)

print(housing_predictions)

from sklearn.metrics import mean_squared_error

final_mse = mean_squared_error(housing_labels, housing_predictions);
final_rmse = np.sqrt(final_mse);

print("RMSE", final_rmse)
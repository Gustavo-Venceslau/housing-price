import sys
sys.path.append('/Users/gustavodealmeida/projects/housing-price')

from pipeline import full_pipeline, housing, housing_labels, strat_test_set
import numpy as np
from sklearn.metrics import mean_squared_error

housing_prepared = full_pipeline.fit_transform(housing)

# Training the Random Forest Regressor model
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# Using Cross-Validation with Grid Search
param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
]

forest_reg = RandomForestRegressor()

grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                          scoring='neg_mean_squared_error',
                          return_train_score=True)
grid_search.fit(housing_prepared, housing_labels)

# Testing the model with test set
final_model = grid_search.best_estimator_

X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)

print("predictions:", final_predictions)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
print("rmse:", final_rmse)

# Confidence Interval for the RMSE

from scipy import stats

confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2
confidence_interval = np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,
                         loc=squared_errors.mean(),
                        scale=stats.sem(squared_errors)))

print("confidence interval:", confidence_interval)

# Saving the model
import joblib
joblib.dump(final_model, "random_forest_model.joblib")
from pipeline import full_pipeline, housing, housing_labels
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR

housing_prepared = full_pipeline.fit_transform(housing)

# Training the SVR model
suport_vector_regressor = SVR(kernel="linear", C=20);
suport_vector_regressor.fit(housing_prepared, housing_labels)

housing_predictions = suport_vector_regressor.predict(housing_prepared);

print("predictions", housing_predictions)

import numpy as np

final_mse = mean_squared_error(housing_labels, housing_predictions);
final_rmse = np.sqrt(final_mse);

print("RMSE", final_rmse)

from sklearn.model_selection import RandomizedSearchCV

# Cross-Validation with Grid Search for different hiperparameters

suport_vector_regressor = SVR();

param_grid = [
    {'kernel': ['linear'], 'C': [10, 100, 1000]},
    {'kernel': ['rbf'], 'C': [1, 10, 100], 'gamma': [0.01, 0.001]},
]

randomized_search = RandomizedSearchCV(suport_vector_regressor, param_grid, cv=5,
                          scoring='neg_mean_squared_error',
                          return_train_score=True)

randomized_search.fit(housing_prepared, housing_labels)

print(randomized_search.best_params_)
print(randomized_search.best_score_)
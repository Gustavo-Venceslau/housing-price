from train_SVM import randomized_search
from pipeline import full_pipeline, strat_test_set
from sklearn.metrics import mean_squared_error
import numpy as np

final_model = randomized_search.best_estimator_

X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)

print("predictions:", final_predictions)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
print("rmse:", final_rmse)
from sklearn.ensemble import GradientBoostingRegressor

gdb_reg = GradientBoostingRegressor(random_state=1)
gdb_reg.fit(housing_prepared, housing_labels)
housing_predictions = gdb_reg.predict(housing_prepared)
gdb_mse = mean_squared_error(housing_labels, housing_predictions)
gdb_rmse = np.sqrt(gdb_mse)
gdb_rmse

from sklearn.model_selection import cross_val_score

gdb_scores = cross_val_score(gdb_reg, housing_prepared, housing_labels,
                                scoring="neg_mean_squared_error", cv=3)
gdb_rmse_scores = np.sqrt(-gdb_scores)
display_scores(gdb_rmse_scores)

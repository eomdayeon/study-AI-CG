from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import VotingRegressor


reg1 = GradientBoostingRegressor(random_state=1)
reg2 = RandomForestRegressor(random_state=1)
ereg = VotingRegressor(estimators=[('gb', reg1), ('rf', reg2)])
ereg = ereg.fit(housing_prepared, housing_labels)
housing_predictions = ereg.predict(housing_prepared)
ereg_mse = mean_squared_error(housing_labels, housing_predictions)
ereg_rmse = np.sqrt(ereg_mse)
ereg_rmse

from sklearn.model_selection import cross_val_score

ereg_scores = cross_val_score(ereg, housing_prepared, housing_labels,
                                scoring="neg_mean_squared_error", cv=3)
ereg_rmse_scores = np.sqrt(-ereg_scores)
display_scores(ereg_rmse_scores)

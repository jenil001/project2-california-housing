import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score


housing = pd.read_csv('housing.csv')

housing['income_cat'] = pd.cut(housing['median_income'],
                             bins=[0, 1.5, 3.0, 4.5, 6.0, np.inf],
                             labels=[1, 2, 3, 4, 5])

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(housing, housing['income_cat']):
    strat_train_set = housing.loc[train_index].drop("income_cat", axis=1)
    strat_test_set = housing.loc[test_index].drop("income_cat", axis=1)

housing = strat_train_set.copy()

housing_labels = housing['median_house_value'].copy()
housing = housing.drop('median_house_value', axis=1)

# print(housing, housing_labels)

num_attribs = housing.drop(['ocean_proximity'], axis=1).columns.tolist()
cat_attribs = ['ocean_proximity']

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('onehot', OneHotEncoder(handle_unknown="ignore"))
])

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", cat_pipeline, cat_attribs)
])

housing_prepared = full_pipeline.fit_transform(housing)
print(housing_prepared.shape)

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)
lin_preds = lin_reg.predict(housing_prepared)
#lin_rmse = root_mean_squared_error(housing_labels, lin_preds)
lin_rmses = -cross_val_score(lin_reg, housing_prepared, housing_labels, scoring="neg_root_mean_squared_error", cv=10)
#print("Linear Regression RMSE:", -lin_rmses.mean())
print(pd.Series(lin_rmses).describe())


dec_reg = DecisionTreeRegressor()
dec_reg.fit(housing_prepared, housing_labels)
dec_preds = dec_reg.predict(housing_prepared)
#dec_rmse = root_mean_squared_error(housing_labels, dec_preds)
dec_rmses = -cross_val_score(dec_reg, housing_prepared, housing_labels, scoring="neg_root_mean_squared_error", cv=10)
#print("Decision Tree RMSE:", -dec_rmses.mean())
print(pd.Series(dec_rmses).describe())


ran_reg = RandomForestRegressor()
ran_reg.fit(housing_prepared, housing_labels)
ran_preds = ran_reg.predict(housing_prepared)
#ran_rmse = root_mean_squared_error(housing_labels, ran_preds)
ran_rmses = -cross_val_score(ran_reg, housing_prepared, housing_labels, scoring="neg_root_mean_squared_error", cv=10)
#print("Random Forest RMSE:", -ran_rmses.mean())
print(pd.Series(ran_rmses).describe())
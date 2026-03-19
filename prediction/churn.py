from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.datasets import load_diabetes

pipe = Pipeline([("scale", StandardScaler), ("model", KNeighborsRegressor)])


# df = pd.read_csv('analytics/sales.csv')
# x,y = df
mod = GridSearchCV(
    estimator=pipe, param_grid={"model_n_neighbours": [1, 2, 3, 4, 5, 6, 7, 8, 9]}
)

print(pd.DataFrame(load_diabetes()))

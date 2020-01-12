from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

data_set = load_boston()
pd.set_option('display.width', 320)
pd.set_option('display.max_columns', None)


class ValuationTool:
    def __init__(self):
        self.boston_data = pd.DataFrame(data=data_set.data, columns=data_set.feature_names)

    def gather_data(self):
        features = self.boston_data.drop(['INDUS', 'AGE'], axis=1)
        log_prices = np.log(data_set.target)
        target = pd.DataFrame(log_prices, columns=['PRICE'])
        property_stat = np.ndarray(shape=(1, 11))
        property_stat = features.mean().values.reshape(1, 11)
        regression = LinearRegression()
        regression.fit(features, target)
        fitted_values = regression.predict(features)
        mse = mean_squared_error(target, fitted_values)
        rmse = np.sqrt(mse)
        print(fitted_values)

valuation = ValuationTool()
valuation.gather_data()

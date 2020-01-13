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
        self.property_stat = np.ndarray(shape=(1, 11))
        self.boston_data = pd.DataFrame(data=data_set.data, columns=data_set.feature_names)

    def gather_data_log_estimate(self,
                                 nr_rooms,
                                 student_per_class,
                                 next_to_river=False,
                                 high_confidence=True):
        """return log estimate property stat is mean values of all feature columns"""
        features = self.boston_data.drop(['INDUS', 'AGE'], axis=1)
        log_prices = np.log(data_set.target)
        target = pd.DataFrame(log_prices, columns=['PRICE'])
        self.property_stat = features.mean().values.reshape(1, 11)
        regression = LinearRegression()
        regression.fit(features, target)
        fitted_values = regression.predict(features)
        mse = mean_squared_error(target, fitted_values)
        rmse = np.sqrt(mse)
        print(fitted_values)
        print(features)
        # rooms in fourth column etc
        self.property_stat[0][4] = nr_rooms
        self.property_stat[0][8] = student_per_class
        # challs river based
        if next_to_river:
            self.property_stat[0][2] = 1
        else:
            self.property_stat[0][2] = 0
        # finding log estimate
        log_estimate = regression.predict(self.property_stat)
        # configure price based function parameter
        # calculation range
        if high_confidence:
            # finding std and for high confidence using two times rms
            upper_bond = log_estimate + 2 * rmse
            lower_bond = log_estimate - 2 * rmse
            interval = 95
        else:
            # simply use rms with out route
            upper_bond = log_estimate + rmse
            lower_bond = log_estimate - rmse
            interval = 68
        return log_estimate, upper_bond, lower_bond, interval

    def conversion_pries(self, rooms, stud_class_room, river=False, high_confi=True):
        """The zillow median price. zillow is popular web site for house price prediction
        The function convert log price into 1990's price
        the upper and lower bond to today price
        round the values to nearest dollar"""
        if (rooms < 1) or (stud_class_room < 1):
            print('un realistic value ')
            return
        zillow_median_price = 583.3
        scale_factor = zillow_median_price / np.median(data_set.target)
        log_estimate, upper, lower, confidence = self.gather_data_log_estimate(nr_rooms=rooms,
                                                                               student_per_class=stud_class_room,
                                                                               next_to_river=river,
                                                                               high_confidence=high_confi)
        # convert to today Dollar
        dollar_log = np.e ** log_estimate * 1000 * scale_factor
        dollar_high = np.e ** upper * 1000 * scale_factor
        dollar_low = np.e ** lower * 1000 * scale_factor

        # round dollar to nearest dollar
        rounded_log = np.around(dollar_log, -3)
        rounded_high = np.around(dollar_high, -3)
        rounded_low = np.around(dollar_low, -3)

        print(f'The estimated property value is {rounded_log}')
        print(f'at {confidence} confidence of value range')
        print(f'usd {rounded_low} at the lower end to usd {rounded_high} at the high end')


valuation = ValuationTool()
valuation.conversion_pries(rooms=2, stud_class_room=20, river=True)

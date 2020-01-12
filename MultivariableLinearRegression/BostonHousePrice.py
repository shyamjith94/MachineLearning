from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

boston_dataset = load_boston()
pd.set_option('display.width', 320)
pd.set_option('display.max_columns', None)


class BostonHouse:
    def __init__(self, data):
        self.boston_dataset = data
        self.reduced_log_mse = 0
        self.reduced_rsquared = 0
        self.original_log_mse = 0
        self.original_rsquared = 0
        self.omitting_log_mse = 0
        self.omitting_rsquared = 0

    def info(self):
        dataset_return = dir(self.boston_dataset)
        print(dataset_return)
        # getting description
        print(boston_dataset.DESCR)
        # accessing data
        print(boston_dataset.data.shape)
        print(type(boston_dataset.data))
        # actual price in thousands
        # print(boston_dataset.target)

    def cleaning_data(self):
        # gathering data
        data = pd.DataFrame(data=self.boston_dataset.data, columns=self.boston_dataset.feature_names)
        data['PRICE'] = self.boston_dataset.target
        # missing data
        print(pd.isnull(data).any())
        self.boston_dataset = data

    def visualization(self, mask):
        """visualizing data Histogram, Distribution abd Bar Chart"""
        plt.figure(figsize=[10, 6])
        plt.ylabel('No. of Houses')
        plt.xlabel('Price in Thousands')
        plt.hist(self.boston_dataset['PRICE'], bins=50, ec='black', color='#46FFD1')
        # plt.show()
        plt.figure(figsize=[10, 6])
        sns.distplot(self.boston_dataset['PRICE'], bins=50, color='#fbc02d')
        # plt.show()
        # number of rooms visualization
        plt.figure(figsize=[10, 6])
        plt.ylabel('No. of Houses')
        plt.xlabel('Average Rooms House')
        plt.hist(self.boston_dataset['RM'], ec='black', color='#BAFF3E')
        # plt.show()
        plt.figure(figsize=[10, 6])
        sns.distplot(self.boston_dataset['RM'], color='#fbc02d')
        # plt.show()
        # number of rooms visualization
        plt.figure(figsize=[10, 6])
        plt.ylabel('No. of Houses')
        plt.xlabel('accessibility to highways')
        plt.hist(self.boston_dataset['RAD'], bins=24, ec='black', color='#3BC3FF')
        # plt.show()
        plt.figure(figsize=[10, 6])
        sns.distplot(self.boston_dataset['RM'], color='#3BC3FF')
        # plt.show()
        # hard coding bar diagram
        frequency = self.boston_dataset['RAD'].value_counts()
        plt.figure(figsize=[10, 6])
        plt.ylabel('No. of Houses')
        plt.xlabel('accessibility to highways')
        plt.bar(frequency.index, height=frequency)
        # plt.show()
        # heat ma on correlation using seaborn
        plt.figure(figsize=[16, 10])
        sns.heatmap(self.boston_dataset.corr(), mask=mask, annot=True)
        plt.show()
        # scatter plot
        plt.figure(figsize=[10, 6])
        nox_dis_correlation = self.boston_dataset['NOX'].corr(self.boston_dataset['DIS'])
        plt.title(f'NOX, DIS With (Correlation {nox_dis_correlation})')
        plt.xlabel('DIS-distance from employment')
        plt.ylabel('NOX-nitric oxide pollution')
        plt.scatter(x=self.boston_dataset['DIS'], y=self.boston_dataset['NOX'], alpha=0.5, s=50, color='indigo')
        plt.show()
        # scatter plot using seaborn
        sns.set()  # set default styling
        sns.set_style('whitegrid')
        sns.set_context('talk')
        sns.jointplot(x=self.boston_dataset['DIS'], y=self.boston_dataset['NOX'], kind='hex', color='indigo',
                      )
        sns.jointplot(x=self.boston_dataset['TAX'], y=self.boston_dataset['RAD'], color='darkred', joint_kws={
            'alpha': 0.5})
        # lmplot will auto plot linear line
        sns.lmplot(x='TAX', y='RAD', data=self.boston_dataset)
        # plot the RM vs PRICE correlation
        rm_price_correlation = self.boston_dataset['RM'].corr(self.boston_dataset['PRICE'])
        plt.title(f'RM, PRICE With (Correlation {rm_price_correlation})')
        plt.xlabel('RM-median number of rooms')
        plt.ylabel('PRICE-property Price in thousands')
        plt.scatter(x=self.boston_dataset['RM'], y=self.boston_dataset['PRICE'], alpha=0.5, s=50, color='indigo')
        # lmplot will auto plot linear line
        sns.lmplot(x='RM', y='PRICE', data=self.boston_dataset)
        plt.show()
        # ploting enter data frame using seaborn pairplot
        # comment due to time taken
        # sns.pairplot(self.boston_dataset, kind='reg', plot_kws={'line_kws': {'color': 'cyan'}})
        # plt.show()

    def static_data(self):
        print('average of rooms in house')
        print(self.boston_dataset['RM'].mean())
        # RAD dont have some values
        unique_rad_values = self.boston_dataset['RAD'].value_counts()
        print(unique_rad_values)
        # descriptive statics
        print('small house price')
        print(self.boston_dataset['PRICE'].min())
        print('larges house price')
        print(self.boston_dataset['PRICE'].max())
        print(self.boston_dataset.describe())
        # correlation 1.0 or -1.0 perfect correlation
        # correlation between room and house price
        print('correlation between room and house price')
        print(self.boston_dataset['PRICE'].corr(self.boston_dataset['RM']))
        print('correlation between house price and teaching ratio')
        print(self.boston_dataset['PRICE'].corr(self.boston_dataset['PTRATIO']))
        print('correlation od data`s')
        print(self.boston_dataset.corr(method='pearson'))
        # creating dumpy array like above to feed data
        mask = np.zeros_like(self.boston_dataset.corr())
        # creating triangle model
        triangle_indices = np.triu_indices_from(mask)
        mask[triangle_indices] = True
        # boston.visualization(mask)
        print(mask)

    def split_shuffle_data(self):
        prices = self.boston_dataset['PRICE']
        features = self.boston_dataset.drop('PRICE', axis=1)
        # tran_test_split return four variable
        x_train, x_test, y_train, y_test = train_test_split(features, prices, test_size=0.2, random_state=10)
        # checking data percentage in train and test
        print('train data set size')
        print(len(x_train) / len(features))
        # checking test data size
        print('test data set size')
        print(x_test.shape[0] / features.shape[0])
        # linear regression
        linear_reg = LinearRegression()
        linear_reg.fit(x_train, y_train)
        print('intercept', linear_reg.intercept_)
        print('coef in data frame')
        coef_df = pd.DataFrame(data=linear_reg.coef_, index=x_train.columns, columns=['coef'])
        print(coef_df)
        print('Train data set r-squared :-', linear_reg.score(x_train, y_train))
        print('Test data set r-squared :-', linear_reg.score(x_test, y_test))
        # -----model evaluation deploy model stage-----
        # data transformation
        print('Skew of the price column')
        print(self.boston_dataset['PRICE'].skew())
        print('price log')
        y_log = np.log(self.boston_dataset['PRICE'])
        print(y_log)
        print('log skew is ', y_log.skew())
        # sns.distplot(y_log)
        # plt.title(f'log price with {y_log.skew()}')
        # plt.show()
        transformed_data = features
        transformed_data['LOG_PRICE'] = y_log
        sns.lmplot(x='LSTAT', y='LOG_PRICE', data=transformed_data, size=7, scatter_kws={'alpha': 0.6},
                   line_kws={'color': 'darkred'})
        plt.show()

    def log_price_regression(self):
        prices = np.log(self.boston_dataset['PRICE'])
        feature = self.boston_dataset.drop('PRICE', axis=1)
        x_train, x_test, y_train, y_test = train_test_split(feature, prices, test_size=0.2, random_state=10)
        linear_reg = LinearRegression()
        linear_reg.fit(x_train, y_train)
        print('intercept log price', linear_reg.intercept_)
        print('coef in data frame based on log price')
        coef_df = pd.DataFrame(data=linear_reg.coef_, index=x_train.columns, columns=['coef'])
        # making reverse calculation chalse river columns
        print(np.e ** 0.08033)
        # p value evaluating
        x_include_constant = sm.add_constant(x_train)
        model = sm.OLS(y_train, x_include_constant)
        result = model.fit()
        print('coef and p values')
        cof_pvalue = pd.DataFrame({'coef': result.params, 'p_values': np.round(result.pvalues, 2)})
        print(cof_pvalue)
        """variance_inflation_factor_check
        with multicollinearity"""
        df_vif = pd.DataFrame()
        for i in range(len(x_include_constant.columns)):
            vif_value = variance_inflation_factor(exog=x_include_constant.values, exog_idx=i)
            df_vif = df_vif.append({
                'coef_columns': x_include_constant.columns[i],
                'vif_value': np.round(vif_value, 2)},
                ignore_index=True)
        df_vif['coef'] = coef_df['coef']

    def model_simplification(self):
        """Model Simiplication & Baysian
            Information Criterion"""
        prices = np.log(self.boston_dataset['PRICE'])
        feature = self.boston_dataset.drop('PRICE', axis=1)
        x_train, x_test, y_train, y_test = train_test_split(feature, prices, test_size=0.2, random_state=10)
        linear_reg = LinearRegression()
        linear_reg.fit(x_train, y_train)
        x_include_constant = sm.add_constant(x_train)
        model = sm.OLS(y_train, x_include_constant)
        result = model.fit()
        original_coef = pd.DataFrame({'coef': result.params, 'p_values': np.round(result.pvalues, 2)})
        print(f'baysian information criterian:- {result.bic}')
        print(f'r-squared is :- {result.rsquared}')

        """excluding feature 'AGE, INDUS' from data frame and check"""

        x_include_constant = sm.add_constant(x_train)
        x_include_constant = x_include_constant.drop(['INDUS'], axis=1)
        model = sm.OLS(y_train, x_include_constant)
        result = model.fit()
        exclude_indus_feature = pd.DataFrame({'coef': result.params, 'p_values': np.round(result.pvalues, 2)})
        print('after exclude feature ')
        print(f'baysian information criterian:- {result.bic}')
        print(f'r-squared is :- {result.rsquared}')

        x_include_constant = sm.add_constant(x_train)
        x_include_constant = x_include_constant.drop(['INDUS', 'AGE'], axis=1)
        model = sm.OLS(y_train, x_include_constant)
        result = model.fit()
        reduced_feature = pd.DataFrame({'coef': result.params, 'p_values': np.round(result.pvalues, 2)})
        print('after exclude feature ')
        print(f'baysian information criterian:- {result.bic}')
        print(f'r-squared is :- {result.rsquared}')

        frames = [original_coef, exclude_indus_feature, reduced_feature]
        frames = pd.concat(frames, axis=1, sort=False)
        print(frames)

    def residual_find_plot(self):
        """Looking the Perfection of model and find residual and plot"""
        # eliminate columns as before
        prices = np.log(self.boston_dataset['PRICE'])
        feature = self.boston_dataset.drop(['PRICE', 'AGE', 'INDUS'], axis=1)
        x_train, x_test, y_train, y_test = train_test_split(feature, prices, test_size=0.2, random_state=10)
        # using stat model to fit
        x_include_constant = sm.add_constant(x_train)
        model = sm.OLS(y_train, x_include_constant)
        result = model.fit()
        # finding residuals manually
        residual = y_train - result.fittedvalues
        # building function
        # result.resid
        print('residual in model')
        print(residual)
        # Graph for actual Vs predicted
        # finding cor-relation
        corr = y_train.corr(result.fittedvalues)
        print(x_train)
        plt.xlabel('Actual Log Price', fontsize=14)
        plt.ylabel('Predicted Log Price', fontsize=14)
        plt.title('Actual vs predicted log prices')
        plt.scatter(x=y_train, y=result.fittedvalues, color='navy', alpha=0.6)
        plt.plot(y_train, y_train, color='red')
        plt.show()
        # using value plot reverse to log
        plt.xlabel('Actual  Price in thousands', fontsize=14)
        plt.ylabel('Predicted  Price in thousands', fontsize=14)
        plt.title('Actual vs predicted  prices in thousands')
        plt.scatter(x=np.e ** y_train, y=np.e ** result.fittedvalues, color='navy', alpha=0.6)
        plt.plot(np.e ** y_train, np.e ** y_train, color='red')
        plt.show()

        # differences in predicted value and fitted values
        plt.scatter(x=result.fittedvalues, y=result.resid, color='navy', alpha=0.6)
        plt.xlabel('Fitted values', fontsize=14)
        plt.ylabel('residual', fontsize=14)
        plt.title('fitted vs residuals ')
        plt.show()

        # distribution of residuals (log prices) - checking for normality
        residual_mean = round(result.resid.mean(), 3)
        residual_skew = round(result.resid.skew(), 3)
        print(f'residual mean \t{residual_mean} and residual skew\t{residual_skew}')

        # residuals in dist plt
        sns.distplot(result.resid, color='navy')
        plt.title('log price model residuals')
        plt.show()

        self.reduced_log_mse = round(result.mse_resid, 3)
        self.reduced_rsquared = round(result.rsquared, 3)

    def original_model_residuals(self):
        """using original model to test residuals with all feature
        plot residuals vs predicted values"""
        prices = self.boston_dataset['PRICE']
        feature = self.boston_dataset.drop('PRICE', axis=1)
        x_train, x_test, y_train, y_test = train_test_split(feature, prices, random_state=10, test_size=0.2)
        x_exclude_constant = sm.add_constant(x_train)
        model = sm.OLS(y_train, x_exclude_constant)
        result = model.fit()
        corr = y_train.corr(result.fittedvalues)
        plt.scatter(x=y_train, y=result.fittedvalues, color='orange')
        plt.plot(y_train, y_train, color='blue')
        plt.xlabel('y_train or actual prices')
        plt.title('actual prices and predicted price')
        plt.ylabel('predicted values')
        plt.show()

        # predicted values and residuals
        plt.scatter(x=result.fittedvalues, y=result.resid, color='indigo')
        plt.title('predicted values and residuals')
        plt.xlabel('predicted values')
        plt.ylabel('residuals')
        plt.show()
        # distribution graph
        residual_mean = round(result.resid.mean(), 3)
        residual_skew = round(result.resid.skew(), 3)
        sns.distplot(result.resid, color='green')
        plt.title(f'residuals skew{residual_skew}, mean is {residual_mean}')
        plt.show()
        print(corr)
        self.original_log_mse = round(result.mse_resid, 3)
        self.original_rsquared = round(result.rsquared, 3)

    def omitting_features(self):
        """model using log prices and omitting features"""
        prices = np.log(self.boston_dataset['PRICE'])
        features = self.boston_dataset.drop(['PRICE', 'INDUS', 'AGE', 'LSTAT', 'RM', 'NOX', 'CRIM'], axis=1)
        x_train, x_test, y_train, y_test = train_test_split(features, prices)
        x_exclude_cnstant = sm.add_constant(x_train)
        model = sm.OLS(y_train, x_exclude_cnstant)
        result = model.fit()
        corr = y_train.corr(result.fittedvalues)
        print(f'omitting feature correlation:-{corr}')

        plt.scatter(x=y_train, y=result.fittedvalues, color='orange')
        plt.plot(y_train, y_train, color='blue')
        plt.xlabel('y_train or actual log prices')
        plt.title('actual log  prices and predicted price omitting features')
        plt.ylabel('predicted log values')
        plt.show()

        plt.scatter(x=result.fittedvalues, y=result.resid, color='indigo')
        plt.title('predicted values and residuals with omitting features')
        plt.xlabel('predicted values')
        plt.ylabel('residuals')
        plt.show()

        self.omitting_log_mse = round(result.mse_resid, 3)
        self.omitting_rsquared = round(result.rsquared, 3)

    def mean_squared_error(self):
        """Calculating mean squares error and rsquared """
        mse_r_squared = pd.DataFrame({
            'r-squared': [self.reduced_rsquared, self.original_rsquared,
                          self.omitting_rsquared],
            'mse': [self.reduced_log_mse, self.original_log_mse, self.omitting_log_mse],
            'rmse': np.sqrt([self.reduced_log_mse, self.original_log_mse, self.omitting_log_mse])},
            index=['reduced', 'original', 'omitting'])
        print(mse_r_squared)
        # if  our estimated a house price 30, calculate upper and lower bond
        print('1 standard deviation is ', np.sqrt(self.reduced_log_mse))
        print('2 standard deviation is ', 2 * np.sqrt(self.reduced_log_mse))
        upper_bound = np.log(30) + 2 * np.sqrt(self.reduced_log_mse)
        print('the upper bond in log prices for a 95% prediction interval is ', upper_bound)
        print('the upper bond in normal prices', np.e ** upper_bound * 1000)

        lower_bound = np.log(30) - 2 * np.sqrt(self.reduced_log_mse)
        print('the lower bond in log prices for a 95% prediction interval is ', lower_bound)
        print('the lower bond in normal prices', np.e ** lower_bound * 1000)


boston = BostonHouse(boston_dataset)
boston.info()
boston.cleaning_data()

# boston.static_data()
# boston.split_shuffle_data()
boston.log_price_regression()
boston.model_simplification()
boston.residual_find_plot()
boston.original_model_residuals()
boston.omitting_features()
boston.mean_squared_error()

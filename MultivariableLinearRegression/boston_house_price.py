from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

boston_dataset = load_boston()
pd.set_option('display.width', 320)
pd.set_option('display.max_columns', None)


class BostonHouse:
    def __init__(self, data):
        self.boston_dataset = data

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



boston = BostonHouse(boston_dataset)
boston.info()
boston.cleaning_data()

boston.static_data()
boston.split_shuffle_data()

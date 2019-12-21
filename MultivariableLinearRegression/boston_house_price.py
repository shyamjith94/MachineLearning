from sklearn.datasets import load_boston
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
        # print(boston_dataset.DESCR)
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

    def visualization(self):
        """visualizing data Histogram, Distribution abd Bar Chart"""
        plt.figure(figsize=[10, 6])
        plt.ylabel('No. of Houses')
        plt.xlabel('Price in Thousands')
        plt.hist(self.boston_dataset['PRICE'], bins=50, ec='black', color='#46FFD1')
        plt.show()
        plt.figure(figsize=[10, 6])
        sns.distplot(self.boston_dataset['PRICE'], bins=50, color='#fbc02d')
        plt.show()
        # number of rooms visualization
        plt.figure(figsize=[10, 6])
        plt.ylabel('No. of Houses')
        plt.xlabel('Average Rooms House')
        plt.hist(self.boston_dataset['RM'], ec='black', color='#BAFF3E')
        plt.show()
        plt.figure(figsize=[10, 6])
        sns.distplot(self.boston_dataset['RM'], color='#fbc02d')
        plt.show()
        # number of rooms visualization
        plt.figure(figsize=[10, 6])
        plt.ylabel('No. of Houses')
        plt.xlabel('accessibility to highways')
        plt.hist(self.boston_dataset['RAD'], bins=24, ec='black', color='#3BC3FF')
        plt.show()
        plt.figure(figsize=[10, 6])
        sns.distplot(self.boston_dataset['RM'], color='#3BC3FF')
        plt.show()
        # hard coding bar diagram
        frequency = self.boston_dataset['RAD'].value_counts()
        plt.figure(figsize=[10, 6])
        plt.ylabel('No. of Houses')
        plt.xlabel('accessibility to highways')
        plt.bar(frequency.index, height=frequency)
        plt.show()

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


boston = BostonHouse(boston_dataset)
boston.info()
boston.cleaning_data()
# boston.visualization()
boston.static_data()

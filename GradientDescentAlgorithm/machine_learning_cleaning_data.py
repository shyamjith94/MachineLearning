import pandas as pd
import re


def cleaning_data():
    data = pd.read_csv("/home/shyam/GITHUB_SHYAM/Research/udumy/machine learning/csv/cost-revenue-dirty.csv")
    data = data[['Production Budget ($)', 'Worldwide Gross ($)']]
    data.rename(
        columns={
            'Production Budget ($)': 'Production_Budget',
            'Worldwide Gross ($)': 'Worldwide_Gross'},
        inplace=True)
    # data['Production_Budget'] = data['Production_Budget'].str.replace(r'\D', '')
    data['Production_Budget'] = data['Production_Budget'].map(lambda x: re.sub(r'\D', '', x)).astype(float)
    data['Worldwide_Gross'] = data['Worldwide_Gross'].map(lambda x: re.sub(r'\D', '', x)).astype(float)
    return data


if __name__ == "__main__":
    cleaning_data()

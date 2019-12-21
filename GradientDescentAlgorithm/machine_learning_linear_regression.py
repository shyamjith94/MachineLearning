import machine_learning_cleaning_data as cleaned_data
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


def plot_map(plot=0, ploating_value=None):
    data = cleaned_data.cleaning_data()
    frame_x = pd.DataFrame(data, columns=['Production_Budget'])
    frame_y = pd.DataFrame(data, columns=['Worldwide_Gross'])
    plt.figure(figsize=(10, 6))
    plt.scatter(frame_x, frame_y, alpha=0.3)
    if plot == 1:
        plt.plot(frame_x, ploating_value, color='red', linewidth=4)
    plt.title("Film Cost With Global Revenue")
    plt.xlabel("Production Budget")
    plt.ylabel("worldWide Gross")
    plt.ylim(0, 3000000000)
    plt.xlim(0, 4500000000)
    # scatter plot for model difference showing original and code
    plt.show()
    return [data, frame_x, frame_y]


def liner_regression(values):
    data = values[0]
    frame_x = values[1]
    frame_y = values[2]
    regression = LinearRegression()
    liner_model = regression.fit(frame_x, frame_y)
    coefficient = regression.coef_
    plot_map(plot=1, ploating_value=regression.predict(frame_x))
    print(coefficient)
    print(regression.intercept_)
    print(regression.score(frame_x, frame_y))


return_list = plot_map()
liner_regression(return_list)

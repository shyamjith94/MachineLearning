import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

"""Out Put Display Setting """
desired_width = 320
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', None)

path = "/home/shyam/GITHUB_SHYAM/Research/udumy/machine learning/csv/lsd-math-score-data.csv"
frame_lsd = pd.read_csv(path)
time = frame_lsd[['Time_Delay_in_Minutes']]
lsd = frame_lsd[['LSD_ppm']]
score = frame_lsd[['Avg_Math_Test_Score']]
predict = 0


def liner_regression():
    global predict
    liner = LinearRegression()
    liner.fit(time, lsd)
    coefficient = liner.coef_
    intercept = liner.intercept_
    scores = liner.score(lsd, score)
    predict = liner.predict(lsd)
    print('coefficient:- ', coefficient[0][0])
    print('intercept:- ', intercept[0])
    print('score:- ', scores)


def plot_normal():
    plt.title("LSD Over Time")
    plt.xlabel("Time in minute")
    plt.ylabel("LSD ppm")
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.text(x=0, y=-1, s='wagner et al. (1968)')
    plt.plot(time, lsd, color='g', lw=3)
    # plt.show()


def plot_scatter():
    plt.scatter(lsd, score, color='blue', s=100, alpha=0.5)
    print(predict)
    plt.plot(lsd, predict)
    plt.title("LSD Over Time")
    plt.ylabel(" Performance Score")
    plt.xlabel("LSD ppm")
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    # plt.show()


liner_regression()
plot_normal()
plot_scatter()
print(frame_lsd)

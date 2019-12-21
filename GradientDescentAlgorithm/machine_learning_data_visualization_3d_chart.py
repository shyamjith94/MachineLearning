"""Example for Data visualization with three D chart """
import numpy as np
import math
import matplotlib.pyplot as plt

"""below import using three D plot tool"""
from mpl_toolkits.mplot3d.axes3d import Axes3D

"""below using for color map in 3D"""
from matplotlib import cm

"""
f(x, y) =  1
        _______
        3-x2-y2 + 1 

"""


def function_f(x, y):
    r = 3 ** (-x ** 2 - y ** 2)
    return 1 / (r + 1)


def partial_fpx(x, y):
    r = 3 ** (-x ** 2 - y ** 2)
    return 2 * x * math.log(3) * r / (r + 1) ** 2


def partial_fpy(x, y):
    r = 3 ** (-x ** 2 - y ** 2)
    return 2 * y * math.log(3) * r / (r + 1) ** 2


def generate_Data():
    x = np.linspace(start=-2, stop=2, num=200)
    y = np.linspace(start=-2, stop=2, num=200)
    # change to 2 dimensional array using meshgrid function
    x, y = np.meshgrid(x, y)
    return x, y


def function_plot():
    """generate three D plot"""
    x_4, y_4 = generate_Data()
    fig = plt.figure(figsize=[16, 12])
    ax = fig.gca(projection='3d')
    # adding label
    ax.set_xlabel('X', fontsize=20)
    ax.set_ylabel('Y', fontsize=20)
    ax.set_zlabel('f(x,y)', fontsize=20)
    """cmap=cm.coolwarm using for color map in 3D plot"""
    ax.plot_surface(x_4, y_4, function_f(x_4, y_4), cmap=cm.winter, alpha=0.4)
    plt.show()


function_plot()

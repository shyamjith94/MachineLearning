""" Graphing 3D Gradient Descent and advanced numpy array"""
import machine_learning_data_visualization_3d_chart as prsfun
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d.axes3d import Axes3D
import sympy


def partial_derivative_mathematical():
    """ partial derivative and mathematical computation with advanced numpy array"""
    """ advanced numpy array practices"""
    test_one = np.array([['captain', 'guitar']])
    print(test_one)
    print(test_one.shape)

    test_two = np.array([['captain', 'guitar'], ['magic', 'dream']])
    print(test_two)
    print(test_two.shape, end="")

    the_root = np.append(arr=test_two, values=test_one, axis=0)
    print(the_root.shape)
    print(the_root, end="")
    the_root_two = np.append(arr=test_two, values=test_one.reshape((2, 1)), axis=1)
    print(the_root_two.shape)
    print(the_root_two, end="")

    multiplier = 0.1
    max_iter = 100
    params = np.array([1.8, 1.0])  # initial guess
    values_array = params.reshape(1, 2)
    for i in range(max_iter):
        gradient_x = prsfun.partial_fpx(params[0], params[1])
        gradient_y = prsfun.partial_fpy(params[0], params[1])
        gradients = np.array([gradient_x, gradient_y])
        params = params - multiplier * gradients
        values_array = np.append(values_array, params.reshape(1, 2), axis=0)
        # print(params, gradients, sep="<-->")

    """print result"""
    print("value of gradient descent array", gradients)
    print("Minimum occurs at x value of", params[0])
    print("Minimum occurs at y value of", params[1])
    """The optimized values I don't have to substitute them into our cost function which is F and then the"""
    print("The cost is ", prsfun.function_f(params[0], params[1]))
    return values_array


def function_plot_include_numpy_array():
    """getting previous array value"""
    value_array = partial_derivative_mathematical()

    """generate th ree D plot"""
    x_4, y_4 = prsfun.generate_Data()
    fig = plt.figure(figsize=[16, 12])
    ax = fig.gca(projection='3d')
    # adding label
    ax.set_xlabel('X', fontsize=20)
    ax.set_ylabel('Y', fontsize=20)
    ax.set_zlabel('f(x,y)', fontsize=20)
    """cmap=cm.coolwarm using for color map in 3D plot"""
    ax.plot_surface(x_4, y_4, prsfun.function_f(x_4, y_4), cmap=cm.winter, alpha=0.4)
    ax.scatter(value_array[:, 0], value_array[:, 1], prsfun.function_f(value_array[:, 0], value_array[:, 1]),
               s=50, color='red')
    plt.show()


function_plot_include_numpy_array()

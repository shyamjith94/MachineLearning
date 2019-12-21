"""divergence overflow and Python tuples the function that we're gonna be looking at in this example it's"""
import machine_learning_gradient_descent_Advanced_function as privious_fun
import numpy as np
import matplotlib.pyplot as plt


def generate_data():
    x_value = np.linspace(start=-2.5, stop=2.5, num=1000)
    return x_value


def diverge_h(x):
    # h(x) = x5 + 2x4 + 2
    return x ** 5 - 2 * x ** 4 + 2


def diverge_dh(x):
    # h(x) = 5x4 - 8*x3
    return 5 * x ** 4 - 8 * x ** 3


def plot_function(list_x, derivative_list):
    x_value = generate_data()
    # advanced cost function chart
    plt.figure(figsize=[15, 5])
    plt.subplot(1, 2, 1)
    plt.xlim(-1.2, 2.5)
    plt.ylim(-1, 4)
    plt.title('Cost Function', fontsize=17)
    plt.xlabel('X', fontsize=17)
    plt.ylabel('h(x)', fontsize=17)
    plt.plot(x_value, diverge_h(x_value), color='blue', linewidth=3)

    plt.scatter(list_x, diverge_h(np.array(list_x)), color='red', s=100, alpha=0.5)
    # advanced derivative cost function chart
    plt.subplot(1, 2, 2)
    plt.xlim(-1, 2)
    plt.ylim(-4, 5)
    plt.grid()
    plt.xlabel('X', fontsize=17)
    plt.ylabel('dh(x)', fontsize=17)
    plt.plot(x_value, diverge_dh(x_value), color='red', linewidth=3)
    plt.scatter(list_x, derivative_list, color='blue', s=100, alpha=0.5)

    plt.show()


local_min, x_list, list_derivative = privious_fun.gradient_descent(diverge_dh, -0.2, 0.01, 0.00, max_iterator=10)
plot_function(x_list, list_derivative)

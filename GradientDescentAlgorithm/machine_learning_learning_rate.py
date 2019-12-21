"""learning rate of gradient descent algorithm"""

import machine_learning_gradient_descent_Advanced_function as privious_fun
import numpy as np
import matplotlib.pyplot as plt


def plot_function(list_x, derivative_list):
    x_value = privious_fun.generate_data(start=-2.5, stop=2.5, num=1000)
    # advanced cost function chart
    plt.figure(figsize=[15, 5])
    plt.subplot(1, 2, 1)
    plt.xlim(-2, 2)
    plt.ylim(0.5, 5.5)
    plt.title('Cost Function', fontsize=17)
    plt.xlabel('X', fontsize=17)
    plt.ylabel('gf(x)', fontsize=17)
    plt.plot(x_value, privious_fun.cost_function_g(x_value), color='blue', linewidth=3)

    plt.scatter(list_x, privious_fun.cost_function_g(np.array(list_x)), color='red', s=100, alpha=0.5)
    # advanced derivative cost function chart
    plt.subplot(1, 2, 2)
    plt.xlim(-2, 2)
    plt.ylim(-6, 8)
    plt.grid()
    plt.xlabel('X', fontsize=17)
    plt.ylabel('dgf(x)', fontsize=17)
    plt.plot(x_value, privious_fun.cost_function_dg(x_value), color='red', linewidth=3)
    plt.scatter(list_x, derivative_list, color='blue', s=100, alpha=0.5)

    plt.show()


local_min, x_list, list_derivative = privious_fun.gradient_descent(privious_fun.cost_function_dg, 1.9, 0.01, 0.25,
                                                                   max_iterator=5)

plot_function(x_list, list_derivative)

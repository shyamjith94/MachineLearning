"""Example two with multiple minima vs initial guess and advanced function"""
import numpy as np
import matplotlib.pyplot as plt


def generate_data(start=-2, end=2, num=1000):
    x_value = np.linspace(start, end, num)
    return x_value


def cost_function_g(x):
    # G effects function
    # g(x) = x4 - 4x2 + 5
    return x ** 4 - 4 * x ** 2 + 5


def cost_function_dg(x):
    # derivative function dg(x)
    return 4 * x ** 3 - 8 * x


def gradient_descent(derivative_fun, initial_guess, multiplier, precision, max_iterator=500):
    # gradient descent function
    new_x = initial_guess
    x_list = [new_x]
    slope_list = [derivative_fun(new_x)]

    for i in range(max_iterator):
        previous_x = new_x
        gradient = derivative_fun(previous_x)
        new_x = previous_x - multiplier * gradient
        step_size = abs(new_x - previous_x)
        print("step_size", step_size)
        x_list.append(new_x)
        slope_list.append(derivative_fun(new_x))
        if step_size < precision:
            break
    return new_x, x_list, slope_list


def plot_function(list_x, derivative_list):
    x_value = generate_data()
    # advanced cost function chart
    plt.figure(figsize=[15, 5])
    plt.subplot(1, 2, 1)
    plt.xlim(-2, 2)
    plt.ylim(0.5, 5.5)
    plt.title('Cost Function', fontsize=17)
    plt.xlabel('X', fontsize=17)
    plt.ylabel('gf(x)', fontsize=17)
    plt.plot(x_value, cost_function_g(x_value), color='blue', linewidth=3)

    plt.scatter(list_x, cost_function_g(np.array(list_x)), color='red', s=100, alpha=0.5)
    # advanced derivative cost function chart
    plt.subplot(1, 2, 2)
    plt.xlim(-2, 2)
    plt.ylim(-6, 8)
    plt.grid()
    plt.xlabel('X', fontsize=17)
    plt.ylabel('dgf(x)', fontsize=17)
    plt.plot(x_value, cost_function_dg(x_value), color='red', linewidth=3)
    plt.scatter(list_x, derivative_list, color='blue', s=100, alpha=0.5)

    plt.show()


local_min, x_list, list_derivative = gradient_descent(cost_function_dg, -0.1, 0.01, 0.001)
plot_function(x_list, list_derivative)

import matplotlib.pyplot as plt
import numpy as np


def derivative(x):
    # slope and & derivatives f(x)
    return 2 * x + 1


def simple_cost_function(x):
    # example of simple cost function
    # f(x) = x2 +x + 1
    return x ** 2 + x + 1


def generate_data():
    # generate data using numpy array with linspace
    x_1 = np.linspace(start=-3, stop=3, num=500)
    return x_1


# gradient descent
def gradient_descent():
    new_x = 3
    x_list = [new_x]
    slope_list = [derivative(new_x)]
    previous_x = 0
    step_multiplier = 0.1
    precision = 0.0001  # So this is how precise I want my answer to be.
    for n in range(500):
        # if increasing loop running So as you can see we're converging on this local minimum by brute force right.
        # We didn't solve our cost function here analytically.
        # What we're doing is we're iterating and going down that valley that cost function.
        # Until we reach the minimum point and at the minimum our slope is equal to zero.
        previous_x = new_x
        gradient = derivative(previous_x)
        new_x = previous_x - step_multiplier * gradient
        # we can say well the step size is gonna be the difference between our new X minus
        step_size = abs(new_x - previous_x)
        print(step_size)
        # adding new value to list for plot
        x_list.append(new_x)
        slope_list.append(derivative(new_x))
        if step_size < precision:
            print("loop run this may times: ", n)
            break

    print("local minimum error occur at: ", new_x)
    print("slope or df(x) or derivative value is this point: ", derivative(new_x))
    print("f(x) value or cost at tis point: ", simple_cost_function(new_x))
    return x_list, slope_list


def plot_graph():
    # calling gradient descent algorithm for plot
    x_list, slope_list = gradient_descent()
    # list cant be plot so convert to numpy array
    x_list_values = np.array(x_list)
    # plot graph based of above generate number
    plt.figure(figsize=[15, 5])
    # plot two graph using subplot
    # first plot
    """Chart Num One"""
    plt.subplot(1, 3, 1)  # want to down change row=2 and column =1
    return_generate_data = generate_data()
    print(simple_cost_function(return_generate_data))
    plt.title('Cost Function')
    plt.xlabel('x', fontsize=10)
    plt.ylabel('F(x)', fontsize=10)
    plt.xlim([-3, 3])
    plt.ylim(0, 8)
    plt.plot(return_generate_data, simple_cost_function(return_generate_data), color='blue', linewidth=3)
    # scatter plot for gradient descent algorithm based
    plt.scatter(x_list, simple_cost_function(x_list_values), color='red', s=100, alpha=0.5)
    # second graph plot
    """Chart Num Two"""
    plt.subplot(1, 3, 2)  # want to down change row=2 and column =1
    plt.grid()
    plt.title('Slope Of The Cost Function')
    plt.xlabel('x', fontsize=10)
    plt.ylabel('derivative df(x)', fontsize=10)
    plt.xlim([-2, 3])
    plt.ylim(-3, 6)
    plt.plot(return_generate_data, derivative(return_generate_data), color='skyblue', linewidth=5)
    # scatter plot for gradient descent algorithm based
    plt.scatter(x_list, slope_list, color='red', s=100, alpha=0.5)
    """Chart Num Three"""
    plt.subplot(1, 3, 3)  # want to down change row=2 and column =1
    plt.title("gradient descent (close up")
    plt.xlabel('x', fontsize=10)
    plt.grid()
    plt.xlim(-0.55, -0.2)
    plt.ylim(-0.3, 0.8)
    plt.plot(return_generate_data, derivative(return_generate_data), color='skyblue', linewidth=5)
    # scatter plot for gradient descent algorithm based
    plt.scatter(x_list, slope_list, color='red', s=100, alpha=0.5)
    plt.show()




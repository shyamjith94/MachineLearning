import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.linear_model import LinearRegression
from matplotlib import cm

"""partial derivative Mean Squared Error with respect of theta_0 and theta_1"""


# $$ \frac{\partial MSE}{partial\theta_0} = -\frac{2}{n} \sum_{i=1}^{n}\big(y^{(i)} - \theta_0 - \theta_1 x^{(i)} \big)
# $$ \frac{\partial MSE}{partial\theta_0} = -\frac{2}{n} \sum_{i=1}^{n}\big(y^{(i)} - \theta_0 - \theta_1 x^{(i)}
# \big) \big(x^{(i)} \big)$$
def function_mse(y, y_hat_values):
    # we can do in three ways commented in here
    # msc_calc = 1 / 7 * sum((y - y_hat_values) ** 2)
    # msc_calc = (1 / y_value.size) * sum((y - y_hat_values) ** 2)
    msc_calc = np.average((y - y_hat_values) ** 2, axis=0)
    return msc_calc


def function_grade(x_value, y_value, thetas):
    # array of theta parameter theta_0 index at 0 and theta_1 at index 1
    n = y_value.size
    # theta_0 slope and theta_1 slope to hold slope values of partial derivative
    theta_0_slope = (-2 / n) * sum(y_value - thetas[0] - thetas[1] * x_value)
    theta_1_slope = (-2 / n) * sum((y_value - thetas[0] - thetas[1] * x_value) * x_value)
    return np.array([theta_0_slope[0], theta_1_slope[0]])


x_value = np.array([[0.1, 1.2, 2.4, 3.2, 4.1, 5.7, 6.2]]).transpose()
y_value = np.array([[1.7, 2.4, 3.5, 3.0, 6.1, 9.4, 8.2]]).transpose()
multiplier = 0.01
thetas = np.array([2.9, 2.9])
# collect data point to scatter plot
plot_values = thetas.reshape(1, 2)
mse_values = function_mse(y_value, thetas[0] + thetas[1]*x_value)

print(thetas)
for i in range(1000):
    thetas = thetas - multiplier * function_grade(x_value, y_value, thetas)
    # append new values to numppy array
    plot_values = np.concatenate((plot_values, thetas.reshape(1, 2)), axis=0)
    mse_values = np.append(arr=mse_values, values=function_mse(y_value, thetas[0] + thetas[1]*x_value))
print('min occurs at theta 0 ', thetas[0])
print('min occurs at theta 1 ', thetas[1])
print('mse is ', function_mse(y_value, thetas[0] + thetas[1]*x_value))

fig = plt.figure(figsize=[16, 12])
ax = fig.gca(projection='3d')
ax.set_title('MSE Cost')
ax.set_xlabel('Theta 0', fontsize=20)
ax.set_ylabel('Theta 1', fontsize=20)
ax.set_zlabel('Cost_MSE', fontsize=20)
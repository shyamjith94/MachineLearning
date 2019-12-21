import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from mpl_toolkits import mplot3d
from matplotlib import cm

"""Mean Squared Error : A Real Cost Function For Regression Problem"""
### $$RSS = \sum_{i=1}^{n} \big(y^{(i)} - h_\theta x^ {(i)} \big)^2 $$
### $$MSE = \frac{1} {n} \sum_{i=1}^{n} \big(y^{(i)} - h_\theta x^ {(i)} \big)^2 $$
### $$MSE = \frac{1} {n} \sum_{i=1}^{n} \big(y - hat{y} \big)^2 $$
### y_hat = theta0 + thea1 * x

# create some sample data
# actual input like x = [0.1, 1.2, 2.4, 3.2, 4.1, 5.7, 6.2]
# actual input like y = [1.7, 2.4, 3.5, 3.0, 6.1, 9.4, 8.2]
# use transpose method and [] make array two dimensional or use reshape(7,1)
x_value = np.array([[0.1, 1.2, 2.4, 3.2, 4.1, 5.7, 6.2]]).transpose()
y_value = np.array([[1.7, 2.4, 3.5, 3.0, 6.1, 9.4, 8.2]]).transpose()
print("x_value shape", x_value.shape)
print("y_value shape", y_value.shape)
# quick LinearRegression
regression = LinearRegression()
regression.fit(x_value, y_value)
theta_0 = regression.intercept_[0]
theta_1 = regression.coef_[0][0]
print("Theta 0: ", theta_0)
print("Theta 1: ", theta_1)
predict_value = regression.predict(x_value)
plt.scatter(x_value, y_value, s=50)
plt.plot(x_value, predict_value, color='red', linewidth=3)
plt.xlabel('x_value')
plt.ylabel('y_value')
# plt.show()
# estimated value  y_hat = theta0 + thea1 * x
y_hat = theta_0 + theta_1 * x_value
print('estimated y_hat: \n', y_hat)


# printing mse for calculated y_hat values of above
def function_mse(y, y_hat_values):
    # we can do in three ways commented in here
    # msc_calc = 1 / 7 * sum((y - y_hat_values) ** 2)
    # msc_calc = (1 / y_value.size) * sum((y - y_hat_values) ** 2)
    msc_calc = np.average((y - y_hat_values) ** 2, axis=0)
    return msc_calc


mse_values = function_mse(y=y_value, y_hat_values=y_hat)
print("mse calculated: ", mse_values)
# mse regression using sklearn method
print('mse using manual calculated ', mean_squared_error(y_value, y_hat))
print('mse regression is ', mean_squared_error(y_value, predict_value))

# 3D plot for mse cost function
# make data for theta
nr_thetas = 200

th_0 = np.linspace(start=-1, stop=3, num=nr_thetas)
th_1 = np.linspace(start=-1, stop=3, num=nr_thetas)
# making two dimensional array
plot_th_0, plot_th_1 = np.meshgrid(th_0, th_1)
print(plot_th_0.shape, th_0.shape)

# calc mse cost function using nested loop
plot_cost = np.zeros((nr_thetas, nr_thetas))
# print(plot_cost)
for i in range(nr_thetas):
    for j in range(nr_thetas):
        # print(plot_th_0[i][j])
        # calculate estimated y values
        y_hat = plot_th_0[i][j] + plot_th_1[i][j] * x_value
        # updating the mse value
        plot_cost[i][j] = mean_squared_error(y_value, y_hat)
print(plot_cost)
print('shape of plot_th_0', plot_th_0.shape)
print('shape of plot_th_1', plot_th_1.shape)
print('shape of plot_cost', plot_cost.shape)

# plot mse in 3D chart
fig = plt.figure(figsize=[16, 12])
ax = fig.gca(projection='3d')
ax.set_title('MSE Cost')
ax.set_xlabel('Theta 0', fontsize=20)
ax.set_ylabel('Theta 1', fontsize=20)
ax.set_zlabel('Cost_MSE', fontsize=20)
ax.plot_surface(plot_th_0, plot_th_1, plot_cost, cmap=cm.hot)
plt.show()
# pull out lowest mse from array
min_mse_plot_cost = plot_cost.min()
print('min value of plot_cost', min_mse_plot_cost)
# fetch index of min cost value
ij_min = np.unravel_index(indices=plot_cost.argmin(), dims=plot_cost.shape)
print('minimum mse occur index ', ij_min)
print('min MSE for theta_0 at plot_th_0[113][87]', plot_th_0[113][87])
print('min MSE for theta_1 at plot_th_1[113][87]', plot_th_1[113][87])




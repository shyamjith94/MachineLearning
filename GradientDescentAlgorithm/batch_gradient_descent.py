"""batch gradient descent with with sympy"""
import machine_learning_data_visualization_3d_chart as prsfun
import numpy as np
import sympy


def partial_derivative_symbolic():
    """ partial derivative and symbolic computation"""
    a, b = sympy.symbols('x, y')

    multiplier = 0.1
    max_iter = 500
    params = np.array([1.8, 1.0])  # initial guess
    for i in range(max_iter):
        gradient_x = sympy.diff(prsfun.function_f(a, b), a).evalf(subs={a: params[0], b: params[1]})
        gradient_y = sympy.diff(prsfun.function_f(a, b), b).evalf(subs={a: params[0], b: params[1]})
        gradients = np.array([gradient_x, gradient_y])
        params = params - multiplier * gradients
        print(params, gradients, sep="<-->")

    """print result"""
    print("value of gradient descent array", gradients)
    print("Minimum occurs at x value of", params[0])
    print("Minimum occurs at y value of", params[1])
    """The optimized values I don't have to substitute them into our cost function which is F and then the"""
    print("The cost is ", prsfun.function_f(params[0], params[1]))


def partial_derivative_mathematical():
    """ partial derivative and mathematical computation"""

    multiplier = 0.1
    max_iter = 500
    params = np.array([1.8, 1.0])  # initial guess
    for i in range(max_iter):
        gradient_x = prsfun.partial_fpx(params[0], params[1])
        gradient_y = prsfun.partial_fpy(params[0], params[1])
        gradients = np.array([gradient_x, gradient_y])
        params = params - multiplier * gradients
        print(params, gradients, sep="<-->")

    """print result"""
    print("value of gradient descent array", gradients)
    print("Minimum occurs at x value of", params[0])
    print("Minimum occurs at y value of", params[1])
    """The optimized values I don't have to substitute them into our cost function which is F and then the"""
    print("The cost is ", prsfun.function_f(params[0], params[1]))


partial_derivative_mathematical()

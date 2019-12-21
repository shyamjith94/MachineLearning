import machine_learning_data_visualization_3d_chart as prsfun
import sympy

""" partial derivative and symbolic computation"""

a, b = sympy.symbols('x, y')
print('Our Cost Function f(x,y):- ', prsfun.function_f(a, b))
print('partial derivative with respective :-', sympy.diff(prsfun.function_f(a, b), a))
print('evaluation of function when value of f(x,y) at x=1.8 and y=1.0 :-',
      prsfun.function_f(a, b).evalf(subs={a: 1.8, b: 1.0}))
"""value of partial derivative"""
print('evaluation of slope when value of diff(f(x,y),x) at x=1.8 and y=1.0:-',
      sympy.diff(prsfun.function_f(a, b), a).evalf(subs={a: 1.8, b: 1.0}))

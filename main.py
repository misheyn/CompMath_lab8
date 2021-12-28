import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar
import scipy.optimize as optimize
from scipy.optimize import LinearConstraint

SOLX = 15

linear_constrain = LinearConstraint([1, 1], 0, 6)


def func1(x):
    return (x - 15) ** 2 + 5


def diff_func1(x):
    return 2 * x - 30


def func2(prm):
    x1, x2 = prm
    return 4 / x1 + 9 / x2 + x1 + x2


def b_func(prm) -> float:
    x1, x2 = prm
    return -(1 / x1 + x2 - 6)


def F(prms, r):
    return func2(prms) + r * b_func(prms)


def newton_method(x0, eps):
    x = x0
    n = 0
    while abs(diff_func1(x)) > eps:
        n += 1
        x = x - (diff_func1(x) / 2)
    return x, n


def scan_method(a, b, h):
    x = xm = a
    y = ym = func1(x)
    n = 0
    x_mins, y_mins = [], []
    while x < b:
        n += 1
        if y < ym:
            ym = y
            xm = x
            x_mins.append(xm)
            y_mins.append(ym)
        x = x + h
        y = func1(x)
    return xm, n, x_mins, y_mins


def local_min_newton(x, e):
    x0 = 199
    minx, count = newton_method(x0, e)
    print("xmin = ", minx, "ymin = ", func1(minx))
    print("number of iteration newton method: ", count)
    min_scp = minimize_scalar(func1, bounds=(2, 200), tol=e)

    plt.title("Single variable function: local min")
    plt.plot(x, func1(x), label='original function')
    plt.scatter(min_scp.x, min_scp.fun, c='g', label="scipy")
    plt.scatter(minx, func1(minx), c='r', label="newton method")
    plt.legend()
    plt.grid()
    plt.show()

    summa = abs(minx - min_scp.x)
    print('err = ', summa, "\n")


def global_min_scan(x, a, b, e):
    n = 1000
    h = (b - a) / n
    minx, count, sol_xm, sol_ym = scan_method(a, b, h)
    print("xmin = ", round(minx, 3), "ymin = ", func1(minx))
    print("number of iteration scan method: ", count)
    min_scp = minimize_scalar(func1, bounds=(a, b), tol=e)

    plt.title("Single variable function: global min")
    plt.plot(x, func1(x), label='original function')
    plt.scatter(sol_xm, sol_ym, label='minimize by scan method')
    plt.scatter(min_scp.x, min_scp.fun, c='g', label="scipy")
    plt.scatter(minx, func1(minx), c='r', label="min by scan method")
    plt.legend()
    plt.grid()
    plt.show()

    summa = abs(minx - min_scp.x)
    print('err = ', summa, "\n")


def min_barriers(x1, r, b, eps):
    while r * b_func(x1) > eps:
        r = b * r
        result = optimize.minimize(F, x1, r)
        x1 = result.x
    print("min = ", x1)
    x = np.linspace(-10, 10, 50)
    y = np.linspace(-10, 10, 50)
    X, Y = np.meshgrid(x, y)
    Z = func2([X, Y])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')
    plt.title('3D graph of a function of two variables')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()


def two_sol():
    initial_guess = [1, 1]
    result = optimize.minimize(func2, initial_guess, constraints=linear_constrain)
    if result.success:
        fitted_params = result.x
        print(fitted_params)
    else:
        raise ValueError(result.message)


X = np.arange(2, 200, 0.01)
e = 10e-3

local_min_newton(X, e)

global_min_scan(X, 2, 200, e)

res = minimize_scalar(func1, bounds=(2, 200), tol=e)
print("xmin = ", res.x, "ymin = ", res.fun)

min_barriers([1, 1], 1, 0.01, 10e-3)
min_barriers([1, 1], 1, 0.01, 10e-5)
min_barriers([1, 1], 1, 0.01, 10e-7)

two_sol()

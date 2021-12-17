import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar


def func1(x):
    return (x - 15) ** 2 + 5


def diff_func1(x):
    return 2 * x - 30


def func2(x1, x2):
    return 4 / x1 + 9 / x2 + x1 + x2


def newton_method(x0, eps):
    x = x0
    n = 0
    while abs(diff_func1(x)) > eps:
        n += 1
        x = x - (diff_func1(x) / 2)
    return x, func1(x), n


def scan_method(a, b, eps):
    x = xm = a
    y = ym = func1(x)
    n = 0
    while x < b:
        n += 1
        if y < ym:
            ym = y
            xm = x
        x = x + eps
        y = func1(x)
    return xm, ym, n


def draw_graphs(x, minx, miny, ttl, lbl):
    min_scp = minimize_scalar(func1, bounds=(2, 200), tol=e)
    plt.title(ttl)
    plt.plot(x, func1(x), label='original function')
    plt.scatter(min_scp.x, min_scp.fun, c='g', label="scipy")
    plt.scatter(minx, miny, c='r', label=lbl)
    plt.legend()
    plt.grid()
    plt.show()

    summa = abs(minx - min_scp.x)
    print('err = ', summa, "\n")


X = np.arange(2, 200, 0.01)
X0 = 55
hi = 0.01
e = 10e-3

res1x, res1y, count1 = newton_method(X0, e)
print("xmin = ", res1x, "ymin = ", res1y)
print("number of iteration newton method: ", count1)
draw_graphs(X, res1x, res1y, "Single variable function", "newton method")

res2x, res2y, count2 = scan_method(2, 200, e)
print("xmin = ", round(res2x, 3), "ymin = ", res2y)
print("number of iteration scan method: ", count2)
draw_graphs(X, res2x, res2y, "Single variable function", "scan method")

res = minimize_scalar(func1, bounds=(2, 200), tol=e)
print("xmin = ", res.x, "ymin = ", res.fun)

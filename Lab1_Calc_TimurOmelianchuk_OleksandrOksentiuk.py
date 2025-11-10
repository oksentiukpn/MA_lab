from typing import Callable
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

# Наша функція
function = lambda x: (np.e ** (-0.5* x)) * x

# x та y
x_values = np.linspace(-5, 5, 100)
y_values = function(x_values)

figure, axes = plt.subplots(figsize=(5, 5))

axes.plot(x_values, y_values)
axes.set_aspect('equal', adjustable='datalim')

# Візуальні налаштування
axes.grid(True)
axes.set_xlabel('x')
axes.set_ylabel('y')
axes.set_title(r'$y = e^{-0.5x} x$')

plt.show()

def numeric_derivative(func: Callable[[float], float], x: float, h: float = 1e-7) -> float:
    """
    Обчислює похідну за стандартною формулою.
    :param func: Функція f(x), яку потрібно диференціювати.
    :param x: Точка, в якій обчислюється похідна.
    :param h: Розмір приросту (default: 1e-7).
    :return: Наближене значення f'(x).
    >>> numeric_derivative(lambda x: x**2, 5)
    10.000000116860974
    >>> numeric_derivative(lambda x: np.sin(x), np.pi)
    np.float64(-0.9999999983634196)
    """
    return (func(x + h) - func(x)) / h

def numeric_derivative_cd(func: Callable[[float], float], x: float, h: float = 1e-7) -> float:
    """
    Обчислює похідну за формулою центральної різниці.
    :param func: Функція f(x), яку потрібно диференціювати.
    :param x: Точка, в якій обчислюється похідна.
    :param h: Розмір приросту (default: 1e-7).
    :return: Наближене значення f'(x).
    >>> numeric_derivative(lambda x: x**2, 5)
    10.000000116860974
    >>> numeric_derivative(lambda x: np.sin(x), np.pi)
    np.float64(-0.9999999983634196)
    """
    return (func(x + h) - func(x - h)) / (2*h)


x_values = np.linspace(0, np.pi / 2, 100) # Інтервал від 0 до 90 градусів
exact_derivs = np.cos(x_values)

# Пряма різниця
forward_derivs = numeric_derivative(np.sin, x_values)
forward_error = np.abs(forward_derivs - exact_derivs)
max_forward_error = np.max(forward_error)

# Центральна різниця
central_derivs = numeric_derivative_cd(np.sin, x_values)
central_error = np.abs(central_derivs - exact_derivs)
max_central_error = np.max(central_error)

print(f"Максимальна похибка (Пряма різниця): {max_forward_error}")
print(f"Максимальна похибка (Центральна різниця): {max_central_error}")

if __name__ == '__main__':
    import doctest
    print(doctest.testmod())

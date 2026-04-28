import numpy as np
import matplotlib.pyplot as plt

def gaussxw(N):
    """
    Calcula los pesos y los puntos de Gauss-Legendre.

    Examples:
        >>> x, w = gaussxw(2)
        >>> print(x)
        [-0.57735027  0.57735027]

    Args:
        N (int): Número de puntos de integración a calcular.

    Returns:
        tuple: Un par (x, w) que contiene los puntos y los pesos originales.
    """
    x, w = np.polynomial.legendre.leggauss(N)
    return x, w

def gaussxwab(limInf, limSup, x, w):
    """
    Escala los puntos y pesos de integración al intervalo [a, b].

    Args:
        limInf (float): Límite inferior de la integral (a).
        limSup (float): Límite superior de la integral (b).
        x (numpy.ndarray): Puntos de integración originales en [-1, 1].
        w (numpy.ndarray): Pesos de integración originales.

    Returns:
        tuple: Un par (xEsc, wEsc) con los puntos y pesos escalados al intervalo deseado.
    """
    xEsc = 0.5 * (limSup - limInf) * x + 0.5 * (limSup + limInf)
    wEsc = 0.5 * (limSup - limInf) * w
    return xEsc, wEsc

def f(x):
    """
    Calcula el valor de la función f(x) = sin(x^2).

    Args:
        x (float or numpy.ndarray): Valor o arreglo de valores de entrada.

    Returns:
        float or numpy.ndarray: El resultado de aplicar la función seno al cuadrado de x.
    """
    return np.sin(x**2)

def calInt(N, limInf, limSup):
    """
    Calcula la integral numérica de f(x) utilizando la cuadratura de Gauss-Legendre.

    Args:
        N (int): Número de puntos para la aproximación.
        limInf (float): Límite inferior de integración.
        limSup (float): Límite superior de integración.

    Returns:
        float: El valor aproximado de la integral definida.
    """
    # Obtener los pesos y puntos originales
    x, w = gaussxw(N)
    
    # Escalarlos al intervalo [limInf, limSup]
    xEsc, wEsc = gaussxwab(limInf, limSup, x, w)
    
    # Realizar la sumatoria de la cuadratura
    resultado = np.sum(wEsc * f(xEsc))
    
    return resultado

# --- Script Principal ---

limInf = 0
limSup = np.pi
numN = np.arange(1, 51)

# Calcular la integral para diferentes valores de N
resInt = [calInt(n, limInf, limSup) for n in numN]

print(f"El valor aproximado de la integral es: {resInt[-1]}")
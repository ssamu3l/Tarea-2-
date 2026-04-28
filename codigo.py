import numpy as np
import matplotlib.pyplot as plt

def gaussxw(N):
    x, w = np.polynomial.legendre.leggauss(N) # Calcular los pesos y los puntos
    return x, w

def gaussxwab(limInf, limSup, x, w):
    return 0.5 * (limSup - limInf) * x + 0.5 * (limSup + limInf), 0.5 * (limSup - limInf) * w # Escalar los pesos y puntos al intervalo [limInf, limSup]

def f(x):
    return np.sin(x**2) # Obtener la funcion seno

def calInt(N, limInf, limSup):
    x, w = gaussxw(N) # Obtener los pesos y puntos originales
    
    xEsc, wEsc = gaussxwab(limInf, limSup, x, w) # Escalarlos al intervalo [0, pi]

    resultado = np.sum(wEsc * f(xEsc)) #
    
    return resultado

limInf = 0 
limSup = np.pi
numN = np.arange(1, 51) 

resInt = [calInt(n, limInf, limSup) for n in numN]

print (f"El valor aproximado de la integral es: {resInt[-1]}")
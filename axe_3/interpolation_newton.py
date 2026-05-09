"""
Interpolation de Newton
"""

import numpy as np
import scipy.interpolate as inter
import matplotlib.pyplot as plt

# calcul des differences divisees
def div_diff(x, y):
    n = len(x)
    tab = np.zeros((n, n))
    tab[:,0] = y
    
    for j in range(1, n):
        for i in range(n-j):
            tab[i][j] = (tab[i+1][j-1] - tab[i][j-1]) / (x[i+j] - x[i])
    return tab

# evaluation du polynome (méthode de Horner)
def eval_newton(x, y, t):
    tab = div_diff(x, y)
    coeff = tab[0,:]
    n = len(x)
    
    # fonction pour un point
    def eval_un(t_val):
        res = coeff[n-1]
        for i in range(n-2, -1, -1):
            res = res * (t_val - x[i]) + coeff[i]
        return res
    
    # si c'est un seul point
    if type(t) in [float, int]:
        return eval_un(t)
    
    # si c'est une liste/array
    res = []
    for ti in t:
        res.append(eval_un(ti))
    return np.array(res)

# affichage moche mais qui marche
def affiche_table(x, y):
    tab = div_diff(x, y)
    n = len(x)
    
    print("\n--- Differences divisees ---")
    for i in range(n):
        ligne = f"x={x[i]:.2f} | "
        for j in range(n-i):
            ligne += f"{tab[i][j]:.4f}  "
        print(ligne)
    print("")

# graphique
def plot_newton(x, y):
    xx = np.linspace(x[0], x[-1], 300)
    yy = eval_newton(x, y, xx)
    
    # spline pour comparer (optionnel)
    spline = inter.CubicSpline(x, y)
    yy_spline = spline(xx)
    
    plt.figure()
    plt.plot(xx, yy, 'b-', label='Newton')
    plt.plot(xx, yy_spline, 'r--', label='Spline')
    plt.plot(x, y, 'ko', markersize=8, label='Points')
    plt.legend()
    plt.grid(True)
    plt.title('Interpolation de Newton')
    plt.show()


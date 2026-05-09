

import numpy as np
import matplotlib.pyplot as plt


def fixed_point_iteration(g, x0, precision=1e-6, max_iter=100, display=False, display_plot: plt.axes=None): #type: ignore
    """Cherche la racine d'une fonction f a l'aide d'une fonction g telle que g(x) = x et ce a partir d'un point de depart x0."""
    #initalization
    i=0
    I=[]
    X=[]
    X_next=[]
    errs=[]
    erreur=float('inf')
    X.append(x0)

    # iteration de point fixe
    while i < max_iter and erreur > precision:
        I.append(i)
        X_next.append(g(X[i]))

        X.append(X_next[i])

        erreur=abs(X_next[i] - X[i])
        errs.append(erreur)

        if  ( errs[i] < precision):
            print(f"\nConverged to {X_next[i]} in {i+1} iterations.")
            break

        i+=1

    #affichage de la table d'iterations
    if (display==True):
        display_plot.axis('tight')
        display_plot.axis('off')

        cell_content= list( zip( I, X[:len(I)], X_next, errs ) )
        the_table= display_plot.table(cellText= cell_content,
                           colLabels=["N° iteration", "x(n)", "x(n+1)", "erreur"],
                            loc='center')
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(7)

    return I, X, X_next, errs



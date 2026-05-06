

import numpy as np


def fixed_point_iteration(g, x0, precision=1e-6, max_iter=100):
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
            return I, X, X_next, errs

        i+=1
    
    #! TABLE D'ITERATIONS A AJOUTER

    return I, X, X_next, errs



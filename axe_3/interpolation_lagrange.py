import numpy as np


def polynome_de_lagrange(X_nuage,Y_nuage,x_test):
    """retourne la valeur du polynome de lagrange en x_test a partir du nuage de points (X_nuage,Y_nuage)"""
    p=0
    n=len(X_nuage)
    for i in range(n):
        l=1
        for j in range(n):
            if i!=j:
                l*= (x_test-X_nuage[j])/(X_nuage[i]-X_nuage[j])

        p+=Y_nuage[i]*l
    return p





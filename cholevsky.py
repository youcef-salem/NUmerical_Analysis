
import numpy as np

def definie_positif(A):
    """Verifie si la matrice A est definie positive en essayant la decomposition de Cholesky."""
    try:
        np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False

def cholesky(A):
    """Retourne la matrice L de la decomposition de Cholesky d'une matrice."""
    # on verifie si l'algorithme de cholesky est applicable a la matrice
    if not definie_positif(A):
        return None
    #sinon application de l'algorithme de cholesky
    else:
        n=len(A)
        L=np.zeros((n,n))
        for i in range(0, n):
            for j in range(0, i + 1):
                sum_part = 0
                if i == j:
                    for k in range(0, j):
                        sum_part += L[j, k] ** 2
                    L[i, j] = np.sqrt(A[j, j] - sum_part)
                else:
                    for k in range(0, j):
                        sum_part += L[i, k] * L[j, k]
                    L[i, j] = (A[i, j] - sum_part) / L[j, j]
        return L


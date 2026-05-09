import numpy as np
from typing import List


def norme_1(A: List[List[float]]) -> float:
    """Norme induite 1 (norme colonne max): max des sommes des valeurs absolues des colonnes."""
    A_np = np.array(A, dtype=float)
    return float(np.max(np.sum(np.abs(A_np), axis=0)))


def norme_inf(A: List[List[float]]) -> float:
    """Norme induite infinie (norme ligne max): max des sommes des valeurs absolues des lignes."""
    A_np = np.array(A, dtype=float)
    return float(np.max(np.sum(np.abs(A_np), axis=1)))


def norme_2(A: List[List[float]]) -> float:
    """Norme induite 2 (norme spectrale): plus grande valeur singuliere de A."""
    A_np = np.array(A, dtype=float)
    return float(np.linalg.norm(A_np, ord=2))

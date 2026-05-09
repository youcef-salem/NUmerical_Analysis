

#la methode de gauss seidel 
def gauss_seidel(A, b, x0=None, eps=1e-6, max_iter=1000):
    n = len(A)

    if x0 is None:
        x = [0.0 for _ in range(n)]
    else:
        x = x0.copy()

    for k in range(max_iter):
        x_old = x.copy()

        for i in range(n):
            somme = 0.0
            for j in range(n):
                if j != i:
                    somme += A[i][j] * x[j]

            x[i] = (b[i] - somme) / A[i][i]

        erreur = max(abs(x[i] - x_old[i]) for i in range(n))

        if erreur < eps:
            return x, k + 1

    return x, max_iter
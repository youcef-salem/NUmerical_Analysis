import numpy as np
import matplotlib.pyplot as plt

# descente de gradient
def descente_gradient(f, grad_f, x0, alpha=0.1, tol=1e-6, max_iter=1000):
    x = np.array(x0, dtype=float)
    hist_f = [f(x)]
    hist_x = [x.copy()]

    for i in range(max_iter):
        g = grad_f(x)
        if max(abs(g)) < tol:  # norme L-infini, plus simple
            break
        x = x - alpha * g
        hist_f.append(f(x))
        hist_x.append(x.copy())
        if i > 500 and abs(hist_f[-1] - hist_f[-100]) < 1e-8: 
            print(f"stop early a {i}")
            break
    return x, hist_f, hist_x

# regression 
def regression():
    print("regression lineaire avec GD")
    np.random.seed(7)
    n = 50
    X = np.linspace(0, 5, n)
    y = 2.5*X + 1.2 + np.random.normal(0, 0.8, n)
    
    def loss(param):
        w,b = param
        return np.mean((w*X + b - y)**2)
    
    def grad(param):
        w,b = param
        err = w*X + b - y
        return np.array([2*np.mean(err*X), 2*np.mean(err)])
    
    # essai avec different pas
    for alpha in [0.01, 0.05, 0.1]:
        x_opt, hist_f, _ = descente_gradient(loss, grad, [0.,0.], alpha=alpha, max_iter=1000)
        w,b = x_opt
        print(f"alpha={alpha}: w={w:.3f}, b={b:.3f}, loss={loss(x_opt):.5f}")
    
    # meilleur non
    x_opt, hist_f, _ = descente_gradient(loss, grad, [0.,0.], alpha=0.05, max_iter=2000)
    w,b = x_opt
    
    print(f"\nresultat final: y = {w:.3f}x + {b:.3f}")
    
    plt.figure()
    plt.scatter(X, y, alpha=0.5)
    xx = np.linspace(0,5,100)
    plt.plot(xx, w*xx + b, 'r-', label='GD')
    plt.legend()
    plt.grid(True)
    plt.show()


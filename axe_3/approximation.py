"""
Approximation - Moindres carrés et Splines
"""
import numpy as np
import scipy.interpolate as inter
import matplotlib.pyplot as plt



def poly_fit(x, y, d):
    """ajuste un polynome degre d aux donnees"""
    n = len(x)
    A = np.zeros((n, d+1))
    for i in range(d+1):
        A[:,i] = x**i

    # numpy fait le boulot
    coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    return coeffs

def eval_poly(c, t):
    """evalue le polynome"""
    t = np.asarray(t)
    res = np.zeros_like(t)
    for i, ci in enumerate(c):
        res += ci * t**i
    return res

# moindres carrés
def test_mc():
    print("\n" + "="*50)
    print("MOINDRES CARRES")
    print("="*50)
    
    # fabrication des donnees
    np.random.seed(42)
    x = np.linspace(0, 5, 25)
    y = 2*x**2 - 3*x + 1 + np.random.normal(0, 2, len(x))
    
    print(f"\n{len(x)} points, fonction cachee: 2x^2 - 3x + 1\n")
    
    # graphique
    plt.figure(figsize=(10,6))
    plt.scatter(x, y, c='k', s=30, label='donnees bruitees')
    
    # vrai fonction
    xx = np.linspace(0,5,200)
    plt.plot(xx, 2*xx**2 - 3*xx + 1, 'k--', alpha=0.5, label='verite terrain')
    
    # essai differents degres
    for d in [1,2,3]:
        c = poly_fit(x, y, d)
        yy = eval_poly(c, xx)
        
        # calcul erreur
        err = np.sqrt(np.mean((eval_poly(c, x) - y)**2))
        
        print(f"degre {d}: coeffs={np.round(c,3)}, RMSE={err:.4f}")
        
        if d==1:
            plt.plot(xx, yy, 'r-', label=f'degre {d} (err={err:.3f})')
        elif d==2:
            plt.plot(xx, yy, 'g-', label=f'degre {d} (err={err:.3f})')
        else:
            plt.plot(xx, yy, 'b-', label=f'degre {d} (err={err:.3f})')
    
    plt.legend()
    plt.grid(True)
    plt.title('Ajustement polynomial')
    plt.show()

# Splines de lissage
def test_spline():
    print("\n" + "="*50)
    print("SPLINES DE LISSAGE")
    print("="*50)
    
    # donnees bruitees
    np.random.seed(0)
    x = np.linspace(0, 2*np.pi, 30)
    y = np.sin(x) + np.random.normal(0, 0.2, len(x))
    
    print(f"\n{len(x)} points autour de sin(x)\n")
    
    xx = np.linspace(0, 2*np.pi, 300)
    plt.figure(figsize=(10,6))
    plt.scatter(x, y, c='k', s=30, label='donnees')
    plt.plot(xx, np.sin(xx), 'k--', alpha=0.5, label='sin(x) reel')
    
    # test differente valeurs de lissage
    for s, col in zip([0.1, 1.0, 5.0], ['red', 'green', 'blue']):
        sp = inter.UnivariateSpline(x, y, s=s)
        yy = sp(xx)
        
        # erreur
        err = np.sqrt(np.mean((sp(x) - y)**2))
        
        print(f"s={s} -> RMSE={err:.4f}")
        plt.plot(xx, yy, color=col, lw=2, label=f's={s} (err={err:.3f})')
    
    plt.legend()
    plt.grid(True)
    plt.title('Lissage par splines')
    plt.show()

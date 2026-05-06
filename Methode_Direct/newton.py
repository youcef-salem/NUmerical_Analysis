
import numpy as np
import matplotlib.pyplot as plt

def newton(f, a, precision=0.001, display=False, display_plot: plt.axes=None): #type: ignore
    """Cherche la racine de f a partir d'un point a avec la methode de newton. Affiche aussi une table d'iterations si demandée."""
    
    #!VERIFICATION DE LA CONVERGENCE A AJOUTER
    
    #initialization
    x=list()
    x_next=list()
    iterations= list()
    erreurs=list()

    # la derivée de f
    def f_derive(x):
        A=f(x- (precision/2))
        B=f(x+ (precision/2))

        if (A==B): #cas ou f'(x) = 0
             print("encountered flat tangent")
             return 999999999999 #!
        else:
            return (B-A)/ precision
    

    #valeurs initiales
    n=0
    x.append(a)
    x_next.append( x[0]- f(x[0])/f_derive(x[0]) )
    erreurs.append( np.abs(x[n]-x_next[n]) )

    #les iterations de newton
    while(erreurs[n]>precision and n<100):
        #journalisation console
        print(f"Precedent: {x[n]} Suivant:{x_next[n]}")
        print(f"erreur: {erreurs[n]}")

        x.append(x_next[n])
        n=n+1
        x_next.append(x[n] - f(x[n]) / f_derive(x[n]))
        erreurs.append( np.abs(x[n]-x_next[n]) )

    #affichage de la table d'iterations
    if (display==True):
        display_plot.axis('tight')    
        display_plot.axis('off')

        cell_content= list( zip( list(range(1,n+1)) , x , x_next , erreurs ) )
        the_table= display_plot.table(cellText= cell_content ,
                           colLabels=["N° iteration", "x(n)", "x(n+1)", "erreur"],
                            loc='center')
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(7)

    return x

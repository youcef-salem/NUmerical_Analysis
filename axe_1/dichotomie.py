import numpy as np
import matplotlib.pyplot as plt

def dichotomie(f,a,b,precision=0.1, display=False, display_plot: plt.axes=None): #type: ignore
    """Cherche la racine de f dans un intevalle [a, b] avec la methode de dichotomie. Affiche aussi une table d'iterations si demandée."""
    #initialization
    i=0
    I=[]
    A=[]
    B=[]
    errs=[]
    
    #la methode de dichotomie
    while (np.abs(b-a) > precision):
        #journalisation console
        print("Iteration ",i,":")
        print(" Intervalle actuel: [ ",a,", ",b,"]")
        erreur= abs(b-a)- precision
        print(" Erreur: ",erreur,"\n")
        
        #choix du nouveau demi-intervalle
        if ( f(a) == 0):
            return a
        elif ( f(b) == 0):
            return b
        elif ( f(a) * f((a+b)/2) < 0 ):
                b= (a+b)/2
        else:
                a= (a+b)/2

        #enregistrement des valeurs de l'iteration
        I.append(i)
        A.append(a)
        B.append(b)
        errs.append(erreur)

        i+=1

    
    #affichage de la table d'iterations
    if (display==True):
        display_plot.axis('tight')    
        display_plot.axis('off')

        cell_content= list( zip( I, A, B, errs ) )
        the_table= display_plot.table(cellText= cell_content ,
                           colLabels=["N° iteration", "valeur a", "valeur b", "erreur"],
                            loc='center')
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(7)

    
    return a

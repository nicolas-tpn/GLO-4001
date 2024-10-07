import scipy.io as sc
from scipy.optimize import fmin
import numpy as np
import matplotlib.pyplot as plt

# Question 2.1 : Génération d'une image
L1 = np.array([-0.4,0,2.4])
L2 = np.array([0,0,2])
L3 = np.array([0.4,0,2.4])

def reprojection(H,focale,L):
    # On considère que la pose que l'on a est de la forme (x,z,theta)
    matrice_intrinseque = np.array([[1,0,0,0],[0,1,0,0],[0,0,1/focale,0]])
    x = H[0]
    z = H[1]
    theta = H[2]
    matrice_extrinseque = np.array([
        [np.cos(theta),0,np.sin(theta),-x],
        [0,1,0,0],
        [-np.sin(theta),0,np.cos(theta),-z],
        [0,0,0,1]])
    
    C = np.array([])
    
    for Li in L:
        C.append(matrice_extrinseque@matrice_intrinseque@Li)

    return C

# Question 2.2 : Localisation par la minimisation de l'erreur de reprojection



L = np.array([L1,L2,L3])
print(reprojection())
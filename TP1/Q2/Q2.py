import scipy.io as sc
from scipy.optimize import fmin
import numpy as np
import matplotlib.pyplot as plt

# Question 2.1 : Génération d'une image
L1 = np.array([-0.4, 0, 2.4])
L2 = np.array([0, 0, 2])
L3 = np.array([0.4, 0, 2.4])


def reprojection(H, focale, L):

    # Passage des points en coordonnées homogènes

    if L.ndim == 1:
        L = L.reshape(1, -1)
    L = np.hstack([L, np.ones((L.shape[0], 1))])

    # On considère que la pose que l'on a est de la forme (x,y,z,theta)
    matrice_intrinseque = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1 / focale, 0]]
    )
    x = H[0]
    y = H[1]
    z = H[2]
    theta = H[3]
    matrice_extrinseque = np.array(
        [
            [np.cos(theta), 0, np.sin(theta), -x],
            [0, 1, 0, -y],
            [-np.sin(theta), 0, np.cos(theta), -z],
            [0, 0, 0, 1],
        ]
    )

    C = np.empty((0, 4))

    for Li in L:
        new_line = matrice_extrinseque @ matrice_intrinseque @ Li
        new_line = new_line.reshape(1, -1)
        C = np.vstack([C, new_line])

    return C


# Question 2.2 : Localisation par la minimisation de l'erreur de reprojection


def somme_des_residuels_au_carre(pose_camera, focale, L, C):
    # pose_camera de la forme : [x,z,theta]
    # L contient les points dans le repère monde
    # C contient les points dans le repère caméra

    somme = 0

    for index, Li in enumerate(L):
        somme += (np.abs(C[index] - reprojection(pose_camera, focale, Li))) ** 2

    return somme


# On décale légèrement la pose par rapport à la pose initiale
pose_initiale_camera = [0.2, 0, 0.19, 0.21]
focale = 1500
L = np.array([L1, L2, L3])
C = reprojection(pose_initiale_camera, focale, L)

pose_solution = fmin(
    somme_des_residuels_au_carre,
    pose_initiale_camera,
    args=(focale, L, C),
    maxiter=1000,
)

print(pose_solution)

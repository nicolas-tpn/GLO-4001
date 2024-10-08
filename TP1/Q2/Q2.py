import scipy.io as sc
from scipy.optimize import fmin
import numpy as np
import matplotlib.pyplot as plt

# Question 2.1 : Génération d'une image
L1 = np.array([-0.4, 0, 2.4])
L2 = np.array([0, 0, 2])
L3 = np.array([0.4, 0, 2.4])
CAM = np.array([0, 0, -3])


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
        somme += (np.linalg.norm(C[index] - reprojection(pose_camera, focale, Li))) ** 2

    return somme


# On décale légèrement la pose par rapport à la pose initiale
pose_initiale_camera = [0.2, 0, 0.19, 0.21]
focale = 1500
L = np.array([L1, L2, L3])
C = reprojection([0, 0, 0, 0], focale, L)


pose_solution = fmin(
    somme_des_residuels_au_carre,
    pose_initiale_camera,
    args=(focale, L, C),
    maxiter=1000,
)


print(pose_solution)


# Question 2.3 : Impact d'une erreur d'association de données

pose_camera = [0,0,-3]

# Inversion de L2 et L3
false_L2 = L3
false_L3 = L2

# Calcul des angles entre caméra et points
vect_L1 = [L1[0]-pose_camera[0], L1[2]-pose_camera[2]]
vect_L2 = [false_L2[0]-pose_camera[0], false_L2[2]-pose_camera[2]]
vect_L3 = [false_L3[0]-pose_camera[0], false_L3[2]-pose_camera[2]]

cos_angle_L1_L2 = np.dot(vect_L1, vect_L2) / (np.linalg.norm(vect_L1) * np.linalg.norm(vect_L2))
angle_L1_L2 = np.arccos(cos_angle_L1_L2)

cos_angle_L2_L3 = np.dot(vect_L2, vect_L3) / (np.linalg.norm(vect_L2) * np.linalg.norm(vect_L3))
angle_L2_L3 = np.arccos(cos_angle_L2_L3)

# Calcul des distances entre points
d_L1_L2 = np.linalg.norm([false_L2[0]-L1[0], false_L2[2]-L1[2]])
d_L2_L3 = np.linalg.norm([false_L2[0]-false_L3[0], false_L2[2]-false_L3[2]])

# Calcul des hauteurs
h_L1_L2 = d_L1_L2/(2*np.tan(angle_L1_L2))
h_L2_L3 = d_L2_L3/(2*np.tan(angle_L2_L3))

# Calcul des médiatrices
point_median_L1_L2 = [(L1[0]+false_L2[0])/2, (L1[2]+false_L2[2])/2]
point_median_L2_L3 = [(false_L3[0]+false_L2[0])/2, (false_L3[2]+false_L2[2])/2]

pente_L1_L2 = (false_L2[2]-L1[2])/(false_L2[0]/L1[0])
pente_L2_L3 = (false_L3[2]-false_L2[2])/(false_L3[0]/false_L2[0])

pente_med_L1_L2 = -1/pente_L1_L2
pente_med_L2_L3 = -1/pente_L2_L3



# Question 2.4 : Impact du bruit sur l'estimation des repères via une approche de type Monte Carlo

plt.plot(L1[0],L1[2],"r+")
plt.plot(point_median_L1_L2[0],point_median_L1_L2[1],"r+")
plt.plot(point_median_L2_L3[0],point_median_L2_L3[1],"r+")
plt.plot(L2[0],L2[2],"r+")
plt.plot(L3[0],L3[2],"r+")
plt.plot(CAM[0],CAM[2],"r+")
plt.show()

# for i in range(1,7):
#     for _ in range(1000):


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

    # ok la partie translation c'est bon en fait, c'est le theta que je comprends pas tellement en fait là, pourquoi c'est directement la quatrième coordonnée ? (à moins que j'ai loupé un truc du cours encore)
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
        # résultat en coordonnées homogènes
        new_line = matrice_intrinseque @ matrice_extrinseque @ Li
        new_line = new_line.reshape(1, -1)
        
        C = np.vstack([C, new_line])

    w = C[:, 3]
    # print("c : ", C)
    C_reel = C[:, :3] / w.reshape(-1, 1)
        
    return C_reel


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
pose_initiale_camera = [0.2, 0, -0.19, 0.21]
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

pose_camera = [0, 0, -3]

# Inversion de L2 et L3
false_L2 = L3
false_L3 = L2

# pour faire de manière générale, avec un bruit de 0 pour 2.3

def estimation_position_bruitee(pose_camera, L1, L2, L3, bruit):
    # Calcul des angles entre caméra et points
    vect_L1 = [L1[0] - pose_camera[0], L1[2] - pose_camera[2]]
    vect_L2 = [L2[0] - pose_camera[0], L2[2] - pose_camera[2]]
    vect_L3 = [L3[0] - pose_camera[0], L3[2] - pose_camera[2]]
    
    cos_angle_L1_L2 = np.dot(vect_L1, vect_L2) / (
        np.linalg.norm(vect_L1) * np.linalg.norm(vect_L2)
    )
    angle_L1_L2 = np.arccos(cos_angle_L1_L2)
    
    # Ici on ajoute un moins devant le résultat pour respecter l'ordre de lecture
    cos_angle_L2_L3 = - np.dot(vect_L2, vect_L3) / (
        np.linalg.norm(vect_L2) * np.linalg.norm(vect_L3)
    )
    angle_L2_L3 = np.arccos(cos_angle_L2_L3)
    
    # Calcul des distances entre points
    d_L1_L2 = np.linalg.norm([L2[0] - L1[0], L2[2] - L1[2]])
    d_L2_L3 = np.linalg.norm([L2[0] - L3[0], L2[2] - L3[2]])
    
    # Calcul des hauteurs
    h_L1_L2 = d_L1_L2 / (2 * np.tan(angle_L1_L2))
    h_L2_L3 = d_L2_L3 / (2 * np.tan(angle_L2_L3))
    
    print(h_L1_L2)
    print(h_L2_L3)

    # Calcul des médiatrices
    point_median_L1_L2 = [(L1[0] + L2[0]) / 2, (L1[2] + L2[2]) / 2]
    point_median_L2_L3 = [(L3[0] + L2[0]) / 2, (L3[2] + L2[2]) / 2]
    
    if L2[2] == L1[2]:
        pente_L1_L2 = None
    else:
        pente_L1_L2 = (L2[2] - L1[2]) / (L2[0] - L1[0])
    
    if L2[2] == L3[2]:
        pente_L2_L3 = None
    else:
        pente_L2_L3 = (L3[2] - L2[2]) / (L3[0] - L2[0])
    
    if pente_L1_L2 is None:
        pente_med_L1_L2 = 0
    else:
        pente_med_L1_L2 = -1 / pente_L1_L2
    
    if pente_L2_L3 is None:
        pente_med_L2_L3 = 0
    else:
        pente_med_L2_L3 = -1 / pente_L2_L3
    
    vect_med_L1_L2 = np.array([pente_med_L1_L2, 1])
    vect_med_L2_L3 = np.array([pente_med_L2_L3, 1])
    
    vect_unit_med_L1_L2 = vect_med_L1_L2/np.linalg.norm(vect_med_L1_L2)
    vect_unit_med_L2_L3 = vect_med_L2_L3/np.linalg.norm(vect_med_L2_L3)
    
    centre_cercle_L1_L2 = point_median_L1_L2-vect_unit_med_L1_L2*h_L1_L2
    centre_cercle_L2_L3 = point_median_L2_L3-vect_unit_med_L2_L3*h_L2_L3
    
    rayon_cercle_L1_L2 = np.sqrt((L2[0] - centre_cercle_L1_L2[0])**2 + (L2[2] - centre_cercle_L1_L2[1])**2)
    rayon_cercle_L2_L3 = np.sqrt((L3[0] - centre_cercle_L2_L3[0])**2 + (L3[2] - centre_cercle_L2_L3[1])**2)

    # Calcul des points d'intersection
    # https://lucidar.me/fr/mathematics/how-to-calculate-the-intersection-points-of-two-circles/

    distance_cercles = np.sqrt((centre_cercle_L2_L3[0] - centre_cercle_L1_L2[0])**2 + (centre_cercle_L2_L3[1] - centre_cercle_L1_L2[1])**2)

    if distance_cercles > rayon_cercle_L1_L2 + rayon_cercle_L2_L3 or distance_cercles < abs(rayon_cercle_L1_L2 - rayon_cercle_L2_L3):
        return 
    
    a = (rayon_cercle_L1_L2**2 - rayon_cercle_L2_L3**2 + distance_cercles**2) / (2 * distance_cercles)
    h = np.sqrt(rayon_cercle_L1_L2**2 - a**2)

    P = centre_cercle_L1_L2 + a * (centre_cercle_L2_L3 - centre_cercle_L1_L2) / distance_cercles

    x3 = P[0] + h *(centre_cercle_L2_L3[1] - centre_cercle_L1_L2[1]) / distance_cercles
    y3 = P[1] - h *(centre_cercle_L2_L3[0] - centre_cercle_L1_L2[0]) / distance_cercles

    x4 = P[0] - h *(centre_cercle_L2_L3[1] - centre_cercle_L1_L2[1]) / distance_cercles
    y4 = P[1] + h *(centre_cercle_L2_L3[0] - centre_cercle_L1_L2[0]) / distance_cercles

    point_intersection = np.array([x4, y4])

    print("Erreur d'estimation : ", np.linalg.norm([x4 - pose_camera[0], y4 - pose_camera[2]]))

    return [rayon_cercle_L1_L2, rayon_cercle_L2_L3, centre_cercle_L1_L2, centre_cercle_L2_L3, point_median_L1_L2, point_median_L2_L3, point_intersection] 

estimation_position = estimation_position_bruitee(pose_camera, L1, false_L2, false_L3, 0)
theta = np.linspace(0, 2*np.pi, 180)

# Points et rayons des cercles (exemple basé sur la fonction estimation_position_bruitee)
x_L1_L2 = estimation_position[0] * np.cos(theta) + estimation_position[2][0]
y_L1_L2 = estimation_position[0] * np.sin(theta) + estimation_position[2][1]
x_L2_L3 = estimation_position[1] * np.cos(theta) + estimation_position[3][0]
y_L2_L3 = estimation_position[1] * np.sin(theta) + estimation_position[3][1]

plt.plot(estimation_position[2][0], estimation_position[2][1], 'bo', label="Centre du cercle L1-L2")
plt.plot(estimation_position[3][0], estimation_position[3][1], 'go', label="Centre du cercle L2-L3")
plt.plot(x_L1_L2, y_L1_L2, 'b-', label="Cercle L1-L2")
plt.plot(x_L2_L3, y_L2_L3, 'g-', label="Cercle L2-L3")
plt.plot(L1[0], L1[2], 'r+', label="Repère L1", markersize=10)
plt.plot(L2[0], L2[2], 'r+', label="Repère L2", markersize=10)
plt.plot(L3[0], L3[2], 'r+', label="Repère L3", markersize=10)
plt.plot(CAM[0], CAM[2], 'kx', label="Caméra", markersize=10)
plt.plot(estimation_position[4][0], estimation_position[4][1], 'c+', label="Point médian L1-L2", markersize=10)
plt.plot(estimation_position[5][0], estimation_position[5][1], 'm+', label="Point médian L2-L3", markersize=10)
plt.plot(estimation_position[6][0], estimation_position[6][1], 'r.', label="Point d'intersection", markersize=10)
plt.axis('scaled')
plt.xlabel('x')
plt.ylabel('z')
plt.title('Tracé des repères, des cercles et de la caméra')
plt.legend(loc='best')
plt.grid(True)

plt.show()

# Question 2.4 : Impact du bruit sur l'estimation des repères via une approche de type Monte Carlo

ecart_type = 3

# Rappel : C contient les points L reprojetés dans le plan image

# print("reproj", reprojection([0, 0, -2, 0], focale, L))

for i in range(1, 8):
    C = reprojection([0, 0, L2[2]-i, 0], focale, L)
    pose_camera = [0.2, 0, L2[2]-i, 0.19]

    for k in range(1000):
        bruit_gaussien = np.random.normal(0, ecart_type)

        # Construction de C bruité
        C_bruit = C.copy()
        C_bruit[:, 0] += bruit_gaussien

        pose_solution = fmin(
            somme_des_residuels_au_carre,
            pose_camera,
            args=(focale, L, C_bruit),
            maxiter=1000,
            disp=False
        )

        plt.plot(pose_solution[0], pose_solution[2], "r.")

plt.show()

import random

import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.io as scio
import tqdm

x = np.array([0, 1.5])
delta_t = 0.5
sigma_x2 = 2**2
sigma_x_dot_2 = 0.2**2
sigma_x_x_dot = 0.2 * 2
z = 0.2

P = np.array([[sigma_x2, sigma_x_x_dot], [sigma_x_x_dot, sigma_x_dot_2]])
P0 = np.array([[sigma_x2, sigma_x_x_dot], [sigma_x_x_dot, sigma_x_dot_2]])
# matrix_lambda = np.array([[z*10**3/182.25*10**3, 0]])
matrix_lambda = np.array([[1/1000, 0]])
Cw = 25*10**-6

phi = np.array([[1, delta_t], [0, 1]])

table = np.empty((0, 4))

x_hat = x

# Question 2.4
z = 5
matrix_lambda = np.array([[0, 1]])
Cw = 0.01


for i in range(300):

    # étape de prédiction
    x_hat = phi @ x_hat # 1
    # P = phi @ P @ np.transpose(phi) + P0 # 2
    P = phi @ P @ np.transpose(phi) # 2

    if i == 243:
        # étape de mise à jour

        # Question 2.3.4
        P = np.array([[P[0][0], 0], [0, P[1][1]]])

        z_hat = matrix_lambda @ x_hat # 3
        r = z - z_hat # 4
        K = P @ np.transpose(matrix_lambda) @ np.linalg.inv(matrix_lambda @ P @ np.transpose(matrix_lambda) + Cw) # 5
        x_hat = x_hat + K @ r # 6
        P = (np.identity(2) - K @ matrix_lambda) @ P # 7
    table = np.vstack((table, np.array([[i, P[0][0], P[0][1], P[1][1]]])))

print(x_hat)

index = table[:, 0]
pos = table[:, 1]      
cov = table[:, 2]        
vit = table[:, 3]  

plt.figure(figsize=(8, 6))  
plt.plot(index, pos, label="Position", marker='o')
plt.plot(index, cov, label="Covariance", marker='s')
plt.plot(index, vit, label="Vitesse", marker='^')

plt.title("Courbes des trois valeurs en fonction de l'index")
plt.xlabel("Index")
plt.ylabel("Valeurs")
plt.legend()
plt.grid(True)
plt.show()

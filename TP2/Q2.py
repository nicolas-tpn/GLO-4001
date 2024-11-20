import random

import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.io as scio
import tqdm

x = np.array([0, 1.5])
delta_t = 0.5
sigma_x2 = 2**2
sigma_x_dot_2 = 0.02**2
sigma_x_x_dot = 0.02 * 2
z = 0.2

P = np.array([[sigma_x2, sigma_x_x_dot], [sigma_x_x_dot, sigma_x_dot_2]])
P0 = np.array([[sigma_x2, sigma_x_x_dot], [sigma_x_x_dot, sigma_x_dot_2]])
matrix_lambda = np.array([[0.2/182.25*10**-3, 0]])
Cw = 25*10**-6


phi = np.array([[1, delta_t], [0, 1]])

for _ in range(100):

    # étape de prédiction
    x_hat = phi @ x # 1
    P = phi @ P @ np.transpose(phi) + P0 # 2

    # étape de mise à jour
    z_hat = matrix_lambda @ x_hat # 3
    r = z_hat - z # 4
    K = P @ np.transpose(matrix_lambda) @ np.linalg.inv(matrix_lambda @ P @ np.transpose(matrix_lambda) + Cw) # 5
    x_hat = x_hat + K @ r # 6
    P = (np.identity(2) - K @ matrix_lambda) @ P # 7


print(P)

# for _ in range(143):
#     P = phi @ P @ np.transpose(phi) + P0

# print(P)

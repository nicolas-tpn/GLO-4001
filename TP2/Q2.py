import random

import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.io as scio
import tqdm

delta_t = 0.5
lambda_x2 = 2**2
lambda_x_dot_2 = 0.02**2
lambda_x_x_dot = 0.02 * 2

P = np.array([[lambda_x2, lambda_x_x_dot], [lambda_x_x_dot, lambda_x_dot_2]])
P0 = np.array([[lambda_x2, lambda_x_x_dot], [lambda_x_x_dot, lambda_x_dot_2]])

phi = np.array([[1, delta_t], [0, 1]])

for _ in range(100):
    P = phi @ P @ np.transpose(phi) + P0

print(P)

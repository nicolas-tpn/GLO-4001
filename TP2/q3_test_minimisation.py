import random

import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.io as scio
import tqdm
from scipy.optimize import minimize
from scipy.optimize import differential_evolution

from numba import njit

import raycast as rc


def wrap_to_pi(angle):
    """
    Wrap an angle in [-pi, pi]
    :param angle: angle to wrap
    :return: wrapped angle
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi


@njit
def norm_pdf(x, mu, sigma):
    return 1 / np.sqrt(2 * np.pi * sigma**2) * np.exp(-((x - mu) ** 2) / (2 * sigma**2))


def to_minimize(initial_guess):
    s_lidar = initial_guess[0]
    s_angle = initial_guess[1]
    r_eff = initial_guess[2]
    
    map_data = scio.loadmat("Carte.mat")
    map_polygon = map_data["Carte"]

    # load trajectory data
    trajectory_data = scio.loadmat("Q1Trajectoire.mat")

    dt = trajectory_data["dT"].item()
    x_pose = trajectory_data["xPose"].squeeze()
    y_pose = trajectory_data["yPose"].squeeze()
    angles_pose = trajectory_data["anglePose"].squeeze()
    poses = np.stack([x_pose, y_pose, angles_pose], axis=0)

    v = trajectory_data["V"].squeeze()
    omega = trajectory_data["omega"].squeeze()
    lidar_measurements = trajectory_data["Lidar"]
    compas = trajectory_data["Compas"].squeeze()
    
    ray_length = 20
    s_v = 0.01
    s_omega = 0.05
    s_compas = 0.01
    n_step = y_pose.shape[0] - 1
    num_particles = 200
    
    # particles = np.zeros((num_particles, 2))
    particles = np.zeros((num_particles, 3))
    weights = np.ones(num_particles)
    positions = np.zeros((n_step, 2))
    
    error = np.zeros(n_step)
    
    # limites de la carte
    x_min, x_max = map_polygon[0, :].min(), map_polygon[0, :].max()
    y_min, y_max = map_polygon[1, :].min(), map_polygon[1, :].max()

    # initialiser particules, avec distribution uniforme dans la carte
    particles[:, 0] = np.random.uniform(x_min, x_max, num_particles)
    particles[:, 1] = np.random.uniform(y_min, y_max, num_particles)
    particles[:, 2] = np.random.uniform(-np.pi, np.pi, num_particles)
    
    for step in tqdm.tqdm(range(n_step)):
        lidar_reading = rc.fast_four_way_raycast(
            map_polygon,
            np.array([x_pose[step], y_pose[step]]),
            angles_pose[step],
            ray_length,
        )
        angle = compas[step] + np.random.normal(0, s_angle)

        x1, y1 = np.min(lidar_reading[0], axis=1)
        x2, y2 = np.min(lidar_reading[1], axis=1)
        x3, y3 = np.min(lidar_reading[2], axis=1)
        x4, y4 = np.min(lidar_reading[3], axis=1)

        m1 = (y3 - y1) / (x3 - x1)
        m2 = (y4 - y2) / (x4 - x2)
        b1 = y1 - m1 * x1
        b2 = y2 - m2 * x2

        x_robot = (b2 - b1) / (m1 - m2)
        y_robot = m1 * x_robot + b1

        for particle_idx in range(num_particles):

            # simulate motion

            x_sim = (
                particles[particle_idx][0]
                + (v[step] + np.random.normal(0, s_v, 1))
                * np.cos(particles[particle_idx][2])
                * dt
            )
            y_sim = (
                particles[particle_idx][1]
                + (v[step] + np.random.normal(0, s_v, 1))
                * np.sin(particles[particle_idx][2])
                * dt
            )
            
            # measurement update
            particles[particle_idx][0] = x_sim
            particles[particle_idx][1] = y_sim
            particles[particle_idx][2] = angle

            # p(z|x)
            proba_x = norm_pdf(x_robot, x_sim, s_lidar)
            proba_y = norm_pdf(y_robot, y_sim, s_lidar)
            
            proba = proba_x * proba_y
            
            weights[particle_idx] *= proba

        # normalize weights
        if np.sum(weights) == 0:
            weights = np.ones(num_particles) / num_particles
        else: # normalize weights
            weights /= np.sum(weights)

        particles = np.array(particles)
        particles = particles.squeeze()

        x_mean = np.average(particles[:, 0], weights=weights)
        y_mean = np.average(particles[:, 1], weights=weights)
        positions[step][0] = x_mean
        positions[step][1] = y_mean
        
        error[step] += np.linalg.norm(np.array([x_mean, y_mean]) - np.array([x_pose[step], y_pose[step]]))

        # apply resampling if necessary
        n_eff = 1 / np.sum(weights ** 2)
        if n_eff < r_eff * num_particles:
            indices = np.random.choice(np.arange(num_particles), size=num_particles, replace=True, p=weights)
            particles = particles[indices, :]
            weights = np.ones(num_particles) / num_particles
            
    return np.sum(error)


if __name__ == "__main__":
    s_lidar = 10
    r_eff = 0.01
    s_angle = 10
    initial_guess = [s_lidar, s_angle, r_eff]
    bounds = [(0, 10), (0, 10), (0, 1)]
    
    result = differential_evolution(to_minimize, bounds)
    # result = minimize(to_minimize, initial_guess, method='L-BFGS-B', bounds=bounds)
    
    # Afficher les résultats 
    print("Valeurs minimisées des variables :", result.x) 
    print("Valeur minimale de la fonction objectif :", result.fun)

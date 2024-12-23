import random

import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.io as scio
import tqdm
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


if __name__ == "__main__":
    # load map
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
    
    # parameters
    ray_length = 20
    s_lidar = 0.01
    s_v = 0.01
    s_omega = 0.05
    s_compas = 0.01
    n_step = y_pose.shape[0] - 1
    # TODO change to show progress
    show_progress = True
    show_every = 10

    # parameters to tune
    s_angle = 0.1  # TODO parameter to tune
    num_particles = 200  # TODO parameter to tune
    r_eff = 0.5  # TODO parameter to tune

    manager = plt.get_current_fig_manager()
    manager.full_screen_toggle()

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

    # initialiser poids
    weights = np.ones(num_particles) / num_particles

    for step in tqdm.tqdm(range(n_step)):
        lidar_reading = rc.fast_four_way_raycast(
            map_polygon,
            np.array([x_pose[step], y_pose[step]]),
            angles_pose[step],
            ray_length,
        )
        angle = compas[step] + np.random.normal(0, s_angle)

        # on a les 4 points de touche des rayons Lidar, on va calculer le point d'intersection des deux droites des points opposés pour situer le robot
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
            particles[particle_idx][2] = wrap_to_pi(angle)

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
        
        error[n_step] += np.linalg.norm(np.array([x_mean, y_mean]) - np.array([x_pose[step], y_pose[step]]))

        # apply resampling if necessary
        n_eff = 1 / np.sum(weights ** 2)
        if n_eff < r_eff * num_particles:
            indices = np.random.choice(np.arange(num_particles), size=num_particles, replace=True, p=weights)
            particles = particles[indices, :]
            weights = np.ones(num_particles) / num_particles

        if show_progress and step % show_every == 0:
            plt.clf()
            plt.title(f"Step {step}")
            plt.xlabel("x (m)")
            plt.ylabel("y (m)")
            plt.axis("equal")

            plt.plot(map_polygon[0, :], map_polygon[1, :], "k-")
            plt.scatter(x1, y1, color="green")
            plt.scatter(x3, y3, color="green")
            plt.scatter(x2, y2, color="green")
            plt.scatter(x4, y4, color="green")
            plt.scatter(x_robot, y_robot, color="red")
            plt.scatter(
                particles[:, 0], particles[:, 1], c=weights, cmap="hot", s=2, alpha=0.5
            )
            plt.colorbar()
            plt.plot(x_pose, y_pose, "r-", label="ground truth")
            plt.plot(positions[:step, 0], positions[:step, 1], "b-", label="estimated")

            plt.legend()
            plt.pause(1e-9)

    # plot trajectory
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.axis("equal")

    plt.plot(map_polygon[0, :], map_polygon[1, :], "k-")
    plt.plot(x_pose, y_pose, "r-", label="ground truth")
    # plt.plot(positions[0, 20:], positions[1, 20:], 'b-', label='estimated')

    plt.legend()
    plt.show()

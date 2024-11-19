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
    return 1 / np.sqrt(2 * np.pi * sigma ** 2) * np.exp(- (x - mu) ** 2 / (2 * sigma ** 2))


if __name__ == '__main__':
    # load map
    map_data = scio.loadmat('data/Carte.mat')
    map_polygon = map_data['Carte']

    # load trajectory data
    trajectory_data = scio.loadmat('data/Q2Trajectoire.mat')

    dt = trajectory_data['dT'].item()
    x_pose = trajectory_data['xPose'].squeeze()
    y_pose = trajectory_data['yPose'].squeeze()
    angles_pose = trajectory_data['anglePose'].squeeze()
    poses = np.stack([x_pose, y_pose, angles_pose], axis=0)

    v = trajectory_data['V'].squeeze()
    omega = trajectory_data['omega'].squeeze()
    lidar_measurements = trajectory_data['Lidar']

    # parameters
    ray_length = 
    s_lidar = 
    s_v = 
    s_omega = 
    s_compas = 
    n_step = y_pose.shape[0] - 1
    # TODO change to show progress
    show_progress = True
    show_every = 10

    # parameters to tune
    s_angle =   # TODO parameter to tune
    num_particles =   # TODO parameter to tune
    r_eff =   # TODO parameter to tune

    for step in tqdm.tqdm(range(n_step)):
        lidar_reading = 
        for particle_idx in range(num_particles):
            # simulate motion

            # measurement update

            # p(z|x)

        # normalize weights

        # apply resampling if necessary

        if show_progress and step % show_every == 0:
            plt.clf()
            plt.title(f'Step {step}')
            plt.xlabel('x (m)')
            plt.ylabel('y (m)')
            plt.axis('equal')

            plt.plot(map_polygon[0, :], map_polygon[1, :], 'k-')
            plt.scatter(particles[0, :], particles[1, :], c=weights, cmap='gray', s=2, alpha=0.5)
            plt.plot(x_pose, y_pose, 'r-', label='ground truth')
            plt.plot(positions[0, :step], positions[1, :step], 'b-', label='estimated')

            plt.legend()
            plt.pause(1e-9)

    # plot trajectory
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.axis('equal')

    plt.plot(map_polygon[0, :], map_polygon[1, :], 'k-')
    plt.plot(x_pose, y_pose, 'r-', label='ground truth')
    plt.plot(positions[0, 20:], positions[1, 20:], 'b-', label='estimated')

    plt.legend()
    plt.show()

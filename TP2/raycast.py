from typing import Optional

import numpy as np
from numba import njit


@njit
def line_line_intersect(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> Optional[np.ndarray]:
    """
    Computes the intersection between two lines defined by two points each
    :param a: first point of the first line
    :param b: second point of the first line
    :param c: first point of the second line
    :param d: second point of the second line
    :return: intersection point if it exists, None otherwise
    """
    ax1, ay1 = a
    ax2, ay2 = b
    bx1, by1 = c
    bx2, by2 = d

    # Compute the denominator of the formula
    d = (by2 - by1) * (ax2 - ax1) - (bx2 - bx1) * (ay2 - ay1)
    if d != 0:
        # Compute the distance along the line segments where the lines intersect
        u_a = ((bx2 - bx1) * (ay1 - by1) - (by2 - by1) * (ax1 - bx1)) / d
        u_b = ((ax2 - ax1) * (ay1 - by1) - (ay2 - ay1) * (ax1 - bx1)) / d
    else:
        return None

    # Check if the intersection is within the bounds of the line segments
    if not (0 <= u_a <= 1 and 0 <= u_b <= 1):
        return None

    # Compute the intersection point
    x = ax1 + u_a * (ax2 - ax1)
    y = ay1 + u_a * (ay2 - ay1)

    return np.array([x, y])


def naive_raycast(m: np.ndarray, pos: np.ndarray, angle: float, ray_length: float) -> np.ndarray:
    """
    Computes all the hits of a ray cast from `pos` with orientation `angle` for `ray_length` units with the map `m`
    :param m: map, a list of points defining a polygon (shape: (2, num_points))
    :param pos: robot's position (shape: (2,))
    :param angle: robot's orientation
    :param ray_length: length of the ray
    :return: list of all hits of the ray with the map (shape: (2, num_hits))
    """
    ray_direction = np.array([np.cos(angle), np.sin(angle)])
    ray_start, ray_end = pos, pos + ray_length * ray_direction
    hits = []
    num_pts = m.shape[1]
    for i in range(num_pts - 1):
        a, b = m[:, i], m[:, i + 1]
        hit = line_line_intersect(a, b, ray_start, ray_end)
        if hit is not None:
            hits.append(hit)
    return np.array(hits).T


def naive_four_way_raycast(m: np.ndarray, pos: np.ndarray, angle: float, ray_length: float) -> [np.ndarray]:
    """
    Uses the same technique as `naive_raycast` but throw 4 rays at 90° from each other
    :param m: map, a list of points defining a polygon (shape: (2, num_points))
    :param pos: robot's position (shape: (2,))
    :param angle: robot's orientation
    :param ray_length: length of the ray
    :return: list of all hits per direction [forward, left, backward, right]
    """
    hits = []
    for delta_angle in [i * np.pi / 2 for i in range(4)]:
        hits.append(naive_raycast(m, pos, angle + delta_angle, ray_length))
    return hits


def fast_raycast(m: np.ndarray, pos: np.ndarray, angle: float, ray_length: float) -> np.ndarray:
    """
    Computes all the hits of a ray cast from `pos` with orientation `angle` for `ray_length` units with the map `m`.
    This algorithm is based on the fact that for a ray to intersect with a line, at least one of its point should be
    in front of `pos`. Also, one of those point must be at the left of the `pos`, and the other on the right.
    This allows to filter a lot of points and only compute the intersection with lines we know we could intersect
    with our ray.
    :param m: map, a list of points defining a polygon (shape: (2, num_points))
    :param pos: robot's position (shape: (2,))
    :param angle: robot's orientation
    :param ray_length: length of the ray
    :return: list of all hits of the ray with the map (shape: (2, num_hits))
    """
    # direction vectors
    forward = np.array([np.cos(angle), np.sin(angle)])
    left = np.array([[np.cos(angle + np.pi / 2)], [np.sin(angle + np.pi / 2)]])

    # compute which points are in front of `pos`
    is_forward = np.dot((m - pos[..., None]).T, forward) > 0
    # convolution allows to get all lines with at least one point in front of `pos`
    # the convolution returns True if either the point to the left or right is in front
    is_forward = np.convolve(is_forward, np.array([1.0, 1.0, 1.0]), 'same') > 0
    forward_pts = np.nonzero(is_forward)[0]

    # compute points on the left of pos
    is_left = np.dot((m[:, forward_pts].T - pos), left)[:, 0] > 0

    # compute starting point of lines that cross
    # np.diff allows to find sequential points that are on different sides of `pos`
    # since left = 1 and right = 0, a non-zero diff means we passed from left to right
    # the same is true for the opposite, left = 0 and right = 1, a non-zero diff means we passed from right to left
    start_pts_idx = np.nonzero(np.diff(is_left))[0]

    # find the hits between the ray and the filtered lines
    # we only need to check the lines in start_pts_idx since we know they might intersect
    hits = []
    ray_end = pos + forward * ray_length
    for pt in forward_pts[start_pts_idx]:
        hit = line_line_intersect(m[:, pt], m[:, pt + 1], pos, ray_end)
        if hit is not None:
            hits.append(hit)
    return np.array(hits).T


def _fast_four_helper(in_front, is_at_left, m, pos, ray_end) -> np.ndarray:
    """
    Compute the hits for a single direction
    :param in_front: buffer of points in front of `pos`
    :param is_at_left: buffer of points on the left of `pos`
    :param m: map, a list of points defining a polygon (shape: (2, num_points))
    :param pos: robot's position (shape: (2,))
    :param ray_end: end of the ray
    :return: list of all hits of the ray with the map (shape: (2, num_hits))
    """
    is_in_front = np.convolve(in_front, np.array([1.0, 1.0, 1.0]), 'same') > 0
    front_pts = np.nonzero(is_in_front)[0]
    start_pts_idx = np.nonzero(np.diff(is_at_left[front_pts]))[0]
    hits = []
    for pt in front_pts[start_pts_idx]:
        hit = line_line_intersect(m[:, pt], m[:, pt + 1], pos, ray_end)
        if hit is not None:
            hits.append(hit)
    return np.array(hits).T


def fast_four_way_raycast(m: np.ndarray, pos: np.ndarray, angle: float, ray_length: float) -> [np.ndarray]:
    """
    Computes all the hits of a ray cast from `pos` with orientation `angle` for `ray_length` units with the map `m`.
    Uses the fast algorithm of `fast_raycast` but throw 4 rays at 90° from each other.
    This allows to reuse computation between rays.
    :param m: map, a list of points defining a polygon (shape: (2, num_points))
    :param pos: robot's position (shape: (2,))
    :param angle: robot's orientation
    :param ray_length: length of the ray
    :return: list of all hits per direction [forward, left, backward, right]
    """
    # direction vectors
    forward = np.array([np.cos(angle), np.sin(angle)])
    left = np.array([np.cos(angle + np.pi / 2), np.sin(angle + np.pi / 2)])

    # forward/backward buffer
    is_forward = np.dot((m - pos[..., None]).T, forward) > 0
    is_backward = 1 - is_forward

    # left/right buffer
    is_left = np.dot((m - pos[..., None]).T, left) > 0
    is_right = 1 - is_left

    # throw 4 rays, reuse position matrices to save compute
    hits = [_fast_four_helper(is_forward, is_left, m, pos, pos + forward * ray_length),  # forward
            _fast_four_helper(is_left, is_backward, m, pos, pos + left * ray_length),  # left
            _fast_four_helper(is_backward, is_right, m, pos, pos - forward * ray_length),  # backward
            _fast_four_helper(is_right, is_forward, m, pos, pos - left * ray_length)]  # right
    return hits


def is_in_map(m: np.ndarray, pos: np.ndarray, max_dist: float = 100) -> bool:
    """
    Checks if `pos` is in `m`
    :param m: map, a list of points defining a polygon (shape: (2, num_points))
    :param pos: robot's position (shape: (2,))
    :return: True if `pos` is in `m`, False otherwise
    """
    hits = fast_raycast(m, pos, 0, max_dist)
    if hits.size == 0:
        return False
    return hits.shape[1] % 2 == 1

import cv2
import numpy as np

from r2_lidar_icp.point_cloud import PointCloud


def draw_base_vector(ax,
                     head,
                     text='',
                     origin=np.array([0., 0.]),
                     text_offset=np.array([0., 0.]),
                     color='tab:red',
                     ha='right',
                     va='top'):
    head_global = origin + head
    text_global = head_global + text_offset

    ax.annotate('', xy=head_global, xytext=origin,
                arrowprops=dict(arrowstyle='->,head_width=0.6, head_length=1', color=color, lw=2))
    ax.text(text_global[0], text_global[1], text, size=30, color=color, ha=ha, va=va)


def draw_frame(ax,
               origin=np.array([0., 0.]),
               x=np.array([1., 0.]),
               y=np.array([0., 1.]),
               color='white',
               name='',
               text_x='',
               text_y=''):
    draw_base_vector(ax, origin=origin, head=x, text=text_x, text_offset=(0., -0.1), ha='right', va='top', color=color)
    draw_base_vector(ax, origin=origin, head=y, text=text_y, text_offset=(-0.1, 0.), ha='right', va='top', color=color)
    ax.text(origin[0], origin[1], name, color=color, size=30, ha='right', va='top')
    return


def draw_point_clouds(ax, pc=None, reference=None, normals_pc=None, normals_reference=None, errors=None, T=None):
    if pc is not None:
        ax.scatter(pc[0], pc[1], alpha=0.2, color='tab:blue', label='Reading')
        if normals_pc is not None:
            ax.quiver(pc[0], pc[1], normals_pc[0], normals_pc[1], color='yellow')
        if errors is not None:
            ax.quiver(pc[0], pc[1], errors[0], errors[1],
                      color='tab:red', alpha=0.4,
                      angles='xy', scale_units='xy', scale=1.)
    if reference is not None:
        ax.scatter(reference[0], reference[1], alpha=0.2, color='tab:green', label='Reference')
        if normals_reference is not None:
            ax.quiver(reference[0], reference[1], normals_reference[0], normals_reference[1], color='red')
    if T is not None:
        ax.quiver(0, 0, T[0, 2], T[1, 2], color='tab:red',
                  angles='xy', scale_units='xy', scale=1.)

    draw_frame(ax, x=[0.2, 0], y=[0, 0.2])
    ax.set_xlabel(r'$\vec{\mathscr{x}}$')
    ax.set_ylabel(r'$\vec{\mathscr{y}}$')
    ax.set_aspect('equal', adjustable='box')


def draw_point_cloud_cv2(pc: PointCloud, img, size: int, color: (int, int, int), scaling_factor=10_000, offset=(0, 0)):
    assert pc.features.shape[0] == 3, 'only works with 2d points'
    tmp = np.copy(pc.features)
    tmp = tmp / tmp[2, :]
    for (x, y) in tmp[:2, :].T:
        x = int(x / scaling_factor * size + size / 2) + offset[0]
        y = int(y / scaling_factor * size + size / 2) + offset[1]
        cv2.circle(img, (x, y), 3, color, -1)

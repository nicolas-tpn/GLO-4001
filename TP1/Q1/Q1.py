import scipy.io as sc
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

data = sc.loadmat("Q1Donnees.mat")
data2 = sc.loadmat("Carte.mat")

z = data["z"].flatten()
t = data["t"].flatten()
g = data["g"].flatten()
Carte = data2["Carte"]

# Question 1.1 : Calibration

t_upper_bound = 12.87
t_lower_bound = 2.86
K1 = np.mean(g[t < 2.75])
g_moy = np.mean(g[(t > t_lower_bound) & (t < t_upper_bound)])
delta_t = t_upper_bound - t_lower_bound
angular_speed_moy = 2 * np.pi / delta_t
K2 = (g_moy - K1) / angular_speed_moy
print("K1", K1, "K2", K2)

# Question 1.2 : Carte 2D locale

points = np.empty((0, 2))
current_angle = 0
for index, timing in enumerate(t):
    # Faire un truc plus propre pour le delta_t que 0.05

    current_angle += (g[index] - K1) / K2 * 0.05
    if 1 / z[index] < 4:
        x = np.cos(current_angle) * 1 / z[index]
        y = np.sin(current_angle) * 1 / z[index]
        point = np.array([[x, y]])

        points = np.append(points, point, axis=0)

# Question 1.3 : Localisation du robot dans la carte globale

# Passage en coordonnées homogènes
adjusted_points = np.hstack([points, np.ones((points.shape[0], 2))])

# Application des matrices de transformation
angle = 30
radian_angle = np.radians(angle)
translation_x = 3
translation_y = 4.9

adjusted_points = np.dot(
    adjusted_points,
    np.array(
        [
            [np.cos(radian_angle), -np.sin(radian_angle), 0, 0],
            [np.sin(radian_angle), np.cos(radian_angle), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    ),
)
adjusted_points = np.dot(
    adjusted_points,
    np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [translation_x, translation_y, 0, 1]]
    ),
)

plt.plot(Carte[0, :], Carte[1, :], "b")
plt.plot(adjusted_points[:, 0], adjusted_points[:, 1], "r.")
plt.axis("equal")
plt.show()


# plt.subplot(2, 1, 1)
# plt.plot(t, z, linewidth=2)
# plt.ylabel("Mesure telemetre (V)")
# plt.subplot(2, 1, 2)
# plt.plot(t, g, linewidth=2)
# plt.ylabel("Mesure gyroscope (V)")
# plt.xlabel("Temps (s)")
# plt.tight_layout()
# plt.show()

# plt.plot(Carte[0, :], Carte[1, :], "b")
# plt.plot(points[:, 0], points[:, 1], "r")
# plt.xlabel("Coordonnées en x (m)")
# plt.ylabel("Coordonnées en y (m)")
# plt.axis("equal")
# plt.savefig("environnement.png")
# plt.show()

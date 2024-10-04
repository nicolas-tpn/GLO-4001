import scipy.io as sc
import numpy as np
import matplotlib.pyplot as plt
data = sc.loadmat('Q1Donnees.mat')
data2 = sc.loadmat('Carte.mat')

z = data['z'].flatten()
t = data['t'].flatten()
g = data['g'].flatten()
Carte = data2['Carte']

# Question 1.1 : Calibration

t_upper_bound = 12.87
t_lower_bound = 2.86
K1 = np.mean(g[t<2.75])
g_moy = np.mean(g[(t>t_lower_bound) & (t<t_upper_bound)])
delta_t = t_upper_bound-t_lower_bound
angular_speed_moy = 2*np.pi/delta_t
K2 = (g_moy-K1)/angular_speed_moy
print("K1", K1, "K2", K2)

# Question 1.2 : Carte 2D locale

points = np.empty((0, 2))
current_angle = 0
for index, timing in enumerate(t[(t>t_lower_bound) & (t<t_upper_bound)]):
# for index, timing in enumerate(t):
    # Faire un truc plus propre pour le delta_t que 0.05
    current_angle += (g[index]-K1)/K2 * 0.05
    x = np.sin(current_angle)*z[index]
    y = np.cos(current_angle)*z[index]
    point = np.array([[x, y]])

    points = np.append(points, point, axis=0)

plt.scatter(points[:, 0], points[:, 1])
plt.show()


plt.subplot(2,1,1)
plt.plot(t, z, linewidth=2)
plt.ylabel('Mesure telemetre (V)')
plt.subplot(2,1,2)
plt.plot(t, g, linewidth=2)
plt.ylabel('Mesure gyroscope (V)')
plt.xlabel('Temps (s)')
plt.tight_layout()
plt.show()

plt.plot(Carte[0,:],Carte[1,:],'b');
plt.xlabel('Coordonnées en x (m)')
plt.ylabel('Coordonnées en y (m)')
plt.axis('equal')
plt.savefig('environnement.png')
plt.show()






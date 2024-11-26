from modulos import robot
from modulos import dhParameters
import sympy as sp
import numpy as np

if __name__ == "__main__":

    # Parámetros de ejemplo
    theta   = [0, 0, 0, 0,0, 0]
    d       = [0.089159, 0, 0, 0.10915, 0.09465, 0.0823]
    a       = [0, -0.425, -0.39225, 0, 0, 0]
    alpha   = [sp.pi/2, 0, 0, sp.pi/2, -sp.pi/2, 0]
    kind    = ['R', 'R', 'R', 'R', 'R', 'R']

    # Creación del objeto
    dh_params = dhParameters(theta, d, a, alpha, kind)
    q_lim = [
        [-np.pi,np.pi],
        [-np.pi,np.pi],
        [-np.pi,np.pi],
        [-np.pi,np.pi],
        [-np.pi,np.pi],
        [-np.pi,np.pi]
    ]

    theta1 = np.array([0,       np.pi / 4,  np.pi,  -np.pi,     -3 * np.pi / 4])
    theta2 = np.array([3*np.pi,  -np.pi / 4,  0,       np.pi / 2,  3 * np.pi / 4])

    UR5 = robot(dh_params,q_lim)
    result = UR5.shortest_angular_distances(theta1,theta2)

    # Mostrar resultados
    print("Theta 1:", theta1)
    print("Theta 2:", theta2)
    print("Diferencias angulares más cortas:", result)
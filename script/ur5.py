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


    UR5 = robot(dh_params,q_lim)

    E=UR5.matrixEuler2Wel(*UR5.asEuler(UR5.tWrist[:3,:3]))
    JA = E @ UR5.jGWrist
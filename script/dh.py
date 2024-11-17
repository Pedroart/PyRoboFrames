#python3 -m tests.dh
import sympy as sp
from pyrobotframes import *


if __name__ == "__main__":

    # Parámetros de ejemplo
    theta   = [0, sp.pi/4, sp.pi/2]
    d       = [0.5, 0, 0]
    a       = [0.3, 0.2, 0.1]
    alpha   = [sp.pi/2, 0, -sp.pi/2]
    kind    = ['R', 'P', 'R']

    # Creación del objeto
    dh_params = dhParameters(theta, d, a, alpha, kind)

    # Representación del objeto
    print(f'{dh_params} \n')
    

    # Acceso a variables articulares simbólicas
    print("Articular Variables (q):")
    print(dh_params.q)

    # Cálculo de la transformación homogénea
    ht = dh_params.homogeneous_transform()
    print()
    print("Homogeneous Transformation Matrix:")
    print(ht)

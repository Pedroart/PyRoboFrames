#python3 -m tests.dh
import sympy as sp
import numpy as np
from modulos import dhParameters


if __name__ == "__main__":

    # Parámetros de ejemplo
    theta   = [0, 0, 0, 0,0, 0]
    d       = [0.089159, 0, 0, 0.10915, 0.09465, 0.0823]
    a       = [0, -0.425, -0.39225, 0, 0, 0]
    alpha   = [sp.pi/2, 0, 0, sp.pi/2, -sp.pi/2, 0]
    kind    = ['R', 'R', 'R', 'R', 'R', 'R']

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

    # Asignar valores a las variables simbólicas q
    valores_q = {
        dh_params.q[0]: 5.27, 
        dh_params.q[1]: 3.31, 
        dh_params.q[2]: 1.02,
        dh_params.q[3]: 3.47, 
        dh_params.q[4]: 2.09, 
        dh_params.q[5]: 1.57
    }

    # Evaluar la matriz de transformación con los valores asignados
    ht_evaluada = ht.subs(valores_q)
    print("Matriz de transformación homogénea evaluada con valores de q:")
    print(ht_evaluada)

    ht_numpy = np.array(ht_evaluada.evalf()).astype(np.float64)

    print("Matriz de transformación homogénea en formato NumPy:")
    print(ht_numpy)
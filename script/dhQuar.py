import numpy as np
import sympy as sp
from scipy.spatial.transform import Rotation as R

def dh_to_quaternion(theta, d, a, alpha):
    # Cuaternión de rotación en el eje z (theta)
    q_z = R.from_euler('z', theta).as_quat()  # [x, y, z, w]

    # Cuaternión de rotación en el eje x (alpha)
    q_x = R.from_euler('x', alpha).as_quat()

    # Producto de cuaterniones para obtener la rotación total
    q_total = (R.from_quat(q_z) * R.from_quat(q_x)).as_quat()

    # Traslaciones
    t_z = np.array([0, 0, d])
    t_x = np.array([a, 0, 0])

    # Traslación total (en el marco rotado por q_z)
    t_total = t_z + R.from_quat(q_z).apply(t_x)

    return q_total, t_total

def forward_kinematics_dh_quaternion(dh_params):
    q_accumulated = np.array([0, 0, 0, 1])  # Cuaternión identidad
    t_accumulated = np.array([0, 0, 0], dtype=float)     # Traslación inicial

    for theta, d, a, alpha in dh_params:
        q, t = dh_to_quaternion(theta, d, a, alpha)

        # Actualizar cuaternión acumulado
        q_accumulated = (R.from_quat(q_accumulated) * R.from_quat(q)).as_quat()

        # Actualizar traslación acumulada
        t_accumulated += R.from_quat(q_accumulated).apply(t)

    return R.from_quat(q_accumulated), t_accumulated


# Definir variables simbólicas para las rotaciones (q1, q2, q3)
q1, q2, q3 = sp.symbols('q1 q2 q3')

# Redefinir los parámetros DH
dh_params = [
    (q1 + np.pi / 6, 2, 1, np.pi / 4),  # Primer eslabón
    (q2 + np.pi / 3, 1, 2, np.pi / 6),  # Segundo eslabón
    (q3 + np.pi / 4, 3, 1, 0)           # Tercer eslabón
]

# Testear forward_kinematics_dh_quaternion
q_final, t_final = forward_kinematics_dh_quaternion(dh_params)

print("Cuaternión final (x, y, z, w):", q_final.as_quat())
print("Traslación final (x, y, z):", t_final)

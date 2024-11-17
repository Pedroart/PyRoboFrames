from scipy.spatial.transform import Rotation as R
import numpy as np

def quaternion_difference(q_current, q_target):
    """
    Calcula la diferencia entre un cuaternión actual y un objetivo.

    Parámetros:
    q_current : np.array
        Cuaternión actual en formato [x, y, z, w].
    q_target : np.array
        Cuaternión objetivo en formato [x, y, z, w].

    Retorna:
    np.array
        Diferencia de cuaterniones en formato [x, y, z, w].
    """
    # Normalizar ambos cuaterniones
    q_current = q_current / np.linalg.norm(q_current)
    q_target = q_target / np.linalg.norm(q_target)

    # Crear rotaciones desde cuaterniones
    r_current = R.from_quat(q_current)
    r_target = R.from_quat(q_target)

    # Calcular el cuaternión de diferencia
    r_diff = r_target * r_current.inv()
    q_diff = r_diff.as_quat()  # En formato [x, y, z, w]

    return q_diff

# Test de orientación
def test_orientation_error():
    """
    Genera cuaterniones a partir de ángulos de Euler en un solo estándar y calcula el error de orientación.
    """
    # Orientación actual en Euler (roll, pitch, yaw) y convertir a cuaternión
    roll_current, pitch_current, yaw_current = np.radians([0, 0, 0])
    q_current = R.from_euler('xyz', [roll_current, pitch_current, yaw_current]).as_quat()  # [x, y, z, w]

    # Orientación objetivo en Euler (roll, pitch, yaw) y convertir a cuaternión
    roll_target, pitch_target, yaw_target = np.radians([90, 90, 0])
    q_target = R.from_euler('xyz', [roll_target, pitch_target, yaw_target]).as_quat()  # [x, y, z, w]



    # Calcular el error de orientación usando cuaterniones
    q_diff = quaternion_difference(q_current, q_target)
    e_o = q_diff[1:]  # Parte imaginaria como error

    # Convertir el cuaternión de diferencia a Euler para ver el error en ángulos
    euler_error = R.from_quat(q_diff).as_euler('xyz', degrees=True)

    # Imprimir resultados
    print("Orientación Actual (Roll, Pitch, Yaw):", np.degrees([roll_current, pitch_current, yaw_current]))
    print("Cuaternión Actual (w, x, y, z):", q_current)

    print("\nOrientación Objetivo (Roll, Pitch, Yaw):", np.degrees([roll_target, pitch_target, yaw_target]))
    print("Cuaternión Objetivo (w, x, y, z):", q_target)

    print("\nError de Orientación (cuaternión imaginario):", e_o)
    print("Error de Orientación en Euler (grados):", euler_error)

# Ejecutar el test
if __name__ == "__main__":
    test_orientation_error()

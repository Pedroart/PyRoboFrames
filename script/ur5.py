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

    UR5.ikine_task(np.array([-0.79,-0.19,-0.005, 0, 1, 0, 0]))

'''    RPYDES = np.array([0,0,1.57])
    XYZDES = np.array([-0.70,-0.19,-0.005])
    E=UR5.matrixEuler2Wel(*UR5.asEuler(UR5.tWrist[:3,:3]))
    JA = E @ UR5.jGWrist
    
    RPYActual = UR5.asEuler(UR5.tWrist[:3,:3])
    XYZActual = [UR5.tWrist[:3,3]]

    # Parámetros de tolerancia
    epsilon_xyz = 1e-3  # Tolerancia para posición
    epsilon_rpy = 1e-1  # Tolerancia para orientación
    max_iterations = 100
    alpha = 10.0  # Tasa de aprendizaje

    # Inicialización
    q = np.zeros(UR5.num_joints)  # Configuración inicial (típicamente en el rango de las q_lim)

    for iteration in range(max_iterations):

        # Calcula la pose actual
        UR5._q = q  # Actualiza el robot con las nuevas articulaciones
        UR5.update()

        RPYActual = UR5.asEuler(UR5.tWrist[:3, :3])
        XYZActual = UR5.tWrist[:3, 3]

        print(f'PRY: {RPYActual}')
        print(f'XYZ: {XYZActual}')
        # Calcula los errores
        e_xyz = XYZDES - XYZActual
        e_rpy = RPYDES - RPYActual

        # Verifica la convergencia
        if np.linalg.norm(e_xyz) < epsilon_xyz  and np.linalg.norm(e_rpy) < epsilon_rpy:
            print(f"Convergencia alcanzada en iteración {iteration}")
            break

        # Calcula el jacobiano analítico
        E = UR5.matrixEuler2Wel(*RPYActual)  # Matriz de conversión para orientación
        JA = E @ UR5.jGWrist  # Jacobiano analítico

        # Calcula el ajuste de q
        error = np.concatenate((e_xyz,0.1*e_rpy))
        delta_q = np.linalg.pinv(JA) @ error  # Pseudo-inversa para resolver el gradiente

        # Actualiza las articulaciones
        q = q + delta_q

    else:
        print("No se alcanzó convergencia dentro del número máximo de iteraciones.")

    print("Configuración final de articulaciones:", q)'''
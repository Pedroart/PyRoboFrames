import sympy as sp
import numpy as np
from . import dhParameters as dhp
from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion
from copy import copy
import time

def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = (end_time - start_time) * 1000  # Execution time in milliseconds
        print(f"{func.__name__}: {execution_time:.2f} ms")
        return result
    return wrapper

class robot:
    
    def __init__(self,params_dh,q_lim):
        self._q_lim = q_lim
        self.params_dh = params_dh
        self.num_joints = params_dh.num_joints
        self._qsym = params_dh.q
        self._q = np.array([0 for x in range(self.num_joints)])
        
        print('Inicio Precalculo de Variables')
        self._tWrist_symbolic = self._funCinematicaDirecta()
        self._tWrist_func = sp.lambdify(self._qsym, self._tWrist_symbolic, "numpy")

        self._jGWrist_symbolic = self._funJacobianoGeometrioOpt()
        self._jGWrist_func = sp.lambdify(self._qsym, self._jGWrist_symbolic, "numpy")
        print('Fin de Precalculo de Variables')
        

        self.update()

    
    @timeit
    def update(self):
        self.tWrist = self._tWrist_func(*self._q)
        self.jGWrist = self._jGWrist_func(*self._q)
        #self.pose = self.TF2xyzquat(self.tWrist)

    
    def TF2xyzquat(_,T):
        """
        Convert a homogeneous transformation matrix into the a vector containing the
        pose of the robot.

        Input:
        T -- A homogeneous transformation
        Output:
        X -- A pose vector in the format [x y z ex ey ez ew], donde la first part
            is Cartesian coordinates and the last part is a quaternion
        """
        quat = R.from_matrix(T[0:3,0:3]).as_quat()
        res = [T[0,3], T[1,3], T[2,3], quat[0], quat[1], quat[2], quat[3]]
        return np.array(res)

    
    def limit_joint_pos(self,q):
        """
        Delimita los valores articulares a los limites articulares del UR5
        """
        
        #Verifica si cada valor articular esta dentro de sus limites
        for i in range(6):
            if q[i] < self._q_lim[i,0]:
                q[i] = self._q_lim[i,0]
            elif q[i] > self._q_lim[i,1]:
                q[i] = self._q_lim[i,1]
            else:
                q[i] = q[i]

        return q

    
    def matrixEuler2Wel(self, alpha, beta, gama ):
        # Matriz de transformación T
        T = np.array([
            [1, 0, np.sin(beta)],
            [0, np.cos(alpha), -np.sin(alpha) * np.cos(beta)],
            [0, np.sin(alpha), -np.cos(alpha) * np.cos(beta)]
        ])
        
        # Matriz inversa directamente calculada
        T_inv = np.linalg.inv(T)
        
        # Construcción eficiente de la matriz completa (6x6)
        full_matrix = np.eye(6)
        full_matrix[3:, 3:] = T_inv     # Inversa de T en la esquina inferior derecha
        
        return full_matrix

    
    def asEuler(_,Matrix):
        rotation = R.from_matrix(Matrix)
        return rotation.as_euler('xyz', degrees=False)

    
    def _funCinematicaDirecta(self, n=None):
        T_total = self.params_dh.homogeneous_transform(n)

        return sp.trigsimp(T_total)

    
    def _funJacobianoGeometrio(self, n=None):
        if(n == None):
            n = self.num_joints
        T_total = sp.eye(4)
        p_n = []
        T_list = []
        
        z = [sp.Matrix([0, 0, 1])]
        for iter in range(n):
            T_list.append(self.params_dh.homogeneous_transform(iter+1))
            z.append(T_list[-1][:3, 2])   # Eje z
        
        p_n = T_list[-1][:3, 3]
        J_linear = []
        J_angular = []

        for i in range(n):
            Jv_i = p_n.diff(self._qsym[i])
            J_linear.append(Jv_i)
        
        J_angular = z[:n]
        return sp.Matrix.vstack(sp.Matrix.hstack(*J_linear), sp.Matrix.hstack(*J_angular))
    
    def _funJacobianoGeometrioOpt(self, n=None):
        if n is None:
            n = self.num_joints

        T_list = [self.params_dh.homogeneous_transform(i + 1) for i in range(n)]
        z = [sp.Matrix([0, 0, 1])] + [T[:3, 2] for T in T_list]
        p_n = T_list[-1][:3, 3]

        # Precomputar derivadas de p_n para reducir accesos y cálculos simbólicos
        #J_linear = [p_n.jacobian([self._qsym[i]]) for i in range(n)]
        J_linear = [
            sp.trigsimp(p_n.jacobian([self._qsym[i]])) for i in range(n)
        ]


        # Evitar crear más listas; construir directamente las filas del jacobiano
        return sp.Matrix.vstack(
            sp.Matrix.hstack(*J_linear),
            sp.Matrix.hstack(*z[:n])
        )

    @timeit
    def quaternion_difference(_,q_current, q_target):
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
        # Crear rotaciones desde cuaterniones
        r_current = R.from_quat(q_current)
        r_target = R.from_quat(q_target)

        # Calcular el cuaternión de diferencia
        r_diff = r_target * r_current.inv()
        q_diff = r_diff.as_quat()  # En formato [x, y, z, w]

        return q_diff

    @timeit
    def jacobian_quar(self, q, delta=0.0001):
        """
        Jacobiano analitico para la posicion. Retorna una matriz de 3x6 y toma como
        entrada el vector de configuracion articular q=[q1, q2, q3, q4, q5, q6]

        """
        
        # Alocacion de memoria
        J = np.zeros((7,self.num_joints))
        # Transformacion homogenea inicial (usando q)
        T = self.tWrist
        # Iteracion para la derivada de cada columna
        for i in range(self.num_joints):
            # Copiar la configuracion articular inicial
            dq = copy(q)
            # Incrementar la articulacion i-esima usando un delta
            dq[i] = dq[i] + delta
            # Transformacion homogenea luego del incremento (q+dq)
            Td = self._tWrist_func(*dq)
            # Aproximacion del Jacobiano de posicion usando diferencias	finitas
            Ji_p = (Td[0:3,3] - T[0:3,3])/delta
            
            # Aproximación del Jacobiano de orientación usando diferencias finitas
            quat_current = R.from_matrix(T[0:3, 0:3]).as_quat()
            quat_perturbed = R.from_matrix(Td[0:3, 0:3]).as_quat()
            quat_diff = self.quaternion_difference(quat_current, quat_perturbed)

            # Dividir por delta para calcular la tasa de cambio
            Ji_o = quat_diff / delta

            J[0:3,i] = Ji_p
            J[3:,i]  = Ji_o

        return J

    @timeit
    def jacobian_quar_optimized(self, q, delta=0.0001):
        J = np.zeros((7, self.num_joints))
        T = self.tWrist
        quat_current = R.from_matrix(T[0:3, 0:3]).as_quat()
        
        dq = np.tile(q, (self.num_joints, 1))  # Crear una copia para cada articulación
        dq[np.arange(self.num_joints), np.arange(self.num_joints)] += delta  # Incrementar

        Td_all = np.array([self._tWrist_func(*dq_i) for dq_i in dq])  # Transformaciones en paralelo
        
        
        quats_perturbed = np.array([R.from_matrix(Td[0:3, 0:3]).as_quat() for Td in Td_all])
        quat_diffs = np.array([self.quaternion_difference(quat_current, q_p) for q_p in quats_perturbed]) / delta
        
        J[0:3, :] = self.jGWrist[0:3,:]
        J[3:, :] = quat_diffs.T

        return J

    @timeit
    def jacobian_position(self, q, delta=0.0001):
        """
        Jacobiano analitico para la posicion. Retorna una matriz de 3x6 y toma como
        entrada el vector de configuracion articular q=[q1, q2, q3, q4, q5, q6]

        """
        
        # Alocacion de memoria
        J = np.zeros((3,self.num_joints))
        # Transformacion homogenea inicial (usando q)
        T = self.tWrist
        # Iteracion para la derivada de cada columna
        for i in range(self.num_joints):
            # Copiar la configuracion articular inicial
            dq = copy(q)
            # Incrementar la articulacion i-esima usando un delta
            dq[i] = dq[i] + delta
            # Transformacion homogenea luego del incremento (q+dq)
            Td = self._tWrist_func(*dq)
            # Aproximacion del Jacobiano de posicion usando diferencias	finitas
            Ji = (Td[0:3,3] - T[0:3,3])/delta
            J[:,i] = Ji

        return J
    
    @timeit
    def ikine_quar(self,xdes):
        epsilon = 0.001 # Tolerancia para la convergencia
        max_iter = 100  # Número máximo de iteraciones
        
        q = self._q.astype(float) 

        for i in range(max_iter):
            

            e_pos = self.pose[0:3]-xdes[0:3]
            # Error de orientación (normalizar cuaterniones)
            q_current = self.pose[3:] / np.linalg.norm(self.pose[3:])
            q_target = xdes[3:] / np.linalg.norm(xdes[3:])
            e_o = self.quaternion_difference(q_current, q_target)

            error = np.concatenate((-e_pos,0.1*e_o))
            
            #dq = np.dot(np.linalg.pinv(self.jacobian_position(self._q)), error)
            
            dq = np.dot(np.linalg.pinv(self.jacobian_quar_optimized(q)) , error)
            q = q+dq
            self._q = q
            self.update()
            

            print(self.pose)
            if np.linalg.norm(error) < epsilon:
                break
        return q

    @timeit
    def ikine_position(self,xdes):
        epsilon = 0.0001  # Tolerancia para la convergencia
        max_iter = 100  # Número máximo de iteraciones
        
        q = self._q.astype(float) 

        for i in range(max_iter):
            
            xcurr = self.tWrist[:3,3]
            error = xdes - xcurr
            
           
            #dq = np.dot(np.linalg.pinv(self.jacobian_position(self._q)), error)
            
            dq = np.dot(np.linalg.pinv(self.jGWrist[:3,:]) , error)
            q = self.limit_joint_pos(q+dq)
            self._q = q
            self.update()

            if np.linalg.norm(error) < epsilon:
                break

            print(xcurr)
           
            
        return q

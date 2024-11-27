import sympy as sp
import numpy as np
from . import dhParameters as dhp
from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion
from copy import copy
import time
from concurrent.futures import ThreadPoolExecutor


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
        self._q = np.array([0 for x in range(self.num_joints)]).astype(float) 
        
        sin_q = [sp.Symbol(f'sin_q{i+1}') for i in range(self.num_joints)]
        cos_q = [sp.Symbol(f'cos_q{i+1}') for i in range(self.num_joints)]

        self.new_inputs = sin_q + cos_q
        subs_dict = {
            trig: val
            for i in range(self.num_joints)
            for trig, val in zip([sp.sin(self._qsym[i]), sp.cos(self._qsym[i])], [sin_q[i], cos_q[i]])
        }

        
        self._tElbow_symbolic = self._funCinematicaDirecta(2).subs(subs_dict)
        self._tElbow_func = sp.lambdify(self.new_inputs, self._tElbow_symbolic, "numpy")

        # Separar las componentes lineales y de orientación
        self._tLElbow_func = sp.lambdify(self.new_inputs, self._tElbow_symbolic[0:3, 3], "numpy")  # Posición del codo
        self._tOElbow_func = sp.lambdify(self.new_inputs, self._tElbow_symbolic[0:3, 0:3], "numpy")  # Orientación del codo

        # Jacobiano lineal hasta la articulación 3
        self._jLElbow_symbolic = self._funJacobianoLineal(2).subs(subs_dict)
        self._jLElbow_func = sp.lambdify(self.new_inputs, self._jLElbow_symbolic, "numpy")



        self._tWrist_symbolic = self._funCinematicaDirecta(4).subs(subs_dict)
        self._tWrist_func = sp.lambdify(self.new_inputs, self._tWrist_symbolic, "numpy")

        self._tLWrist_func = sp.lambdify(self.new_inputs, self._tWrist_symbolic[0:3,3], "numpy")
        self._tOWrist_func = sp.lambdify(self.new_inputs, self._tWrist_symbolic[0:3,0:3], "numpy")

        self._jLWrist_symbolic = self._funJacobianoLineal(4).subs(subs_dict)
        self._jLWrist_func = sp.lambdify(self.new_inputs, self._jLWrist_symbolic, "numpy")
        
        self.update()

    
    def _compute_trig_inputs(self,q=None):
        if q is None:
            q = self._q
        sin_values = np.sin(q)  # Vectorizado
        cos_values = np.cos(q)  # Vectorizado
        return np.concatenate((sin_values, cos_values))  # Concatenar sinos y cosenos


    #@timeit
    def update(self):
        self._inputs = self._compute_trig_inputs()
        #self.tWrist = self._tWrist_func(*self._inputs)
        self.tLWrist = self._tLWrist_func(*self._inputs).T[0]
        #self.tOWrist = self._tOWrist_func(*self._inputs)

        self.jLWrist = self._jLWrist_func(*self._inputs)

        #self.pose = self.TF2xyzquat()
        
        #self.jCWrist = self.jacobian_quar_optimized()

        
        #self.tWrist = self._tWrist_func(*self._inputs)
        self.tLElbow = self._tLElbow_func(*self._inputs).T[0]
        #self.tOWrist = self._tOWrist_func(*self._inputs)

        self.jLElbow = self._jLElbow_func(*self._inputs)

        #self.pose = self.TF2xyzquat()
        
        #self.jCWrist = self.jacobian_quar_optimized()
        

    def TF2xyzquat(self):
        """
        Convert a homogeneous transformation matrix into the a vector containing the
        pose of the robot.

        Input:
        T -- A homogeneous transformation
        Output:
        X -- A pose vector in the format [x y z ex ey ez ew], donde la first part
            is Cartesian coordinates and the last part is a quaternion
        """
        quat = R.from_matrix(self.tOWrist).as_quat()
        print(self.tLWrist)
        res = [self.tLWrist[0], self.tLWrist[1], self.tLWrist[2], quat[0], quat[1], quat[2], quat[3]]
        
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

    def _funCinematicaDirecta(self, n=None):
        T_total = self.params_dh.homogeneous_transform(n)
        return T_total

    def _funJacobianoLineal(self, n=None):
        if n is None:
            n = self.num_joints

        J_linear = sp.zeros(3, self.num_joints)
        # Posición del extremo en función de las articulaciones
        p_n = self.params_dh.homogeneous_transform(n)[:3, 3]

        # Usar jacobiano simbólico en lugar de diferenciar en bucle
        qsym_mat = sp.Matrix(self._qsym[:n])
        J_linear[:, :n] = p_n.jacobian(qsym_mat)
        #return sp.trigsimp(J_linear)
        return J_linear


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

        # Calcular el cuaternión de diferencia
        #r_diff = 
        #q_diff = r_diff.as_quat()  # En formato [x, y, z, w]

        return (R.from_quat(q_target) * R.from_quat(q_current).inv()).as_quat()

    @timeit
    def _quaternion_difference(_,r_current, r_target):
        r_diff = np.dot(r_target, np.linalg.inv(r_current))
        q_diff = R.from_matrix(r_diff).as_quat()  # En formato [x, y, z, w]

        return q_diff
    
    @timeit
    def jacobian_quar_optimized(self, delta=0.0001):
        
        quat_current = self.pose[3:]
        
        dq = np.tile(self._q, (self.num_joints, 1)).astype(np.float64)  # Crear una copia para cada articulación
        dq[np.arange(self.num_joints), np.arange(self.num_joints)] += delta  # Incrementar
        
        Td_all = np.array([self._tOWrist_func(*self._compute_trig_inputs(dq_i)) for dq_i in dq])  # Transformaciones
        
        quats_perturbed = R.from_matrix(Td_all[:, 0:3, 0:3] ).as_quat()  # Efficient batch conversion
        quat_diffs = (quats_perturbed - quat_current) / delta  # Assuming quaternion_difference is subtraction
        
        return quat_diffs.T
    
    @timeit
    def ikine_quar(self,xdes):
        epsilon = 1e-3 # Tolerancia para la convergencia
        max_iter = 1e3  # Número máximo de iteraciones
        
        q = self._q.astype(float) 

        for i in range(max_iter):
            

            e_pos = self.pose[0:3]-xdes[0:3]
            # Error de orientación (normalizar cuaterniones)
            q_current = self.pose[3:]
            q_target = xdes[3:]
            e_o = self.quaternion_difference(q_current, q_target)

            error = np.concatenate((-e_pos,0.1*e_o))
            
            dq = np.dot(np.linalg.pinv(  np.vstack((self.jLWrist,self.jCWrist))   ) , error)
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
            
            xcurr = self.tLWrist
            error = xdes - xcurr
            
            dq = np.dot(np.linalg.pinv(self.jLWrist) , error)
            self._q += dq
            print(dq)
            print(self._q)
            self.update()

            if np.linalg.norm(error) < epsilon:
                print('Error Minimo')
                break

            
           
            
        return self._q

    def regularized_pseudoinverse(self,J, u=np.sqrt(0.001)):
        return J.T @ np.linalg.inv(J @ J.T + u**2 * np.eye(J.shape[0]))

    def shortest_angular_distances(_,theta_array1, theta_array2):
        """
        Calcula la diferencia angular más corta entre múltiples pares de ángulos en radianes usando numpy.

        Args:
            theta_array1 (numpy.ndarray): Array de ángulos iniciales en radianes.
            theta_array2 (numpy.ndarray): Array de ángulos finales en radianes.

        Returns:
            numpy.ndarray: Array de diferencias angulares más cortas para cada par en radianes.
        """
        if theta_array1.shape != theta_array2.shape:
            raise ValueError("Los arrays de ángulos deben tener la misma forma.")
        
        delta_thetas = (theta_array2 - theta_array1 + np.pi) % (2 * np.pi) - np.pi
        return delta_thetas

    #@timeit
    def generalized_task_augmentation(self,q, J_tasks, errors, deltaT=1, u=np.sqrt(0.001)):
        P = np.eye(len(q))
        q_dot = np.zeros(len(q))
        
        for J, r_dot in zip(J_tasks, errors):
            Ji_hash = np.linalg.pinv(J)
            q_dot += P @ Ji_hash @ r_dot
            P = P @ (np.eye(len(q)) - Ji_hash @ J)

        return q + q_dot * deltaT, q_dot
    
    @timeit
    def ikine_task(self,xdes,send = None):
        epsilon = 0.001 # Tolerancia para la convergencia
        max_iter = 100  # Número máximo de iteraciones
        
        qin = self._q.astype(float) 
        print(qin)
        q = self._q.astype(float) 

        for i in range(max_iter):
            

            e_pos = xdes - self.tLWrist
            e_securiti = np.array([0,0, 0.5 - self.tLElbow[2]])

            print(e_securiti)
            if np.linalg.norm(e_pos) < epsilon and e_securiti<0.5:
                print('Error Minimo')
                break

            J_tasks = [
                self.jLWrist,
                #self.jLElbow,
            ]
            errors = [
                e_pos,
                #e_securiti,
            ]
            q, qd = self.generalized_task_augmentation(q, J_tasks, errors, deltaT=0.1)
            self._q = q
            self.update()

            
        
        #self._q = self.shortest_angular_distances(qin,self._q)
        #self.update()

        if send is not None:
            send(self._q) 
        
        print(self._q)
        return self._q
    

    ##########################
    # Deprecated
    ##########################

    '''
    def matrixEuler2Wel(self, alpha, beta, gama ):
        # Matriz de transformación T
        T = np.array([
            
            [0, -np.sin(alpha), np.cos(alpha) * np.sin(beta)],
            [0,  np.cos(alpha), np.sin(alpha) * np.sin(beta)],
            [1,  0,             np.cos(beta)]
        ])
        
        # Matriz inversa directamente calculada
        #T_inv = np.linalg.inv(T)
        T_inv = T
        
        # Construcción eficiente de la matriz completa (6x6)
        full_matrix = np.eye(6)
        full_matrix[3:, 3:] = T_inv     # Inversa de T en la esquina inferior derecha
        
        return full_matrix
    '''
    
    '''
    def asEuler(_,Matrix):
        rotation = R.from_matrix(Matrix)
        return rotation.as_euler('zyx', degrees=False)
    '''

    '''
    def _funJacobianoGeometrio(self, n=None):
        if n is None:
            n = self.num_joints

        # Posición del extremo en función de las articulaciones
        p_n = self.params_dh.homogeneous_transform()[:3, 3]

        # Usar jacobiano simbólico en lugar de diferenciar en bucle
        qsym_mat = sp.Matrix(self._qsym[:n])
        J_linear = p_n.jacobian(qsym_mat)
        
        #return sp.trigsimp(J_linear)
        return J_linear

    def _funJacobianoGeometrioOpt(self, n=None):
        if n is None:
            n = self.num_joints

        T_list = [self.params_dh.homogeneous_transform(i + 1) for i in range(n)]
        z = [sp.Matrix([0, 0, 1])] + [T[:3, 2] for T in T_list]
        p_n = T_list[-1][:3, 3] 

        J_linear = [
            p_n.jacobian([self._qsym[i]]) for i in range(n)
        ]


        # Evitar crear más listas; construir directamente las filas del jacobiano
        return sp.Matrix.vstack(
            sp.Matrix.hstack(*J_linear),
            sp.Matrix.hstack(*z[:n])
        )
    '''

    '''def _funJacobianQuat(self, q, delta=0.0001):
        J = np.zeros((7, self.num_joints))
        T = self.tWrist
        quat_current = R.from_matrix(T[0:3, 0:3]).as_quat()
        
        dq = np.tile(q, (self.num_joints, 1))  # Crear una copia para cada articulación
        dq[np.arange(self.num_joints), np.arange(self.num_joints)] += delta  # Incrementar

        Td_all = np.array([self._tWrist_func(*dq_i) for dq_i in dq])  # Transformaciones en paralelo
        
        
        quats_perturbed = np.array([R.from_matrix(Td[0:3, 0:3]).as_quat() for Td in Td_all])
        quat_diffs = np.array([self.quaternion_difference(quat_current, q_p) for q_p in quats_perturbed]) / delta
        
        J[0:3, :] = self.jLWrist[0:3,:]
        J[3:, :] = quat_diffs.T

        return J
    '''

    '''@timeit
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
            Td = self._tWrist_func(*self._compute_trig_inputs(dq))
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
    '''
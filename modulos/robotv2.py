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
        
        self.__qsym = params_dh.q
        self._q = np.array([0 for x in range(self.num_joints)]).astype(float) 
        sin_q = [sp.Symbol(f'sin_q{i+1}') for i in range(self.num_joints)]
        cos_q = [sp.Symbol(f'cos_q{i+1}') for i in range(self.num_joints)]
        self.__new_inputs = sin_q + cos_q
        subs_dict = {
            trig: val
            for i in range(self.num_joints)
            for trig, val in zip([sp.sin(self.__qsym[i]), sp.cos(self.__qsym[i])], [sin_q[i], cos_q[i]])
        }

        # ###############
        # Elbow
        # ###############        
        self._tElbow_symbolic = self.__funCinematicaDirecta(2).subs(subs_dict)
        self._tElbow_func = sp.lambdify(self.__new_inputs, self._tElbow_symbolic, "numpy")

        self._tLElbow_func = sp.lambdify(self.__new_inputs, self._tElbow_symbolic[0:3, 3], "numpy")  # Posición del codo
        self._tOElbow_func = sp.lambdify(self.__new_inputs, self._tElbow_symbolic[0:3, 0:3], "numpy")  # Orientación del codo

        self._jLElbow_symbolic = self.__funJacobianoLineal(2).subs(subs_dict)
        self._jLElbow_func = sp.lambdify(self.__new_inputs, self._jLElbow_symbolic, "numpy")

        # ###############
        # Wrist
        # ###############

        self._tWrist_symbolic = self.__funCinematicaDirecta(4).subs(subs_dict)
        self._tWrist_func = sp.lambdify(self.__new_inputs, self._tWrist_symbolic, "numpy")

        self._tLWrist_func = sp.lambdify(self.__new_inputs, self._tWrist_symbolic[0:3,3], "numpy")
        self._tOWrist_func = sp.lambdify(self.__new_inputs, self._tWrist_symbolic[0:3,0:3], "numpy")

        self._jLWrist_symbolic = self.__funJacobianoLineal(4).subs(subs_dict)
        self._jLWrist_func = sp.lambdify(self.__new_inputs, self._jLWrist_symbolic, "numpy")
        
        self.update()

    
    def update(self):
        self._inputs = self.__compute_trig_inputs()

        self.tLWrist = self._tLWrist_func(*self._inputs).T[0]
        self.jLWrist = self._jLWrist_func(*self._inputs)

        self.tLElbow = self._tLElbow_func(*self._inputs).T[0]
        self.jLElbow = self._jLElbow_func(*self._inputs)

    def __compute_trig_inputs(self,q=None):
        if q is None:
            q = self._q
        sin_values = np.sin(q)  # Vectorizado
        cos_values = np.cos(q)  # Vectorizado
        return np.concatenate((sin_values, cos_values))  # Concatenar sinos y cosenos

    def generalized_task_augmentation(self,J_tasks, errors):
        P = np.eye(self.num_joints)
        q_dot = np.zeros(self.num_joints)
        
        for J, r_dot in zip(J_tasks, errors):
            Ji_hash = np.linalg.pinv(J)
            q_dot += P @ Ji_hash @ r_dot
            P = P @ (np.eye(self.num_joints) - Ji_hash @ J)

        return q_dot

    @timeit
    def ikine_task(self,xdes,funSend = None):
        epsilon = [1e-3] # Tolerancia para la convergencia
        max_iter = 1000  # Número máximo de iteraciones
        
        for i in range(max_iter):
            errors = [
                xdes - self.tLWrist,
            ]
            
            
            if np.linalg.norm(errors[0])<epsilon[0]:
                break

            J_tasks = [
                self.jLWrist,
            ]
            
            qd = self.generalized_task_augmentation(J_tasks, errors)
            #print(f'valores q actuales: {self._q}')
            self._q += 0.1*qd
            #print(f'valores q estimado: {self._q}')
            self.update()
        
        return self._q

    def __funCinematicaDirecta(self, n=None):
        T_total = self.params_dh.homogeneous_transform(n)
        return T_total

    def __funJacobianoLineal(self, n=None):
        if n is None:
            n = self.num_joints

        J_linear = sp.zeros(3, self.num_joints)
        # Posición del extremo en función de las articulaciones
        p_n = self.params_dh.homogeneous_transform(n)[:3, 3]

        # Usar jacobiano simbólico en lugar de diferenciar en bucle
        qsym_mat = sp.Matrix(self.__qsym[:n])
        J_linear[:, :n] = p_n.jacobian(qsym_mat)
        #return sp.trigsimp(J_linear)
        return J_linear
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
        # EndEfector
        # ###############        
        self._tEndEfector_symbolic = self.__funCinematicaDirecta().subs(subs_dict)
        self._tEndEfector_func = sp.lambdify(self.__new_inputs, self._tEndEfector_symbolic, "numpy")

        self._tLEndEfector_func = sp.lambdify(self.__new_inputs, self._tEndEfector_symbolic[0:3, 3], "numpy")  # Posición del codo
        self._tOEndEfector_func = sp.lambdify(self.__new_inputs, self._tEndEfector_symbolic[0:3, 0:3], "numpy")  # Orientación del codo

        self._jLEndEfector_symbolic = self.__funJacobianoLineal().subs(subs_dict)
        self._jLEndEfector_func = sp.lambdify(self.__new_inputs, self._jLEndEfector_symbolic, "numpy")

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

        self.tLEndEfector = self._tLEndEfector_func(*self._inputs).T[0]
        self.jLEndEfector = self._jLEndEfector_func(*self._inputs)

    def limit_joint_pos(self):
        """
        Delimita los valores articulares a los limites articulares del UR5
        """
        
        #Verifica si cada valor articular esta dentro de sus limites
        for i in range(6):
            if self._q[i] < self._q_lim[i,0]:
                self._q[i] = self._q_lim[i,0]
            elif self._q[i] > self._q_lim[i,1]:
                self._q[i] = self._q_lim[i,1]
            else:
                pass


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
        epsilon = [1e-4,1e-3] # Tolerancia para la convergencia
        max_iter = 30  # Número máximo de iteraciones
        
        for i in range(max_iter):
            

            epos = xdes - self.tLWrist

            # Limitar el módulo a 0.1
            #modulo = np.linalg.norm(epos)  # Calcula el módulo
            #if modulo > 0.5:
            #    epos = epos * (0.5 / modulo)  # Escala el vector para que su módulo sea 0.1

            
            
            theta= self._q[0]+np.pi/2
            rotation_matrix = np.array([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta),  np.cos(theta), 0],
                [0,0,1]
            ])
            eo = rotation_matrix @ np.array([0.00,-0.1,-0.08])
            
            eorien = (xdes+eo) - self.tLEndEfector
            
            errors = [
                epos,
                10*eorien,
            ]
            
            #print(errors[1])
            #print(np.linalg.norm(errors[1])<epsilon[1])
            if np.linalg.norm(errors[0])<epsilon[0] and np.linalg.norm(errors[1]*0.1)<epsilon[1]:
                break
            
                
            jOri = self.jLEndEfector
            #jOri[:,:3] = 0
            #print(jOri)
            J_tasks = [
                self.jLWrist,
                jOri,
            ]


            qd = self.generalized_task_augmentation(J_tasks, errors)
            #print(f'valores q actuales: {self._q}')
            self._q += 0.5*qd
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
import sympy as sp
import numpy as np
from . import dhParameters as dhp
from scipy.spatial.transform import Rotation as R

class robot:

    def __init__(self,params_dh):
        self.params_dh = params_dh
        self.num_joints = params_dh.num_joints
        self._qsym = params_dh.q
        self._q = np.array([0 for x in range(self.num_joints)])
        
        self._tWrist_symbolic = self._funCinematicaDirecta()
        self._tWrist_func = sp.lambdify(self._qsym, self._tWrist_symbolic, "numpy")

        self._jWrist_symbolic = self._funJacobianGeometrico()
        self._jWrist_func = sp.lambdify(self._qsym, self._jWrist_symbolic, "numpy")


        self.update()

    @property
    def q(self):
        """
        Getter para acceder al valor de q.
        """
        return self._q

    @q.setter
    def q(self, value):
        """
        Setter para actualizar el valor de q y realizar alguna acción.
        """
        self._q = value
        self.update()

    def update(self):
        self.tWrist = self._tWrist_func(*self._q)
        self.jGWrist = self._jWrist_func(*self._q)
        M = self.matrixEuler2Wel(self.tWrist[:3,:3])
        self.jAWrist = M @ self.jGWrist
        

    def matrixEuler2Wel(self, Matrix):
        alpha, beta, _ = self.asEuler(Matrix)
        
        # Matriz de transformación T
        T = np.array([
            [1, 0, np.sin(beta)],
            [0, np.cos(alpha), -np.sin(alpha) * np.cos(beta)],
            [0, np.sin(alpha), -np.cos(alpha) * np.cos(beta)]
        ])
        
        # Matriz inversa directamente calculada
        T_inv = np.linalg.inv(T)
        
        # Construcción eficiente de la matriz completa (6x6)
        full_matrix = np.zeros((6, 6))
        full_matrix[:3, :3] = np.eye(3)  # Identidad en la esquina superior izquierda
        full_matrix[3:, 3:] = T_inv     # Inversa de T en la esquina inferior derecha
        
        return full_matrix

    def asEuler(_,Matrix):
        rotation = R.from_matrix(Matrix)
        return rotation.as_euler('xyz', degrees=False)

    def _funCinematicaDirecta(self, n=None):
        T_total = self.params_dh.homogeneous_transform(n)
        return T_total

    def _funJacobianGeometrico(self,n=None):
        if(n == None):
            n = self.num_joints
        
        T_total = sp.eye(4)  # Transformación acumulada inicial (identidad)
        J = sp.zeros(6, n)  # Matriz jacobiana (6 filas x n columnas)

        z_i = T_total[:3, 2]  # Eje z de la articulación i
        o_i = T_total[:3, 3]  # Posición del origen de la articulación i
        o_n = self.params_dh.homogeneous_transform()[:3, 3]
        

        for i in range(n):
            T_i = self.params_dh.homogeneous_transform(i+1)
            T_total = T_i  # Acumula la transformación
            
            if self.params_dh.kind[i] == 'R':  # Articulación rotacional
                J[:3, i] = z_i.cross(o_n - o_i)  # Velocidad lineal
                J[3:, i] = z_i  # Velocidad angular
            elif self.params_dh.kind[i] == 'P':  # Articulación prismática
                # Columna del jacobiano para articulación prismática
                J[:3, i] = z_i  # Velocidad lineal
                J[3:, i] = sp.Matrix([0, 0, 0])  # Sin velocidad angular
            
            z_i = T_total[:3, 2]  # Eje z de la articulación i
            o_i = T_total[:3, 3]  # Posición del origen de la articulación i

        
        return J
        

'''if __name__ == "__main__":

    # Parámetros de ejemplo
    theta   = [0, 0, 0]
    d       = [1, 1, 1]
    a       = [1, 1, 0]
    alpha   = [0, sp.pi, 0]
    kind    = ['R', 'R', 'P']

    # Creación del objeto
    ur5 = robot(dhp.dhParameters(theta, d, a, alpha, kind))
    ur5._q = [0,0,3.14]
    ur5.update()
    print(ur5.jGWrist)
    print(ur5.jAWrist)'''
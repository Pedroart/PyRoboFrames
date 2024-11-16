import sympy as sp
import numpy as np
import time  as tm

class dhParameters:
    
    def __init__(self,theta, d, a, alpha,kind):
        """
        Inicializa los parametros con la convención Denavit–Hartenberg

        Args:
            theta (List float): Ángulo theta.
            d (List float): Desplazamiento d.
            a (List float): Longitud a.
            alpha (List float): Ángulo alpha.
            kind (str): Tipo de Movimiento Rotacional(R) o Prismatico(P).
        Raises:
            ValueError: Si las listas no tienen el mismo tamaño.
            ValueError: Si los elementos de 'kind' no son 'R' o 'P'.
        """
        # Verifica que todas las listas tengan el mismo tamaño
        if not (len(theta) == len(d) == len(a) == len(alpha) == len(kind)):
            raise ValueError("Todas las listas deben tener el mismo tamaño.")

        # Inicializa los parámetros
        self.theta = theta
        self.d = d
        self.a = a
        self.alpha = alpha
        self.kind = kind
        
        # Crea un vector simbólico para las variables articulares
        self.num_joints = len(theta)
        self.q = sp.symbols(f'q1:{self.num_joints+1}')  # Genera q1, q2, ..., qn según el tamaño de las listas
        
        self.params_dh = []

        #Construye la Matriz de Operacion
        for i in range(len(self.kind)):
            if self.kind[i] == 'R':  # Rotacional
                self.params_dh.append([self.q[i] + self.theta[i], self.d[i], self.a[i], self.alpha[i]])
            elif self.kind[i] == 'P':  # Prismático
                self.params_dh.append([self.theta[i], self.q[i] + self.d[i], self.a[i], self.alpha[i]])
        
    def __repr__(self):
        """
        Representación legible del objeto para depuración.
        """
        return (f"DHParameters(\n"
                f"  theta={self.theta},\n"
                f"  d={self.d},\n"
                f"  a={self.a},\n"
                f"  alpha={self.alpha},\n"
                f"  kind={self.kind},\n"
                f"  q={self.q}\n"
                f"  params={self.params_dh}\n"
                f")")
    
    def dh_matrix(self, theta, d, a, alpha):
        return sp.Matrix([
            [sp.cos(theta), -sp.sin(theta) * sp.cos(alpha), sp.sin(theta) * sp.sin(alpha), a * sp.cos(theta)],
            [sp.sin(theta), sp.cos(theta) * sp.cos(alpha), -sp.cos(theta) * sp.sin(alpha), a * sp.sin(theta)],
            [0, sp.sin(alpha), sp.cos(alpha), d],
            [0, 0, 0, 1]
        ])

    def homogeneous_transform(self,n= None):
        if(n == None):
            n = self.num_joints
        
        if not 0 < n <= self.num_joints:
            erro = f'El parametro N debe estar dentro del rango [0:{self.num_joints}]'
            raise ValueError(erro)

        T = np.eye(4)

        for param in self.params_dh:
            theta, d, a, alpha = param
            T = T @ self.dh_matrix(theta, d, a, alpha)
        return sp.sympify(T)

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
    print(dh_params)

    # Acceso a variables articulares simbólicas
    print(dh_params.q)

    ht = dh_params.homogeneous_transform()
    #print(ht)

    # Asignar valores a las variables simbólicas q
    valores_q = {dh_params.q[0]: sp.pi/6, dh_params.q[1]: 0.1, dh_params.q[2]: sp.pi/3}

    # Evaluar la matriz de transformación con los valores asignados
    ht_evaluada = ht.subs(valores_q)
    print("Matriz de transformación homogénea evaluada con valores de q:")
    print(ht_evaluada)

    ht_numpy = np.array(ht_evaluada.evalf()).astype(np.float64)

    print("Matriz de transformación homogénea en formato NumPy:")
    print(ht_numpy)

    
    ht_fun = sp.lambdify(dh_params.q, ht , "numpy")
    
    # Inicialización de parámetros de medición de tiempo
    n_evaluaciones = 100
    tiempos = []

    for _ in range(n_evaluaciones):
        # Generar valores aleatorios para las variables simbólicas q
        q_random = np.random.randint(1000, 9999, size=dh_params.num_joints)
        
        # Medir tiempo de evaluación
        tinit = tm.time()
        ht_fun(*q_random)  # Evaluar la función con los valores aleatorios
        t_eval = tm.time() - tinit
        
        # Almacenar tiempo
        tiempos.append(t_eval)

    # Calcular métricas
    tiempo_promedio = np.mean(tiempos)
    tiempo_minimo = np.min(tiempos)
    tiempo_maximo = np.max(tiempos)

    # Mostrar resultados
    print(f"Tiempo promedio de evaluación: {tiempo_promedio:.6f} segundos")
    print(f"Tiempo mínimo de evaluación: {tiempo_minimo:.6f} segundos")
    print(f"Tiempo máximo de evaluación: {tiempo_maximo:.6f} segundos")

    '''
    NumPy:
        Tiempo promedio de evaluación: 0.000213 segundos
        Tiempo mínimo de evaluación:   0.000128 segundos
        Tiempo máximo de evaluación:   0.000906 segundos
    '''
    
    '''
    SmyPy
        Tiempo promedio de evaluación: 0.000062 segundos
        Tiempo mínimo de evaluación:   0.000057 segundos
        Tiempo máximo de evaluación:   0.000141 segundos
    '''
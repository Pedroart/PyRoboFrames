import sympy as sp

# Variables simbólicas
theta1, theta2 = sp.symbols('theta1 theta2')
a, d = sp.symbols('a d')

# Ejemplo: Matriz homogénea simbólica (solo un ejemplo sencillo)
Rz = sp.Matrix([
    [sp.cos(theta1), -sp.sin(theta1), 0],
    [sp.sin(theta1), sp.cos(theta1), 0],
    [0, 0, 1]
])


# Matriz homogénea completa
T = sp.inv_quick(sp.Matrix.vstack(Rz, sp.inv_quick(Rz)))

# Simplificar trigonométricamente
T_simplified = sp.trigsimp(T)

# Extraer términos comunes
subexpr, simplified_expr = sp.cse(T_simplified)

# Mostrar resultados
print("Subexpresiones comunes:")
for i, expr in enumerate(subexpr):
    print(f"x{i} = {expr[1]}")

print("\nExpresión simplificada:")
sp.pprint(simplified_expr)

import sympy as sp
import numpy as np
import dhParameters as dhp

class Robot:

    def __init__(self,params_dh):
        self.params_dh = params_dh
        

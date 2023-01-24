import sys
sys.path.append('C:/Users/logan/Desktop/PhDResearch')
import header
import numpy as np 
import sympy 

phi, b1, b2, a = sympy.symbols('a phi b1 b2')
Htest = sympy.Matrix([[a*sympy.sin(phi)],[a*sympy.cos(phi)]])
A = np.array([[1,0,0],[0,1,0],[0,0,1]]) # state variables are phase and all bias states 
G = np.array([[0,1],[1,0]])             # process noise from IMU and clock phase noise 
C = np.array([[1,0,0]])                 # only measurement is phase 
D = np.array([1])                       # noise input 

nx = np.transpose(A).shape[1]                       # number of states must match columns of A 
ng = np.transpose(G).shape[1]                       # number of noise variables must match the colummns of G 
sys = header.ssSystem(A,np.zeros((nx,nx)),G,C,D)

print(sys.A)
 
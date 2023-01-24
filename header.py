from dataclasses import dataclass
import numpy as np 
import sympy


@dataclass
class ssSystem:
    """Class for holding dynamic and msmt info for a State-Space system"""
    A: np.ndarray
    B: np.ndarray
    G: np.ndarray
    C: np.ndarray
    D: np.ndarray

@dataclass
class SymSystem:
    """Class for holding dynamic and msmt info for a State-Space system in symbolic form"""
    A: sympy.matrices.dense.MutableDenseMatrix  # state transition matrix 
    B: sympy.matrices.dense.MutableDenseMatrix  # control input matrix 
    G: sympy.matrices.dense.MutableDenseMatrix  # the process noise input matrix 
    C: sympy.matrices.dense.MutableDenseMatrix  # the observation matrix
    D: sympy.matrices.dense.MutableDenseMatrix  # the msmt. noise matrix 

@dataclass
class KF:
    """Class for holding Kalman filter information. Note that all varables are updated at each time step."""
    x_hat: np.ndarray   # the msmt. update state estimate 
    P_hat: np.ndarray   # the x_hat's covariance 
    x_bar: np.ndarray   # the time update state estimate 
    P_bar: np.ndarray   # x_bar's covariance 
    z: np.ndarray       # the msmt. vector
    V: np.ndarray       # the covariance of the msmt. noise 
    # 'u,' the noise vector, is not a feature of the class since it is generated 
    W: np.ndarray       # the covariance of the process noise 
    L: np.ndarray       # the Kalman gain
    Phi: np.ndarray     # the state transition matrix 
    Gamma: np.ndarray   # the process noise transition matrix 

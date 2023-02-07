import numpy as np 
import sympy 
import control as ct
import header as h
import initialize as init

def KalmanFilter(sys: h.SymSystem, init_args: dict, traj: np.ndarray, stats: dict, Ts: float, T: float):
    # convert symbolic, non-linear system to a numeric one. 
    # It will need to be relinearized by passing a column of "traj"
    # as the "initials" field in "init_args"
    sys_num = init.initialize(init_args)
    ss_sys  = ct.ss(sys_num.A,sys_num.B,sys_num.C,sys_num.D)
    sysd  = ct.matlab.c2d(ss_sys,Ts,mehtod='zoh',prewarp_frequency=None)
    Phik  = sysd.A 
    GamUk = sysd.B
    # GamWk = sysd.G
    Hk    = sysd.C
    D     = sysd.D
    W     = stats.W
    V     = stats.V

    nx = Phik.shape[0]
    xbar = np.zeros((Phik.shape[1],int(T*Ts))) 
    xhat = np.zeros((Phik.shape[1],int(T*Ts)+1)) # plus one is from initial conditions
    xhat[:,0] = stats.x0
    u = np.zeros((GamUk.shape[1],int(T*Ts)))
    Pbar = np.zeros((nx,int(T*Ts)*nx))
    Phat = np.zeros((nx,int(T*Ts)*nx)+1)
    Phat[0:nx-1,0:nx-1] = stats.P0

    zk = np.zeros((Phik.shape[1],int(T*Ts))) 

    for k in range(1,int(T*Ts),1):
        # time update 
        xbar[:,k] = Phik*xhat[:,k-1] + GamUk*u[:,k-1]
        Pbar[:,k*nx:(k+1)*nx-1] = Phik*Phat*np.transpose(Phik) + GamWk*W*np.transpose(GamWk) # ASSUMES CONSTANT W AND V 
        # measurement update 
        K = Pbar[:,k*nx:(k+1)*nx-1]*np.transpose(Hk)*np.linalg.inv((V+Hk*Pbar[:,k*nx:(k+1)*nx-1]*np.transpose(Hk))) # ASSUMES CONSTANT W AND V 
        xhat[:,k] = xbar[:,k] + K*(zk[:,k]-Hk*xbar[:,k])
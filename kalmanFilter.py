import numpy as np 
import sympy 
import scipy as sp
import header as h
import simMeasurements as sm
import initialize as init
import pandas as pd

def KalmanFilter(sys: h.SymSystem, init_args: dict, traj: np.matrix, stats: dict, Ts: float, T: float, u: np.matrix, opt: str):
    # convert symbolic, non-linear system to a numeric one. 
    # It will need to be relinearized by passing a column of "traj"
    # as the "initials" field in "init_args"
    sys_num = init.initialize(init_args)
    Phik  = sys_num.A 
    GamUk = sys_num.B
    GamWk = sys_num.G
    Hk    = np.matrix(sys_num.C)
    # D     = sys_num.D
    W     = stats["W"]
    V     = stats["V"]

    nx = Phik.shape[0]
    xbar = np.matrix(np.zeros((Phik.shape[1],int(T*Ts)))) 
    xhat = np.matrix(np.zeros((Phik.shape[1],int(T*Ts)+1))) # plus one is from initial conditions
    xhat[:,0] = np.transpose(np.matrix(stats["x0"]))
    Pbar = np.matrix(np.zeros((nx,int(T*Ts)*nx)))
    Phat = np.matrix(np.zeros((nx,int(T*Ts)*nx+1)))
    Phat[0:nx,0:nx] = np.matrix(stats["P0"])

    if opt == "S":
        zk = sm.simMeas(V, init_args, traj.shape[1], traj) # simulates measurements of the system
        # convert array into dataframe
        DF = pd.DataFrame(zk)
        # save the dataframe as a csv file
        DF.to_csv("zk.csv")
    if opt == "R":
        zk = np.matrix(pd.read_csv('zk.csv'))

    for k in range(1,int(T*Ts),1):
        # time update 
        xbar[:,k] = Phik*xhat[:,k-1] + GamUk*u[:,k-1]
        Pbar[:,k*nx:(k+1)*nx] = Phik*Phat[:,(k-1)*nx:k*nx]*np.transpose(Phik) + GamWk*W*np.transpose(GamWk) # ASSUMES CONSTANT W AND V
        # measurement update 
        K = Pbar[:,k*nx:(k+1)*nx]*np.transpose(Hk)*np.linalg.inv((V+Hk*Pbar[:,k*nx:(k+1)*nx]*np.transpose(Hk))) # ASSUMES CONSTANT W AND V 
        xhat[:,k] = xbar[:,k] + K*(zk[:,k]-Hk*xbar[:,k])
        Phat[:,k*nx:(k+1)*nx] = np.linalg.inv(Pbar[:,k*nx:(k+1)*nx]) + np.transpose(Hk)*np.linalg.inv(V)*Hk
        # relinearize for next epoch of KF 
        fill = np.matrix([[1],[np.pi/2]]) # these are the signal amplitude and inital phase. They are constant for now.
        initals_hold = np.array(np.concatenate((traj[:,k],fill), axis=0))
        init_args["initials"] = initals_hold
        sys_num = init.initialize(init_args)
        Phik  = sys_num.A 
        GamUk = sys_num.B
        GamWk = sys_num.G
        Hk    = np.matrix(sys_num.C)
        # D     = sys_num.D
    return xhat, Phat, K
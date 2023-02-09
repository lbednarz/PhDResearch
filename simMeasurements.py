import numpy as np
import header as h
import initialize as init
from typing import Optional

def simMeas(V: np.ndarray, init_args: dict, len: int, traj: Optional[np.matrix]):
    sys_num = init.initialize(init_args)
    Hk = sys_num.C.astype(float)
    zk = np.matrix(np.zeros((Hk.shape[0],len)))
    w = np.matrix(np.random.randn(Hk.shape[0], len))
    fill = np.matrix([[1],[np.pi/2]]) # these are the signal amplitude and inital phase. They are constant for now.
    for k in range(1,len,1):
        x = traj[:,k].astype(float)
        initals_hold= np.array(np.concatenate((traj[:,k],fill), axis=0))
        init_args["initials"] = initals_hold
        sys_num = init.initialize(init_args)
        zk[:,k] = sys_num.C.astype(float)*x + V*w[:,k]
    return zk
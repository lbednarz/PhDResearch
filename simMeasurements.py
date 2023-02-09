import numpy as np
import header as h
import initialize as init
from typing import Optional

def simMeas(V: np.ndarray, init_args: dict, len: int, traj: Optional[np.ndarray]):
    sys_num = init.initialize(init_args)
    Hk = sys_num.H
    zk = np.Array(np.zeros((Hk.shape[0],len)))
    w = np.random.randn(Hk.shape[0], len)
    for k in range(0,len,1):
        x = traj[:,k]
        zk[:,k] = Hk*x + V*w[:,k]
        init_args["initials"] = traj[:,k+1]
        sys_num = init.initialize(init_args)
    return zk
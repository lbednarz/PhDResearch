import numpy as np
import sympy 
import dataclasses
import header 

#def initialize(sys: header.SymSystem, initials: np.ndarray, var_list: list, fields: list): 
def initialize(init_args: dict):  # NOTE a dataclass would be much safer here  
    # take numeric list of initial values of each variable in var_list and map them 
    # pull args from dict 
    sys = init_args["sys"]
    initials = init_args["initials"]
    var_list = init_args["var_list"]
    fields = init_args["fields"] 
    init_map = dict.fromkeys(var_list)
    i = 0
    for var in var_list:
        init_map[var] = initials[i]
        i = i + 1
    # sub in conditions from map and return numeric arrays for the system
    sys_dict = dataclasses.asdict(sys)
    dict_out = dict.fromkeys(fields)
    keys = sys_dict.keys()
    i = 0
    for key in keys:
        mat = sys_dict[key]
        mat_vars = mat.free_symbols
        for var in mat_vars:
            mat = mat.subs(var, init_map[var])
        dict_out[fields[i]] = np.array(mat).astype(np.float64)
        i = i + 1 
    return dict_out 



# initals = np.array([1, np.pi/2, .001, np.pi, 14, .2])
# initals = np.transpose(initals)
# var_list = list([a, phi, b, phi_0, fd, fddot])
# init_map = dict.fromkeys(var_list)
# i = 0
# for var in var_list:
#     init_map[var] = initals[i]
#     i = i + 1
# aa = dataclasses.asdict(sys)
# feilds = list(["A", "B", "G", "C", "D"]) # form elements of control system
# dict_out = dict.fromkeys(feilds)
# keys = aa.keys()
# i = 0
# for key in keys:
#     mat = aa[key]
#     mat_vars = mat.free_symbols
#     for var in mat_vars:
#         mat = mat.subs(var, init_map[var])
#     aa[key] = mat
#     dict_out[feilds[i]] = np.array(mat).astype(np.float64)
#     i = i + 1 
# print(dict_out)

    

    
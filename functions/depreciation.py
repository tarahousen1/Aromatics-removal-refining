# DEPRECIATION FUNCTION


import numpy as np

def depgendb(cost, salvage, life, factor):
    if life < 1:
        raise ValueError("Invalid life value. Life must be at least 1 period.")
    
    if isinstance(cost, (int, float)):
        cost = np.array([cost])
    if isinstance(salvage, (int, float)):
        salvage = np.array([salvage])
    if isinstance(life, (int, float)):
        life = np.array([life])
    if isinstance(factor, (int, float)):
        factor = np.array([factor])
        
    if cost.size != salvage.size or cost.size != life.size or cost.size != factor.size:
        raise ValueError("Inputs must have the same size.")
    
    oldlife = 0
    if life == 1:
        life = 2
        oldlife = 1

    if np.any(life <= 0):
        raise ValueError("Invalid life value. Life must be at least 1 period.")

    cs = cost - salvage
    span = np.arange(1, life[0])
    yr = np.tile(span, (len(cost), 1))
    n = life - 1
    d = (cost[:, None] * factor[:, None] / life) * (1 - factor[:, None] / life) ** (yr - 1)
    len_d = len(d)
    totald = np.cumsum(d, axis=1)
    i = np.where(totald > cs[:, None])
    if len(i[0]) > 0:
        idx = i[0][0]
        if idx == 0:
            d[idx, i[1][0]] = cs[i[0][0]]
        else:
            d[idx, i[1][0]] = cs[i[0][0]] - totald[i[0][0], i[1][0] - 1]
        zs = np.arange(i[1][0] + 1, len_d)
        d[i[0][0], zs] = 0

    sumd = np.sum(d, axis=1)
    d = np.hstack((d, np.where(sumd[:, None] > cs[:, None], np.zeros((len(cost), 1)), cs[:, None] - sumd[:, None])))
    if oldlife == 1:
        d = d[0, 0]
    return d.tolist()

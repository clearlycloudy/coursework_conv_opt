import cvxpy as cp
import numpy as np
from scipy.io import loadmat
import numpy.linalg as linalg
import math
import copy

def gen_seq_exhaustive(dim):
    
    q = [[1], [-1]]

    while len(q[0]) < dim:
        qq =[]
        for i in q:
            a = copy.deepcopy(i)
            b = copy.deepcopy(i)
            a.append(1)
            b.append(-1)
            s = len(i)
            assert(len(a)==len(b))
            assert(len(a)==s+1)
            qq.append(a)
            qq.append(b)
        q = qq

    ret = np.zeros((len(q), dim))
    idx = 0
    for i in q:
        ret[idx,:] = i
        idx += 1

    return ret

def solve_d_exhaustive(W):
    dim = W.shape[0]
    samples = gen_seq_exhaustive(dim)
    best_val = math.inf
    best_x = None
    for i in range(0, samples.shape[0]):
        x = samples[i,:].T
        v = (x.T).dot(W).dot(x)
        if v < best_val:
            best_val = v
            best_x = x
            
    print("problem size:", W.shape[0])
    print("best_objective (exhaustive): ", best_val)
    print("-----")
    return best_val, best_x
    
m = loadmat('../data/hw4data.mat')
w5 = np.array(m['W5'])
w10 = np.array(m['W10'])

solve_d_exhaustive(w5)
solve_d_exhaustive(w10)

import cvxpy as cp
import numpy as np
from scipy.io import loadmat
import numpy.linalg as linalg
import math

#generate samples each having dim numbers in {-1,1}
def gen_seq(dim, samples):
    return np.sign(np.random.rand(samples, dim) - 0.5)
    
def greedy(x,W):
    xx = x
    val = (xx.T).dot(W).dot(xx)

    while True:
        idx = None
        for i in range(0, xx.size):
            y = xx
            y[i] = -y[i]
            v = (y.T).dot(W).dot(y)
            if v < val:
                val = v
                idx = i
        if idx is None:
            break
        else:
            xx[i] = -xx[i]

    return val, xx

def solve_d_a(W):
    dim = W.shape[0]
    K = 100
    samples = gen_seq(dim, K)
    best_val = math.inf
    best_x = None
    for i in range(0,samples.shape[0]):
        x = samples[i,:].T
        val, xx = greedy(x, W)
        if val < best_val:
            best_val = val
            best_x = xx
            
    print("problem size:", W.shape[0])
    print("best_objective: ", best_val)
    print("-----")
    return best_val, best_x
    
m = loadmat('../data/hw4data.mat')
w5 = np.array(m['W5'])
w10 = np.array(m['W10'])
w50 = np.array(m['W50'])

solve_d_a(w5)
solve_d_a(w10)
solve_d_a(w50)

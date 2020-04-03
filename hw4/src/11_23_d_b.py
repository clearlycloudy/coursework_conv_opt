import cvxpy as cp
import numpy as np
from scipy.io import loadmat
import numpy.linalg as linalg
import math

def solve_heur(W):
    
    #dual of original:
    dim = W.shape[0]
    v = cp.Variable((dim,1))
    constraints = [W + cp.diag(v) >> 0]
    prob = cp.Problem(cp.Maximize( -cp.sum(v) ),
                      constraints)
    prob.solve()
    
    lower_bound = 0
    if prob.status not in ["infeasible", "unbounded"]:
        print("Optimal value: %s" % prob.value)
        lower_bound = prob.value

    print("lower_bound:", lower_bound)

    #dual of relaxed:
    X = cp.Variable((dim,dim))
    constraints = [X >> 0, cp.diag(X) == np.ones((dim,))]
    prob = cp.Problem(cp.Minimize( cp.trace(cp.matmul(W,X)) ),
                      constraints)
    prob.solve()
    
    if prob.status not in ["infeasible", "unbounded"]:
        print("Optimal value: %s" % prob.value)

    ret = prob.variables()[0].value
    eigenValues, eigenVectors = linalg.eig(ret)

    idx = eigenValues.argsort()[::-1]
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]
    
    x_approx = np.sign(eigenVectors[0])[:,np.newaxis]
    p_heuristic = (x_approx.T).dot(W).dot(x_approx)
    print("heuristic objective: ", p_heuristic)

    return p_heuristic, x_approx.T
    
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

def solve_d_b(W):
    
    print("performing greedy search")
    dim = W.shape[0]
    best_val = math.inf
    best_x = None
    _, xs = solve_heur(W)
    
    for i in range(0,xs.shape[0]):
        val, xx = greedy(xs[i,:].T, W)
        if val < best_val:
            best_val = val
            best_x = xx
            
    print("problem size:", W.shape[0])
    print("best objective (heuristic+greedy): ", best_val )
    print("-----")
    return best_val, best_x

m = loadmat('../data/hw4data.mat')
w5 = np.array(m['W5'])
w10 = np.array(m['W10'])
w50 = np.array(m['W50'])

solve_d_b(w5)
solve_d_b(w10)
solve_d_b(w50)

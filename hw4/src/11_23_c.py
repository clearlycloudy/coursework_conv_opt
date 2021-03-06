import cvxpy as cp
import numpy as np
from scipy.io import loadmat
import numpy.linalg as linalg
import math

def solve(W):
    print("problem size:", W.shape[0])
    
    #dual of original:
    print("dual of original:")
    dim = W.shape[0]
    v = cp.Variable((dim,1))
    constraints = [W + cp.diag(v) >> 0]
    prob = cp.Problem(cp.Maximize( -cp.sum(v) ),
                      constraints)
    prob.solve()

    # print("prob.status:", prob.status)
    
    lower_bound = 0
    
    if prob.status not in ["infeasible", "unbounded"]:
        print("Optimal value: %s" % prob.value)
        lower_bound = prob.value

    print("lower_bound: ", lower_bound)

    #dual of relaxed:
    
    #restrict to PSD for randomized sampling
    #on proper covariance matrix later
    X = cp.Variable((dim,dim))
    
    constraints = [X >> 0, cp.diag(X) == np.ones((dim,))]
    prob = cp.Problem(cp.Minimize( cp.trace(cp.matmul(W,X)) ),
                      constraints)
    prob.solve(solver=cp.SCS, max_iters=4000,
               eps=1e-11, warm_start=True)

    # print("prob.status:", prob.status)
    
    if prob.status not in ["infeasible", "unbounded"]:
        print("Optimal value: %s" % prob.value)
    
    ret = prob.variables()[0].value

    K = 100 #number of samples
    xs_approx = np.random.multivariate_normal(np.zeros((dim,)),
                                              ret, size=(K))
    xs_approx = np.sign(xs_approx) #shape: (K,dim)

    p_best = math.inf
    for i in range(0,K):
        p = (xs_approx[i,:].dot(W)).dot(xs_approx[i,:].T)
        if p < p_best:
            p_best = p

    print("best objective (randomized): ", p_best, "size: ", K)
    print("-----")

    return p_best, xs_approx
    
m = loadmat('../data/hw4data.mat')
w5 = np.array(m['W5'])
w10 = np.array(m['W10'])
w50 = np.array(m['W50'])

solve(w5)
solve(w10)
solve(w50)

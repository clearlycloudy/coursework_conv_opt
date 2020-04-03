# need these packages:
import cvxpy as cp
import numpy as np
import matplotlib.pylab as pl
import scipy.sparse as sps
import seaborn as sns
from scipy.io import loadmat
from os.path import dirname, join as pjoin
import numpy.linalg as linalg

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

    print("prob.status:", prob.status)
    
    lower_bound = 0
    
    if prob.status not in ["infeasible", "unbounded"]:
        # Otherwise, problem.value is inf or -inf, respectively.
        print("Optimal value: %s" % prob.value)
        lower_bound = prob.value

    lower_bound = -lower_bound
    print("lower_bound:", lower_bound)

    # for variable in prob.variables():
    #     print("Variable %s: value %s" % (variable.name(), variable.value))

    #dual of relaxed:
    print("dual of relaxed:")
    X = cp.Variable((dim,dim))
    constraints = [X >> 0, cp.diag(X) == np.ones((dim,))]
    prob = cp.Problem(cp.Minimize( cp.trace(cp.matmul(W,X)) ),
                      constraints)
    prob.solve()

    print("prob.status:", prob.status)
    
    if prob.status not in ["infeasible", "unbounded"]:
        # Otherwise, problem.value is inf or -inf, respectively.
        print("Optimal value: %s" % prob.value)

    # for variable in prob.variables():
    #     print("Variable %s: value %s" % (variable.name(), variable.value))

    ret = prob.variables()[0].value
    eigenValues, eigenVectors = linalg.eig(ret)

    idx = eigenValues.argsort()[::-1]
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]
    print("eigenvalues:",eigenValues)
    print("eigenVectors:",eigenVectors)
    print("eigenvec w/ largest eigenvalue: ")
    print(eigenVectors[0])
    print("sign(eigenvec) w/ largest eigenvalue: ")
    print(np.sign(eigenVectors[0]))
    x_approx = np.sign(eigenVectors[0])[:,np.newaxis]
    p_heuristic = (x_approx.T).dot(W).dot(x_approx)
    print("heuristic objective: ", p_heuristic)
    print("heuristic objective - lower_bound: ", p_heuristic - lower_bound)
    print("-----")
    
m = loadmat('../data/hw4data.mat')
w5 = np.array(m['W5'])
w10 = np.array(m['W10'])
w50 = np.array(m['W50'])

# print("w5:", w5.shape)
# print("w10:", w10)
# print("w50:", w50)
solve(w5)
solve(w10)
solve(w50)

import cvxpy as cp
import numpy as np
import matplotlib.pylab as pl
import scipy.sparse as sps
import seaborn as sns
from scipy.io import loadmat
from os.path import dirname, join as pjoin
import numpy.linalg as linalg

def objective(t, A, B, x, c, b, s):

    obj = 0.0
    obj += x/(c-x)
    obj += -1.0/t * np.sum(np.log(-(x-c)))
    obj += -1.0/t * np.sum(np.log(x))
    obj += -1.0/t * np.sum(np.log(-(B.dot(x)-b)))

    return obj

def objective_phase1(t, A, B, x, c, b, s):

    #assume y is concatenated in x as last element
    y = x[-1,0]
    
    obj = 0.0
    obj += -1.0/t * np.sum(np.log(-(x[:-1,0]-c[:-1,0]-y)))
    obj += -1.0/t * np.sum(np.log(x[:-1,0]+y))
    obj += -1.0/t * np.sum(np.log(-(B.dot(x)-b-y)))
    obj += y

    return obj

def constraint_max_val(t, A, B, x, c, b, s):
    
    val = np.amax( np.array([np.amax(np.log(-(x-c))),
                              np.amax(np.log(x)),
                              np.amax(np.log(-(B.dot(x)-b)))]) )

    return val
    
def residual(t, A, B, x, c, b, s):

    L = x.size
    r = np.zeros((L+A.shape[0], 1))
    gradient = np.zeros((L, 1))
    
    for i in range(0,L):
        gradient[i,0] += 1.0/(c[i,1]-x[i,1]) + x[i,1]/((c[i,1]-x[i,1])**2)
        gradient[i,0] += -1.0/t * 1.0/(x[i,1]-c[i,1])
        gradient[i,0] += -1.0/t * 1.0/(x[i,1])
        gradient[i,0] += -1.0/t * ( B[0,i]/(B[0,:].dot(x)-b[0,0]) + 
                                    B[1,i]/(B[1,:].dot(x)-b[1,0]) +
                                    B[2,i]/(B[2,:].dot(x)-b[2,0]) )
        
    r[0:x.size,0] = -1.0 * gradient
    r[x.size:,0] = s - A.dot(x)

    return r

def residual_phase1(t, A, B, x, c, b, s):
    
    #assume y is concatenated in x as last element
    y = x[-1,0]
    
    L = x.size
    r = np.zeros((L+A.shape[0], 1))
    gradient = np.zeros((L, 1))
    
    for i in range(0,L-1):
        gradient[i,0] += -1.0/t * 1.0/(x[i,1]-c[i,1]-y)
        gradient[i,0] += -1.0/t * 1.0/(x[i,1]+y)
        gradient[i,0] += -1.0/t * ( B[0,i]/(B[0,:].dot(x)-b[0,0]-y) + 
                                    B[1,i]/(B[1,:].dot(x)-b[1,0]-y) +
                                    B[2,i]/(B[2,:].dot(x)-b[2,0]-y) )

    for i in range(0,L-1):
        gradient[L-1,1] += 1.0/t * 1.0/(x[i,1]-c[i,1]-y)
        gradient[L-1,1] += 1.0/t * -1.0/(x[i,1]+y)
        gradient[L-1,1] += 1.0/t * ( 1.0/(B[0,:].dot(x)-b[0,0]-y) + 
                                     1.0/(B[1,:].dot(x)-b[1,0]-y) +
                                     1.0/(B[2,:].dot(x)-b[2,0]-y) )
        
    r[0:x.size,1] = -1.0 * gradient
    r[x.size:,1] = s - A.dot(x)

    return r

def kkt_matrix(t, A, B, x, c, b, s):

    L = x.size
    m = np.zeros((L + A.shape[0], L + A.shape[0]))

    for i in range(0,L):
        m[i,i] += 2.0/((c[i,1]-x[i,1])**2) + 2*x[i,1]/((c[i,1]-x[i,1])**3)
        m[i,i] += 1/t * 1.0/((x[i,1]-c[i])**2)
        m[i,i] += 1/t * 1.0/((x[i,1])**2)

    for i in range(0,L):
        for j in range(0,L):
            m[i,j] +=  1/t * ( (B[0,i]*B[0,j])/((B[0,:].dot(x)-b[0,0])**2) +
                               (B[1,i]*B[1,j])/((B[1,:].dot(x)-b[1,0])**2) +
                               (B[2,i]*B[2,j])/((B[2,:].dot(x)-b[2,0])**2) )

    m[L:L+A.shape[0],0:A.shape[1]] = A
    m[0:A.shape[1],0:A.shape[0]] = A.T

    return m

def phase1_init_point(A, B, x, c, b, s):
    
    #init y to be feasible, make it > max of inequality functions
    y = constraint_max_val(A, B, x, c, b, s)
    y += 100.0

    return x, y
    
def kkt_matrix_phase1(t, A, B, x, c, b, s):

    # y assumed to be concatenated to x as last element
    y = x[-1,1]
    
    L = x.size
    m = np.zeros((L + A.shape[0], L + A.shape[0]))

    #df^2/(dx_i dx_j):
    
    for i in range(0,L-1):
        m[i,i] += 1.0/t * 1.0/((x[i,1]-c[i,1]-y)**2)
        m[i,i] += 1.0/t * 1.0/((x[i,1]+y)**2)
    
    for i in range(0,L-1):
        for j in range(0,L-1):
            m[i,j] +=  1.0/t * ( (B[0,i]*B[0,j])/((B[0,:].dot(x)-b[0,0]-y)**2) +
                                 (B[1,i]*B[1,j])/((B[1,:].dot(x)-b[1,0]-y)**2) +
                                 (B[2,i]*B[2,j])/((B[2,:].dot(x)-b[2,0]-y)**2) )

    #df^2/dy^2:
    
    idx_y = m.shape[0]-1
    
    for i in range(0,L-1):
        m[idx_y, idx_y] += 1.0/t * 1.0/((x[i,1]-c[i,1]-y)**2)
        m[idx_y, idx_y] += 1.0/t * 1.0/((x[i,1]+y)**2)

    for i in range(0,L-1):
        m[idx_y, idx_y] +=  1.0/t * ( 1.0/((B[0,:].dot(x)-b[0,0]-y)**2) +
                                      1.0/((B[1,:].dot(x)-b[1,0]-y)**2) +
                                      1.0/((B[2,:].dot(x)-b[2,0]-y)**2) )

    #df/(dx_i d_y):
    for i in range(0,L-1):
        m[i, idx_y] += -1.0/t * 1.0/((x[i,1]-c[i,1]-y)**2)
        m[idx_y, i] += -1.0/t * 1.0/((x[i,1]-c[i,1]-y)**2)
        
        m[i, idx_y] += 1.0/t * 1.0/((x[i,1]+y)**2)
        m[idx_y, i] += 1.0/t * 1.0/((x[i,1]+y)**2)

        m[i, idx_y] +=  -1.0/t * ( B[0,i]/((B[0,:].dot(x)-b[0,0]-y)**2) +
                                   B[1,i]/((B[1,:].dot(x)-b[1,0]-y)**2) +
                                   B[2,i]/((B[2,:].dot(x)-b[2,0]-y)**2) )
        m[idx_y, i] +=  -1.0/t * ( B[0,i]/((B[0,:].dot(x)-b[0,0]-y)**2) +
                                   B[1,i]/((B[1,:].dot(x)-b[1,0]-y)**2) +
                                   B[2,i]/((B[2,:].dot(x)-b[2,0]-y)**2) )
        
    
    m[L:L+A.shape[0],0:A.shape[1]] = A
    m[0:A.shape[1],0:A.shape[0]] = A.T

    return m
    
def formulate():

    N = 8
    L = 13
    
    A = np.zeros((N-1,L))
    B = np.zeros((3,L))
    x = np.zeros((L,1)) #to be initialized in phase1
    c = np.zeros((L,1))
    b = np.zeros((3,1))
    s = np.zeros((N-1,1))

    #node1, links: out: 1,2,3, in:
    A[0,0] = 1.
    A[0,1] = 1.
    A[0,2] = 1.

    #node2, links: out: 4,6, in: 1
    A[1,3] = 1.
    A[1,5] = 1.
    A[1,0] = -1.

    #node3, links: out: 5,8, in: 3
    A[2,4] = 1.
    A[2,7] = 1.
    A[2,2] = -1.

    #node4, links: out: 7, in: 2,4,5
    A[3,6] = 1.
    A[3,1] = -1.
    A[3,3] = -1.
    A[3,4] = -1.

    #node5, links: out: 9,10,12, in: 7
    A[4,8] = 1.
    A[4,9] = 1.
    A[4,11] = 1.
    A[4,6] = -1.

    #node6, links: out: 11, in: 6,9
    A[5,10] = 1.
    A[5,5] = -1.
    A[5,8] = -1.

    #node7, links: out: 13, in: 8,10
    A[6,12] = 1.
    A[6,7] = -1.
    A[6,9] = -1.
    
    B[0,3] = 1
    B[0,5] = 1

    B[1,4] = 1
    B[1,7] = 1

    B[2,8] = 1
    B[2,9] = 1
    B[2,11] = 1

    c[:,0] = 1
    
    b[:,0] = 1
    
    s[0:3,0] = 1
    s[3:,0] = 0
    
    return [A, B, x, c, b, s]

def solve_kkt_phase1(t, A, B, x, c, b, s):
    kkt_m = kkt_matrix_phase1(t, A, B, x, c, b, s)
    ret = residual_phase1(t, A, B, x, c, b, s)
    return linalg.solve(kkt_m, res)
        
def solve_inner_phase1(t, A, B, x, c, b, s):

    eps1 = 1e-7
    eps2 = 1e-7
    
    while True:
        print("loop inner, phase 1")
        y = solve_kkt_phase1(t, A, B, x, c, b, s)
    
        v = y[0:x.size]
        w = y[x.size:]
    
        delta_x = v
        delta_v = w - v
        
        #backtracking line search
        beta = 0.9
        alpha = 0.5
        h = 1
        res_1 = np.concatenate(x+h*delta_x, v+h*delta_v)
        res_2 = np.concatenate(x, v)
        assert(res_1.size == res_2.size)
        assert(res_1.size == y.size)
        
        while linalg.norm(res_1) > (1-alpha*t)*linalg.norm(res_2):
            h = beta * h
            
        x = x + h * delta_x
        v = v + h * delta_v

        print("inner loop, phase1: objective: ", objective_phase1(t, A, B, x, c, b, s))
        
        if linalg.norm(np.concatenate(x, v)) <= eps1 and A.dot(x)-b <= eps2:
            break

    return x, objective_phase1(t, A, B, x, c, b, s)

def solve_phase1(A, B, x, c, b, s):

    x, y = phase1_init_point(A, B, x, c, b, s)

    #augment x -> [x[:], y] for feasibility search
    xx = np.copy(x)
    xx = np.concatenate(xx, np.array([[y]]))
    AA = np.copy(A)
    AA = np.concatenate(AA, np.zeros((A.shape[0],1)))
    BB = np.copy(B)
    BB = np.concatenate(BB, np.zeros((B.shape[0],1)))
    
    t = 1.0
    mu = 10.0
    m = AA.shape[0]
    eps = 1e-7
    
    while True:
        print("loop outer, phase1")
        xx, obj = solve_inner_phase1(t, AA, BB, xx, c, b, s)
        
        t = mu * t
        if m/t <= eps:
            break

    return xx, obj

def solve_kkt(t, A, B, x, c, b, s):
    kkt_m = kkt_matrix(t, A, B, x, c, b, s)
    res = residual(t, A, B, x, c, b, s)
    return linalg.solve(kkt_m, res)

def solve_inner(t, A, B, x, c, b, s):

    eps1 = 1e-7
    eps2 = 1e-7
    
    while True:
        print("loop inner")
        y = solve_kkt(t, A, B, x, c, b, s)
    
        v = y[0:x.size]
        w = y[x.size:]
    
        delta_x = v
        delta_v = w - v
        
        #backtracking line search
        beta = 0.9
        alpha = 0.5
        h = 1
        res_1 = np.concatenate(x+h*delta_x, v+h*delta_v)
        res_2 = np.concatenate(x, v)
        assert(res_1.size == res_2.size)
        assert(res_1.size == y.size)
        
        while linalg.norm(res_1) > (1-alpha*t)*linalg.norm(res_2):
            h = beta * h
            
        x = x + h * delta_x
        v = v + h * delta_v

        print("inner loop: objective: ", objective(t, A, B, x, c, b, s))
        
        if linalg.norm(np.concatenate(x, v)) <= eps1 and A.dot(x)-b <= eps2:
            break

    return x, objective(t, A, B, x, c, b, s)
        
def solve(A, B, x, c, b, s):

    t = 1.0
    mu = 10.0
    xs, _ = solve_phase1(A, B, x, c, b, s)
    print("phase1 xs: ", xs)
    # x = xs[:-1,0]
    # m = A.shape[0]
    # eps = 1e-7
    
    # while True:
    #     print("loop outer")
    #     x, obj = solve_inner(t, A, B, x, c, b, s)
        
    #     t = mu * t
    #     if m/t <= eps:
    #         break

    # return x, obj

prob = formulate()
solution, objective = solve(*prob)

print("objective achieved: ", objective)
print("solution: ", solution)

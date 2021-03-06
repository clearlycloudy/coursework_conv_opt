import cvxpy as cp
import numpy as np
import matplotlib.pylab as pl
import scipy.sparse as sps
import seaborn as sns
from scipy.io import loadmat
from os.path import dirname, join as pjoin
import numpy.linalg as linalg
from scipy import linalg as scipy_linalg

def objective(t, A, B, x, c, b, s):

    obj = 0.0
    obj += x/(c-x)
    obj += -1.0/t * np.sum(np.log(-(x-c)))
    obj += -1.0/t * np.sum(np.log(x))
    obj += -1.0/t * np.sum(np.log(-(B.dot(x)-b)))

    return obj

def objective_phase1(t, A, B, x, c, b, s, dim_x_orig):

    #assume y1,y2,y3 are concatenated in x
    ys = x[dim_x_orig:,0]
    y1 = ys[0:dim_x_orig]
    y2 = ys[dim_x_orig:2*dim_x_orig]
    y3 = ys[2*dim_x_orig:]
        
    obj = 0.0
    obj += -1.0/t * np.sum(np.log(-(x[0:dim_x_orig,0]-c[0:dim_x_orig,0]-y1)))
    obj += -1.0/t * np.sum(np.log(x[0:dim_x_orig,0]+y2))
    obj += -1.0/t * np.sum(np.log(-(B.dot(x)-b-y3)))
    obj += np.sum(ys)

    return obj

def constraint_max_val(A, B, x, c, b, s):
    
    val = np.amax( np.array([np.amax(x-c),
                             np.amax(-x),
                             np.amax(B.dot(x)-b)]) )
        
    return val

def grad(t, A, B, x, c, b, s):
    
    L = x.size
    gradient = np.zeros((L, 1))
    
    for i in range(0,L):
        gradient[i,0] += 1.0/(c[i,0]-x[i,0]) + x[i,0]/(np.power(c[i,0]-x[i,0]),2)
        gradient[i,0] += -1.0/t * 1.0/(x[i,0]-c[i,0])
        gradient[i,0] += -1.0/t * 1.0/(x[i,0])
        gradient[i,0] += -1.0/t * ( B[0,i]/(B[0,:].dot(x)-b[0,0]) + 
                                    B[1,i]/(B[1,:].dot(x)-b[1,0]) +
                                    B[2,i]/(B[2,:].dot(x)-b[2,0]) )
    return gradient

def grad_phase1(t, A, B, x, c, b, s, dim_x_orig):

    #assume y's are concatenated in x
    y1 = x[dim_x_orig: 2*dim_x_orig, :]
    y2 = x[2*dim_x_orig: 3*dim_x_orig, :]
    y3 = x[3*dim_x_orig:, :]

    gradient = np.zeros((x.size, 1))

    for i in range(0,dim_x_orig):
        #df/dx_i
        gradient[i,0] += -1.0/(x[i,0]-c[i,0]-y1[i,0])
        gradient[i,0] += -1.0/(x[i,0]+y2[i,0])
        gradient[i,0] += -1.0 * ( B[0,i]/(B[0,:].dot(x)-b[0,0]-y3[0,0]+1e-15) + 
                                  B[1,i]/(B[1,:].dot(x)-b[1,0]-y3[1,0]+1e-15) +
                                  B[2,i]/(B[2,:].dot(x)-b[2,0]-y3[2,0]+1e-15) )

    #df/dy1
    gradient[dim_x_orig: 2*dim_x_orig,:] = t - 1.0/(-x[0:dim_x_orig,:] + c[0:dim_x_orig,:] + y1 + 1e-15)
    
    #df/dy2
    gradient[2*dim_x_orig: 3*dim_x_orig,:] = t - 1.0/(x[0:dim_x_orig,:] + y2 + 1e-15)
    
    #df/dy3
    gradient[3*dim_x_orig:, :] = t + 1.0/(B.dot(x)-b-y3 + 1e-15)
    
    return gradient

def residual_prim(t, A, B, x, c, b, s, v):
    return A.dot(x)-s

def residual_dual(t, A, B, x, c, b, s, v):
    return grad(t, A, B, x, c, b, s) + (A.T).dot(v)

def residual_dual_phase1(t, A, B, x, c, b, s, v, dim_x_orig):
    return grad_phase1(t, A, B, x, c, b, s, dim_x_orig) + (A.T).dot(v)

def kkt_rhs(t, A, B, x, c, b, s):

    L = x.size
    r = np.zeros((L+A.shape[0], 1))

    gradient = grad(t, A, B, x, c, b, s)

    r[0:x.size,:] = -1.0 * gradient
    r[x.size:,:] = s - A.dot(x)

    return r

def kkt_rhs_phase1(t, A, B, x, c, b, s, dim_x_orig):
    # #assume feasibility variables y1,y2,y3 are concatenated in x

    r = np.zeros((x.size+A.shape[0], 1))

    gradient = grad_phase1(t, A, B, x, c, b, s, dim_x_orig)

    r[0:x.size,:] = -1.0 * gradient
    r[x.size:,:] = s - A.dot(x)

    return r

def kkt_matrix(t, A, B, x, c, b, s):

    L = x.size
    m = np.zeros((L + A.shape[0], L + A.shape[0]))

    for i in range(0,L):
        m[i,i] += 2.0/(np.power(c[i,0]-x[i,0]),2) + 2*x[i,0]/(np.power(c[i,0]-x[i,0]),3)
        m[i,i] += 1/t * 1.0/(np.power(x[i,0]-c[i]),2)
        m[i,i] += 1/t * 1.0/(np.power(x[i,0]),2)

    for i in range(0,L):
        for j in range(0,L):
            m[i,j] +=  1/t * ( (B[0,i]*B[0,j])/(np.power(B[0,:].dot(x)-b[0,0]),2) +
                               (B[1,i]*B[1,j])/(np.power(B[1,:].dot(x)-b[1,0]),2) +
                               (B[2,i]*B[2,j])/(np.power(B[2,:].dot(x)-b[2,0]),2) )

    m[L:L+A.shape[0],0:A.shape[1]] = A
    m[0:A.shape[1],L:L+A.shape[0]] = A.T

    return m

def phase1_init_point(A, B, x, c, b, s):
    
    #init y to be feasible, make it > max of inequality functions
    # y = constraint_max_val(A, B, x, c, b, s)

    y1 = x-c
    y2 = -x
    y3 = B.dot(x)-b

    y1 += 1.0
    y2 += 1.0
    y3 += 0.5

    return x, y1, y2, y3
    
def kkt_matrix_phase1(t, A, B, x, c, b, s, dim_x_orig):

    #assume y's are concatenated in x
    y1 = x[dim_x_orig: 2*dim_x_orig, :]
    y2 = x[2*dim_x_orig: 3*dim_x_orig, :]
    y3 = x[3*dim_x_orig:, :]
    
    #hessian:
    m = np.zeros((x.size + A.shape[0], x.size + A.shape[0]))
    
    #df^2/(dx_i dx_j):
    for i in range(0,dim_x_orig):
        m[i,i] += 1.0/np.power(x[i,0]-c[i,0]-y1[i,0],2)
        m[i,i] += 1.0/np.power(x[i,0]+y2[i,0],2)

    for i in range(0,dim_x_orig):
        for j in range(0,dim_x_orig):
            m[i,j] +=  1.0 * ( (B[0,i]*B[0,j])/np.power(B[0,:].dot(x)-b[0,0]-y3[0,0],2) +
                               (B[1,i]*B[1,j])/np.power(B[1,:].dot(x)-b[1,0]-y3[1,0],2) +
                               (B[2,i]*B[2,j])/np.power(B[2,:].dot(x)-b[2,0]-y3[2,0],2) )

    #df^2/dy1^2:
    for i in range(0, dim_x_orig):
        m[dim_x_orig+i, dim_x_orig+i] += 1.0/np.power(x[i,0]-c[i,0]-y1[i,0],2)

    #df^2/dy2^2:
    for i in range(0, dim_x_orig):
        m[2*dim_x_orig+i, 2*dim_x_orig+i] += 1.0/np.power(x[i,0]+y2[i,0],2)

    #df^2/dy3^2:
    for i in range(0, 3):
        m[3*dim_x_orig+i, 3*dim_x_orig+i] += 1.0/np.power(B[i,:].dot(x)-b[i,0]-y3[i,0],2)


    for i in range(0,dim_x_orig):
        
        #df/(dx_i d_y1_i):
        v1 = -1.0/np.power(-x[i,0]+c[i,0]+y1[i,0],2)
        m[i, dim_x_orig+i] += v1
        m[dim_x_orig+i, i] += v1

        #df/(dx_i d_y2_i):
        v2 = 1.0/np.power(x[i,0]+y2[i,0],2)
        m[i, 2*dim_x_orig+i] += v2
        m[2*dim_x_orig+i, i] += v2

        #df/(dx_i d_y3_k):
        for k in range(0,3):
            v3 = -1.0 * B[k,i] / np.power(B[k,:].dot(x)-b[k,0]-y3[k,0],2)
            m[i, 3*dim_x_orig+k] += v3
            m[3*dim_x_orig+k, i] += v3

    m[x.shape[0]: x.shape[0]+A.shape[0], 0:A.shape[1]] = A
    m[0:A.shape[1], x.shape[0]:x.shape[0]+A.shape[0]] = A.T

    ax = sns.heatmap(m)
    pl.show()

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
    
    B[0,3] = 1.
    B[0,5] = 1.

    B[1,4] = 1.
    B[1,7] = 1.

    B[2,8] = 1.
    B[2,9] = 1.
    B[2,11] = 1.

    c[:,0] = 1.
    
    b[:,0] = 1.
    
    s[0,0] = 1.2
    s[1,0] = 0.6
    s[2,0] = 0.6
    s[3:,0] = 0.

    # ax = sns.heatmap(B)
    # pl.show()

    return [A, B, x, c, b, s]

def solve_kkt_phase1(t, A, B, x, c, b, s, dim_x_orig):
    kkt_m = kkt_matrix_phase1(t, A, B, x, c, b, s, dim_x_orig)
    res = kkt_rhs_phase1(t, A, B, x, c, b, s, dim_x_orig)

    # ax = sns.heatmap(res)
    # pl.show()
    # ax = sns.heatmap(kkt_m)
    # pl.show()
    # print("phase 1, kkt matrix rank:", np.rank(kkt_m))
    # return scipy_linalg.solve(kkt_m, res, assume_a='sym')
    return linalg.solve(kkt_m, res)
        
def solve_inner_phase1(t, A, B, x, c, b, s, v, dim_x_orig):

    eps1 = 1e-7
    eps2 = 1e-7

    print("x:", x)
            
    while True:
        print("loop inner, phase1, t: ", t)
        xyv = solve_kkt_phase1(t, A, B, x, c, b, s, dim_x_orig)
        
        delta_x = xyv[0:3*dim_x_orig+3,:]
        v_new = xyv[3*dim_x_orig+3:,:]
        if v is None:
            v = v_new
        delta_v = v_new - v

        # delta_x = delta_xy[0:dim_x_orig,:]
        # print("v_new:", v_new)
        # print("delta_v: ",  delta_v)
        # print("delta_x", delta_x)
        print("x_new:", (x+delta_x)[0:dim_x_orig,:])
        # print("y_new:", (x+delta_x)[dim_x_orig:,:])
        
        # print("inner loop, phase1, kkt solve: feasibility var: ", delta_x[-1,0])
        # print(x+delta_x)
        
        #backtracking line search
        beta = 0.95
        alpha = 0.45
        h = 1.0
        
        # print(delta_x)
        # print(delta_v)
        loopc = 0
        while True:
            loopc+= 1
            assert(v.shape == delta_v.shape)

            res_prim_next = residual_prim(t, A, B, x+h*delta_x, c, b, s, v+h*delta_v)
            res_prim_cur = residual_prim(t, A, B, x, c, b, s, v)

            res_dual_phase1_next = residual_dual_phase1(t, A, B, x+h*delta_x, c, b, s, v+h*delta_v, dim_x_orig)
            res_dual_phase1_cur = residual_dual_phase1(t, A, B, x, c, b, s, v, dim_x_orig)

            r1 = np.concatenate((res_prim_next,res_dual_phase1_next), axis=0)
            r2 = np.concatenate((res_prim_cur,res_dual_phase1_cur), axis=0)

            if loopc % 10 == 0:
                print("loopc:", loopc, ", t:", t)
                # print("inner loop, phase1: linesearch: objective: ", objective_phase1(t, A, B, x+h*delta_x, c, b, s, dim_x_orig))

                # print(x+h*delta_x)

                print("x:", x[0:dim_x_orig,:])
                print("delta x:", delta_x[0:dim_x_orig,:])
                print("y:", x[dim_x_orig:,:])
                        
                print("r1: ", linalg.norm(r1), ", r2: ", linalg.norm(r2))

                print("true res: ", np.linalg.norm(A.dot(x+h*delta_x)-s,2))
                pass
            if linalg.norm(r1) <= (1-alpha*t)*linalg.norm(r2):
            # if linalg.norm(r1) <= 0.95 * linalg.norm(r2) or (linalg.norm(r1) <= linalg.norm(r2) and loopc > 7500):

                print("r1: ", linalg.norm(r1), ", r2: ", linalg.norm(r2))

                print("true res: ", np.linalg.norm(A.dot(x+h*delta_x)-s,2))
                
                # print("inner loop, phase1: linesearch: objective: ", objective_phase1(t, A, B, x+h*delta_x, c, b, s))

                # print("feasibility var: " , (x+h*delta_x)[-1,0])

                # print("r1: ", linalg.norm(r1), ", r2: ", linalg.norm(r2))

                # print("true res: ", np.linalg.norm(A.dot(x+h*delta_x)-s,2))
                                                   
                # print(x+h*delta_x)
                print("exit line search")
                break
            
            h = beta * h
            
        x = x + h * delta_x
        v = v + h * delta_v

        print("inner loop, phase1: objective: ", objective_phase1(t, A, B, x, c, b, s, dim_x_orig))

        res_prim_cur = residual_prim(t, A, B, x, c, b, s, v)
        res_dual_phase1_cur = residual_dual_phase1(t, A, B, x, c, b, s, v, dim_x_orig)
        r = np.concatenate((res_prim_cur,res_dual_phase1_cur), axis=0)

        print("true res: ", np.linalg.norm(A.dot(x)-s,2))
        
        # if linalg.norm(r) <= eps1 and np.linalg.norm(A.dot(x)-s,2) <= eps2:
        if np.linalg.norm(A.dot(x)-s,2) <= eps2:
            break

    return x, v, objective_phase1(t, A, B, x, c, b, s, dim_x_orig)

def solve_phase1(A, B, x, c, b, s):

    x, y1, y2, y3 = phase1_init_point(A, B, x, c, b, s)

    dim_x_orig = x.shape[0]
    
    #augment x -> [x, y1, y2, y3] for feasibility search
    xx = np.copy(x)

    dims_y = y1.shape[0] + y2.shape[0] + y3.shape[0]
    
    xx = np.concatenate((xx, np.array(y1,ndmin=2), np.array(y2,ndmin=2), np.array(y3,ndmin=2)), axis=0)
    assert(xx.shape == (x.shape[0] + dims_y, 1))

    AA = np.copy(A)
    AA = np.concatenate((AA, np.zeros((A.shape[0],dims_y))), axis=1)
    assert(AA.shape == (A.shape[0], x.shape[0] + dims_y))

    BB = np.copy(B)
    BB = np.concatenate((BB, np.zeros((B.shape[0],dims_y))), axis=1)
    assert(BB.shape == (B.shape[0], x.shape[0] + dims_y))
    
    # ax = sns.heatmap(AA)
    # pl.show()
    
    t = 0.1
    mu = 2.0
    m = AA.shape[0]
    eps = 1e-7

    v = None
    
    while True:
        print("loop outer, phase1")
        xx, v, obj = solve_inner_phase1(t, AA, BB, xx, c, b, s, v, dim_x_orig)
        
        t = mu * t
        if m/t <= eps:
            break

    return xx, v, obj

def solve_kkt(t, A, B, x, c, b, s):
    kkt_m = kkt_matrix(t, A, B, x, c, b, s)
    res = kkt_rhs(t, A, B, x, c, b, s)
    # return scipy_linalg.solve(kkt_m, res, assume_a='sym')
    return linalg.solve(kkt_m, res)

def solve_inner(t, A, B, x, c, b, s, v):

    eps1 = 1e-7
    eps2 = 1e-7
        
    while True:
        print("loop inner")
        y = solve_kkt(t, A, B, x, c, b, s)
    
        delta_x = y[0:x.size,:]
        delta_v = y[x.size:,:]

        assert(v.shape == delta_v.shape)
                
        #backtracking line search
        beta = 0.9
        alpha = 0.3
        h = 1.0
        # res_1 = np.concatenate(x+h*delta_x, v+h*delta_v)
        # res_2 = np.concatenate(x, v)
        # assert(res_1.size == res_2.size)
        # assert(res_1.size == y.size)
        
        # while linalg.norm(res_1) > (1-alpha*t)*linalg.norm(res_2):
        #     h = beta * h
        while True:
            res_1 = np.concatenate((x+h*delta_x, v+h*delta_v), axis=0)
            res_2 = np.concatenate((x, v), axis=0)
            
            assert(res_1.size == res_2.size)
            assert(res_1.size == y.size)
            
            if linalg.norm(res_1) <= (1-alpha*t)*linalg.norm(res_2):
                break

        x = x + h * delta_x
        v = v + h * delta_v

        print("inner loop: objective: ", objective(t, A, B, x, c, b, s))
        
        if linalg.norm(np.concatenate(x, v)) <= eps1 and A.dot(x)-s <= eps2:
            break

    return x, v, objective(t, A, B, x, c, b, s)
        
def solve(A, B, x, c, b, s):


        
    t = 1.0
    mu = 10.0
    xs, v, obj = solve_phase1(A, B, x, c, b, s)
    print("phase1 xs: ", xs)
    print("phase1 v: ", v)
    print("phase1 objective value: ", obj)
    # x = xs[:-1,0]
    # m = A.shape[0]
    # eps = 1e-7
    #v = np.zeros((A.shape[0],1))
    
    # while True:
    #     print("loop outer")
    #     x, v, obj = solve_inner(t, A, B, x, c, b, s)
        
    #     t = mu * t
    #     if m/t <= eps:
    #         break

    # return x, obj

prob = formulate()
solution, objective = solve(*prob)

print("objective achieved: ", objective)
print("solution: ", solution)

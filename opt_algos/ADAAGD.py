import numpy as np
import quadprog as qp
import cvxpy as cvx
import dccp
import time

def projector(y, A, b):
    """
    ref: https://github.com/wwehner/projgrad/blob/master/projgrad_algo2.m

    Parameters
    ----------
    y: the solution got from gradient descent
    A: the matrix in constraints
    b: the value vector in constraints

    Returns
    -------
    x: the projected solution

    """
    n = len(y)
    H = np.identity(n)
    x, _, _, _, _, _ = qp.solve_qp(H, y, -A, -b, 0)
    return x

def find_R(z, A, b):
    """
    ref: https://github.com/cvxgrp/dccp

    Parameters
    ----------
    z: a possible solution in the convex set
    A: the matrix in constraints
    b: the value vector in constrains

    Returns
    -------
    R: the maximum value of (x - y) norm 2. Or the negative of minimum value of - (x - y) norm 2

    """
    x = cvx.Variable(z.shape)
    y = cvx.Variable(z.shape)

    prob = cvx.Problem(cvx.Maximize(cvx.norm(x - y, 2)), [A@x <= b, A@y <= b])
    result = prob.solve(method='dccp')
    return result[0]

def project_z(grad_s, z0, D, A, b):
    """

    Parameters
    ----------
    grad_history: gradient for each x
    z0: z0
    D: D_t
    A: the matrix in constraints
    b: the value vector in constrains

    Returns
    -------
    x: the solution to this optimization problem

    """
    a = z0.T@D - grad_s
    x, _, _, _, _, _ = qp.solve_qp(D, a.T, -A, -b, 0)
    return x


def ADAAGD_plus(model, max_iterations=1e4, epsilon=1e-5,
                   x0=None, A=None, b=None, R=None):
    """
    ref: Adaptive Gradient Methods for Constrained Convex Optimization and Variational Inequalities

    Parameters
    ----------
    model
    max_iterations
    epsilon

    Returns
    -------

    """
    # data from model
    grad_F = model.grad_F
    d = model.d
    # F = model.F

    # initialization
    if x0 is not None:
        y0 = x0
    else:
        y0 = np.random.normal(loc=0, scale=1, size=d)

    # make sure the initial  iterate is feasible
    z0 = projector(y0, A, b)
    z_previous = z0
    y_previous = z0

    # initialization of D1 and R
    D_current = np.identity(len(z0))
    if R is None:
        R = find_R(z0, A.T, b)
        #print(R)

    # keep track of x
    x_history = []
    objective_history = []
    y_history = []
    grad_history = []
    grad_s = np.zeros_like(z0)

    for k in range(1, int(max_iterations)):
        t0 = time.time()
        a_current = k
        A_current = k*(k+1)/2
        x_current = (k-1)/(k+1) * y_previous + a_current/A_current * z_previous
        x_history.append(x_current)

        grad = grad_F(x_current)
        grad_history.append(grad)
        grad_s += k * grad

        z_current = project_z(grad_s, z0, D_current, A, b)

        y_current = (k-1)/(k+1) * y_previous + a_current/A_current * z_current
        y_history.append(y_current.tolist())

        objective = model.F(y_current)
        objective_history.append(objective)
        print("Objective value = ", objective)

        # next D
        D_next = D_current + np.diag(np.sqrt(1 + np.square(z_current - z_previous)/R**2))

        print('time', str(time.time() - t0))

        if np.linalg.norm(y_current - y_previous) <= epsilon*np.linalg.norm(y_previous):
            break

        y_previous = y_current
        z_previous = z_current
        D_current = D_next

    print('ADAAGD+ finished after ' + str(k) + ' iterations')

    return {'solution': y_current.tolist(),
            'x_history': y_history,
            'objective_history': objective_history,
            }

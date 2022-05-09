import numpy as np
import quadprog as qp

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

def find_R(x, A, b):
    """

    Parameters
    ----------
    x: a possible solution in the convex set
    A: the matrix in constraints
    b: the value vector in constrains

    Returns
    -------
    R: the maximum value of (x - y) norm 2. Or the negative of minimum value of - (x - y) norm 2

    """
    n = len(x)
    I = np.identity(n)
    H = np.block([
        [I, -I],
        [-I, I]
    ])
    # TODO: figure out the right params to fill in
    _, value, _, _, _, _ = qp.solve_qp(-H, 0, np.hstack(-A, -A), np.stack(-b, -b), 0)
    R = -value
    return R

def project_x(x_current, grad, D, A, b):
    """

    Parameters
    ----------
    x_current: x_t
    grad: gradient at current x_t
    D: D_t
    A: the matrix in constraints
    b: the value vector in constrains

    Returns
    -------
    x: the solution to this optimization problem

    """
    a = x_current.T@D - grad
    x, _, _, _, _, _ = qp.solve_qp(D, a.T, -A, -b, 0)
    return x


def ADAGRAD_plus (model, max_iterations=1e4, epsilon=1e-5,
                   x0=None, A=None, b=None):
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
        y_current = x0
    else:
        y_current = np.random.normal(loc=0, scale=1, size=d)

    # make sure the initial  iterate is feasible
    x_current = projector(y_current, A, b)

    # initialization of D0 and R
    D_current = np.identity(len(x_current))
    R = find_R(x_current, A, b)

    # keep track of history
    x_history = []
    x_return_lst = []

    for k in range(int(max_iterations)):

        x_history.append(x_current)

        # next x
        x_next = project_x(x_current, grad_F, D_current, A, b)

        # next D
        D_next = np.identity(len(x_current))
        for i in D_next.shape[0]:
            D_next[i,i] = D_current[i,i] * np.sqrt(1 + np.square(x_next[i] - x_current[i])/R**2)

        # returned x
        x_return = sum(x_history)/k
        x_return_lst.append(x_return)

        # relative error stopping condition
        if (k > 0) & (np.linalg.norm(x_return - x_return_lst[k-1]) <= epsilon*np.linalg.norm(x_return)):
            break

        x_current = x_next
        D_current = D_next

    print('ADAGRAD+ finished after ' + str(k) + ' iterations')

    return {'solution': x_return,
            'x_history': x_history,
            'x_return_lst': x_return_lst}
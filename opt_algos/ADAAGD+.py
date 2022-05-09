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

def ADAAGD_plus (model, max_iterations=1e4, epsilon=1e-5):
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

    print('ADAAGD+ finished after ' + str(k) + ' iterations')

    return {'solution': x_current,
            'x_history': x_history}

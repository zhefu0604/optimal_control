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

    #TODO: check the usage of qp.solve
    x = qp.solve_qp(H, y, -A, -b, 0)

    return x

def projected_gradient_descent(model, eta, max_iterations=1e4, epsilon=1e-5,
                     x0=None, A=None, b=None):
    """
    Gradient descent

    Parameters
    ----------
    model: optimization model object
    eta: learning rate
    max_iterations: maximum number of gradient iterations
    epsilon: tolerance for stopping condition
    x0: where to start (otherwise random)
    A: the matrix in constraints
    b: the value vector in constraints

    Output
    ------
    solution: final x value
    x_history: x values from each iteration
    """

    # data from model
    grad_F = model.grad_F
    d = model.d
    # F = model.F

    # initialization
    if x0 is not None:
        x = x0
    else:
        x = np.random.normal(loc=0, scale=1, size=d)
    # make sure the initial  iterate is feasible
    x_current = projector(x, A, b)


    # keep track of history
    x_history = []

    for k in range(int(max_iterations)):

        x_history.append(x_current)

        # gradient update
        y_next = x_current - eta * grad_F(x_current)

        # project the solution to the convex set
        x_next = projector(y_next, A, b)

        # relative error stoping condition
        if np.linalg.norm(x_next - x_current) <= epsilon*np.linalg.norm(x_current):
            #  if np.linalg.norm(beta_next) <= epsilon:
            break

        x_current = x_next

    print('GD finished after ' + str(k) + ' iterations')

    return {'solution': x_current,
            'beta_history': x_history}

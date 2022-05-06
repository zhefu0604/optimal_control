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

def accelerated_gradient_descent(model, eta, max_iterations=1e4, epsilon=1e-5,
                                 x0=None, A=None, b=None):
    """
    Nesterov's accelerated gradient descent

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
        y_current = x0
    else:
        y_current = np.random.normal(loc=0, scale=1, size=d)

    # make sure the initial  iterate is feasible
    x_current = projector(y_current, A, b)

    z_current = x_current
    t_current = 1.0

    # history
    x_history = []

    for k in range(int(max_iterations)):
        # history
        x_history.append(x_current)

        # gradient update
        t_next = .5*(1 + np.sqrt(1 + 4*t_current**2))
        y_next = z_current - eta * grad_F(z_current)
        z_next = y_next + (t_current - 1.0)/(t_next)*(y_next - x_current)

        # projection
        x_next = projector(z_next, A, b)

        # relative error stoping condition
        if np.linalg.norm(x_next - x_current) <= epsilon*np.linalg.norm(x_current):
            break

        # restarting strategies
        # TODO: to ensure that after the projection, this if statement still valid.
        if np.dot(z_current - x_next, x_next - x_current) > 0:
            z_next = x_next
            t_next = 1

        x_current = x_next
        z_current = z_next
        t_current = t_next

    print('accelerated GD finished after ' + str(k) + ' iterations')

    return {'solution': x_current,
            'beta_history': x_history}

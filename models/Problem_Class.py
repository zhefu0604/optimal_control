from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import numpy as np


class OptimalControl(object):
    def __init__(self, time, x0, v0, xl, vl):
        # Fuel model parameters
        self.C0 = 0.245104874164936
        self.C1 = 0.00389174029574479
        self.C2 = 0.0
        self.C3 = 3.30050221372029e-05
        self.p0 = 0.014702037699570
        self.p1 = 0.051202817419139
        self.p2 = 0.001873784068826
        self.q0 = 0.0
        self.q1 = 0.018538661491842
        self.P0 = 0
        self.Q0 = 0
        self.beta0 = 0.194462963040005 

        self.time = time
        self.d = len(time)
        self.del_t = self.time[-1]/self.d
        self.x0 = x0
        self.v0 = v0
        self.xl = xl
        self.vl = vl



    def F(self, U):
        X, V = self.system_solve(U)
        E = self.fuel_model(V, U)
        z = self.del_t * np.sum(E)
        return z


    def grad_F(self, U): 
        X, V = self.system_solve(U)
        sol = solve_ivp(self.f_adjoint, [self.time[-1], self.time[0]], [0, 0], t_eval= np.flip(self.time)) ## todo: check if we need to flip time 
        PQ = np.flip(sol.y)
        P = PQ[0, :]
        Q = PQ[1, :]
        V = self.V(self.time)
        dz = -1 * Q + (self.p0 + self.p1 * V + self.p2 * V**2) + (U > 0) * (2*self.q0*U + 2*self.q1*V*U)
        return dz


    def system_solve(self, U):
        self.U = interp1d(self.time, U, kind='linear', assume_sorted=True)
        T = self.del_t * np.ones([self.d, self.d])
        T[0, :] = 0
        T = np.tril(T)
        TU = np.matmul(T, U)
        V = self.v0 * np.ones(self.d) + TU;
        X = self.x0 * np.ones(self.d) + self.v0 * self.time + np.matmul(T, TU);
        self.X = interp1d(self.time, X, kind='linear', assume_sorted=True)
        self.V = interp1d(self.time, V, kind='linear', assume_sorted=True)

        return X, V 


    def f_adjoint(self, t, PQ): 
        P = PQ[0]
        P_dot = 0
        Q_dot = (self.C1 + 2 * self.C2 * self.V(t) + 3 * self.C3*self.V(t)**2 + (self.p1 + 2*self.p2*self.V(t)) * self.U(t) + (self.U(t) > 0) * self.q1 * self.U(t)**2) - P
        
        return np.array([P_dot, Q_dot])



    def fuel_model(self, V, U):
        V = np.maximum(V, 0);   
        fc = self.C0 + self.C1*V + self.C2*V**2 + self.C3*V**3 + self.p0*U + self.p1*U*V + self.p2*U*V**2 + self.q0*np.maximum(U,0)**2 + self.q1*V*np.maximum(U,0)**2

        return np.maximum(fc, self.beta0)

    def initialize(self, mode = "smooth"): 
        if mode == "smooth": 
            U = np.diff(self.vl) / self.del_t
            # U = np.convolve(np.ones(200), np.ones(50)/50, mode=m));
        return np.append(U, U[-1])

    def constraints(self, eps = 5, gamma = 120, u_max = 5, u_min = -5): ## Au <= b
        T = self.del_t * np.ones([self.d, self.d])
        T[0, :] = 0
        T = np.tril(T)
        TT = T @ T
        A = np.concatenate((TT, -TT,  -T, np.identity(self.d), -np.identity(self.d)), axis=0)
        b = np.concatenate((self.xl - self.x0 - self.v0 * self.time - eps, 
            gamma - self.xl + self.x0 + self.v0 * self.time, 
            self.v0 * np.ones(self.d), 
            u_max * np.ones(self.d), 
            -u_min * np.ones(self.d)), axis=0)
        return A, b


class OptimalControlConvex(object):
    def __init__(self, time, x0, v0, xl, vl):
        # Fuel model parameters
        self.C0 = 0.245104874164936
        self.C1 = 0.00389174029574479
        self.C2 = 0.0
        self.C3 = 3.30050221372029e-05
        self.p0 = 0.014702037699570
        self.p1 = 0.051202817419139
        self.p2 = 0.001873784068826
        self.q0 = 0.0
        self.q1 = 0.018538661491842
        self.P0 = 0
        self.Q0 = 0
        self.beta0 = 0.194462963040005 

        self.time = time
        self.d = len(time)
        self.del_t = self.time[-1]/self.d
        self.x0 = x0
        self.v0 = v0
        self.xl = xl
        self.vl = vl



    def F(self, U):
        X, V = self.system_solve(U)
        E = V**2 + U**2
        z = self.del_t * np.sum(E)
        return z


    def grad_F(self, U): 
        X, V = self.system_solve(U)
        sol = solve_ivp(self.f_adjoint, [self.time[-1], self.time[0]], [0, 0], t_eval= np.flip(self.time)) ## todo: check if we need to flip time 
        PQ = np.flip(sol.y)
        P = PQ[0, :]
        Q = PQ[1, :]
        dz = -1 * Q + 2 * U**2
        return dz


    def system_solve(self, U):
        self.U = interp1d(self.time, U, kind='linear', assume_sorted=True)
        T = self.del_t * np.ones([self.d, self.d])
        T[0, :] = 0
        T = np.tril(T)
        TU = np.matmul(T, U)
        V = self.v0 * np.ones(self.d) + TU;
        X = self.x0 * np.ones(self.d) + self.v0 * self.time + np.matmul(T, TU);
        self.X = interp1d(self.time, X, kind='linear', assume_sorted=True)
        self.V = interp1d(self.time, V, kind='linear', assume_sorted=True)

        return X, V 


    def f_adjoint(self, t, PQ): 
        P = PQ[0]
        P_dot = 0
        Q_dot = 2 * self.V(t) - P
        
        return np.array([P_dot, Q_dot])



    def fuel_model(self, V, U):
        V = np.maximum(V, 0);   
        fc = self.C0 + self.C1*V + self.C2*V**2 + self.C3*V**3 + self.p0*U + self.p1*U*V + self.p2*U*V**2 + self.q0*np.maximum(U,0)**2 + self.q1*np.maximum(U,0)**2*V

        return np.maximum(fc, self.beta0)

    def initialize(self, mode = "smooth"): 
        if mode == "smooth": 
            U = np.diff(self.vl) / self.del_t
            # U = np.convolve(np.ones(200), np.ones(50)/50, mode=m));
        return np.append(U, U[-1])

    def constraints(self, eps = 5, gamma = 120): ## Au <= b
        T = self.del_t * np.ones([self.d, self.d])
        T[0, :] = 0
        T = np.tril(T)
        TT = T @ T
        A = np.concatenate((TT, -TT,  -T), axis=0)
        b = np.concatenate((self.xl - self.x0 - self.v0 * self.time - eps, 
            gamma - self.xl + self.x0 + self.v0 * self.time, 
            self.v0 * np.ones(self.d)), axis=0)
        return A, b



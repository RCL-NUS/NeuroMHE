"""
This file includes 3 classes that define the controller, MHE and DMHE respectively
--------------------------------------------------------------------------------------
Wang Bingheng, at Control and Simulation Lab, ECE Dept. NUS, Singapore
First version: 19 Dec. 2020
Second version: 31 Aug. 2021
Third version: 10 May 2022
Should you have any question, please feel free to contact the author via:
wangbingheng@u.nus.edu
"""

from curses.ascii import ctrl
from casadi import *
from numpy import linalg as LA
import numpy as np
import math

class Controller:
    """
    Geometric flight controller on SE(3) [1]
    [1] Lee, T., Leok, M. and McClamroch, N.H., 2010, December. 
        Geometric tracking control of a quadrotor UAV on SE (3). 
        In 49th IEEE conference on decision and control (CDC) (pp. 5420-5425). IEEE.
    """
    def __init__(self, uav_para, ctrl_gain, x):
        # Quadrotor's inertial parameters (mass, rotational inertia)
        self.m    = uav_para[0]
        self.J    = np.diag([uav_para[1], uav_para[2], uav_para[3]])
        # Control gain variables
        self.kp   = np.diag([ctrl_gain[0], ctrl_gain[1], ctrl_gain[2]])
        self.kv   = np.diag([ctrl_gain[3], ctrl_gain[4], ctrl_gain[5]])
        self.kr   = np.diag([ctrl_gain[6], ctrl_gain[7], ctrl_gain[8]])
        self.kw   = np.diag([ctrl_gain[9], ctrl_gain[10], ctrl_gain[11]])
        # Z direction vector free of coordinate
        self.z    = np.array([[0, 0, 1]]).T
        # Gravitational acceleration in Singapore
        self.g    = 9.78      
        self.ref  = SX.sym('ref', x.numel(), 1)

    def skew_sym(self, v):
        v_cross = np.array([
            [0, -v[2, 0], v[1, 0]],
            [v[2, 0], 0, -v[0, 0]],
            [-v[1, 0], v[0, 0], 0]]
        )
        return v_cross

    def vee_map(self, v):
        vect = np.array([[v[2, 1], v[0, 2], v[1, 0]]]).T
        return vect

    def trace(self, v):
        v_trace = v[0, 0] + v[1, 1] + v[2, 2]
        return v_trace
    
    # Rotation matrix in MRP from inertial frame to body frame
    def R_mrp(self,r):
        r_s = self.skew_sym(r)
        den = (1 + mtimes(transpose(r), r))**2
        num = 1 - mtimes(transpose(r), r)
        R   = np.identity(3) - 4*num/den*r_s + 8/den*mtimes(r_s, r_s)
        return R
    
    def geometric_ctrl(self,x,ref_p,ref_v,ref_a,ref_j,ref_s,b1_c,df_mh, dtau_mh):
        # get position, velocity, and MRP from the current state
        p  = np.array([[x[0,0], x[1,0], x[2,0]]]).T
        v  = np.array([[x[3,0], x[4,0], x[5,0]]]).T
        Rb = np.array([
            [x[6, 0], x[7, 0], x[8, 0]],
            [x[9, 0], x[10, 0], x[11, 0]],
            [x[12, 0], x[13, 0], x[14, 0]]]
        )
        w  = np.array([[x[15,0], x[16,0], x[17,0]]]).T
        # trajectory tracking errors
        ep = p - ref_p
        ev = v - ref_v
        # desired force in inertial frame
        Fd = -np.matmul(self.kp, ep) - np.matmul(self.kv, ev) + self.m*self.g*self.z + self.m*ref_a - df_mh

        # desired total thruster force fd
        fd = np.inner(Fd.T, np.transpose(np.matmul(Rb, self.z)))
        # construct the desired rotation matrix (from body frame to inertial frame)
        b3d = Fd/LA.norm(Fd)
        b2d = np.matmul(self.skew_sym(b3d), b1_c)/LA.norm(np.matmul(self.skew_sym(b3d), b1_c))
        b1d = np.matmul(self.skew_sym(b2d), b3d)
        Rbd = np.hstack((b1d, b2d, b3d))
        # compute the desired angular velocity
        den = LA.norm(ref_a + self.g*self.z)
        hw  = (ref_j - np.inner(b3d.T, ref_j.T)*b3d)/den
        wd  = np.reshape(np.vstack((-np.inner(hw.T, b2d.T), np.inner(hw.T, b1d.T), np.zeros(1))),(3,1)) # since the desired yaw angle is constant
        num = ref_s - (np.inner(np.transpose(np.matmul(self.skew_sym(wd), b3d)), ref_a.T) \
            + np.inner(b3d.T, ref_s.T))*b3d + np.inner(b3d.T, ref_j.T)*np.matmul(self.skew_sym(w), b3d)
        hdw = num/den - np.matmul(self.skew_sym(w), b3d) - np.matmul(self.skew_sym(w), np.matmul(self.skew_sym(w), b3d))
        dwd = np.reshape(np.vstack((-np.inner(hdw.T, b2d.T), np.inner(hdw.T, b1d.T), 0)),(3,1))
        # attitude tracking errors
        en  = np.matmul(np.transpose(Rbd), Rb) - np.matmul(np.transpose(Rb), Rbd)
        er  = self.vee_map(en)/2
        ew  = w - np.matmul(np.transpose(Rb), np.matmul(Rbd, wd))
        # desired control torque
        tau = - dtau_mh -np.matmul(self.kr, er) - np.matmul(self.kw, ew) \
            + np.matmul(self.skew_sym(w), np.matmul(self.J, w)) \
            - np.matmul(self.J, (np.matmul(np.matmul(self.skew_sym(w), np.transpose(Rb)), np.matmul(Rbd, wd)) \
                - np.matmul(np.transpose(Rb), np.matmul(Rbd, dwd))))
        # control input
        u   = np.vstack((fd, tau))
        R_B_dh = np.array([[Rbd[0,0], Rbd[0,1], Rbd[0,2], Rbd[1,0], Rbd[1,1], Rbd[1,2], Rbd[2,0], Rbd[2,1], Rbd[2,2]]]).T
        return u, R_B_dh, wd

class MHE:
    def __init__(self, horizon, dt_sample, r11):
        self.N = horizon
        self.DT = dt_sample
        self.r11 = r11

    def SetStateVariable(self, xa):
        self.state = xa
        self.n_state = xa.numel()

    def SetOutputVariable(self):
        assert hasattr(self, 'state'), "Define the state variable first!"
        self.output = vertcat(self.state[0,0], self.state[1,0], self.state[2,0],
                              self.state[3,0], self.state[4,0], self.state[5,0],
                              self.state[9,0], self.state[10,0], self.state[11,0],
                              self.state[12,0], self.state[13,0], self.state[14,0],
                              self.state[15,0], self.state[16,0], self.state[17,0],
                              self.state[18,0], self.state[19,0], self.state[20,0])
        self.y_fn   = Function('y',[self.state], [self.output], ['x0'], ['yf'])
        self.n_output = self.output.numel()

    def SetControlVariable(self, u):
        self.ctrl = u
        self.n_ctrl = u.numel()

    # def SetMRPVariable(self, r):
    #     self.r = r

    def SetNoiseVariable(self, eta):
        self.noise = eta
        self.n_noise = eta.numel()

    def SetModelDyn(self, dymh):
        assert hasattr(self, 'state'), "Define the state variable first!"
        assert hasattr(self, 'ctrl'), "Define the control variable first!"
        assert hasattr(self, 'noise'), "Define the noise variable first!"
        # discrete-time dynamic model based on 4th-order Runge-Kutta method
        self.ModelDyn = self.state + self.DT*dymh
        self.MDyn_fn  = Function('MDyn', [self.state, self.ctrl, self.noise], [self.ModelDyn],
                                 ['s', 'c', 'n'], ['MDynf'])

    def SetArrivalCost(self, m):
        assert hasattr(self, 'state'), "Define the state variable first!"
        self.P0        = diag(self.weight_para[0, 0:24])
        # Define filter priori
        error_a        = self.state - m
        self.cost_a    = 1/2 * mtimes(mtimes(transpose(error_a), self.P0), error_a)
        self.cost_a_fn = Function('cost_a', [self.state, self.weight_para], [self.cost_a], ['s','tp'], ['cost_af'])

    def SetCostDyn(self):
        assert hasattr(self, 'state'), "Define the state variable first!"
        assert hasattr(self, 'output'), "Define the output variable first!"
        assert hasattr(self, 'noise'), "Define the noise variable first!"
        # Tunable parameters
        self.weight_para = SX.sym('t_para', 1, 49)
        self.horizon1    = SX.sym('h1') # horizon - 1
        self.horizon2    = self.horizon1 - 1 # horizon - 2
        self.index       = SX.sym('ki')
        self.gamma_r     = self.weight_para[0, 24]
        self.gamma_q     = self.weight_para[0, 42] 
        r                = horzcat(self.r11[0, 0], self.weight_para[0, 25:42]) # fix the fisrt entry and tune the remaining entries
        R_t              = diag(r) # make sure the weight matrix is positive semidefinite
        self.R           = R_t*self.gamma_r**(self.horizon1-self.index)
        self.R_fn        = Function('R_fn', [self.weight_para, self.horizon1, self.index], \
                            [self.R], ['tp','h1', 'ind'], ['R_fnf'])
        Q_t1             = diag(self.weight_para[0, 43:49])
        self.Q           = Q_t1*self.gamma_q**(self.horizon2-self.index)
        # Measurement variable
        self.measurement = SX.sym('y', self.n_output, 1)

        # Discrete dynamics of the running cost (time-derivative of the running cost) based on Euler-method
        error_running    = self.measurement -self.output
        self.dJ_running  = 1/2*(mtimes(mtimes(error_running.T, self.R), error_running) +
                               mtimes(mtimes(self.noise.T, self.Q), self.noise))
        self.dJ_fn       = Function('dJ_running', [self.state, self.measurement, self.noise, self.weight_para, self.horizon1, self.index],
                              [self.dJ_running], ['s', 'm', 'n', 'tp', 'h1', 'ind'], ['dJrunf'])
        self.dJ_T        = 1/2*mtimes(mtimes(error_running.T, self.R), error_running)
        self.dJ_T_fn     = Function('dJ_T', [self.state, self.measurement, self.weight_para, self.horizon1, self.index], [self.dJ_T],
                                ['s', 'm', 'tp', 'h1', 'ind'], ['dJ_Tf'])

    def MHEsolver(self, Y, m, xmhe_traj, ctrl, weight_para, time, print_level=0):
        assert hasattr(self, 'state'), "Define the state variable first!"
        assert hasattr(self, 'noise'), "Define the noise variable first!"
        assert hasattr(self, 'MDyn_fn'), "Define the model dynamics function first!"
        assert hasattr(self, 'dJ_fn'), "Define the cost dynamics function first!"
        # arrival cost setting
        self.SetArrivalCost(m) # m: filter priori which is an MHE estimate at t-N
        self.diffKKT_general()
        """
        Formulate MHE as a nonlinear programming problem solved by CasADi nlpsol() function
        """
        # Start with an empty NLP
        w   = [] # optimal trajectory list
        w0  = [] # initial guess of optimal trajectory
        lbw = [] # lower boundary of optimal variables
        ubw = [] # upper boundary of optimal variables
        g   = [] # equality or inequality constraints
        lbg = [] # lower boundary of constraints
        ubg = [] # upper boundary of constraints

        # Initial state for the arrival cost
        Xk  = SX.sym('X0', self.n_state, 1)
        w  += [Xk]
        X_hatmh = []
        for i in range(len(m)):
            X_hatmh += [m[i,0]]
        w0 += X_hatmh
        lbw+= self.n_state*[-1e20]
        ubw+= self.n_state*[1e20]
        # Formulate the NLP
        # time_mhe = self.N*self.DT
        if time < self.N:
            # Full-information estimator
            self.horizon = time + 1
        else:
            # Moving horizon estimation
            self.horizon = self.N + 1 # note that we start from t-N, so there are N+1 data points

        J = self.cost_a_fn(s=Xk, tp=weight_para)['cost_af']

        for k in range(self.horizon-1):
            # New NLP variables for the process noise
            Nk   = SX.sym('N_' + str(k), self.n_noise, 1)
            w   += [Nk]
            lbw += self.n_noise*[-1e20]
            ubw += self.n_noise*[1e20]
            w0  += self.n_noise*[0] # because of zero-mean noise

            # Integrate the cost function till the end of horizon
            J    += self.dJ_fn(s=Xk, m=Y[len(Y)-self.horizon+k], n=Nk, tp=weight_para, h1=self.horizon-1, ind=k)['dJrunf']
            Xnext = self.MDyn_fn(s=Xk, c=ctrl[len(ctrl)-self.horizon+1+k], n=Nk)['MDynf']
            # Next state based on the discrete model dynamics and current state
            Xk    = SX.sym('X_' + str(k + 1), self.n_state, 1)
            w    += [Xk]
            lbw  += self.n_state*[-1e20]
            ubw  += self.n_state*[1e20]
            X_guess = []
            for ix in range(self.n_state):
                X_guess += [xmhe_traj[k, ix]]
            w0 += X_guess
            # Add equality constraint
            g    += [Xnext - Xk]
            lbg  += self.n_state*[0]
            ubg  += self.n_state*[0]

        # Add the final cost
        J += self.dJ_T_fn(s=Xk, m=Y[-1], tp=weight_para, h1=self.horizon-1, ind=self.horizon-1)['dJ_Tf']

        # Create an NLP solver
        opts = {'ipopt.print_level': print_level, 'ipopt.sb': 'yes', 'print_time': print_level}
        prob = {'f': J, 'x': vertcat(*w), 'g': vertcat(*g)}
        solver = nlpsol('solver', 'ipopt', prob, opts)

        # Solve the NLP
        sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
        w_opt = sol['x'].full().flatten()

        # Take the optimal noise and state
        sol_traj1 = np.concatenate((w_opt, self.n_noise * [0])) # since the dimension of xk is larger than wk by 1, we need to add one more dimension to wk
        sol_traj = np.reshape(sol_traj1, (-1, self.n_state + self.n_noise))
        state_traj_opt = sol_traj[:, 0:self.n_state]
        noise_traj_opt = np.delete(sol_traj[:, self.n_state:], -1, 0)

        # compute the co-states based on the noise_traj_opt
        costate_traj_opt = np.zeros((self.horizon, self.n_state))

        for ico in range(self.horizon - 1, 0, -1):
            curr_s      = state_traj_opt[ico, :]
            curr_n      = noise_traj_opt[ico-1,:]
            curr_m      = Y[len(Y) - self.horizon + ico]
            curr_c      = ctrl[len(ctrl) - self.horizon + ico-1]  # the index of control should be less than that of measurement by 1
            lembda_curr = np.reshape(costate_traj_opt[ico, :], (24,1))
            mat_F       = self.F_fn(x0=curr_s, u0=curr_c, n0=curr_n)['Ff'].full()
            mat_H       = self.H_fn(x0=curr_s)['Hf'].full()
            R_curr      = self.R_fn(tp=weight_para, h1=self.horizon - 1, ind=ico)['R_fnf'].full()
            y_curr      = self.y_fn(x0=curr_s)['yf'].full()
            lembda_pre  = np.matmul(np.transpose(mat_F), lembda_curr) + np.matmul(np.matmul(np.transpose(mat_H), R_curr), (curr_m - y_curr))
            costate_traj_opt[(ico - 1):ico, :] = np.transpose(lembda_pre)

        # Output
        opt_sol = {"state_traj_opt": state_traj_opt,
                   "noise_traj_opt": noise_traj_opt,
                   "costate_traj_opt": costate_traj_opt}
        return opt_sol

    def diffKKT_general(self):
        assert hasattr(self, 'state'), "Define the state variable first!"
        assert hasattr(self, 'output'), "Define the output variable first!"
        assert hasattr(self, 'noise'), "Define the noise variable first!"
        assert hasattr(self, 'MDyn_fn'), "Define the model dynamics function first!"
        assert hasattr(self, 'dJ_fn'), "Define the cost dynamics function first!"
        # Define co-state variables
        self.costate      = SX.sym('lambda', self.n_state, 1) # lambda at k
        self.cos_pre      = SX.sym('lampre', self.n_state, 1) # lambda at k-1

        # Differentiate the dynamics to get the system Jacobian
        self.F            = jacobian(self.ModelDyn, self.state)
        self.F_fn         = Function('F',[self.state, self.ctrl, self.noise], [self.F], ['x0', 'u0', 'n0'], ['Ff']) 
        self.G            = jacobian(self.ModelDyn, self.noise)
        self.G_fn         = Function('G',[self.state, self.ctrl, self.noise], [self.G], ['x0', 'u0', 'n0'], ['Gf'])
        self.H            = jacobian(self.output, self.state)
        self.H_fn         = Function('H',[self.state], [self.H], ['x0'], ['Hf'])

        # Definition of Lagrangian
        self.Lbar         = self.dJ_running - mtimes(transpose(self.costate), self.ModelDyn) \
            + mtimes(transpose(self.cos_pre), self.state) # k=t-N,...,t-1
        self.L0           = self.cost_a + self.Lbar
        self.LbarT        = self.dJ_T # k=t

        # First-order derivative of arrival Lagrangian, k=t-N
        self.dL0x         = jacobian(self.L0, self.state)

        # First-order derivative of path Lagrangian
        self.dLbarx       = jacobian(self.Lbar, self.state) # k=t-N+1,...,t-1
        self.dLbare       = jacobian(self.Lbar, self.noise) # k=t-N,...t-1

        # First-order derivative of final Lagrangian, k=t
        self.dLbarTx      = jacobian(self.LbarT, self.state)

        # Second-order derivative of arrival Lagrangian, k=t-N
        self.ddL0xw       = jacobian(self.dL0x, self.weight_para)
        self.ddL0xw_fn    = Function('ddL0xw', [self.state, self.costate, self.ctrl, self.noise, self.measurement, self.weight_para, self.horizon1, self.index], [self.ddL0xw], ['x0', 'c0', 'u0', 'n0', 'm0', 'tp', 'h1', 'ind'], ['ddL0xwf'])

        # Second-order derivative of path Lagrangian
        self.ddLbarxx     = jacobian(self.dLbarx, self.state) # k=t-N,...,t-1
        self.ddLbarxx_fn  = Function('ddLxx', [self.state, self.costate, self.ctrl, self.noise, self.measurement, self.weight_para, self.horizon1, self.index], [self.ddLbarxx], ['x0', 'c0', 'u0', 'n0', 'm0', 'tp', 'h1', 'ind'], ['ddLxxf'])
        self.ddLbarxw     = jacobian(self.dLbarx, self.weight_para) # k=t-N+1,...,t-1
        self.ddLbarxw_fn  = Function('ddLxw', [self.state, self.costate, self.ctrl, self.noise, self.measurement, self.weight_para, self.horizon1, self.index], [self.ddLbarxw], ['x0', 'c0', 'u0', 'n0', 'm0', 'tp', 'h1', 'ind'], ['ddLxwf'])
        self.ddLbarxe     = jacobian(self.dLbarx, self.noise) # k=t-N,...,t-1
        self.ddLbarxe_fn  = Function('ddLxe', [self.state, self.costate, self.ctrl, self.noise, self.weight_para, self.horizon1, self.index], [self.ddLbarxe], ['x0', 'c0', 'u0', 'n0', 'tp', 'h1', 'ind'], ['ddLxef'])
        self.ddLbaree     = jacobian(self.dLbare, self.noise) # k=t-N,...,t-1
        self.ddLbaree_fn  = Function('ddLee', [self.state, self.costate, self.ctrl, self.noise, self.weight_para, self.horizon1, self.index], [self.ddLbaree], ['x0', 'c0', 'u0', 'n0', 'tp', 'h1', 'ind'], ['ddLeef'])
        self.ddLbarew     = jacobian(self.dLbare, self.weight_para) # k=t-N,...,t-1
        self.ddLbarew_fn  = Function('ddLew', [self.state, self.costate, self.ctrl, self.noise, self.weight_para, self.horizon1, self.index], [self.ddLbarew], ['x0', 'c0', 'u0', 'n0', 'tp', 'h1', 'ind'], ['ddLewf'])
        
        # Second-order derivative of final Lagrangian, k=t
        self.ddLbarTxw    = jacobian(self.dLbarTx, self.weight_para)
        self.ddLbarTxw_fn = Function('ddLTxw', [self.state, self.costate, self.ctrl, self.noise, self.weight_para, self.horizon1, self.index], [self.ddL0xw], ['x0', 'c0', 'u0', 'n0', 'tp', 'h1', 'ind'], ['ddLTxwf'])

    
    def GetAuxSys_general(self, state_traj_opt, costate_traj_opt, noise_traj_opt, weight_para, Y, ctrl):
        # statement = [hasattr(self, 'A_fn'), hasattr(self, 'D_fn'), hasattr(self, 'E_fn'), hasattr(self, 'F0_fn')]
        horizon = np.size(state_traj_opt, 0)
        self.diffKKT_general()

        # Initialize the coefficient matrices of the auxiliary MHE system:
        matF, matG, matH = [], [], []
        matddLxx, matddLxe, matddLxw, matddLee, matddLew = [], [], [], [], []

        # Solve the above coefficient matrices
        for k in range(horizon-1):
            curr_s    = state_traj_opt[k, :] # current state
            curr_cs   = costate_traj_opt[k, :] # current costate, length = horizon, but with the last value being 0
            curr_n    = noise_traj_opt[k,:] # current noise
            curr_m    = Y[len(Y) - horizon + k] # current measurement
            curr_c    = ctrl[len(ctrl) - horizon + 1 + k] # current control force
            matF     += [self.F_fn(x0=curr_s, u0=curr_c, n0=curr_n)['Ff'].full()]
            matG     += [self.G_fn(x0=curr_s, u0=curr_c, n0=curr_n)['Gf'].full()]
            matH     += [self.H_fn(x0=curr_s)['Hf'].full()]
            matddLxx += [self.ddLbarxx_fn(x0=curr_s, c0=curr_cs, u0=curr_c, n0=curr_n, m0=curr_m, tp=weight_para, h1=horizon-1, ind=k)['ddLxxf'].full()]
            if k == 0:
                matddLxw += [self.ddL0xw_fn(x0=curr_s, c0=curr_cs, u0=curr_c, n0=curr_n, m0=curr_m, tp=weight_para, h1=horizon-1, ind=k)['ddL0xwf'].full()]
            else:
                matddLxw += [self.ddLbarxw_fn(x0=curr_s, c0=curr_cs, u0=curr_c, n0=curr_n, m0=curr_m, tp=weight_para, h1=horizon-1, ind=k)['ddLxwf'].full()]
            matddLxe += [self.ddLbarxe_fn(x0=curr_s, c0=curr_cs, u0=curr_c, n0=curr_n, tp=weight_para, h1=horizon-1, ind=k)['ddLxef'].full()]
            matddLee += [self.ddLbaree_fn(x0=curr_s, c0=curr_cs, u0=curr_c, n0=curr_n, tp=weight_para, h1=horizon-1, ind=k)['ddLeef'].full()]
            matddLew += [self.ddLbarew_fn(x0=curr_s, c0=curr_cs, u0=curr_c, n0=curr_n, tp=weight_para, h1=horizon-1, ind=k)['ddLewf'].full()]
        curr_s    = state_traj_opt[-1, :]
        curr_cs   = costate_traj_opt[-1, :]
        curr_c    = np.zeros((self.n_ctrl,1)) # the value does not matter as it is not used in the below
        curr_n    = np.zeros((self.n_noise,1)) # the value does not matter as it is not used in the below
        curr_m    = Y[-1]
        matddLxx += [self.ddLbarxx_fn(x0=curr_s, c0=curr_cs, u0=curr_c, n0=curr_n, m0=curr_m, tp=weight_para, h1=horizon-1, ind=horizon-1)['ddLxxf'].full()]
        matddLxw += [self.ddLbarxw_fn(x0=curr_s, c0=curr_cs, u0=curr_c, n0=curr_n, m0=curr_m, tp=weight_para, h1=horizon-1, ind=horizon-1)['ddLxwf'].full()]

        auxSys = {"matF": matF,
                  "matG": matG,
                  "matH": matH,
                  "matddLxx": matddLxx,
                  "matddLxw": matddLxw,
                  "matddLxe": matddLxe,
                  "matddLee": matddLee,
                  "matddLew": matddLew
                 }
        return auxSys


"""
The KF_gradient_solver class solves for the explicit solutions of the gradients of optimal trajectories
w.r.t the tunable parameters 
"""
class KF_gradient_solver:
    def __init__(self, ref, xa, r11):
        self.n_xmhe = xa.numel()
        self.rN11   = r11
        self.ref    = ref
        self.xa     = xa
        self.x      = vertcat(self.xa[0:6,0], self.xa[9:21,0]) 
        self.traj_e = self.x - self.ref
        w_p, w_a    = 1, 0.01
        weight      = np.array([w_p, w_p, w_p, w_p, w_p, w_p, w_a, w_a, w_a, w_a, w_a, w_a, w_a, w_a, w_a, w_a, w_a, w_a])
        self.loss   = mtimes(mtimes(transpose(self.traj_e), np.diag(weight)), self.traj_e)

    def SetPara(self, weight_para, horizon1, index):
        P0          = np.diag(weight_para[0, 0:24])
        gamma_r     = weight_para[0, 24]
        gamma_q     = weight_para[0, 42]
        r           = np.hstack((self.rN11[0,0], weight_para[0, 25:42]))
        r           = np.reshape(r, (1, 18))
        R_t         = np.diag(r[0])
        Q_t1        = np.diag(weight_para[0,43:49])
        R           = R_t*gamma_r**(horizon1-index)
        Q           = Q_t1*gamma_q**(horizon1-1-index)
        self.n_para = np.size(weight_para)
        return P0, R, Q

    def GradientSolver_general(self, M, matF, matG, matddLxx, matddLxw, matddLxe, matddLee, matddLew, weight_para): 
        self.horizon = len(matddLxx)
        P0,R, Q      = self.SetPara(weight_para, self.horizon-1, 0)

        """-------------------------Forward Kalman filter-----------------------------"""
        # Initialize the state and covariance matrix
        X_KF    = self.horizon*[np.zeros((self.n_xmhe, self.n_para))]
        C       = self.horizon*[np.zeros((self.n_xmhe, self.n_xmhe))]
        S       = self.horizon*[np.zeros((self.n_xmhe, self.n_xmhe))]
        T       = self.horizon*[np.zeros((self.n_xmhe, self.n_xmhe))]
        F_bar   = (self.horizon-1)*[np.zeros((self.n_xmhe, self.n_xmhe))]
        if self.horizon == 1: 
            S_k = -matddLxx[0]
            T_k = -matddLxw[0]
        else:
            S_k = np.matmul(np.matmul(matddLxe[0], LA.inv(matddLee[0])), np.transpose(matddLxe[0]))-matddLxx[0]
            T_k = np.matmul(np.matmul(matddLxe[0], LA.inv(matddLee[0])), matddLew[0])-matddLxw[0]
        S[0]    = S_k
        T[0]    = T_k
        P_k     = LA.inv(P0)
        M_bar   = np.matmul(P_k, T_k)+M
        C_k     = np.matmul(LA.inv(np.identity(self.n_xmhe)-np.matmul(P_k, S_k)), P_k)
        C[0]    = C_k
        X_KFk   = np.matmul((np.identity(self.n_xmhe)+np.matmul(C_k, S_k)), M_bar)
        X_KF[0] = X_KFk

        for k in range(self.horizon-1):
            F_bark    = matF[k]-np.matmul(np.matmul(matG[k], LA.inv(matddLee[k])), np.transpose(matddLxe[k]))
            F_bar[k]  = F_bark
            # state predictor
            X_kk1     = np.matmul(F_bark, X_KF[k]) - np.matmul(np.matmul(matG[k], LA.inv(matddLee[k])), matddLew[k])
            # error covariance
            P_k       = np.matmul(np.matmul(F_bark, C_k), np.transpose(F_bark)) + np.matmul(np.matmul(matG[k], LA.inv(matddLee[k])), np.transpose(matG[k]))
            # Kalman gain
            if k < self.horizon-2:
                S_k   = np.matmul(np.matmul(matddLxe[k+1], LA.inv(matddLee[k+1])), np.transpose(matddLxe[k+1]))-matddLxx[k+1]
            else:
                S_k   = -matddLxx[k+1]
            C_k       = np.matmul(LA.inv(np.identity(self.n_xmhe)-np.matmul(P_k, S_k)), P_k)
            C[k+1]    = C_k
            S[k+1]    = S_k
            # state corrector
            if k < self.horizon-2:
                T_k   = np.matmul(np.matmul(matddLxe[k+1], LA.inv(matddLee[k+1])), matddLew[k+1])-matddLxw[k+1]
            else:
                T_k   = -matddLxw[k+1]
            T[k+1]    = T_k
            X_KFk1    = np.matmul((np.identity(self.n_xmhe)+np.matmul(C_k, S_k)), X_kk1) + np.matmul(C_k, T_k)
            X_KF[k+1] = X_KFk1
     
        """-------------------------Backward costate gradient--------------------------"""
        LAMBDA      = self.horizon*[np.zeros((self.n_xmhe, self.n_para))]
        Lambda_last = np.zeros((self.n_xmhe, self.n_para))
        LAMBDA[-1]  = Lambda_last

        for k in range((self.horizon-1), 0, -1):
            if k == self.horizon-1:
                LAMBDA_pre = np.matmul(S[k], X_KF[k]) + T[k]
            else:
                LAMBDA_pre = np.matmul((np.identity(self.n_xmhe)+np.matmul(S[k], C[k])), np.matmul(np.transpose(F_bar[k]), LAMBDA[k])) + np.matmul(S[k], X_KF[k]) + T[k]
            LAMBDA[k-1] = LAMBDA_pre
        
        """-------------------------Forward state gradient-----------------------------"""
        X_opt = self.horizon*[np.zeros((self.n_xmhe, self.n_para))]
        for k in range(self.horizon):
            if k < self.horizon-1:
                X_optk  = X_KF[k] + np.matmul(np.matmul(C[k], np.transpose(F_bar[k])), LAMBDA[k])
            else:
                X_optk  = X_KF[k]
            X_opt[k] = X_optk

        gra_opt = {"state_gra_traj": X_opt,
                   "costate_gra_traj": LAMBDA}
        return gra_opt

    def loss_tracking(self, xpa, ref):
        loss_fn = Function('loss', [self.xa, self.ref], [self.loss], ['xpa0', 'ref0'], ['lossf'])
        loss_track = loss_fn(xpa0=xpa, ref0=ref)['lossf'].full()
        return loss_track

    def ChainRule(self, ref, xmhe_traj, X_opt):
        # Define the gradient of loss w.r.t state
        Ddlds = jacobian(self.loss, self.xa)
        Ddlds_fn = Function('Ddlds', [self.xa, self.ref], [Ddlds], ['xpa0', 'ref0'], ['dldsf'])
        # Initialize the parameter gradient
        dp = np.zeros((1, self.n_para))
        # Initialize the loss
        loss_track = 0
        Kloss = 20
        # Positive coefficient in the loss
        for t in range(self.horizon):
            x_mhe = xmhe_traj[t, :]
            x_mhe = np.reshape(x_mhe, (self.n_xmhe, 1))
            dloss_track = Kloss * self.loss_tracking(x_mhe, ref[len(ref)-self.horizon+t])
            loss_track +=  dloss_track
            dlds =  Kloss * Ddlds_fn(xpa0=x_mhe, ref0=ref[len(ref)-self.horizon+t])['dldsf'].full()
            dxdp = X_opt[t]
            dp  += np.matmul(dlds, dxdp)
           
        return dp, loss_track



































"""
This file includes 2 classes that define the MHE and the Kalman filter-based gradient solver respectively
--------------------------------------------------------------------------------------
Wang Bingheng, at Control and Simulation Lab, ECE Dept. NUS, Singapore
1st version: 31 Aug. 2021
2nd version: 10 May 2022
3rd version: 10 Oct. 2022 after receiving the reviewers' comments
Should you have any question, please feel free to contact the author via:
wangbingheng@u.nus.edu
"""

from casadi import *
from numpy import linalg as LA
import numpy as np

class MHE:
    def __init__(self, horizon, dt_sample, r11):
        self.N = horizon
        self.DT = dt_sample
        self.r11 = r11

    def SetStateVariable(self, xa):
        self.state = xa
        self.n_state = xa.numel()

    def SetOutputVariable(self, y):
        assert hasattr(self, 'state'), "Define the state variable first!"
        self.output = y
        self.y_fn   = Function('y',[self.state], [self.output], ['x0'], ['yf'])
        self.n_output = self.output.numel()

    def SetNoiseVariable(self, w):
        self.noise = w
        self.n_noise = w.numel()
    
    # def SetQuaternion(self,q):
    #     self.q = q
    #     self.n_q = q.numel()

    def SetModelDyn(self, dymh):
        assert hasattr(self, 'state'), "Define the state variable first!"
        assert hasattr(self, 'noise'), "Define the noise variable first!"
        # discrete-time dynamic model based on Euler or 4th-order Runge-Kutta method
        self.ModelDyn = self.state + self.DT*dymh
        self.MDyn_fn  = Function('MDyn', [self.state, self.noise], [self.ModelDyn],
                                 ['s','n'], ['MDynf'])

    def SetArrivalCost(self, x_hat):
        assert hasattr(self, 'state'), "Define the state variable first!"
        self.P0        = diag(self.weight_para[0, 0:12])
        # Define an MHE priori
        error_a        = self.state - x_hat # previous mhe estimate at t-N
        self.cost_a    = 1/2 * mtimes(mtimes(transpose(error_a), self.P0), error_a)
        self.cost_a_fn = Function('cost_a', [self.state, self.weight_para], [self.cost_a], ['s','tp'], ['cost_af'])

    def SetCostDyn(self):
        assert hasattr(self, 'state'), "Define the state variable first!"
        assert hasattr(self, 'output'), "Define the output variable first!"
        assert hasattr(self, 'noise'), "Define the noise variable first!"
        # Tunable parameters
        self.weight_para = SX.sym('t_para', 1, 25) # dimension: P: 12 + R: 5 + forgetting factor1: 1 + Q: 6 + forgetting factor2: 1 = 25 
        self.horizon1    = SX.sym('h1') # horizon - 1
        self.horizon2    = self.horizon1 - 1 # horizon - 2
        self.index       = SX.sym('ki')
        self.gamma_r     = self.weight_para[0, 12]
        self.gamma_q     = self.weight_para[0, 18] 
        r                = horzcat(self.r11[0, 0], self.weight_para[0, 13:18]) # fix the fisrt entry and tune the remaining entries
        R_t              = diag(r) 
        self.R           = R_t*self.gamma_r**(self.horizon1-self.index)
        self.R_fn        = Function('R_fn', [self.weight_para, self.horizon1, self.index], \
                            [self.R], ['tp','h1', 'ind'], ['R_fnf'])
        Q_t1             = diag(self.weight_para[0, 19:25])
        self.Q           = Q_t1*self.gamma_q**(self.horizon2-self.index)
        # Measurement variable
        self.measurement = SX.sym('ym', self.n_output, 1)

        # Discrete dynamics of the running cost (time-derivative of the running cost) 
        estimtate_error  = self.measurement -self.output
        self.dJ_running  = 1/2*(mtimes(mtimes(estimtate_error.T, self.R), estimtate_error) +
                               mtimes(mtimes(self.noise.T, self.Q), self.noise))
        self.dJ_fn       = Function('dJ_running', [self.state, self.measurement, self.noise, self.weight_para, self.horizon1, self.index],
                              [self.dJ_running], ['s', 'm', 'n', 'tp', 'h1', 'ind'], ['dJrunf'])
        # the terminal cost regarding x_N
        self.dJ_T        = 1/2*mtimes(mtimes(estimtate_error.T, self.R), estimtate_error)
        self.dJ_T_fn     = Function('dJ_T', [self.state, self.measurement, self.weight_para, self.horizon1, self.index], [self.dJ_T],
                                ['s', 'm', 'tp', 'h1', 'ind'], ['dJ_Tf'])

    def MHEsolver(self, Y, x_hat, xmhe_traj, noise_traj, weight_para, time):
        assert hasattr(self, 'state'), "Define the state variable first!"
        assert hasattr(self, 'noise'), "Define the noise variable first!"
        assert hasattr(self, 'MDyn_fn'), "Define the model dynamics function first!"
        assert hasattr(self, 'dJ_fn'), "Define the cost dynamics function first!"
        # arrival cost setting
        self.SetArrivalCost(x_hat) # x_hat: MHE estimate at t-N, obtained by the previous MHE
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
        for i in range(len(x_hat)): # convert an array to a list
            X_hatmh += [x_hat[i,0]]
        w0 += X_hatmh
        lbw+= self.n_state*[-1e20] # value less than or equal to -1e19 stands for no lower bound. See IPOPT documentation
        ubw+= self.n_state*[1e20] # value greater than or equal to 1e19 stands for no upper bound
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
            W_guess = []
            if self.horizon <=3:
                W_guess += self.n_noise*[0]
            else:
                if k<self.horizon-3: # initial guess based on the previous MHE solution
                    for iw in range(self.n_noise):
                        W_guess += [noise_traj[k+1,iw]]
                else:
                    for iw in range(self.n_noise):
                        W_guess += [noise_traj[-1,iw]]
            
            w0  += W_guess 
            # Integrate the cost function till the end of horizon
            J    += self.dJ_fn(s=Xk, m=Y[len(Y)-self.horizon+k], n=Nk, tp=weight_para, h1=self.horizon-1, ind=k)['dJrunf']
            Xnext = self.MDyn_fn(s=Xk, n=Nk)['MDynf']
            # Next state based on the discrete model dynamics and current state
            Xk    = SX.sym('X_' + str(k + 1), self.n_state, 1)
            w    += [Xk]
            lbw  += self.n_state*[-1e20]
            ubw  += self.n_state*[1e20]
            X_guess = []
            if k<self.horizon-3:
                for ix in range(self.n_state):
                    X_guess += [xmhe_traj[k+2, ix]]
            else:
                for ix in range(self.n_state):
                    X_guess += [xmhe_traj[-1, ix]]
            
            w0 += X_guess
            # Add equality constraint
            g    += [Xk - Xnext] # pay attention to this order! The order should be the same as that defined in the paper!
            lbg  += self.n_state*[0]
            ubg  += self.n_state*[0]

        # Add the final cost
        J += self.dJ_T_fn(s=Xk, m=Y[-1], tp=weight_para, h1=self.horizon-1, ind=self.horizon-1)['dJ_Tf']

        # Create an NLP solver
        opts = {}
        opts['ipopt.tol'] = 1e-6
        opts['ipopt.print_level'] = 0
        opts['print_time'] = 0
        opts['ipopt.warm_start_init_point']='yes'
        opts['ipopt.max_iter']=1e3
        opts['ipopt.acceptable_tol']=1e-5
        opts['ipopt.mu_strategy']='adaptive'
        # opts = {'ipopt.print_level': print_level, 'ipopt.sb': 'yes', 'print_time': print_level}
        prob = {'f': J, 'x': vertcat(*w), 'g': vertcat(*g)}
        solver = nlpsol('solver', 'ipopt', prob, opts)

        # Solve the NLP
        sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
        w_opt = sol['x'].full().flatten() # convert to a row array (one dimensional)

        # Take the optimal noise, state, and costate
        sol_traj1 = np.concatenate((w_opt, self.n_noise * [0])) # sol_traj1 = [x0,w0,x1,w1,...,xk,wk,...xn-1,wn-1,xn,wn] note that we added a wn
        sol_traj = np.reshape(sol_traj1, (-1, self.n_state + self.n_noise)) # sol_traj = [[x0,w0],[x1,w1],...[xk,wk],...[xn-1,wn-1],[xn,wn]] 
        state_traj_opt = sol_traj[:, 0:self.n_state] # each xk is a row vector
        noise_traj_opt = np.delete(sol_traj[:, self.n_state:], -1, 0) # delete the last one as we have added it to make the dimensions of x and w equal
        
        # Compute the co-states using the KKT conditions
        costate_traj_opt = np.zeros((self.horizon, self.n_state))

        for i in range(self.horizon - 1, 0, -1):
            curr_s      = state_traj_opt[i, :]
            curr_n      = noise_traj_opt[i-1,:] # its index should be i and begin from self.horizon - 2. 
            curr_m      = Y[len(Y) - self.horizon + i]
            lembda_curr = np.reshape(costate_traj_opt[i, :], (self.n_state,1)) # computation of lembda should have been a piece-wise function like the code in the function of 'GradientSolver_general'
            mat_F       = self.F_fn(x0=curr_s, n0=curr_n)['Ff'].full()
            mat_H       = self.H_fn(x0=curr_s)['Hf'].full()
            R_curr      = self.R_fn(tp=weight_para, h1=self.horizon - 1, ind=i)['R_fnf'].full()
            y_curr      = self.y_fn(x0=curr_s)['yf'].full()
            lembda_pre  = np.matmul(np.transpose(mat_F), lembda_curr) + np.matmul(np.matmul(np.transpose(mat_H), R_curr), (curr_m - y_curr))
            costate_traj_opt[(i - 1):i, :] = np.transpose(lembda_pre) # Actually, this kind of costate is NOT used in training. We use 'costate_traj_ipopt' instead.
        
        # Alternatively, we can compute the co-states (Lagrange multipliers) from IPOPT itself. These two co-state trajectories are very similar to each other!
        lam_g = sol['lam_g'].full().flatten() # Lagrange multipilers for bounds on g
        costate_traj_ipopt = np.reshape(lam_g, (-1,self.n_state))
        
        # Output
        opt_sol = {"state_traj_opt": state_traj_opt,
                   "noise_traj_opt": noise_traj_opt,
                   "costate_traj_opt": costate_traj_opt,
                   "costate_ipopt": costate_traj_ipopt}
        return opt_sol

    def diffKKT_general(self):
        assert hasattr(self, 'state'), "Define the state variable first!"
        assert hasattr(self, 'output'), "Define the output variable first!"
        assert hasattr(self, 'noise'), "Define the noise variable first!"
        assert hasattr(self, 'MDyn_fn'), "Define the model dynamics function first!"
        assert hasattr(self, 'dJ_fn'), "Define the cost dynamics function first!"
        # Define co-state variables
        self.costate      = SX.sym('lambda', self.n_state, 1) # lambda at k
        self.cos_pre      = SX.sym('lampre', self.n_state, 1) # lambda at k-1, in fact, it will not appear in all the 2nd-derivative terms

        # Differentiate the dynamics to get the system Jacobian
        self.F            = jacobian(self.ModelDyn, self.state)
        self.F_fn         = Function('F',[self.state, self.noise], [self.F], ['x0','n0'], ['Ff']) 
        self.G            = jacobian(self.ModelDyn, self.noise)
        self.G_fn         = Function('G',[self.state, self.noise], [self.G], ['x0','n0'], ['Gf'])
        self.H            = jacobian(self.output, self.state)
        self.H_fn         = Function('H',[self.state], [self.H], ['x0'], ['Hf'])

        # Definition of Lagrangian
        self.Lbar0        = self.dJ_running - mtimes(transpose(self.costate), self.ModelDyn) # arrival Lagrangian_bar
        self.Lbar_k       = self.dJ_running - mtimes(transpose(self.costate), self.ModelDyn) \
            + mtimes(transpose(self.cos_pre), self.state) # k=t-N+1,...,t-1
        self.L0           = self.cost_a + self.Lbar0 # arrival Lagrangian, k=t-N
        self.LbarT        = self.dJ_T + mtimes(transpose(self.cos_pre), self.state) # terminal Lagrangian, k=t, the cos_pre will disappear in all the second derivatives!

        # First-order derivative of arrival Lagrangian, k=t-N
        self.dL0x         = jacobian(self.L0, self.state) # this is used to calculate ddL0xp
        self.dLbar0x      = jacobian(self.Lbar0, self.state)
        self.dLbar0w      = jacobian(self.Lbar0, self.noise)

        # First-order derivative of path Lagrangian, k=t-N+1,...,t-1
        self.dLbarx       = jacobian(self.Lbar_k, self.state) 
        self.dLbarw       = jacobian(self.Lbar_k, self.noise) 

        # First-order derivative of terminal Lagrangian, k=t
        self.dLbarTx      = jacobian(self.LbarT, self.state)

        # Second-order derivative of arrival Lagrangian, k=t-N
        self.ddL0xp       = jacobian(self.dL0x, self.weight_para)
        self.ddL0xp_fn    = Function('ddL0xp', [self.state, self.costate, self.noise, self.measurement, self.weight_para, self.horizon1, self.index], [self.ddL0xp], ['x0', 'c0', 'n0', 'm0', 'tp', 'h1', 'ind'], ['ddL0xpf'])
        self.ddLbar0xx    = jacobian(self.dLbar0x, self.state)
        self.ddLbar0xx_fn = Function('ddL0xx', [self.state, self.costate, self.noise, self.measurement, self.weight_para, self.horizon1, self.index], [self.ddLbar0xx], ['x0', 'c0', 'n0', 'm0', 'tp', 'h1', 'ind'], ['ddL0xxf'])
        self.ddLbar0xw    = jacobian(self.dLbar0x, self.noise)
        self.ddLbar0xw_fn = Function('ddL0xw', [self.state, self.costate, self.noise, self.measurement, self.weight_para, self.horizon1, self.index], [self.ddLbar0xw], ['x0', 'c0', 'n0', 'm0', 'tp', 'h1', 'ind'], ['ddL0xwf'])
        self.ddLbar0ww    = jacobian(self.dLbar0w, self.noise)
        self.ddLbar0ww_fn = Function('ddL0ww', [self.state, self.costate, self.noise, self.measurement, self.weight_para, self.horizon1, self.index], [self.ddLbar0ww], ['x0', 'c0', 'n0', 'm0', 'tp', 'h1', 'ind'], ['ddL0wwf'])
        self.ddLbar0wp    = jacobian(self.dLbar0w, self.weight_para)
        self.ddLbar0wp_fn = Function('ddL0wp', [self.state, self.costate, self.noise, self.measurement, self.weight_para, self.horizon1, self.index], [self.ddLbar0wp], ['x0', 'c0', 'n0', 'm0', 'tp', 'h1', 'ind'], ['ddL0wpf'])
        # note that when k=t-N, ddL0xx = P + ddLbarxx, for all k, ddLxw = ddLbarxw, ddLww = ddLbarww

        # Second-order derivative of path Lagrangian, k=t-N+1,...,t-1
        self.ddLbarxx     = jacobian(self.dLbarx, self.state) 
        self.ddLbarxx_fn  = Function('ddLxx', [self.state, self.costate, self.noise, self.measurement, self.weight_para, self.horizon1, self.index], [self.ddLbarxx], ['x0', 'c0', 'n0', 'm0', 'tp', 'h1', 'ind'], ['ddLxxf'])
        self.ddLbarxp     = jacobian(self.dLbarx, self.weight_para) 
        self.ddLbarxp_fn  = Function('ddLxp', [self.state, self.costate, self.noise, self.measurement, self.weight_para, self.horizon1, self.index], [self.ddLbarxp], ['x0', 'c0', 'n0', 'm0', 'tp', 'h1', 'ind'], ['ddLxpf'])
        self.ddLbarxw     = jacobian(self.dLbarx, self.noise) 
        self.ddLbarxw_fn  = Function('ddLxw', [self.state, self.costate, self.noise, self.measurement, self.weight_para, self.horizon1, self.index], [self.ddLbarxw], ['x0', 'c0', 'n0', 'm0', 'tp', 'h1', 'ind'], ['ddLxwf'])
        self.ddLbarww     = jacobian(self.dLbarw, self.noise) 
        self.ddLbarww_fn  = Function('ddLww', [self.state, self.costate, self.noise, self.measurement, self.weight_para, self.horizon1, self.index], [self.ddLbarww], ['x0', 'c0', 'n0', 'm0', 'tp', 'h1', 'ind'], ['ddLwwf'])
        self.ddLbarwp     = jacobian(self.dLbarw, self.weight_para) 
        self.ddLbarwp_fn  = Function('ddLwp', [self.state, self.costate, self.noise, self.measurement, self.weight_para, self.horizon1, self.index], [self.ddLbarwp], ['x0', 'c0', 'n0', 'm0', 'tp', 'h1', 'ind'], ['ddLwpf'])
        
        # Second-order derivative of terminal Lagrangian, k=t
        self.ddLbarTxx    = jacobian(self.dLbarTx, self.state)
        self.ddLbarTxx_fn = Function('ddLTxx', [self.state, self.measurement, self.weight_para, self.horizon1, self.index], [self.ddLbarTxx], ['x0', 'm0', 'tp', 'h1', 'ind'], ['ddLTxxf'])
        self.ddLbarTxp    = jacobian(self.dLbarTx, self.weight_para)
        self.ddLbarTxp_fn = Function('ddLTxp', [self.state, self.measurement, self.weight_para, self.horizon1, self.index], [self.ddLbarTxp], ['x0', 'm0', 'tp', 'h1', 'ind'], ['ddLTxpf'])

    
    def GetAuxSys_general(self, state_traj_opt, costate_traj_opt, noise_traj_opt, weight_para, Y):
        # statement = [hasattr(self, 'A_fn'), hasattr(self, 'D_fn'), hasattr(self, 'E_fn'), hasattr(self, 'F0_fn')]
        horizon = np.size(state_traj_opt, 0)
        self.diffKKT_general()

        # Initialize the coefficient matrices of the auxiliary MHE system:
        matF, matG, matH = [], [], []
        matddLxx, matddLxw, matddLxp, matddLww, matddLwp = [], [], [], [], []

        # Solve the above coefficient matrices
        for k in range(horizon-1):
            curr_s    = state_traj_opt[k, :] # current state
            curr_cs   = costate_traj_opt[k, :] # current costate, length = horizon, but with the last value being 0
            curr_n    = noise_traj_opt[k,:] # current noise
            curr_m    = Y[len(Y) - horizon + k] # current measurement
            matF     += [self.F_fn(x0=curr_s, n0=curr_n)['Ff'].full()]
            matG     += [self.G_fn(x0=curr_s, n0=curr_n)['Gf'].full()]
            matH     += [self.H_fn(x0=curr_s)['Hf'].full()]
            if k == 0: # note that P is included only in the arrival cost
                matddLxx += [self.ddLbar0xx_fn(x0=curr_s, c0=curr_cs, n0=curr_n, m0=curr_m, tp=weight_para, h1=horizon-1, ind=k)['ddL0xxf'].full()]
                matddLxp += [self.ddL0xp_fn(x0=curr_s, c0=curr_cs, n0=curr_n, m0=curr_m, tp=weight_para, h1=horizon-1, ind=k)['ddL0xpf'].full()]
                matddLxw += [self.ddLbar0xw_fn(x0=curr_s, c0=curr_cs, n0=curr_n, m0=curr_m, tp=weight_para, h1=horizon-1, ind=k)['ddL0xwf'].full()]
                matddLww += [self.ddLbar0ww_fn(x0=curr_s, c0=curr_cs, n0=curr_n, m0=curr_m, tp=weight_para, h1=horizon-1, ind=k)['ddL0wwf'].full()]
                matddLwp += [self.ddLbar0wp_fn(x0=curr_s, c0=curr_cs, n0=curr_n, m0=curr_m, tp=weight_para, h1=horizon-1, ind=k)['ddL0wpf'].full()]
            else:
                matddLxx += [self.ddLbarxx_fn(x0=curr_s, c0=curr_cs, n0=curr_n, m0=curr_m, tp=weight_para, h1=horizon-1, ind=k)['ddLxxf'].full()]
                matddLxp += [self.ddLbarxp_fn(x0=curr_s, c0=curr_cs, n0=curr_n, m0=curr_m, tp=weight_para, h1=horizon-1, ind=k)['ddLxpf'].full()]
                matddLxw += [self.ddLbarxw_fn(x0=curr_s, c0=curr_cs, n0=curr_n, m0=curr_m, tp=weight_para, h1=horizon-1, ind=k)['ddLxwf'].full()]
                matddLww += [self.ddLbarww_fn(x0=curr_s, c0=curr_cs, n0=curr_n, m0=curr_m, tp=weight_para, h1=horizon-1, ind=k)['ddLwwf'].full()]
                matddLwp += [self.ddLbarwp_fn(x0=curr_s, c0=curr_cs, n0=curr_n, m0=curr_m, tp=weight_para, h1=horizon-1, ind=k)['ddLwpf'].full()]
        curr_s    = state_traj_opt[-1, :]
        curr_m    = Y[-1]
        matddLxx += [self.ddLbarTxx_fn(x0=curr_s, m0=curr_m, tp=weight_para, h1=horizon-1, ind=horizon-1)['ddLTxxf'].full()]
        matddLxp += [self.ddLbarTxp_fn(x0=curr_s, m0=curr_m, tp=weight_para, h1=horizon-1, ind=horizon-1)['ddLTxpf'].full()]

        auxSys = {"matF": matF,
                  "matG": matG,
                  "matH": matH,
                  "matddLxx": matddLxx,
                  "matddLxp": matddLxp,
                  "matddLxw": matddLxw,
                  "matddLww": matddLww,
                  "matddLwp": matddLwp
                 }
        return auxSys
    
    def Observability(self, matF, matH):
        rank = np.zeros(len(matF))
        n = len(matF)
        for i in range(n):
            ok0 = np.matmul(matH[i],LA.matrix_power(matF[i],0))
            O   = ok0.flatten()
            for k in range(1,self.n_state):
                ok= np.matmul(matH[i],LA.matrix_power(matF[i],k))
                o = ok.flatten()
                O = np.concatenate((O,o))
            O = np.reshape(O,(-1,self.n_state))
            rank_i = LA.matrix_rank(O)
            rank[i] = rank_i
        return rank
    
    def Hessian_Lagrange(self, matddLxx, matddLxw, matddLww):
        horizon   = len(matddLxx)
        min_eig_k = np.zeros(horizon)
        min4_eig_k = np.zeros(horizon)
        for k in range(horizon-1):
            H_k = np.vstack((
                np.hstack((matddLxx[k],matddLxw[k])),
                np.hstack((matddLxw[k].T,matddLww[k]))
            ))
            # H_k = np.vstack((
            #     np.hstack((matddLxx[k],np.zeros((19,6)))),
            #     np.hstack((np.zeros((19,6)).T,matddLww[k]))
            # ))
            w, v = LA.eig(H_k)
            min_eig_k[k] = np.min(np.real(w))
            min4_eig_k[k] = np.sort(np.real(w))[5]
            inv_H_k = LA.inv(matddLww[k]) # H_k is not guaranteed to be positive-definite and nonsingular
        w, v = LA.eig(matddLxx[-1])
        # min_eig_k[-1] = np.min(np.real(w))
        min_eig = np.min(min_eig_k)
        # min4_eig_k[-1] = np.sort(np.real(w))[2]
        min4_eig = np.max(min4_eig_k)
        return min_eig, min4_eig


"""
The KF_gradient_solver class solves for the explicit solutions of the gradients of optimal trajectories
w.r.t the tunable parameters 
"""
class KF_gradient_solver:
    def __init__(self, xa, para):
        self.n_xmhe = xa.numel()
        self.n_para = para.numel()
        self.x_t    = SX.sym('x_t',6,1)
        self.xa     = xa
        self.dismhe = vertcat(self.xa[3:6,0], self.xa[9:12,0]) 
        self.est_e  = self.dismhe - self.x_t
        w_f, w_t    = 1, 10 # 1, 10 for n_start + 4000
        weight      = np.array([w_f, w_f, w_f, w_t, w_t, w_t])
        self.loss   = mtimes(mtimes(transpose(self.est_e), np.diag(weight)), self.est_e)
        self.Kloss  = 15 # 1.25 for fast training set
    
    def q_2_rotation(self,q): # from body frame to inertial frame
        q = q/norm_2(q) # normalization
        q0, q1, q2, q3 = q[0,0], q[1,0], q[2,0], q[3,0]
        R = vertcat(
        horzcat( 2 * (q0 ** 2 + q1 ** 2) - 1, 2 * q1 * q2 - 2 * q0 * q3, 2 * q0 * q2 + 2 * q1 * q3),
        horzcat(2 * q0 * q3 + 2 * q1 * q2, 2 * (q0 ** 2 + q2 ** 2) - 1, 2 * q2 * q3 - 2 * q0 * q1),
        horzcat(2 * q1 * q3 - 2 * q0 * q2, 2 * q0 * q1 + 2 * q2 * q3, 2 * (q0 ** 2 + q3 ** 2) - 1)
        )
        return R

    def GradientSolver_general(self, Xhat, matF, matG, matddLxx, matddLxp, matddLxw, matddLww, matddLwp, P0): 
        self.horizon = len(matddLxx)
        """-------------------------Forward Kalman filter-----------------------------"""
        # Initialize the state and covariance matrix
        X_KF    = self.horizon*[np.zeros((self.n_xmhe, self.n_para))]
        C       = self.horizon*[np.zeros((self.n_xmhe, self.n_xmhe))]
        S       = self.horizon*[np.zeros((self.n_xmhe, self.n_xmhe))]
        T       = self.horizon*[np.zeros((self.n_xmhe, self.n_xmhe))]
        F_bar   = (self.horizon-1)*[np.zeros((self.n_xmhe, self.n_xmhe))]
        if self.horizon == 1: 
            S_k = -matddLxx[0]
            T_k = -matddLxp[0]
        else:
            S_k = np.matmul(np.matmul(matddLxw[0], LA.inv(matddLww[0])), np.transpose(matddLxw[0]))-matddLxx[0]
            T_k = np.matmul(np.matmul(matddLxw[0], LA.inv(matddLww[0])), matddLwp[0])-matddLxp[0]
        S[0]    = S_k
        T[0]    = T_k
        P_k     = LA.inv(P0)
        C_k     = np.matmul(LA.inv(np.identity(self.n_xmhe)-np.matmul(P_k, S_k)), P_k)
        C[0]    = C_k
        X_KFk   = np.matmul((np.identity(self.n_xmhe)+np.matmul(C_k, S_k)), Xhat) + np.matmul(C_k,T_k)
        X_KF[0] = X_KFk

        for k in range(self.horizon-1):
            F_bark    = matF[k]-np.matmul(np.matmul(matG[k], LA.inv(matddLww[k])), np.transpose(matddLxw[k]))
            F_bar[k]  = F_bark
            # state predictor
            X_kk1     = np.matmul(F_bar[k], X_KF[k]) - np.matmul(np.matmul(matG[k], LA.inv(matddLww[k])), matddLwp[k]) # X_kk1: X^hat_{k+1|k}
            # error covariance
            P_k1       = np.matmul(np.matmul(F_bar[k], C[k]), np.transpose(F_bar[k])) + np.matmul(np.matmul(matG[k], LA.inv(matddLww[k])), np.transpose(matG[k])) # P_k1: P_{k+1}
            # Kalman gain
            if k < self.horizon-2:
                S_k1   = np.matmul(np.matmul(matddLxw[k+1], LA.inv(matddLww[k+1])), np.transpose(matddLxw[k+1]))-matddLxx[k+1] # S_k1: S_{k+1}
            else:
                S_k1   = -matddLxx[k+1] # matddLxw_{t} does not exist
            S[k+1]    = S_k1
            C_k1      = np.matmul(LA.inv(np.identity(self.n_xmhe)-np.matmul(P_k1, S[k+1])), P_k1)
            C[k+1]    = C_k1
            
            # state corrector
            if k < self.horizon-2:
                T_k1  = np.matmul(np.matmul(matddLxw[k+1], LA.inv(matddLww[k+1])), matddLwp[k+1])-matddLxp[k+1] # T_k1: T_{k+1}
            else:
                T_k1  = -matddLxp[k+1]
            T[k+1]    = T_k1
            X_KFk1    = np.matmul((np.identity(self.n_xmhe)+np.matmul(C[k+1], S[k+1])), X_kk1) + np.matmul(C[k+1], T[k+1])
            X_KF[k+1] = X_KFk1
     
        """-------------------------Backward costate gradient--------------------------"""
        LAMBDA      = self.horizon*[np.zeros((self.n_xmhe, self.n_para))]
        Lambda_last = np.zeros((self.n_xmhe, self.n_para))
        LAMBDA[-1]  = Lambda_last

        for k in range((self.horizon-1), 0, -1):
            if k == self.horizon-1: # the length of F_bar is (horizon - 1)
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

    def loss_tracking(self, xpa, gt):
        loss_fn = Function('loss', [self.xa, self.x_t], [self.loss], ['xpa0', 'gt0'], ['lossf'])
        loss_track = loss_fn(xpa0=xpa, gt0=gt)['lossf'].full()
        return loss_track

    def loss_horizon(self, xmhe_traj, gt, N, time):
        loss_track = 0
        if time < N:
            horizon = time + 1
        else:
            horizon = N + 1
        for k in range(horizon):
            x_mhe = xmhe_traj[k, :]
            dloss_track = self.Kloss * self.loss_tracking(x_mhe, gt[len(gt)-horizon+k])
            loss_track +=  dloss_track
        return loss_track
        
    def ChainRule(self, gt, xmhe_traj, X_opt):
        # Define the gradient of loss w.r.t state
        Ddlds = jacobian(self.loss, self.xa)
        Ddlds_fn = Function('Ddlds', [self.xa, self.x_t], [Ddlds], ['xpa0', 'gt0'], ['dldsf'])
        # Initialize the parameter gradient
        dp = np.zeros((1, self.n_para))
        # Initialize the loss
        loss_track = 0
        
        for t in range(self.horizon):
            x_mhe = xmhe_traj[t, :]
            x_mhe = np.reshape(x_mhe, (self.n_xmhe, 1))
            dloss_track = self.Kloss * self.loss_tracking(x_mhe, gt[len(gt)-self.horizon+t])
            loss_track +=  dloss_track
            dlds =  self.Kloss * Ddlds_fn(xpa0=x_mhe, gt0=gt[len(gt)-self.horizon+t])['dldsf'].full()
            dxdp = X_opt[t]
            dp  += np.matmul(dlds, dxdp)
           
        return dp, loss_track



































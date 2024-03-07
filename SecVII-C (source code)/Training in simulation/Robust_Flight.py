"""
This file includes 3 classes that define the controller, MHE and DMHE respectively
--------------------------------------------------------------------------------------
Wang Bingheng, at Control and Simulation Lab, ECE Dept. NUS, Singapore
1st version: 10 May,2022
2nd version: 10 Oct. 2022 after receiving the reviewers' comments
Should you have any question, please feel free to contact the author via:
wangbingheng@u.nus.edu
"""

from casadi import *
from numpy import linalg as LA
import numpy as np
from scipy import linalg as sLA


class Controller:
    """
    Geometric flight controller on SE(3) [1][2]
    [1] Lee, T., Leok, M. and McClamroch, N.H., 2010. 
        Control of complex maneuvers for a quadrotor UAV using geometric methods on SE (3). 
        arXiv preprint arXiv:1003.2005.
    [2] Lee, T., Leok, M. and McClamroch, N.H., 2010, December. 
        Geometric tracking control of a quadrotor UAV on SE (3). 
        In 49th IEEE conference on decision and control (CDC) (pp. 5420-5425). IEEE.
    L1-Adaptive flight controller [3]
    [3] Wu, Z., Cheng, S., Ackerman, K.A., Gahlawat, A., Lakshmanan, A., Zhao, P. and Hovakimyan, N., 2022, May. 
        L 1 Adaptive Augmentation for Geometric Tracking Control of Quadrotors. 
        In 2022 International Conference on Robotics and Automation (ICRA) (pp. 1329-1336). IEEE.
    """
    def __init__(self, uav_para, ctrl_gain, dt_sample):
        # Quadrotor's inertial parameters (mass, rotational inertia)
        self.m    = uav_para[0]
        self.J    = np.diag([uav_para[1], uav_para[2], uav_para[3]])
        # Control gain variables
        self.kp   = np.diag([ctrl_gain[0,0], ctrl_gain[0,1], ctrl_gain[0,2]])
        self.kv   = np.diag([ctrl_gain[0,3], ctrl_gain[0,4], ctrl_gain[0,5]])
        self.kr   = np.diag([ctrl_gain[0,6], ctrl_gain[0,7], ctrl_gain[0,8]]) # control gain for attitude tracking error
        self.kw   = np.diag([ctrl_gain[0,9], ctrl_gain[0,10], ctrl_gain[0,11]])
        # Unit direction vector free of coordinate
        self.ex   = np.array([[1, 0, 0]]).T
        self.ey   = np.array([[0, 1, 0]]).T
        self.ez   = np.array([[0, 0, 1]]).T
        # Gravitational acceleration in Singapore
        self.g    = 9.78      
        self.dt   = dt_sample

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
    
    def lowpass_filter(self,time_const,curr_i,prev_i):
        alpha       = self.dt/(self.dt+time_const)
        y_filter    = (1-alpha)*prev_i + alpha*curr_i
        return y_filter
    
    def position_ctrl(self,x,ref_p,ref_v,ref_a,df_Imh):
        # Get the system state from the feedback
        p  = np.array([[x[0,0], x[1,0], x[2,0]]]).T
        v  = np.array([[x[3,0], x[4,0], x[5,0]]]).T
        """
        Position controller
        """
        # Trajectory tracking errors
        ep = p - ref_p
        ev = v - ref_v
        # Desired robust control force in inertial frame
        Fd = -np.matmul(self.kp, ep) - np.matmul(self.kv, ev) + self.m*self.g*self.ez + self.m*ref_a -df_Imh
        
        return Fd
    
    def attitude_ctrl(self,x,Fd,v_prev,a_lpf_prev,j_lpf_prev,ref_p,ref_v,ref_a,ref_j,ref_s,b1_d,dtau_mh):
        p  = np.array([[x[0,0], x[1,0], x[2,0]]]).T
        v  = np.array([[x[3,0], x[4,0], x[5,0]]]).T
        Rb = np.array([
            [x[6, 0], x[7, 0], x[8, 0]],
            [x[9, 0], x[10, 0], x[11, 0]],
            [x[12, 0], x[13, 0], x[14, 0]]]
        ) # rotation matrix from body frame to inertial frame
        # Trajectory tracking errors
        ep = p - ref_p
        ev = v - ref_v
        fd = np.inner(Fd.T, np.transpose(np.matmul(Rb, self.ez)))
        """
        Attitude controller
        """
        # Construct the desired rotation matrix (from body frame to inertial frame)
        b3c = Fd/LA.norm(Fd)
        b2c = np.matmul(self.skew_sym(b3c), b1_d)/LA.norm(np.matmul(self.skew_sym(b3c), b1_d))
        b1c = np.matmul(self.skew_sym(b2c), b3c)
        Rbd = np.hstack((b1c, b2c, b3c))
        R_B_dh = np.array([[Rbd[0,0], Rbd[0,1], Rbd[0,2], Rbd[1,0], Rbd[1,1], Rbd[1,2], Rbd[2,0], Rbd[2,1], Rbd[2,2]]]).T
        # Compute the desired angular velocity and angular acceleration，see Appendix F in the 2nd version of [1] for details
        a      = (v-v_prev)/self.dt # acclearation based on 1st-order backward differentiation
        time_const  = 0.025 # used in the low-pass filter for the acceleration
        a_lpf  = self.lowpass_filter(time_const,a,a_lpf_prev)
        j      = (a_lpf-a_lpf_prev)/self.dt # jerk based on 1st-order backward differentiation
        time_const  = 0.05 # used in the low-pass filter for the jerk
        j_lpf  = self.lowpass_filter(time_const,j,j_lpf_prev)
        A      = -Fd
        dA     = np.matmul(self.kp, ev) + np.matmul(self.kv, (a_lpf-ref_a)) - self.m*ref_j
        ddA    = np.matmul(self.kp, (a_lpf-ref_a)) + np.matmul(self.kv, (j_lpf-ref_j)) - self.m*ref_s
        db3c   = -dA/LA.norm(A) + np.inner(A.T,dA.T)*A/(LA.norm(A)**3)
        ddb3c  = -ddA/LA.norm(A) + 2*np.inner(A.T,dA.T)*dA/(LA.norm(A)**3) + (LA.norm(dA)**2+np.inner(A.T,ddA.T))*A/(LA.norm(A)**3) - 3*np.inner(A.T,dA.T)**2*A/(LA.norm(A)**5)
        C      = np.matmul(self.skew_sym(b1_d),b3c)
        dC     = np.matmul(self.skew_sym(b1_d),db3c)
        ddC    = np.matmul(self.skew_sym(b1_d),ddb3c)
        db2c   = -dC/LA.norm(C) + np.inner(C.T,dC.T)*C/(LA.norm(C)**3)
        ddb2c  = -ddC/LA.norm(C) + 2*np.inner(C.T,dC.T)*dC/(LA.norm(C)**3) + (LA.norm(dC)**2+np.inner(C.T,ddC.T))*C/(LA.norm(C)**3) - 3*np.inner(C.T,dC.T)**2*C/(LA.norm(C)**5)
        db1c   = np.matmul(self.skew_sym(db2c),b3c) + np.matmul(self.skew_sym(b2c),db3c)
        ddb1c  = np.matmul(self.skew_sym(ddb2c),b3c) + 2*np.matmul(self.skew_sym(db2c),db3c) + np.matmul(self.skew_sym(b2c),ddb3c)
        dRbd   = np.hstack((db1c, db2c, db3c))
        ddRbd  = np.hstack((ddb1c, ddb2c, ddb3c))
        omegad = self.vee_map(np.matmul(Rbd.T,dRbd))
        # Bound the desired angular rate for stability concern
        if LA.norm(omegad)>=10:
            omegad = omegad/LA.norm(omegad)*10
        domegad= self.vee_map(np.matmul(Rbd.T,ddRbd)-LA.matrix_power(self.skew_sym(omegad),2))
        omega  = np.array([[x[15,0], x[16,0], x[17,0]]]).T
        # attitude tracking errors
        er  = 1/2*self.vee_map(np.matmul(Rbd.T, Rb) - np.matmul(Rb.T, Rbd))
        ew  = omega - np.matmul(Rb.T, np.matmul(Rbd, omegad))
        # desired control torque
        tau = -np.matmul(self.kr, er) - np.matmul(self.kw, ew) \
            + np.matmul(self.skew_sym(omega), np.matmul(self.J, omega)) \
            - np.matmul(self.J, (np.matmul(np.matmul(self.skew_sym(omega), Rb.T), np.matmul(Rbd, omegad)) \
                - np.matmul(Rb.T, np.matmul(Rbd, domegad)))) -dtau_mh
        # control input
        u   = np.vstack((fd,tau))
        
        return u, R_B_dh,omegad,domegad, a_lpf, j_lpf
    
 
    def L1_adaptive_law(self,x,Rb,z_hat):
        # Piecewise-constant adaptation law
        z  = np.array([[x[3,0],x[4,0],x[5,0]]]).T
        B_R  = 1/self.m*np.matmul(Rb,self.ez)
        B_Rp = np.hstack((1/self.m*np.matmul(Rb,self.ex), 1/self.m*np.matmul(Rb,self.ey)))
        B_bar = np.hstack((B_R, B_Rp))
        a_s = np.array([[-1,-1,-1]]) # -10 for 100 Hz
        A_s = np.diag(a_s[0]) # diagonal Hurwitz matrix
        PHI = np.matmul(LA.inv(A_s),(sLA.expm(self.dt*A_s)-np.identity(3)))
        mu  = np.matmul(sLA.expm(self.dt*A_s),(z_hat-z))
        sigma_hat  = -np.matmul(LA.inv(B_bar),np.matmul(LA.inv(PHI),mu))
        sig_hat_m  = np.reshape(sigma_hat[0,0],(1,1))
        sig_hat_um = np.reshape(sigma_hat[1:3,0],(2,1))
        return sig_hat_m, sig_hat_um, A_s

    # Disturbance model 
    def dis(self,p,L0):
        K_stiff = 50 # stiffness of the elastic band, 60 is used for training. The stiffness is estimated to be between 50 and 60, as manually measured by a low-cost, spring-loaded force sensor.
        p_offset = np.array([[0,0,0.175]]).T
        p_band   = p - p_offset
        dir_band = -p_band/LA.norm(p_band)
        length   = LA.norm(p_band)
        if length > L0:
            norm_tension = K_stiff *( (length - L0) )
        else:
            norm_tension = 0
        
        if norm_tension > 0:
            df  = norm_tension * dir_band
        else:
            df  = np.zeros((3,1)) 
        
        return df

class MHE:
    def __init__(self, horizon, dt_sample, r11):
        self.N = horizon
        self.DT = dt_sample
        self.r11 = r11

    def SetStateVariable(self, xp):
        self.state = xp
        self.n_state = xp.numel()

    def SetOutputVariable(self, y):
        assert hasattr(self, 'state'), "Define the state variable first!"
        self.output = y
        self.y_fn   = Function('y',[self.state], [self.output], ['x0'], ['yf'])
        self.n_output = self.output.numel()

    def SetControlVariable(self, u):
        self.ctrl = u
        self.n_ctrl = u.numel()

    def SetNoiseVariable(self, wf):
        self.noise = wf
        self.n_noise = wf.numel()
    
    def SetRotationVariable(self,R):
        self.R_B   = R

    def SetModelDyn(self, dymh, dyukf):
        assert hasattr(self, 'state'), "Define the state variable first!"
        assert hasattr(self, 'ctrl'), "Define the control variable first!"
        assert hasattr(self, 'noise'), "Define the noise variable first!"
        # discrete-time dynamic model based on 4th-order Runge-Kutta method
        self.ModelDyn = self.state + self.DT*dymh
        self.MDyn_fn  = Function('MDyn', [self.state, self.ctrl, self.R_B, self.noise], [self.ModelDyn],
                                 ['s', 'c', 'R', 'n'], ['MDynf'])
        self.Modelukf = self.state + self.DT*dyukf
        self.MDyn_ukf_fn = Function('MDyn_ukf',[self.state, self.ctrl, self.R_B], [self.Modelukf],['s', 'c', 'R'], ['MDyn_ukff'])

    def SetArrivalCost(self, x_hat):
        assert hasattr(self, 'state'), "Define the state variable first!"
        self.P0        = diag(self.weight_para[0, 0:6])
        # Define filter priori
        error_a        = self.state - x_hat
        self.cost_a    = 1/2 * mtimes(mtimes(transpose(error_a), self.P0), error_a)
        self.cost_a_fn = Function('cost_a', [self.state, self.weight_para], [self.cost_a], ['s','tp'], ['cost_af'])

    def SetCostDyn(self):
        assert hasattr(self, 'state'), "Define the state variable first!"
        assert hasattr(self, 'output'), "Define the output variable first!"
        assert hasattr(self, 'noise'), "Define the noise variable first!"
        # Tunable parameters
        self.weight_para = SX.sym('t_para', 1, 14)
        self.horizon1    = SX.sym('h1') # horizon - 1
        self.horizon2    = self.horizon1 - 1 # horizon - 2
        self.index       = SX.sym('ki')
        self.gamma_r     = self.weight_para[0, 6]
        self.gamma_q     = self.weight_para[0, 10] 
        # r                = horzcat(self.r11[0, 0], self.weight_para[0, 7:9]) # fix the fisrt entry and tune the remaining entries
        r                = self.weight_para[0, 7:10]
        R_t              = diag(r) # make sure the weight matrix is positive semidefinite
        self.R           = R_t*self.gamma_r**(self.horizon1-self.index)
        self.R_fn        = Function('R_fn', [self.weight_para, self.horizon1, self.index], \
                            [self.R], ['tp','h1', 'ind'], ['R_fnf'])
        Q_t1             = diag(self.weight_para[0, 11:14])
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

    def MHEsolver(self, Y, R_seq, x_hat, xmhe_traj, ctrl, noise_traj, weight_para, time):
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
        lbw+= self.n_state*[-1e20] # value less than or equal to -1e19 stands for no lower bound
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
                if k<self.horizon-3:
                    for iw in range(self.n_noise):
                        W_guess += [noise_traj[k+1,iw]]
                else:
                    for iw in range(self.n_noise):
                        W_guess += [noise_traj[-1,iw]]
            
            w0  += W_guess 
            # Integrate the cost function till the end of horizon
            J    += self.dJ_fn(s=Xk, m=Y[len(Y)-self.horizon+k], n=Nk, tp=weight_para, h1=self.horizon-1, ind=k)['dJrunf']
            Xnext = self.MDyn_fn(s=Xk,c=ctrl[len(ctrl)-self.horizon+1+k], R=R_seq[len(R_seq)-self.horizon+k], n=Nk)['MDynf']
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
        opts['ipopt.tol'] = 1e-8
        opts['ipopt.print_level'] = 0
        opts['print_time'] = 0
        opts['ipopt.warm_start_init_point']='yes'
        opts['ipopt.max_iter']=1e3
        opts['ipopt.acceptable_tol']=1e-7
        opts['ipopt.mu_strategy']='adaptive'
        # opts = {'ipopt.print_level': print_level, 'ipopt.sb': 'yes', 'print_time': print_level}
        prob = {'f': J, 'x': vertcat(*w), 'g': vertcat(*g)}
        solver = nlpsol('solver', 'ipopt', prob, opts)

        # Solve the NLP
        sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
        w_opt = sol['x'].full().flatten() # convert to a row array
        lam_g = sol['lam_g'].full().flatten() # row vector of Lagrange multipilers for bounds on g

        # Take the optimal noise, state, and costate
        sol_traj1 = np.concatenate((w_opt, self.n_noise * [0])) # sol_traj1 = [x0,w0,x1,w1,...,xk,wk,...xn-1,wn-1,xn,wn] note that we added a wn
        sol_traj = np.reshape(sol_traj1, (-1, self.n_state + self.n_noise)) # sol_traj = [[x0,w0],[x1,w1],...[xk,wk],...[xn-1,wn-1],[xn,wn]] 
        state_traj_opt = sol_traj[:, 0:self.n_state] # each xk is a row vector
        noise_traj_opt = np.delete(sol_traj[:, self.n_state:], -1, 0) # delete the last one as we have added it to make the dimensions of x and w equal
        costate_traj_ipopt = np.reshape(lam_g, (-1,self.n_state))
        
        # compute the co-states using the KKT conditions
        costate_traj_opt = np.zeros((self.horizon, self.n_state))

        for i in range(self.horizon - 1, 0, -1):
            curr_s      = state_traj_opt[i, :]
            curr_n      = noise_traj_opt[i-1,:]
            curr_m      = Y[len(Y) - self.horizon + i]
            curr_c      = ctrl[len(ctrl) - self.horizon + i-1]  # the index of control should be less than that of measurement by 1
            curr_R      = R_seq[len(R_seq) - self.horizon + i]
            lembda_curr = np.reshape(costate_traj_opt[i, :], (self.n_state,1))
            mat_F       = self.F_fn(x0=curr_s, u0=curr_c, R0=curr_R, n0=curr_n)['Ff'].full()
            mat_H       = self.H_fn(x0=curr_s)['Hf'].full()
            R_curr      = self.R_fn(tp=weight_para, h1=self.horizon - 1, ind=i)['R_fnf'].full()
            y_curr      = self.y_fn(x0=curr_s)['yf'].full()
            lembda_pre  = np.matmul(np.transpose(mat_F), lembda_curr) + np.matmul(np.matmul(np.transpose(mat_H), R_curr), (curr_m - y_curr))
            costate_traj_opt[(i - 1):i, :] = np.transpose(lembda_pre)

        # Output
        opt_sol = {"state_traj_opt": state_traj_opt,
                   "noise_traj_opt": noise_traj_opt,
                   "costate_traj_opt": costate_traj_opt,
                   "costate_ipopt": costate_traj_ipopt}
        return opt_sol

    def diffKKT_general(self):
        assert hasattr(self, 'state'), "Define the state variable first!"
        assert hasattr(self, 'output'), "Define the output variable first!"
        assert hasattr(self, 'ctrl'), "Define the control variable first!"
        assert hasattr(self, 'noise'), "Define the noise variable first!"
        assert hasattr(self, 'MDyn_fn'), "Define the model dynamics function first!"
        assert hasattr(self, 'dJ_fn'), "Define the cost dynamics function first!"
        # Define co-state variables
        self.costate      = SX.sym('lambda', self.n_state, 1) # lambda at k
        self.cos_pre      = SX.sym('lampre', self.n_state, 1) # lambda at k-1, in fact, it will not appear in all the 2nd-derivate terms

        # Differentiate the dynamics to get the system Jacobian
        self.F            = jacobian(self.ModelDyn, self.state)
        self.F_fn         = Function('F',[self.state, self.ctrl, self.R_B, self.noise], [self.F], ['x0','u0','R0','n0'], ['Ff']) 
        self.G            = jacobian(self.ModelDyn, self.noise)
        self.G_fn         = Function('G',[self.state, self.ctrl, self.R_B, self.noise], [self.G], ['x0','u0','R0','n0'], ['Gf'])
        self.H            = jacobian(self.output, self.state)
        self.H_fn         = Function('H',[self.state], [self.H], ['x0'], ['Hf'])

        # Definition of Lagrangian
        self.Lbar0        = self.dJ_running - mtimes(transpose(self.costate), self.ModelDyn) # arrival Lagrangian_bar
        self.Lbar_k       = self.dJ_running - mtimes(transpose(self.costate), self.ModelDyn) \
            + mtimes(transpose(self.cos_pre), self.state) # k=t-N+1,...,t-1
        self.L0           = self.cost_a + self.Lbar0 # arrival Lagrangian, k=t-N
        self.LbarT        = self.dJ_T # terminal Lagrangian, k=t

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
        self.ddL0xp_fn    = Function('ddL0xp', [self.state, self.costate, self.ctrl, self.R_B, self.noise, self.measurement, self.weight_para, self.horizon1, self.index], [self.ddL0xp], ['x0', 'c0', 'u0', 'R0', 'n0', 'm0', 'tp', 'h1', 'ind'], ['ddL0xpf'])
        self.ddLbar0xx    = jacobian(self.dLbar0x, self.state)
        self.ddLbar0xx_fn = Function('ddL0xx', [self.state, self.costate, self.ctrl, self.R_B, self.noise, self.measurement, self.weight_para, self.horizon1, self.index], [self.ddLbar0xx], ['x0', 'c0', 'u0', 'R0', 'n0', 'm0', 'tp', 'h1', 'ind'], ['ddL0xxf'])
        self.ddLbar0xw    = jacobian(self.dLbar0x, self.noise)
        self.ddLbar0xw_fn = Function('ddL0xw', [self.state, self.costate, self.ctrl, self.R_B, self.noise, self.measurement, self.weight_para, self.horizon1, self.index], [self.ddLbar0xw], ['x0', 'c0', 'u0', 'R0', 'n0', 'm0', 'tp', 'h1', 'ind'], ['ddL0xwf'])
        self.ddLbar0ww    = jacobian(self.dLbar0w, self.noise)
        self.ddLbar0ww_fn = Function('ddL0ww', [self.state, self.costate, self.ctrl, self.R_B, self.noise, self.measurement, self.weight_para, self.horizon1, self.index], [self.ddLbar0ww], ['x0', 'c0', 'u0', 'R0', 'n0', 'm0', 'tp', 'h1', 'ind'], ['ddL0wwf'])
        self.ddLbar0wp    = jacobian(self.dLbar0w, self.weight_para)
        self.ddLbar0wp_fn = Function('ddL0wp', [self.state, self.costate, self.ctrl, self.R_B, self.noise, self.measurement, self.weight_para, self.horizon1, self.index], [self.ddLbar0wp], ['x0', 'c0', 'u0', 'R0', 'n0', 'm0', 'tp', 'h1', 'ind'], ['ddL0wpf'])
        # note that when k=t-N, ddL0xx = P + ddLbarxx, for all k, ddLxw = ddLbarxw, ddLww = ddLbarww

        # Second-order derivative of path Lagrangian, k=t-N+1,...,t-1
        self.ddLbarxx     = jacobian(self.dLbarx, self.state) 
        self.ddLbarxx_fn  = Function('ddLxx', [self.state, self.costate, self.ctrl, self.R_B, self.noise, self.measurement, self.weight_para, self.horizon1, self.index], [self.ddLbarxx], ['x0', 'c0', 'u0', 'R0', 'n0', 'm0', 'tp', 'h1', 'ind'], ['ddLxxf'])
        self.ddLbarxp     = jacobian(self.dLbarx, self.weight_para) 
        self.ddLbarxp_fn  = Function('ddLxp', [self.state, self.costate, self.ctrl, self.R_B, self.noise, self.measurement, self.weight_para, self.horizon1, self.index], [self.ddLbarxp], ['x0', 'c0', 'u0', 'R0', 'n0', 'm0', 'tp', 'h1', 'ind'], ['ddLxpf'])
        self.ddLbarxw     = jacobian(self.dLbarx, self.noise) 
        self.ddLbarxw_fn  = Function('ddLxw', [self.state, self.costate, self.ctrl, self.R_B, self.noise, self.measurement, self.weight_para, self.horizon1, self.index], [self.ddLbarxw], ['x0', 'c0', 'u0', 'R0', 'n0', 'm0', 'tp', 'h1', 'ind'], ['ddLxwf'])
        self.ddLbarww     = jacobian(self.dLbarw, self.noise) 
        self.ddLbarww_fn  = Function('ddLww', [self.state, self.costate, self.ctrl, self.R_B, self.noise, self.measurement, self.weight_para, self.horizon1, self.index], [self.ddLbarww], ['x0', 'c0', 'u0', 'R0', 'n0', 'm0', 'tp', 'h1', 'ind'], ['ddLwwf'])
        self.ddLbarwp     = jacobian(self.dLbarw, self.weight_para) 
        self.ddLbarwp_fn  = Function('ddLwp', [self.state, self.costate, self.ctrl, self.R_B, self.noise, self.measurement, self.weight_para, self.horizon1, self.index], [self.ddLbarwp], ['x0', 'c0', 'u0', 'R0', 'n0', 'm0', 'tp', 'h1', 'ind'], ['ddLwpf'])
        
        # Second-order derivative of terminal Lagrangian, k=t
        self.ddLbarTxx    = jacobian(self.dLbarTx, self.state)
        self.ddLbarTxx_fn = Function('ddLTxx', [self.state, self.measurement, self.weight_para, self.horizon1, self.index], [self.ddLbarTxx], ['x0', 'm0', 'tp', 'h1', 'ind'], ['ddLTxxf'])
        self.ddLbarTxp    = jacobian(self.dLbarTx, self.weight_para)
        self.ddLbarTxp_fn = Function('ddLTxp', [self.state, self.measurement, self.weight_para, self.horizon1, self.index], [self.ddLbarTxp], ['x0', 'm0', 'tp', 'h1', 'ind'], ['ddLTxpf'])

    
    def GetAuxSys_general(self, state_traj_opt, costate_traj_opt, noise_traj_opt, weight_para, Y, ctrl, R_seq):
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
            curr_c    = ctrl[len(ctrl) - horizon + 1 + k] # current control force
            curr_R    = R_seq[len(R_seq) - self.horizon + k]
            matF     += [self.F_fn(x0=curr_s, u0=curr_c, n0=curr_n)['Ff'].full()]
            matG     += [self.G_fn(x0=curr_s, u0=curr_c, n0=curr_n)['Gf'].full()]
            matH     += [self.H_fn(x0=curr_s)['Hf'].full()]
            if k == 0: # note that P is included only in the arrival cost
                matddLxx += [self.ddLbar0xx_fn(x0=curr_s, c0=curr_cs, u0=curr_c, R0=curr_R, n0=curr_n, m0=curr_m, tp=weight_para, h1=horizon-1, ind=k)['ddL0xxf'].full()]
                matddLxp += [self.ddL0xp_fn(x0=curr_s, c0=curr_cs, u0=curr_c, R0=curr_R, n0=curr_n, m0=curr_m, tp=weight_para, h1=horizon-1, ind=k)['ddL0xpf'].full()]
                matddLxw += [self.ddLbar0xw_fn(x0=curr_s, c0=curr_cs, u0=curr_c, R0=curr_R, n0=curr_n, m0=curr_m, tp=weight_para, h1=horizon-1, ind=k)['ddL0xwf'].full()]
                matddLww += [self.ddLbar0ww_fn(x0=curr_s, c0=curr_cs, u0=curr_c, R0=curr_R, n0=curr_n, m0=curr_m, tp=weight_para, h1=horizon-1, ind=k)['ddL0wwf'].full()]
                matddLwp += [self.ddLbar0wp_fn(x0=curr_s, c0=curr_cs, u0=curr_c, R0=curr_R, n0=curr_n, m0=curr_m, tp=weight_para, h1=horizon-1, ind=k)['ddL0wpf'].full()]
            else:
                matddLxx += [self.ddLbarxx_fn(x0=curr_s, c0=curr_cs, u0=curr_c, R0=curr_R, n0=curr_n, m0=curr_m, tp=weight_para, h1=horizon-1, ind=k)['ddLxxf'].full()]
                matddLxp += [self.ddLbarxp_fn(x0=curr_s, c0=curr_cs, u0=curr_c, R0=curr_R, n0=curr_n, m0=curr_m, tp=weight_para, h1=horizon-1, ind=k)['ddLxpf'].full()]
                matddLxw += [self.ddLbarxw_fn(x0=curr_s, c0=curr_cs, u0=curr_c, R0=curr_R, n0=curr_n, m0=curr_m, tp=weight_para, h1=horizon-1, ind=k)['ddLxwf'].full()]
                matddLww += [self.ddLbarww_fn(x0=curr_s, c0=curr_cs, u0=curr_c, R0=curr_R, n0=curr_n, m0=curr_m, tp=weight_para, h1=horizon-1, ind=k)['ddLwwf'].full()]
                matddLwp += [self.ddLbarwp_fn(x0=curr_s, c0=curr_cs, u0=curr_c, R0=curr_R, n0=curr_n, m0=curr_m, tp=weight_para, h1=horizon-1, ind=k)['ddLwpf'].full()]
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



"""
The KF_gradient_solver class solves for the explicit solutions of the gradients of optimal trajectories
w.r.t the tunable parameters 
"""
class KF_gradient_solver:
    def __init__(self, xp, para):
        self.n_xmhe = xp.numel()
        self.n_para = para.numel()
        self.gt    = SX.sym('gt',3,1)
        self.xp     = xp
        self.x      = self.xp[0:3]
        self.traj_e = self.x - self.gt
        self.Kloss  = 3e2
        w_p, w_v  = 1, 1
        weight      = np.array([w_v, w_v, w_v])
        # weight      = np.array([w_x, w_y, w_z, 1.5*w_x, 1.5*w_y, 1.5*w_z, 2*w_x, 2*w_y, 2*w_z])
        # weight      = np.array([w_x, w_y, w_z])
        self.loss   = mtimes(mtimes(transpose(self.traj_e), np.diag(weight)), self.traj_e)
        

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
            # corrector of the estimation error covariance 
            if k < self.horizon-2:
                S_k1   = np.matmul(np.matmul(matddLxw[k+1], LA.inv(matddLww[k+1])), np.transpose(matddLxw[k+1]))-matddLxx[k+1] # S_k1: S_{k+1}
            else:
                S_k1   = -matddLxx[k+1] # matddLxw_{t} does not exist
            S[k+1]    = S_k1
            C_k1      = np.matmul(LA.inv(np.identity(self.n_xmhe)-np.matmul(P_k1, S[k+1])), P_k1)
            C[k+1]    = C_k1
            
            # state corrector (from which the Kalman gain can be extracted using the matrix inversion lemma)
            if k < self.horizon-2: # the last index is always smaller than the dimension by 1
                T_k1   = np.matmul(np.matmul(matddLxw[k+1], LA.inv(matddLww[k+1])), matddLwp[k+1])-matddLxp[k+1] # T_k1: T_{k+1}
            else:
                T_k1   = -matddLxp[k+1]
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
    
    def loss_horizon(self, xmhe_traj, Gt, N, time):
        loss_track = 0
        if time < N:
            horizon = time + 1
        else:
            horizon = N + 1
        for k in range(horizon):
            x_mhe = xmhe_traj[k, :]
            dloss_track = self.Kloss * self.loss_tracking(x_mhe, Gt[len(Gt)-horizon+k])
            loss_track +=  dloss_track
        return loss_track

    def loss_tracking(self, xp, gt):
        loss_fn = Function('loss', [self.xp, self.gt], [self.loss], ['xp0', 'gt0'], ['lossf'])
        loss_track = loss_fn(xp0=xp, gt0=gt)['lossf'].full()
        return loss_track

    def ChainRule(self, Gt, xmhe_traj, X_opt):
        # Define the gradient of loss w.r.t state
        Ddlds = jacobian(self.loss, self.xp)
        Ddlds_fn = Function('Ddlds', [self.xp, self.gt], [Ddlds], ['xp0', 'gt0'], ['dldsf'])
        # Initialize the parameter gradient
        dp = np.zeros((1, self.n_para))
        # Initialize the loss
        loss_track = 0 
        # Positive coefficient in the loss
        for t in range(self.horizon):
            x_mhe = xmhe_traj[t, :]
            x_mhe = np.reshape(x_mhe, (self.n_xmhe, 1))
            dloss_track = self.Kloss * self.loss_tracking(x_mhe, Gt[len(Gt)-self.horizon+t])
            loss_track +=  dloss_track
            dlds =  self.Kloss * Ddlds_fn(xp0=x_mhe, gt0=Gt[len(Gt)-self.horizon+t])['dldsf'].full()
            dxdp = X_opt[t]
            dp  += np.matmul(dlds, dxdp)
           
        return dp, loss_track



































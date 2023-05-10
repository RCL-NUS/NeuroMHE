"""
This file defines the simulation environment for a quad-rotor uav
-------------------------------------------------------------
Wang Bingheng, modified on 10 May. 2022 at Control and Simulation Lab, ECE Dept. NUS
----------------
2nd version on 10 Oct. 2022 after receiving the reviewers' comments
"""

from casadi import *
import numpy as np
from numpy import linalg as LA

class quadrotor:
    def __init__(self, uav_para, dt_sample):
        # Velocity in inertial frame
        self.v    = SX.sym('v',3,1)
        # Quaternion 
        self.q    = SX.sym('q',4,1)
        # Angular velocity in body frame
        self.omega= SX.sym('omega',3,1)
        # Disturbance force in body frame to be estimated (including the propeller force and the residual force)
        self.df   = SX.sym('df',3,1)
        # Disturbance torque in body frame to be estimated
        self.dtau = SX.sym('dt',3,1)
        # Process noise
        self.w    = SX.sym('w',6,1)
        # Quadrotor's inertial parameters (mass, rotational inertia)
        self.m    = uav_para[0]
        self.J    = np.diag([uav_para[1], uav_para[2], uav_para[3]])
        # Discretization step in MHE
        self.dt   = dt_sample
        # Z direction vector free of coordinate
        self.z    = vertcat(0, 0, 1)
        # Gravitational acceleration
        self.g    = 9.81

    def skew_sym(self, v):
        v_cross = vertcat(
            horzcat(0, -v[2,0], v[1,0]),
            horzcat(v[2,0], 0, -v[0,0]),
            horzcat(-v[1,0], v[0,0], 0)
        )
        return v_cross
    
    def q_2_rotation(self,q): # from body frame to inertial frame
        q = q/norm_2(q) # normalization
        q0, q1, q2, q3 = q[0,0], q[1,0], q[2,0], q[3,0]
        R = vertcat(
        horzcat( 2 * (q0 ** 2 + q1 ** 2) - 1, 2 * q1 * q2 - 2 * q0 * q3, 2 * q0 * q2 + 2 * q1 * q3),
        horzcat(2 * q0 * q3 + 2 * q1 * q2, 2 * (q0 ** 2 + q2 ** 2) - 1, 2 * q2 * q3 - 2 * q0 * q1),
        horzcat(2 * q1 * q3 - 2 * q0 * q2, 2 * q0 * q1 + 2 * q2 * q3, 2 * (q0 ** 2 + q3 ** 2) - 1)
        )
        return R
       

    # Reduced body dynamics, state:v,df,omega,dtau, input:noise, output:v,omega
    def uav_dywb(self,x,noise):
        # velocity
        v      = vertcat(x[0,0], x[1,0], x[2,0])
        # disturbance force 
        df     = vertcat(x[3,0], x[4,0], x[5,0])
        # rotation matrix
        # R_b_w  = self.q_2_rotation(q)
        # angular velocity
        omega  = vertcat(x[6,0], x[7,0], x[8,0])
        # disturbance torque
        dtau   = vertcat(x[9,0], x[10,0], x[11,0])
        # process noise for disturbance force
        wf     = vertcat(noise[0,0], noise[1,0], noise[2,0])
        # process noise for disturbance torque
        wt     = vertcat(noise[3,0], noise[4,0], noise[5,0])
        # dynamics model augmented by the random walk model of the disturbance
        # dv     = 1/self.m * (-self.m*self.g*self.z + mtimes(R_b_w, df))
        dv     = 1/self.m * (-self.m*self.g*self.z +df)
        ddf    = wf
        omega_s= self.skew_sym(omega)# skew symmetric matrix w_s of the angular velocity 
        domega = mtimes(inv(self.J), (-mtimes(omega_s, mtimes(self.J, omega)) +dtau))
        ddtau  = wt
        dyn_wb = vertcat(dv, ddf, domega, ddtau)
        return dyn_wb
    
    def model(self):
        # state for whole body dynamics augmented with disturbance
        self.xa   = vertcat(self.v, self.df, self.omega, self.dtau)
        # output
        self.y    = vertcat(self.v, self.omega)
        # 4-order Runge-Kutta discretization of dynamics model used in MHE
        # k1        = self.uav_dywb(self.xa, self.q, self.w)
        # k2        = self.uav_dywb(self.xa + self.dt/2*k1, self.q, self.w)
        # k3        = self.uav_dywb(self.xa + self.dt/2*k2, self.q, self.w)
        # k4        = self.uav_dywb(self.xa + self.dt*k3, self.q, self.w)
        k1        = self.uav_dywb(self.xa, self.w)
        k2        = self.uav_dywb(self.xa + self.dt/2*k1, self.w)
        k3        = self.uav_dywb(self.xa + self.dt/2*k2, self.w)
        k4        = self.uav_dywb(self.xa + self.dt*k3, self.w)
        self.dyn = (k1 + 2*k2 + 2*k3 + k4)/6  # when only k1 is used, it is the Euler method.

















    






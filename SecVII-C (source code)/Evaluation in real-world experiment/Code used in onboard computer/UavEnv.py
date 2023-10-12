"""
This file defines the simulation environment for a quad-rotor uav
-------------------------------------------------------------
Wang Bingheng, 10 May. 2022 at Control and Simulation Lab, ECE Dept. NUS
-------------------------------------------------------------
1st version: 10 May,2022
2nd version: 10 Oct. 2022 after receiving the reviewers' comments
"""

from casadi import *
import numpy as np
import math
from numpy import linalg as LA
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class quadrotor:
    def __init__(self, uav_para, dt_sample):
        # Position in inertial frame
        self.p    = SX.sym('p',3,1)
        # Velocity in inertial frame
        self.v    = SX.sym('v',3,1)
        # Rotation matrix from body frame to inertial frame
        self.r11, self.r12, self.r13 = SX.sym('r11'), SX.sym('r12'), SX.sym('r13')
        self.r21, self.r22, self.r23 = SX.sym('r21'), SX.sym('r22'), SX.sym('r23')
        self.r31, self.r32, self.r33 = SX.sym('r31'), SX.sym('r32'), SX.sym('r33')
        self.r    = vertcat(self.r11, self.r12, self.r13, 
                            self.r21, self.r22, self.r23,
                            self.r31, self.r32, self.r33)
        self.R_B      = vertcat(
            horzcat(self.r11, self.r12, self.r13),
            horzcat(self.r21, self.r22, self.r23),
            horzcat(self.r31, self.r32, self.r33)
        )
        # Angular velocity in body frame
        self.omegax, self.omegay, self.omegaz    = SX.sym('omegax'), SX.sym('omegay'), SX.sym('omegaz')
        self.omega    = vertcat(self.omegax, self.omegay, self.omegaz)
        # Total thruster force
        self.f     = SX.sym('f')
        # Control torque in body frame
        self.tau   = SX.sym('tau',3,1)
        # Control input u: 4-by-1 vecter
        self.u     = vertcat(self.f, self.tau)
        # Disturbance force in inertial frame
        self.df    = SX.sym('df',3,1)
        # Disturbance torque in body frame
        self.dtau  = SX.sym('dtau',3,1)
        # Process noise for disturbance force
        self.wf    = SX.sym('wf',3,1)
        # Quadrotor's inertial parameters (mass, rotational inertia)
        self.m     = uav_para[0]
        self.J     = np.diag([uav_para[1], uav_para[2], uav_para[3]])
        # Discretization step in MHE and geometric controller
        self.dt    = dt_sample
        # Unit direction vectors free of coordinate
        self.ex    = vertcat(1, 0, 0)
        self.ey    = vertcat(0, 1, 0)
        self.ez    = vertcat(0, 0, 1)
        # Gravitational acceleration in Singapore
        self.g     = 9.78
        # Polynomial coefficients of reference trajectory
        self.polyc = SX.sym('c',1,8)
        # Time in polynomial
        self.time  = SX.sym('t')
        # Initial time in polynomial
        self.time0 = SX.sym('t0')
        #-----------variables used in L1-AC-----------#
        # Predicted state z_hat
        self.z_hat = SX.sym('zhat',3,1)
        # Matched disturbance in body frame
        self.dm    = SX.sym('dm',1,1)
        # Unmatched disturbance in body frame
        self.dum   = SX.sym('dum',2,1)
        # Hurwitz matrix
        self.As    = SX.sym('As',3,3)

    def dir_cosine(self, Euler):
        # Euler angles for roll, pitch and yaw
        gamma, theta, psi = Euler[0,0], Euler[1,0], Euler[2,0]
        R_x = np.array([[1, 0, 0],
                        [0, math.cos(gamma), math.sin(gamma)],
                        [0, -math.sin(gamma),  math.cos(gamma)]])
        R_y = np.array([[ math.cos(theta), 0, -math.sin(theta)],
                        [0,                1,              0],
                        [math.sin(theta), 0, math.cos(theta)]])
        R_z = np.array([[math.cos(psi), math.sin(psi), 0],
                        [-math.sin(psi),  math.cos(psi), 0],
                        [0,                          0, 1]])
        # Rotation matrix from world frame to body frame, X->Z->Y
        R_wb= np.matmul(np.matmul(R_y, R_z), R_x)
        # Rotation matrix from body frame to world frame, Y->Z->X
        R_bw= np.transpose(R_wb)
        R_Bv = np.array([[R_bw[0,0], R_bw[0,1], R_bw[0,2], R_bw[1,0], R_bw[1,1], R_bw[1,2], R_bw[2,0], R_bw[2,1], R_bw[2,2]]]).T
        return R_Bv, R_bw, R_wb
    
    def skew_sym(self, v):
        v_cross = vertcat(
            horzcat(0, -v[2,0], v[1,0]),
            horzcat(v[2,0], 0, -v[0,0]),
            horzcat(-v[1,0], v[0,0], 0)
        )
        return v_cross

    def polytraj(self,coeff,time,time0):
        time_vec   = vertcat(1,
                             self.time-self.time0,
                             (self.time-self.time0)**2,
                             (self.time-self.time0)**3,
                             (self.time-self.time0)**4,
                             (self.time-self.time0)**5,
                             (self.time-self.time0)**6,
                             (self.time-self.time0)**7)
        polyp      = mtimes(self.polyc, time_vec)
        polyp_fn   = Function('ref_p',[self.polyc,self.time,self.time0],[polyp],['pc0','t0','ti0'],['ref_pf'])
        ref_p      = polyp_fn(pc0=coeff,t0=time,ti0=time0)['ref_pf'].full()
        polyv      = jacobian(polyp, self.time)
        polyv_fn   = Function('ref_v',[self.polyc,self.time,self.time0],[polyv],['pc0','t0','ti0'],['ref_vf'])
        ref_v      = polyv_fn(pc0=coeff,t0=time,ti0=time0)['ref_vf'].full()
        polya      = jacobian(polyv, self.time)
        polya_fn   = Function('ref_a',[self.polyc,self.time,self.time0],[polya],['pc0','t0','ti0'],['ref_af'])
        ref_a      = polya_fn(pc0=coeff,t0=time,ti0=time0)['ref_af'].full()
        polyj      = jacobian(polya, self.time)
        polyj_fn   = Function('ref_j',[self.polyc,self.time,self.time0],[polyj],['pc0','t0','ti0'],['ref_jf'])
        ref_j      = polyj_fn(pc0=coeff,t0=time,ti0=time0)['ref_jf'].full()
        polys      = jacobian(polyj, self.time)
        polys_fn   = Function('ref_s',[self.polyc,self.time,self.time0],[polys],['pc0','t0','ti0'],['ref_sf'])
        ref_s      = polys_fn(pc0=coeff,t0=time,ti0=time0)['ref_sf'].full()

        return ref_p, ref_v, ref_a, ref_j, ref_s


    
    def ref_exp1(self, Coeffx_exp1, Coeffy_exp1, Coeffz_exp1, time, t_switch):
        if time <5+t_switch: # S->A
            ref_px, ref_vx, ref_ax, ref_jx, ref_sx = self.polytraj(Coeffx_exp1[0,:],time,t_switch)
            ref_py, ref_vy, ref_ay, ref_jy, ref_sy = self.polytraj(Coeffy_exp1[0,:],time,t_switch)
            ref_pz, ref_vz, ref_az, ref_jz, ref_sz = self.polytraj(Coeffz_exp1[0,:],time,t_switch)
        elif time >=5+t_switch and time <8+t_switch: # A->A
            ref_px, ref_vx, ref_ax, ref_jx, ref_sx = self.polytraj(Coeffx_exp1[1,:],time,5+t_switch)
            ref_py, ref_vy, ref_ay, ref_jy, ref_sy = self.polytraj(Coeffy_exp1[1,:],time,5+t_switch)
            ref_pz, ref_vz, ref_az, ref_jz, ref_sz = self.polytraj(Coeffz_exp1[1,:],time,5+t_switch)
        elif time >=8+t_switch and time <10+t_switch: # A->B
            ref_px, ref_vx, ref_ax, ref_jx, ref_sx = self.polytraj(Coeffx_exp1[2,:],time,8+t_switch)
            ref_py, ref_vy, ref_ay, ref_jy, ref_sy = self.polytraj(Coeffy_exp1[2,:],time,8+t_switch)
            ref_pz, ref_vz, ref_az, ref_jz, ref_sz = self.polytraj(Coeffz_exp1[2,:],time,8+t_switch)
        elif time >=10+t_switch and time <12+t_switch: # B->C
            ref_px, ref_vx, ref_ax, ref_jx, ref_sx = self.polytraj(Coeffx_exp1[3,:],time,10+t_switch)
            ref_py, ref_vy, ref_ay, ref_jy, ref_sy = self.polytraj(Coeffy_exp1[3,:],time,10+t_switch)
            ref_pz, ref_vz, ref_az, ref_jz, ref_sz = self.polytraj(Coeffz_exp1[3,:],time,10+t_switch)
        elif time >=12+t_switch and time <15+t_switch: # C->C
            ref_px, ref_vx, ref_ax, ref_jx, ref_sx = self.polytraj(Coeffx_exp1[4,:],time,12+t_switch)
            ref_py, ref_vy, ref_ay, ref_jy, ref_sy = self.polytraj(Coeffy_exp1[4,:],time,12+t_switch)
            ref_pz, ref_vz, ref_az, ref_jz, ref_sz = self.polytraj(Coeffz_exp1[4,:],time,12+t_switch)
        elif time >=15+t_switch and time <17+t_switch: # C->D
            ref_px, ref_vx, ref_ax, ref_jx, ref_sx = self.polytraj(Coeffx_exp1[5,:],time,15+t_switch)
            ref_py, ref_vy, ref_ay, ref_jy, ref_sy = self.polytraj(Coeffy_exp1[5,:],time,15+t_switch)
            ref_pz, ref_vz, ref_az, ref_jz, ref_sz = self.polytraj(Coeffz_exp1[5,:],time,15+t_switch)
        elif time >=17+t_switch and time <19+t_switch: # D->E
            ref_px, ref_vx, ref_ax, ref_jx, ref_sx = self.polytraj(Coeffx_exp1[6,:],time,17+t_switch)
            ref_py, ref_vy, ref_ay, ref_jy, ref_sy = self.polytraj(Coeffy_exp1[6,:],time,17+t_switch)
            ref_pz, ref_vz, ref_az, ref_jz, ref_sz = self.polytraj(Coeffz_exp1[6,:],time,17+t_switch)
        elif time >=19+t_switch and time <22+t_switch: # E->E
            ref_px, ref_vx, ref_ax, ref_jx, ref_sx = self.polytraj(Coeffx_exp1[7,:],time,19+t_switch)
            ref_py, ref_vy, ref_ay, ref_jy, ref_sy = self.polytraj(Coeffy_exp1[7,:],time,19+t_switch)
            ref_pz, ref_vz, ref_az, ref_jz, ref_sz = self.polytraj(Coeffz_exp1[7,:],time,19+t_switch)
        elif time >=22+t_switch and time <24+t_switch: # E->B
            ref_px, ref_vx, ref_ax, ref_jx, ref_sx = self.polytraj(Coeffx_exp1[8,:],time,22+t_switch)
            ref_py, ref_vy, ref_ay, ref_jy, ref_sy = self.polytraj(Coeffy_exp1[8,:],time,22+t_switch)
            ref_pz, ref_vz, ref_az, ref_jz, ref_sz = self.polytraj(Coeffz_exp1[8,:],time,22+t_switch)
        elif time >=24+t_switch and time <26+t_switch: # B->F
            ref_px, ref_vx, ref_ax, ref_jx, ref_sx = self.polytraj(Coeffx_exp1[9,:],time,24+t_switch)
            ref_py, ref_vy, ref_ay, ref_jy, ref_sy = self.polytraj(Coeffy_exp1[9,:],time,24+t_switch)
            ref_pz, ref_vz, ref_az, ref_jz, ref_sz = self.polytraj(Coeffz_exp1[9,:],time,24+t_switch) 
        elif time >=26+t_switch and time <29+t_switch: # F->F
            ref_px, ref_vx, ref_ax, ref_jx, ref_sx = self.polytraj(Coeffx_exp1[10,:],time,26+t_switch)
            ref_py, ref_vy, ref_ay, ref_jy, ref_sy = self.polytraj(Coeffy_exp1[10,:],time,26+t_switch)
            ref_pz, ref_vz, ref_az, ref_jz, ref_sz = self.polytraj(Coeffz_exp1[10,:],time,26+t_switch)
        else:
            ref_px, ref_vx, ref_ax, ref_jx, ref_sx = self.polytraj(Coeffx_exp1[11,:],time,29+t_switch)
            ref_py, ref_vy, ref_ay, ref_jy, ref_sy = self.polytraj(Coeffy_exp1[11,:],time,29+t_switch)
            ref_pz, ref_vz, ref_az, ref_jz, ref_sz = self.polytraj(Coeffz_exp1[11,:],time,29+t_switch)
        ref_p = np.reshape(np.vstack((ref_px, ref_py, ref_pz)), (3,1))
        ref_v = np.reshape(np.vstack((ref_vx, ref_vy, ref_vz)), (3,1))
        ref_a = np.reshape(np.vstack((ref_ax, ref_ay, ref_az)), (3,1))
        ref_j = np.reshape(np.vstack((ref_jx, ref_jy, ref_jz)), (3,1))
        ref_s = np.reshape(np.vstack((ref_sx, ref_sy, ref_sz)), (3,1))
        return ref_p, ref_v, ref_a, ref_j, ref_s
    
    def ref_exp1_cont(self, Coeffx_exp1, Coeffy_exp1, Coeffz_exp1, time, t_switch):
        if time <5+t_switch:
            ref_px, ref_vx, ref_ax, ref_jx, ref_sx = self.polytraj(Coeffx_exp1[0,:],time,t_switch)
            ref_py, ref_vy, ref_ay, ref_jy, ref_sy = self.polytraj(Coeffy_exp1[0,:],time,t_switch)
            ref_pz, ref_vz, ref_az, ref_jz, ref_sz = self.polytraj(Coeffz_exp1[0,:],time,t_switch)
        elif time >=5+t_switch and time <7+t_switch:
            ref_px, ref_vx, ref_ax, ref_jx, ref_sx = self.polytraj(Coeffx_exp1[2,:],time,5+t_switch)
            ref_py, ref_vy, ref_ay, ref_jy, ref_sy = self.polytraj(Coeffy_exp1[2,:],time,5+t_switch)
            ref_pz, ref_vz, ref_az, ref_jz, ref_sz = self.polytraj(Coeffz_exp1[2,:],time,5+t_switch)
        elif time >=7+t_switch and time <9+t_switch:
            ref_px, ref_vx, ref_ax, ref_jx, ref_sx = self.polytraj(Coeffx_exp1[3,:],time,7+t_switch)
            ref_py, ref_vy, ref_ay, ref_jy, ref_sy = self.polytraj(Coeffy_exp1[3,:],time,7+t_switch)
            ref_pz, ref_vz, ref_az, ref_jz, ref_sz = self.polytraj(Coeffz_exp1[3,:],time,7+t_switch)
        elif time >=9+t_switch and time <11+t_switch:
            ref_px, ref_vx, ref_ax, ref_jx, ref_sx = self.polytraj(Coeffx_exp1[5,:],time,9+t_switch)
            ref_py, ref_vy, ref_ay, ref_jy, ref_sy = self.polytraj(Coeffy_exp1[5,:],time,9+t_switch)
            ref_pz, ref_vz, ref_az, ref_jz, ref_sz = self.polytraj(Coeffz_exp1[5,:],time,9+t_switch)
        elif time >=11+t_switch and time <13+t_switch:
            ref_px, ref_vx, ref_ax, ref_jx, ref_sx = self.polytraj(Coeffx_exp1[6,:],time,11+t_switch)
            ref_py, ref_vy, ref_ay, ref_jy, ref_sy = self.polytraj(Coeffy_exp1[6,:],time,11+t_switch)
            ref_pz, ref_vz, ref_az, ref_jz, ref_sz = self.polytraj(Coeffz_exp1[6,:],time,11+t_switch)
        elif time >=13+t_switch and time <15+t_switch:
            ref_px, ref_vx, ref_ax, ref_jx, ref_sx = self.polytraj(Coeffx_exp1[8,:],time,13+t_switch)
            ref_py, ref_vy, ref_ay, ref_jy, ref_sy = self.polytraj(Coeffy_exp1[8,:],time,13+t_switch)
            ref_pz, ref_vz, ref_az, ref_jz, ref_sz = self.polytraj(Coeffz_exp1[8,:],time,13+t_switch)
        elif time >=15+t_switch and time <17+t_switch: 
            ref_px, ref_vx, ref_ax, ref_jx, ref_sx = self.polytraj(Coeffx_exp1[9,:],time,15+t_switch)
            ref_py, ref_vy, ref_ay, ref_jy, ref_sy = self.polytraj(Coeffy_exp1[9,:],time,15+t_switch)
            ref_pz, ref_vz, ref_az, ref_jz, ref_sz = self.polytraj(Coeffz_exp1[9,:],time,15+t_switch) 
        else:
            ref_px, ref_vx, ref_ax, ref_jx, ref_sx = self.polytraj(Coeffx_exp1[11,:],time,17+t_switch)
            ref_py, ref_vy, ref_ay, ref_jy, ref_sy = self.polytraj(Coeffy_exp1[11,:],time,17+t_switch)
            ref_pz, ref_vz, ref_az, ref_jz, ref_sz = self.polytraj(Coeffz_exp1[11,:],time,17+t_switch)
        ref_p = np.reshape(np.vstack((ref_px, ref_py, ref_pz)), (3,1))
        ref_v = np.reshape(np.vstack((ref_vx, ref_vy, ref_vz)), (3,1))
        ref_a = np.reshape(np.vstack((ref_ax, ref_ay, ref_az)), (3,1))
        ref_j = np.reshape(np.vstack((ref_jx, ref_jy, ref_jz)), (3,1))
        ref_s = np.reshape(np.vstack((ref_sx, ref_sy, ref_sz)), (3,1))
        return ref_p, ref_v, ref_a, ref_j, ref_s

    
    def ref_exp2(self, Coeffx_exp2, Coeffy_exp2, Coeffz_exp2, time, t_switch):
        if time <4+t_switch:
            ref_px, ref_vx, ref_ax, ref_jx, ref_sx = self.polytraj(Coeffx_exp2[0,:],time,t_switch)
            ref_py, ref_vy, ref_ay, ref_jy, ref_sy = self.polytraj(Coeffy_exp2[0,:],time,t_switch)
            ref_pz, ref_vz, ref_az, ref_jz, ref_sz = self.polytraj(Coeffz_exp2[0,:],time,t_switch)
        elif time >=4+t_switch and time <20+t_switch:
            ref_px, ref_vx, ref_ax, ref_jx, ref_sx = self.polytraj(Coeffx_exp2[1,:],time,4+t_switch)
            ref_py, ref_vy, ref_ay, ref_jy, ref_sy = self.polytraj(Coeffy_exp2[1,:],time,4+t_switch)
            ref_pz, ref_vz, ref_az, ref_jz, ref_sz = self.polytraj(Coeffz_exp2[1,:],time,4+t_switch)
        else:
            ref_px, ref_vx, ref_ax, ref_jx, ref_sx = self.polytraj(Coeffx_exp2[2,:],time,20+t_switch)
            ref_py, ref_vy, ref_ay, ref_jy, ref_sy = self.polytraj(Coeffy_exp2[2,:],time,20+t_switch)
            ref_pz, ref_vz, ref_az, ref_jz, ref_sz = self.polytraj(Coeffz_exp2[2,:],time,20+t_switch)
        
        ref_p = np.reshape(np.vstack((ref_px, ref_py, ref_pz)), (3,1))
        ref_v = np.reshape(np.vstack((ref_vx, ref_vy, ref_vz)), (3,1))
        ref_a = np.reshape(np.vstack((ref_ax, ref_ay, ref_az)), (3,1))
        ref_j = np.reshape(np.vstack((ref_jx, ref_jy, ref_jz)), (3,1))
        ref_s = np.reshape(np.vstack((ref_sx, ref_sy, ref_sz)), (3,1))
        return ref_p, ref_v, ref_a, ref_j, ref_s
    
    def ref_exp2_linearoscillation(self, Coeffx_exp2_lo, Coeffy_exp2_lo, Coeffz_exp2_lo, time, t_switch):
        if time <4+t_switch:
            ref_px, ref_vx, ref_ax, ref_jx, ref_sx = self.polytraj(Coeffx_exp2_lo[0,:],time,t_switch)
            ref_py, ref_vy, ref_ay, ref_jy, ref_sy = self.polytraj(Coeffy_exp2_lo[0,:],time,t_switch)
            ref_pz, ref_vz, ref_az, ref_jz, ref_sz = self.polytraj(Coeffz_exp2_lo[0,:],time,t_switch)
        elif time >=4+t_switch and time <6+t_switch: # hover 1
            ref_px, ref_vx, ref_ax, ref_jx, ref_sx = self.polytraj(Coeffx_exp2_lo[1,:],time,4+t_switch)
            ref_py, ref_vy, ref_ay, ref_jy, ref_sy = self.polytraj(Coeffy_exp2_lo[1,:],time,4+t_switch)
            ref_pz, ref_vz, ref_az, ref_jz, ref_sz = self.polytraj(Coeffz_exp2_lo[1,:],time,4+t_switch)
        elif time >=6+t_switch and time <11+t_switch:
            ref_px, ref_vx, ref_ax, ref_jx, ref_sx = self.polytraj(Coeffx_exp2_lo[2,:],time,6+t_switch)
            ref_py, ref_vy, ref_ay, ref_jy, ref_sy = self.polytraj(Coeffy_exp2_lo[2,:],time,6+t_switch)
            ref_pz, ref_vz, ref_az, ref_jz, ref_sz = self.polytraj(Coeffz_exp2_lo[2,:],time,6+t_switch)
        elif time >=11+t_switch and time <13+t_switch: # hover 2
            ref_px, ref_vx, ref_ax, ref_jx, ref_sx = self.polytraj(Coeffx_exp2_lo[3,:],time,11+t_switch)
            ref_py, ref_vy, ref_ay, ref_jy, ref_sy = self.polytraj(Coeffy_exp2_lo[3,:],time,11+t_switch)
            ref_pz, ref_vz, ref_az, ref_jz, ref_sz = self.polytraj(Coeffz_exp2_lo[3,:],time,11+t_switch)
        elif time >=13+t_switch and time <17+t_switch:
            ref_px, ref_vx, ref_ax, ref_jx, ref_sx = self.polytraj(Coeffx_exp2_lo[4,:],time,13+t_switch)
            ref_py, ref_vy, ref_ay, ref_jy, ref_sy = self.polytraj(Coeffy_exp2_lo[4,:],time,13+t_switch)
            ref_pz, ref_vz, ref_az, ref_jz, ref_sz = self.polytraj(Coeffz_exp2_lo[4,:],time,13+t_switch)
        elif time >=17+t_switch and time <19+t_switch: # hover 3
            ref_px, ref_vx, ref_ax, ref_jx, ref_sx = self.polytraj(Coeffx_exp2_lo[5,:],time,17+t_switch)
            ref_py, ref_vy, ref_ay, ref_jy, ref_sy = self.polytraj(Coeffy_exp2_lo[5,:],time,17+t_switch)
            ref_pz, ref_vz, ref_az, ref_jz, ref_sz = self.polytraj(Coeffz_exp2_lo[5,:],time,17+t_switch)
        elif time >=19+t_switch and time <22+t_switch: 
            ref_px, ref_vx, ref_ax, ref_jx, ref_sx = self.polytraj(Coeffx_exp2_lo[6,:],time,19+t_switch)
            ref_py, ref_vy, ref_ay, ref_jy, ref_sy = self.polytraj(Coeffy_exp2_lo[6,:],time,19+t_switch)
            ref_pz, ref_vz, ref_az, ref_jz, ref_sz = self.polytraj(Coeffz_exp2_lo[6,:],time,19+t_switch)
        elif time >=22+t_switch and time <24+t_switch: # hover 4
            ref_px, ref_vx, ref_ax, ref_jx, ref_sx = self.polytraj(Coeffx_exp2_lo[7,:],time,22+t_switch)
            ref_py, ref_vy, ref_ay, ref_jy, ref_sy = self.polytraj(Coeffy_exp2_lo[7,:],time,22+t_switch)
            ref_pz, ref_vz, ref_az, ref_jz, ref_sz = self.polytraj(Coeffz_exp2_lo[7,:],time,22+t_switch)
        
        else:
            ref_px, ref_vx, ref_ax, ref_jx, ref_sx = self.polytraj(Coeffx_exp2_lo[8,:],time,24+t_switch)
            ref_py, ref_vy, ref_ay, ref_jy, ref_sy = self.polytraj(Coeffy_exp2_lo[8,:],time,24+t_switch)
            ref_pz, ref_vz, ref_az, ref_jz, ref_sz = self.polytraj(Coeffz_exp2_lo[8,:],time,24+t_switch)
        
        ref_p = np.reshape(np.vstack((ref_px, ref_py, ref_pz)), (3,1))
        ref_v = np.reshape(np.vstack((ref_vx, ref_vy, ref_vz)), (3,1))
        ref_a = np.reshape(np.vstack((ref_ax, ref_ay, ref_az)), (3,1))
        ref_j = np.reshape(np.vstack((ref_jx, ref_jy, ref_jz)), (3,1))
        ref_s = np.reshape(np.vstack((ref_sx, ref_sy, ref_sz)), (3,1))
        return ref_p, ref_v, ref_a, ref_j, ref_s

    # Augmented velocity dynamics used in MHE  
    def uav_dyp(self,x,f,R_B,noise):
        # Velocity
        v      = vertcat(x[0,0], x[1,0], x[2,0])
        # Disturbance force 
        df     = vertcat(x[3,0], x[4,0], x[5,0])
        # Process noise for disturbance force
        wf     = vertcat(noise[0,0], noise[1,0], noise[2,0])
        # Dynamics model augmented by the random walk model of the disturbance
        dv     = 1/self.m * (-self.m*self.g*self.ez + mtimes(R_B, f*self.ez) + df)
        ddf    = wf
        dyn_wb = vertcat(dv, ddf)
        return dyn_wb
    
    def uav_ukf_dyp(self,x,f,R_B):
        # Velocity
        v      = vertcat(x[0,0], x[1,0], x[2,0])
        # Disturbance force 
        df     = vertcat(x[3,0], x[4,0], x[5,0])
        # Process noise for disturbance force used in UKF (this is the standard setting)
        wf     = np.zeros((3,1))
        # Dynamics model augmented by the random walk model of the disturbance
        dv     = 1/self.m * (-self.m*self.g*self.ez + mtimes(R_B, f*self.ez) + df)
        ddf    = wf
        dyn_ukf = vertcat(dv, ddf)
        return dyn_ukf

    def model(self):
        # State for position dynamics augmented with disturbance
        self.xp   = vertcat(self.v, self.df)
        # Quadrotor state
        self.x    = vertcat(self.p, self.v, self.r, self.omega)
        # Output
        self.y    = self.v
        # Skew symmetric matrix omega_s of the angular velocity 
        omega_s   = self.skew_sym(self.omega)
        # Dynamics model
        dp        = self.v
        dv        = 1/self.m * (-self.m*self.g*self.ez + mtimes(self.R_B, self.f*self.ez) + self.df)
        dr11, dr12, dr13 = self.r12 * self.omegaz - self.r13 * self.omegay, self.r13 * self.omegax - self.r11 * self.omegaz, self.r11 * self.omegay - self.r12 * self.omegax
        dr21, dr22, dr23 = self.r22 * self.omegaz - self.r23 * self.omegay, self.r23 * self.omegax - self.r21 * self.omegaz, self.r21 * self.omegay - self.r22 * self.omegax
        dr31, dr32, dr33 = self.r32 * self.omegaz - self.r33 * self.omegay, self.r33 * self.omegax - self.r31 * self.omegaz, self.r31 * self.omegay - self.r32 * self.omegax        
        domega    = mtimes(LA.inv(self.J), (-mtimes(omega_s, mtimes(self.J, self.omega)) + self.tau +self.dtau))
        xdot      = vertcat(dp, dv, dr11, dr12, dr13, dr21, dr22, dr23, dr31, dr32, dr33, domega)
        self.dywb = Function('Dywb', [self.x, self.u, self.df, self.dtau], [xdot], ['x0', 'u0', 'df0', 'dtau0'], ['xdotf'])
        # 4-order Runge-Kutta discretization of the augmented position model used in MHE (symbolic computation)
        k1        = self.uav_dyp(self.xp, self.f, self.R_B, self.wf)
        k2        = self.uav_dyp(self.xp + self.dt/2*k1, self.f, self.R_B, self.wf)
        k3        = self.uav_dyp(self.xp + self.dt/2*k2, self.f, self.R_B, self.wf)
        k4        = self.uav_dyp(self.xp + self.dt*k3, self.f, self.R_B, self.wf)
        self.dymh = (k1 + 2*k2 + 2*k3 + k4)/6
        # 4-order Runge-KUtta discretization of the augmented position model used in UKF (symbolic computation)
        k1_ukf    = self.uav_ukf_dyp(self.xp,self.f,self.R_B)
        k2_ukf    = self.uav_ukf_dyp(self.xp + self.dt/2*k1_ukf,self.f,self.R_B)
        k3_ukf    = self.uav_ukf_dyp(self.xp + self.dt/2*k2_ukf,self.f,self.R_B)
        k4_ukf    = self.uav_ukf_dyp(self.xp + self.dt*k3_ukf,self.f,self.R_B)
        self.dyukf= (k1_ukf + 2*k2_ukf + 2*k3_ukf + k4_ukf)/6
        # Dynamics model used in the L1-AC state predictor
        f_z       = -self.g*self.ez
        B_R       = 1/self.m*mtimes(self.R_B,self.ez)
        B_Rp      = horzcat(1/self.m*mtimes(self.R_B,self.ex),1/self.m*mtimes(self.R_B,self.ey))    
        z         = self.v
        dz_hat    = f_z + mtimes(B_R, (self.f + self.dm)) + mtimes(B_Rp,self.dum) + mtimes(self.As,(self.z_hat-z))
        self.dyzhat = Function('Dyzhat',[self.z_hat,self.R_B,self.v,self.f,self.dm,self.dum,self.As],[dz_hat],['z_hat0','Rb0','v0','f0','dm0','dum0','As0'],['dzhatf'])


    """
    The step function takes the control (force and torque) as the input and 
    returns the new states in the next step
    """
    def step(self, x, u, df, dtau, dt):
        self.model()
        # define discrete-time dynamics using 4-th order Runge-Kutta
        k1    = self.dywb(x0=x, u0=u, df0=df, dtau0=dtau)['xdotf'].full()
        k2    = self.dywb(x0=x+dt/2*k1, u0=u, df0=df, dtau0=dtau)['xdotf'].full()
        k3    = self.dywb(x0=x+dt/2*k2, u0=u, df0=df, dtau0=dtau)['xdotf'].full()
        k4    = self.dywb(x0=x+dt*k3, u0=u, df0=df, dtau0=dtau)['xdotf'].full()
        xdot  = (k1 + 2*k2 + 2*k3 + k4)/6
        x_new = x + dt*xdot
        # components
        p_new     = np.array([[x_new[0,0], x_new[1,0], x_new[2,0]]]).T
        v_new     = np.array([[x_new[3,0], x_new[4,0], x_new[5,0]]]).T
        omega_new = np.array([[x_new[15,0], x_new[16,0], x_new[17,0]]]).T
        R_B_new   = np.array([
                           [x_new[6,0], x_new[7,0], x_new[8,0]],
                           [x_new[9,0], x_new[10,0],x_new[11,0]],
                           [x_new[12,0],x_new[13,0],x_new[14,0]]
                           ])
        R_B_h_new = np.array([[x_new[6,0], x_new[7,0], x_new[8,0], 
                               x_new[9,0], x_new[10,0],x_new[11,0], 
                               x_new[12,0],x_new[13,0],x_new[14,0]]]).T
        # Y->Z->X rotation from {b} to {I}
        gamma   = np.arctan(R_B_new[2, 1]/R_B_new[1, 1])
        theta   = np.arctan(R_B_new[0, 2]/R_B_new[0, 0])
        psi     = np.arcsin(-R_B_new[0, 1])
        Euler_new = np.array([[gamma, theta, psi]]).T
        output = {"p_new":p_new,
                  "v_new":v_new,
                  "R_new":R_B_h_new,
                  "omega_new":omega_new,
                  "Euler":Euler_new
                 }
        return output
    
    """
    Predictor function for the L1-AC state predictor
    """
    def predictor_L1(self,z_hat,Rb,v,u,dm,dum,As,dt):
        self.model()
        # define discrete-time dynamics using 4-th order Runge-Kutta
        k1    = self.dyzhat(z_hat0=z_hat,Rb0=Rb,v0=v,f0=u,dm0=dm,dum0=dum,As0=As)['dzhatf'].full()
        k2    = self.dyzhat(z_hat0=z_hat+dt/2*k1,Rb0=Rb,v0=v,f0=u,dm0=dm,dum0=dum,As0=As)['dzhatf'].full()
        k3    = self.dyzhat(z_hat0=z_hat+dt/2*k2,Rb0=Rb,v0=v,f0=u,dm0=dm,dum0=dum,As0=As)['dzhatf'].full()
        k4    = self.dyzhat(z_hat0=z_hat+dt*k3,Rb0=Rb,v0=v,f0=u,dm0=dm,dum0=dum,As0=As)['dzhatf'].full()
        dz_hat= (k1 + 2*k2 + 2*k3 + k4)/6
        z_hat_new = z_hat + dt*dz_hat
        return z_hat_new
    
    # Disturbance model (random walk model)
    def dis_noise(self,x,dpara):
        # Polynomisl coefficients
        c_xv, c_xp, c_xc = dpara[0], dpara[1], dpara[2]
        c_yv, c_yp, c_yc = dpara[3], dpara[4], dpara[5]
        c_zv, c_zp, c_zc = dpara[6], dpara[7], dpara[8]
        # Quadrotor state
        px, py, pz       = x[0,0], x[1,0], x[2,0]
        vx, vy, vz       = x[3,0],x[4,0],x[5,0]
       
        # Polynomials of variance parameters
        vfx     = c_xv*vx**2 + c_xp*px**2 + c_xc
        vfy     = c_yv*vy**2 + c_yp*py**2 + c_yc
        vfz     = c_zv*vz**2 + c_zp*pz**2 + c_zc
        vf_inv  = np.array([[1/vfx, 1/vfy, 1/vfz]]).T # compute the inverse of the variance for illustration purpose
        # State-dependent process noise for generating the disturbance force and troque
        wfx     = np.random.normal(0,vfx,1)
        wfy     = np.random.normal(0,vfy,1)
        wfz     = np.random.normal(0,vfz,1)
        w       = np.vstack((wfx,wfy,wfz))
        
        return w

    def dis(self,w,df,dt):
        # Current disturbance
        dfx, dfy, dfz           = df[0,0], df[1,0], df[2,0] 
        # Process noise of disturbance dynamics
        wfx, wfy, wfz = w[0,0], w[1,0], w[2,0]
        # Update the disturbance force
        dfx_new = np.clip(dfx + dt*wfx,-0.25*self.m*self.g,0.25*self.m*self.g) 
        dfy_new = np.clip(dfy + dt*wfy,-0.25*self.m*self.g,0.25*self.m*self.g) 
        dfz_new = np.clip(dfz + dt*wfz,-0.5*self.m*self.g,0.1*self.m*self.g)
        df_new  = np.vstack((dfx_new, dfy_new, dfz_new))
        # if LA.norm(df_new)>=5*self.m*self.g:
        #     df_new = df_new/LA.norm(df_new)*6*self.m*self.g
        
        return df_new

    # below functions are for demo
    # get the position of the centroid and the four vertexes of the quadrotor within a trajectory
    def get_quadrotor_position(self, wing_len, state_traj):

        # thrust_position in body frame
        r1 = vertcat(wing_len*0.5 / sqrt(2), wing_len*0.5 / sqrt(2), 0)
        r2 = vertcat(-wing_len*0.5 / sqrt(2), wing_len*0.5 / sqrt(2), 0)
        r3 = vertcat(-wing_len*0.5 / sqrt(2), -wing_len*0.5 / sqrt(2), 0)
        r4 = vertcat(wing_len*0.5 / sqrt(2), -wing_len*0.5 / sqrt(2), 0)

        # horizon
        horizon = np.size(state_traj, 1)
        position = np.zeros((15,horizon))
        for t in range(horizon):
            # position of COM
            rc = state_traj[0:3, t]
            # altitude of quaternion
            r = state_traj[6:9, t]

            # direction cosine matrix from body to inertial
            R = self.R_mrp(self.r)
            R_fn = Function('R',[self.r], [R], ['r0'], ['Rf'])
            CIB = np.transpose(R_fn(r0=r)['Rf'].full())

            # position of each rotor in inertial frame
            r1_pos = rc + mtimes(CIB, r1).full().flatten()
            r2_pos = rc + mtimes(CIB, r2).full().flatten()
            r3_pos = rc + mtimes(CIB, r3).full().flatten()
            r4_pos = rc + mtimes(CIB, r4).full().flatten()

            # store
            position[0:3,t] = rc
            position[3:6,t] = r1_pos
            position[6:9,t] = r2_pos
            position[9:12,t] = r3_pos
            position[12:15,t] = r4_pos

        return position
    
    def play_animation(self, wing_len, state_traj, position_ref, dt ,save_option=0, title='NeuroMHE'):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X (m)', fontsize=10, labelpad=5)
        ax.set_ylabel('Y (m)', fontsize=10, labelpad=5)
        ax.set_zlabel('Z (m)', fontsize=10, labelpad=5)
        ax.set_zlim(-0.5, 3)
        ax.set_ylim(-3, 3)
        ax.set_xlim(-1, 1)
        ax.set_title(title, pad=20, fontsize=15)

        # data
        position = self.get_quadrotor_position(wing_len, state_traj)
        sim_horizon = np.size(position, 0)

        # animation
        line_traj, = ax.plot(position[0,:1], position[1,:1], position[2,:1])
        c_x, c_y, c_z = position[0:3,0]
        r1_x, r1_y, r1_z = position[3:6,0]
        r2_x, r2_y, r2_z = position[6:9,0]
        r3_x, r3_y, r3_z = position[9:12,0]
        r4_x, r4_y, r4_z = position[12:15,0]
        line_arm1, = ax.plot([c_x, r1_x], [c_y, r1_y], [c_z, r1_z], linewidth=2, color='red', marker='o', markersize=3)
        line_arm2, = ax.plot([c_x, r2_x], [c_y, r2_y], [c_z, r2_z], linewidth=2, color='blue', marker='o', markersize=3)
        line_arm3, = ax.plot([c_x, r3_x], [c_y, r3_y], [c_z, r3_z], linewidth=2, color='orange', marker='o', markersize=3)
        line_arm4, = ax.plot([c_x, r4_x], [c_y, r4_y], [c_z, r4_z], linewidth=2, color='green', marker='o', markersize=3)

        line_traj_ref, = ax.plot(position_ref[0, :1], position_ref[1, :1], position_ref[2, :1], color='gray', alpha=0.5)

        # time label
        time_template = 'time = %.2fs'
        time_text = ax.text2D(0.66, 0.55, "time", transform=ax.transAxes)

        # customize
        if position_ref is not None:
            plt.legend([line_traj, line_traj_ref], ['actual', 'desired'], ncol=1, loc='best',
                       bbox_to_anchor=(0.35, 0.25, 0.5, 0.5))

        def update_traj(num):
            # customize
            time_text.set_text(time_template % (num * dt))

            # trajectory
            line_traj.set_data(position[:num, 0], position[:num, 1])
            line_traj.set_3d_properties(position[:num, 2])

            # uav
            c_x, c_y, c_z = position[num, 0:3]
            r1_x, r1_y, r1_z = position[num, 3:6]
            r2_x, r2_y, r2_z = position[num, 6:9]
            r3_x, r3_y, r3_z = position[num, 9:12]
            r4_x, r4_y, r4_z = position[num, 12:15]

            line_arm1.set_data_3d([c_x, r1_x], [c_y, r1_y],[c_z, r1_z])
            #line_arm1.set_3d_properties()

            line_arm2.set_data_3d([c_x, r2_x], [c_y, r2_y],[c_z, r2_z])
            #line_arm2.set_3d_properties()

            line_arm3.set_data_3d([c_x, r3_x], [c_y, r3_y],[c_z, r3_z])
            #line_arm3.set_3d_properties()

            line_arm4.set_data_3d([c_x, r4_x], [c_y, r4_y],[c_z, r4_z])
            #line_arm4.set_3d_properties()

            # trajectory ref
            num=sim_horizon-1
            line_traj_ref.set_data_3d(position_ref[0,:num], position_ref[1,:num],position_ref[2,:num])
            #line_traj_ref.set_3d_properties()

            return line_traj, line_arm1, line_arm2, line_arm3, line_arm4, line_traj_ref, time_text

        ani = animation.FuncAnimation(fig, update_traj, sim_horizon, interval=dt*1000, blit=True)

        if save_option != 0:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=-1)
            ani.save('training'+title + '.mp4', writer=writer, dpi=300)
            print('save_success')

        plt.show()



















    






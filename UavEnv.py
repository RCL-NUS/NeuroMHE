"""
This file defines the simulation environment for a quad-rotor uav
-------------------------------------------------------------
Wang Bingheng, modified on 10 May. 2022 at Control and Simulation Lab, ECE Dept. NUS
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
        # # Modified Rodrigues parameters (MRP)
        # self.r    = SX.sym('r',3,1)
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
        self.wx, self.wy, self.wz    = SX.sym('wx'), SX.sym('wy'), SX.sym('wz')
        self.w    = vertcat(self.wx, self.wy, self.wz)
        # Total thruster force
        self.f    = SX.sym('f')
        # Control torque in body frame
        self.tau  = SX.sym('tau',3,1)
        # Control input u: 4-by-1 vecter
        self.u    = vertcat(self.f, self.tau)
        # Disturbance force in inertial frame
        self.df   = SX.sym('df',3,1)
        # Process noise for disturbance force
        self.etaf = SX.sym('etaf',3,1)
        # Disturbance torque in body frame
        self.dtau = SX.sym('dt',3,1)
        # Process noise for disturbance torque
        self.etat = SX.sym('etat',3,1)
        # Process noise
        self.eta  = vertcat(self.etaf, self.etat)
        # Quadrotor's inertial parameters (mass, rotational inertia)
        self.m    = uav_para[0]
        self.J    = np.diag([uav_para[1], uav_para[2], uav_para[3]])
        # Discretization step in MHE
        self.dt   = dt_sample
        # Z direction vector free of coordinate
        self.z    = vertcat(0, 0, 1)
        # Gravitational acceleration in Singapore
        self.g    = 9.78
        # Polynomial coefficients of reference trajectory
        self.polyc = SX.sym('c',1,8)
        # Time in polynomial
        self.time = SX.sym('t')
        # Initial time in polynomial
        self.time0 = SX.sym('t0')

    def dir_cosine(self, Euler):
        # Euler angles for roll, pitch and yaw
        gamma, theta, psi = Euler[0], Euler[1], Euler[2]
        # Initial rotation matrix from body frame to inertial frame, X->Z->Y
        R_x = np.array([[1, 0, 0],
                        [0, math.cos(gamma), -math.sin(gamma)],
                        [0, math.sin(gamma),  math.cos(gamma)]])
        R_y = np.array([[ math.cos(theta), 0, math.sin(theta)],
                        [0,                1,              0],
                        [-math.sin(theta), 0, math.cos(theta)]])
        R_z = np.array([[math.cos(psi), -math.sin(psi), 0],
                        [math.sin(psi),  math.cos(psi), 0],
                        [0,                          0, 1]])
        # rotation matrix from world frame to body frame
        R_wb= np.matmul(np.matmul(R_y, R_z), R_x)
        R_bw= np.transpose(R_wb)
        R_Bv0 = np.array([[R_bw[0,0], R_bw[0,1], R_bw[0,2], R_bw[1,0], R_bw[1,1], R_bw[1,2], R_bw[2,0], R_bw[2,1], R_bw[2,2]]]).T
        return R_Bv0, R_bw

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


    def skew_sym(self, v):
        v_cross = vertcat(
            horzcat(0, -v[2], v[1]),
            horzcat(v[2], 0, -v[0]),
            horzcat(-v[1], v[0], 0)
        )
        return v_cross
    
    # Augmented whole body dynamics, state:x, control:u, disturbance:d
    def uav_dywb(self,x,u,noise):
        # position
        p      = vertcat(x[0,0], x[1,0], x[2,0])
        # velocity
        v      = vertcat(x[3,0], x[4,0], x[5,0])
        # rotation matrix
        r11, r12, r13 = x[9,0], x[10], x[11]
        r21, r22, r23 = x[12,0], x[13,0], x[14,0]
        r31, r32, r33 = x[15,0], x[16,0], x[17,0]
        R_B  = vertcat(
            horzcat(r11, r12, r13),
            horzcat(r21, r22, r23),
            horzcat(r31, r32, r33)
        )
        # angular velocity
        wx, wy, wz     = x[18,0], x[19,0], x[20,0]
        w      = vertcat(wx, wy, wz)
        
        # total thruster force
        f      = u[0,0]
        # control torque in body frame
        tau    = vertcat(u[1,0], u[2,0], u[3,0])
        # disturbance force 
        df     = vertcat(x[6,0], x[7,0], x[8,0])
        # disturbance torque
        dtau   = vertcat(x[21,0], x[22,0], x[23,0])
        # process noise for disturbance force
        etaf   = vertcat(noise[0,0], noise[1,0], noise[2,0])
        # process noise for disturbance torque
        etat   = vertcat(noise[3,0], noise[4,0], noise[5,0])
        # skew symmetric matrix w_s of the angular velocity 
        w_s    = self.skew_sym(w)
        # # rotation matrix given the current MRP
        # R      = self.R_mrp(r)
        # # mapping matrix in rotational kinematics given the current MRP
        # G      = self.G_r(r)
        # dynamics model augmented by the random walk model of the disturbance
        dp     = v
        dv     = 1/self.m * (-self.m*self.g*self.z + mtimes(R_B, f*self.z) + df)
        ddf    = etaf
        dr11, dr12, dr13 = r12 * wz - r13 * wy, r13 * wx - r11 * wz, r11 * wy - r12 * wx
        dr21, dr22, dr23 = r22 * wz - r23 * wy, r23 * wx - r21 * wz, r21 * wy - r22 * wx
        dr31, dr32, dr33 = r32 * wz - r33 * wy, r33 * wx - r31 * wz, r31 * wy - r32 * wx
        dw     = mtimes(LA.inv(self.J), (-mtimes(w_s, mtimes(self.J, w)) + tau +dtau))
        ddtau  = etat
        dyn_wb = vertcat(dp, dv, ddf, dr11, dr12, dr13, dr21,
                           dr22, dr23, dr31, dr32, dr33, dw, ddtau)
        return dyn_wb
    
    def model(self):
        # state for whole body dynamics augmented with disturbance
        self.xa   = vertcat(self.p, self.v, self.df, self.r, self.w, self.dtau)
        # state for whole body dynamics
        self.x    = vertcat(self.p, self.v, self.r, self.w)
        # state for positional dynamics augmented with disturbance force
        self.xpa  = vertcat(self.p, self.v, self.df)
        # state for positional dynamics
        self.xp   = vertcat(self.p, self.v)
        # # rotation matrix given the current MRP
        # R         = self.R_mrp(self.r)
        # # mapping matrix in rotational kinematics given the current MRP
        # G         = self.G_r(self.r)
        # skew symmetric matrix w_s of the angular velocity 
        w_s       = self.skew_sym(self.w)
        # dynamics model
        dp        = self.v
        dv        = 1/self.m * (-self.m*self.g*self.z + mtimes(self.R_B, self.f*self.z) + self.df)
        dr11, dr12, dr13 = self.r12 * self.wz - self.r13 * self.wy, self.r13 * self.wx - self.r11 * self.wz, self.r11 * self.wy - self.r12 * self.wx
        dr21, dr22, dr23 = self.r22 * self.wz - self.r23 * self.wy, self.r23 * self.wx - self.r21 * self.wz, self.r21 * self.wy - self.r22 * self.wx
        dr31, dr32, dr33 = self.r32 * self.wz - self.r33 * self.wy, self.r33 * self.wx - self.r31 * self.wz, self.r31 * self.wy - self.r32 * self.wx        
        dw        = mtimes(LA.inv(self.J), (-mtimes(w_s, mtimes(self.J, self.w)) + self.tau +self.dtau))
        self.xdot = vertcat(dp, dv, dr11, dr12, dr13, dr21, dr22, dr23, dr31, dr32, dr33, dw)
        self.dywb = Function('Dywb', [self.x, self.u, self.df, self.dtau], [self.xdot], ['x0', 'u0', 'df0', 'dtau0'], ['xdotf'])
        # output
        self.yp   = vertcat(self.p, self.v, self.r, self.w)
        # 4-order Runge-Kutta discretization of dynamics model used in MHE
        k1        = self.uav_dywb(self.xa, self.u, self.eta)
        k2        = self.uav_dywb(self.xa + self.dt/2*k1, self.u, self.eta)
        k3        = self.uav_dywb(self.xa + self.dt/2*k2, self.u, self.eta)
        k4        = self.uav_dywb(self.xa + self.dt*k3, self.u, self.eta)
        self.dymh = (k1 + 2*k2 + 2*k3 + k4)/6

    # Disturbance model (random walk model)
    def dis(self,x,Euler,df,dtau,dpara,dt):
        # polynomisl coefficients
        c_xv, c_xp, c_xc = dpara[0], dpara[1], dpara[2]
        c_yv, c_yp, c_yc = dpara[3], dpara[4], dpara[5]
        c_zv, c_zp, c_zc = dpara[6], dpara[7], dpara[8]
        c_xw, c_xa, caxc = dpara[9], dpara[10], dpara[11]
        c_yw, c_ya, cayc = dpara[12], dpara[13], dpara[14]
        c_zw, c_za, cazc = dpara[15], dpara[16], dpara[17]
        px, py, pz       = x[0,0], x[1,0], x[2,0]
        vx, vy, vz       = x[3,0], x[4,0], x[5,0]
        roll, pitch, yaw = Euler[0,0], Euler[1,0], Euler[2,0]
        wx, wy, wz       = x[15,0], x[16,0], x[17,0]
        dfx, dfy, dfz    = df[0,0], df[1,0], df[2,0]
        dtaux, dtauy, dtauz = dtau[0,0], dtau[1,0], dtau[2,0]
        # polynomials of variance parameters
        vfx     = c_xv*vx**2 + c_xp*px**2 + c_xc
        vfy     = c_yv*vy**2 + c_yp*py**2 + c_yc
        vfz     = c_zv*vz**2 + c_zp*pz**2 + c_zc
        vf_inv  = np.array([[1/vfx, 1/vfy, 1/vfz]]).T
        vtx     = c_xw*wx**2 + c_xa*roll**2 + caxc
        vty     = c_yw*wy**2 + c_ya*pitch**2 + cayc
        vtz     = c_zw*wz**2 + c_za*yaw**2 + cazc
        vt_inv  = np.array([[1/vtx, 1/vty, 1/vtz]]).T
        # state-dependent process noise for the disturbance force and troque
        etafx   = np.random.normal(0,vfx,1)
        etafy   = np.random.normal(0,vfy,1)
        etafz   = np.random.normal(0,vfz,1)
        etatx   = np.random.normal(0,vtx,1)
        etaty   = np.random.normal(0,vty,1)
        etatz   = np.random.normal(0,vtz,1)
        # update the disturbance force
        dfx_new = dfx + dt*etafx
        dfy_new = dfy + dt*etafy
        dfz_new = dfz + dt*etafz # make sure it is downward
        df_new  = np.vstack((dfx_new, dfy_new, dfz_new))
        
        dtx_new = dtaux + dt*etatx
        dty_new = dtauy + dt*etaty
        dtz_new = dtauz + dt*etatz
        dt_new  = np.vstack((dtx_new, dty_new, dtz_new))
        # set saturation for the disturbance to avoid instability
        f_max   = np.array([[3,3,self.m*self.g]]).T
        f_min   = np.array([[-3,-3,-25]]).T
        tau_max = np.array([[0.005,0.005,0.005]]).T
        tau_min = np.array([[-0.005,-0.005,-0.005]]).T
        df_new  = np.clip(df_new,f_min,f_max)
        dt_new  = np.clip(dt_new,tau_min,tau_max)
        return df_new, vf_inv, dt_new, vt_inv

    """
    The step function takes control (force and torque) as input and 
    returns the new states in the next step
    """
    def step(self, x, u, df, dtau,dt):
        self.model()
        # define discrete-time dynamics using 4-th order Runge-Kutta
        k1    = self.dywb(x0=x, u0=u, df0=df, dtau0=dtau)['xdotf'].full()
        k2    = self.dywb(x0=x+dt/2*k1, u0=u, df0=df, dtau0=dtau)['xdotf'].full()
        k3    = self.dywb(x0=x+dt/2*k2, u0=u, df0=df, dtau0=dtau)['xdotf'].full()
        k4    = self.dywb(x0=x+dt*k3, u0=u, df0=df, dtau0=dtau)['xdotf'].full()
        xdot  = (k1 + 2*k2 + 2*k3 + k4)/6
        x_new = x + dt*xdot
        # components
        p_new = np.array([[x_new[0,0], x_new[1,0], x_new[2,0]]]).T
        v_new = np.array([[x_new[3,0], x_new[4,0], x_new[5,0]]]).T
        w_new = np.array([[x_new[15,0], x_new[16,0], x_new[17,0]]]).T
        R_B_new = np.array([
                           [x_new[6,0], x_new[7,0], x_new[8,0]],
                           [x_new[9,0], x_new[10,0],x_new[11,0]],
                           [x_new[12,0],x_new[13,0],x_new[14,0]]
                           ])
        R_B_h_new = np.array([[x_new[6,0], x_new[7,0], x_new[8,0], 
                               x_new[9,0], x_new[10,0],x_new[11,0], 
                               x_new[12,0],x_new[13,0],x_new[14,0]]]).T
        # Y->Z->X rotation from {b} to {I}
        gamma   = np.arctan(-R_B_new[2, 1]/R_B_new[1, 1])
        theta   = np.arctan(-R_B_new[0, 2]/R_B_new[0, 0])
        psi     = np.arcsin(R_B_new[0, 1])
        Euler_new = np.array([[gamma, theta, psi]]).T
        output = {"x_new":x_new,
                  "p_new":p_new,
                  "v_new":v_new,
                  "R_new":R_B_h_new,
                  "w_new":w_new,
                  "Euler":Euler_new
                 }
        return output
    
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



















    






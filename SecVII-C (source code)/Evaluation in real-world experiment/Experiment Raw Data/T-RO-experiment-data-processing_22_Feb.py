"""
Data processing of the physical experiments for the T-RO submission (22-0575)
============================================================================
Wang, Bingheng, 22 Feb. 2023 at Control & Simulation Lab, NUS, Singapore
--------------------Data Source & Coordinate Systems------------------------
________________Data Source_______________________________Coordinate________ 
1.IMU (accelerometer,from sensor_combined)             North-East-Down
2.vehicle_local_position                               North-East-Down
3.position_setpoint_triplet                            North-East-Down
* Data is recored at 50Hz
----------------------------------------------------------------------------
Position dynamics in NED frame:
dot v = g - R*f*z/m + df/m, 
where g = [0;0;9.78] and R is a rotation matrix from body NED frame to world NED frame *
* body NED frame is fixed in the quadrotor body and coincides with world NED frame when the attitude is zero.
Position dynamics in East-North-Up (ENU) frame (used in the T-RO paper):
dot v = g + R*f*z/m + df/m,
where g = [0;0;-9.78] and R is the rotation matrix from body ENU frame to world ENU frame
--------------------Computation of the ground truth data--------------------
Step1:  Computation based on the position dynamics in NED frame:
        df_groundtruth = m*R*a_imu + R*f*z,
        where a_imu is the acceleration data from IMU expressed in body NED frame. 
        Note that a_imu includes the gravity acceleration g!
Step2:  Transformation from NED frame to ENU frame:
        R_enu = [0,1,0;
                 1,0,0;
                 0,0,-1]
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.axes as axp
from matplotlib.pyplot import MultipleLocator
from sklearn.metrics import mean_squared_error
import matplotlib.ticker
import os
from matplotlib import rc
from numpy import linalg as LA
import pandas as pd
from matplotlib.ticker import FormatStrFormatter
from scipy.signal import kaiserord, lfilter, firwin, freqz
import math
from matplotlib.patches import Rectangle
import UavEnv

uav_para = np.array([1.0, 0.029, 0.029, 0.055])
frequency = 25
dt_mhe = 1/frequency
uav = UavEnv.quadrotor(uav_para, dt_mhe)
mass = 1.0 # The true mass of the quadrotor including the onboard computer (Intel NUC), PX4 controller, and battery.
R_enu = np.array([[0,1,0],
                  [1,0,0],
                  [0,0,-1]])
z     = np.array([[0,0,1]]).T
g     = np.array([[0,0,9.78]]).T # the gravity acceleration constant in Singapore
offset = np.array([[0,0,0.17]]).T # the offset height of the tension sensor from the ground
sample_rate = 50
dt          = 1/sample_rate # note that we recorded the data at 50Hz
#------------------------------------------------
# Create a FIR filter and apply it to raw ground truth data.
# The FIR code is copied from the open-source code available at https://scipy-cookbook.readthedocs.io/items/FIRFilter.html
#------------------------------------------------

# The Nyquist rate of the signal.
nyq_rate = sample_rate / 2.0

# The desired width of the transition from pass to stop,
# relative to the Nyquist rate.  We'll design the filter
# with a 5 Hz transition width.
width = 5.0/nyq_rate

# The desired attenuation in the stop band, in dB.
ripple_db = 30.0 

# Compute the order and Kaiser parameter for the FIR filter.
N, beta = kaiserord(ripple_db, width)
print('order=',N)
# The cutoff frequency of the filter.
cutoff_hz = 2.0 

# Use firwin with a Kaiser window to create a lowpass FIR filter.
taps = firwin(N, cutoff_hz/nyq_rate, window=('kaiser', beta))

class OOMFormatter(matplotlib.ticker.ScalarFormatter):
    def __init__(self,order=0,fformat="%1.1f",offset=True,mathText=True):
        self.oom = order
        self.fformat = fformat
        matplotlib.ticker.ScalarFormatter.__init__(self, useOffset=offset,useMathText=mathText)
    def _set_order_of_magnitude(self):
        self.orderOfMagnitude = self.oom
    def _set_format(self,vmin=None,vmax=None):
        self.format = self.fformat
        if self._useMathText:
            self.format = r'$\mathdefault{%s}$'% self.format 

def Quaternion2Rotation(quaternion):
    # convert a vertor from body frame to inertial frame
    quaternion = quaternion/LA.norm(quaternion)
    q0, q1, q2, q3 = quaternion[0], quaternion[1], quaternion[2], quaternion[3]
    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)
    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1
    # 3x3 rotation matrix from body to inertial frame
    R_b = np.array([[r00, r01, r02],
                    [r10, r11, r12],
                    [r20, r21, r22]])
    return R_b
#------------load polynomial coefficients of the reference trajectory-------------#
Coeffx_exp1, Coeffy_exp1, Coeffz_exp1 = np.zeros((12,8)), np.zeros((12,8)), np.zeros((12,8))
Coeffx_exp2, Coeffy_exp2, Coeffz_exp2 = np.zeros((3,8)), np.zeros((3,8)), np.zeros((3,8))
for i in range(12):
    Coeffx_exp1[i,:] = np.load('Ref_exp1/coeffx'+str(i+1)+'.npy')
    Coeffy_exp1[i,:] = np.load('Ref_exp1/coeffy'+str(i+1)+'.npy')
    Coeffz_exp1[i,:] = np.load('Ref_exp1/coeffz'+str(i+1)+'.npy')
for i in range(3):
    Coeffx_exp2[i,:] = np.load('Ref_exp2/coeffx'+str(i+1)+'.npy')
    Coeffy_exp2[i,:] = np.load('Ref_exp2/coeffy'+str(i+1)+'.npy')
    Coeffz_exp2[i,:] = np.load('Ref_exp2/coeffz'+str(i+1)+'.npy')


"""------------Plot figures---------------"""
font1 = {'family':'Times New Roman',
         'weight':'normal',
         'style':'normal', 'size':7}

rc('font', **font1)        
cm_2_inch = 2.54

print("===========================================")
print("Please choose an experiment dataset to plot")
print("a: cable experiment")
print("b: downwash experiment")
experiment = input("enter 'a' or 'b' without the quotation mark:")
print("======================================")
print("Please choose a controller to evaluate")
print("a: NeuroMHE            + PD Controller")
print("b: DMHE                + PD Controller")
print("c: L1 Adaptive Control + PD Controller")
print("d: UKF                 + PD Controller")
print("e: PD Controller alone")
print("f: plot trajectories together & training loss")
controller = input("enter 'a', or 'b',... without the quotation mark:")
print("======================================")

if experiment == 'a':
    T_end  = 33
    if controller == 'a':
        dataset={'a':"NeuroMHE/log_10_2023-2-22-11-32-38_vehicle_local_position_0.csv",
             'b':"NeuroMHE/log_10_2023-2-22-11-32-38_position_setpoint_triplet_0.csv",
             'c':"NeuroMHE/log_10_2023-2-22-11-32-38_sensor_combined_0.csv",
             'd':"NeuroMHE/log_10_2023-2-22-11-32-38_vehicle_attitude_0.csv",
             'e':"DMHE/log_11_2023-2-22-11-44-46_position_setpoint_triplet_0.csv"}
        tensionset = {'a':"22_Feb_2023_Tension_data/22-Feb-2023-Exp1-controller_a.csv"}
        datatension = pd.read_csv(tensionset['a'],header=None, names=['world time','dt','force'])
        local_position = pd.read_csv(dataset['a'])
        position_setpoint = pd.read_csv(dataset['b'])
        sensor         = pd.read_csv(dataset['c'])
        attitude       = pd.read_csv(dataset['d'])
        data_dmhe      = pd.read_csv(dataset['e'])
        dataframe_a    = pd.DataFrame(local_position)
        dataframe_b    = pd.DataFrame(position_setpoint)
        dataframe_c    = pd.DataFrame(sensor)
        dataframe_d    = pd.DataFrame(attitude)
        dataframe_e    = pd.DataFrame(datatension)
        dataframe_dmhe = pd.DataFrame(data_dmhe)
        # load the position data in NED frame
        position_x     = dataframe_a['x']
        position_y     = dataframe_a['y']
        position_z     = dataframe_a['z']
        # load the computed total force command
        f_cmd = dataframe_b['current.vx']
        # load the forgetting factors
        gamma_r = dataframe_b['current.vy']
        gamma_q = dataframe_b['current.vz']
        g_r_dmhe = dataframe_dmhe['current.vy']
        g_q_dmhe = dataframe_dmhe['current.vz']
        # load the NeuroMHE estimation data in NED frame
        dfx_mhe = dataframe_b['current.a_x']
        dfy_mhe = dataframe_b['current.a_y']
        dfz_mhe = dataframe_b['current.a_z']
        n_start = 121 # During the initial interval n_start*dt, the quadrotor is still on the ground and the estimator does not begin to work
        j = 0
        Time1      = []
        time       = 0
        position_enu   = np.zeros((4,len(g_r_dmhe)-n_start))
        ref_enu        = np.zeros((4,len(g_r_dmhe)-n_start))
        # ref_enu_for_rmse_a3 = np.zeros((3,len(Ref_x)))
        t_switch = 0

        for k in range(n_start,len(g_r_dmhe),1):
            Time1 += [time]
            pos_ned      = np.array([[position_x[k],position_y[k],position_z[k]]]).T
            pos_enu      = np.matmul(R_enu,pos_ned) # transfer to ENU frame
            position_enu[0:3,j:j+1] = pos_enu
            position_enu[3,j:j+1] = LA.norm(pos_enu)
            if time <T_end:
                ref_p, ref_v, ref_a, ref_j, ref_s = uav.ref_exp1(Coeffx_exp1, Coeffy_exp1, Coeffz_exp1, time, t_switch) 
            else:
                ref_p, ref_v, ref_a, ref_j, ref_s = np.array([[1,-1,0.12]]).T, np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1))
            ref_enu[0:3,j:j+1]=ref_p
            ref_enu[3,j:j+1]=LA.norm(ref_p)
            time += dt
            j += 1
        np.save('ref_enu_neuromhe',ref_enu)
        np.save('pos_enu_neuromhe',position_enu)
        np.save('Time_track_neuromhe',Time1)
        
        # fig, (ax1,ax2,ax3) = plt.subplots(3, sharex=True,figsize=(8/cm_2_inch,8/cm_2_inch),dpi=600)
        # ax1.plot(Time1,position_enu[0,:],linewidth=0.5)
        # ax1.plot(Time1,ref_enu[0,:],linewidth=0.5)
        # ax2.plot(Time1,position_enu[1,:],linewidth=0.5)
        # ax2.plot(Time1,ref_enu[1,:],linewidth=0.5)
        # ax3.plot(Time1,position_enu[2,:],linewidth=0.5)
        # ax3.plot(Time1,ref_enu[2,:],linewidth=0.5)
        # leg=ax1.legend(['Actual', 'Desired'],loc='lower left',prop=font1)
        # leg.get_frame().set_linewidth(0.5)
        # ax1.set_ylabel('$p_{x}$ [m]',labelpad=0,**font1)
        # ax2.set_ylabel('$p_{y}$ [m]',labelpad=0,**font1)
        # ax3.set_ylabel('$p_{z}$ [m]',labelpad=0,**font1)
        # ax1.grid(b=True,axis='both',linestyle='--', linewidth=0.5)
        # ax2.grid(b=True,axis='both',linestyle='--', linewidth=0.5)
        # ax3.grid(b=True,axis='both',linestyle='--', linewidth=0.5)
        # for axis in ['top', 'bottom', 'left', 'right']:
        #     ax1.spines[axis].set_linewidth(0.5)
        # for axis in ['top', 'bottom', 'left', 'right']:
        #     ax2.spines[axis].set_linewidth(0.5)
        # for axis in ['top', 'bottom', 'left', 'right']:
        #     ax3.spines[axis].set_linewidth(0.5)
        # plt.savefig('NeuroMHE/position-axes-NeuroMHE.png')
        # plt.show()

        # plt.figure(2,figsize=(4.5/cm_2_inch,4/cm_2_inch),dpi=600)
        # ax = plt.axes(projection="3d")
        # ax.plot3D(ref_enu[0,:], ref_enu[1,:], ref_enu[2,:], linewidth=1.5, linestyle='--',color='black')
        # ax.plot3D(position_enu[0,:], position_enu[1,:], position_enu[2,:], linewidth=1,color='orange')
        # # leg=plt.legend(['Desired','Actual'],prop=font1,loc=(0.5,0.2),labelspacing = 0.1)
        # # leg.get_frame().set_linewidth(0.5)
        # ax.set_xlabel('x [m]',labelpad=-10,**font1)
        # ax.set_ylabel('y [m]',labelpad=-10,**font1)
        # ax.set_zlabel('z [m]',labelpad=-8,**font1)
        # ax.tick_params(axis='x',which='major',pad=-5)
        # ax.tick_params(axis='y',which='major',pad=-5)
        # ax.tick_params(axis='z',which='major',pad=-3)
        # for t in ax.xaxis.get_major_ticks(): 
        #     t.label.set_font('Times New Roman') 
        #     t.label.set_fontsize(7)
        # for t in ax.yaxis.get_major_ticks(): 
        #     t.label.set_font('Times New Roman') 
        #     t.label.set_fontsize(7)
        # for t in ax.zaxis.get_major_ticks(): 
        #     t.label.set_font('Times New Roman') 
        #     t.label.set_fontsize(7)
        # ax.view_init(30,60)
        # ax.xaxis._axinfo["grid"].update({"linewidth":0.5,"linestyle":'--'})
        # ax.yaxis._axinfo["grid"].update({"linewidth":0.5,"linestyle":'--'})
        # ax.zaxis._axinfo["grid"].update({"linewidth":0.5,"linestyle":'--'})
        # plt.savefig('NeuroMHE/tracking_NeuroMHE.png')
        # plt.show()

        
       
        Df_Imhe_enu = np.zeros((4,len(g_r_dmhe)))
        for k in range(len(g_r_dmhe)):
            Dfmhe_ned = np.array([[dfx_mhe[k],dfy_mhe[k],dfz_mhe[k]]]).T
            Df_Imhe_enu[0:3,k:k+1] = np.matmul(R_enu,Dfmhe_ned)
            Df_Imhe_enu[3,k:k+1]   = LA.norm(np.matmul(R_enu,Dfmhe_ned))
        np.save('Df_Imhe_neuromhe',Df_Imhe_enu[:,n_start:len(g_r_dmhe)])
        
        # load the accelerometer data in NED frame
        acc_x   = dataframe_c['accelerometer_m_s2[0]']
        acc_y   = dataframe_c['accelerometer_m_s2[1]']
        acc_z   = dataframe_c['accelerometer_m_s2[2]']
        Acc_b_ned   = np.zeros((3,len(acc_x)))
        for k in range(len(acc_x)):
            Acc_b_ned[:,k:k+1] = np.array([[acc_x[k],acc_y[k],acc_z[k]]]).T
        # load the quaternion
        Q0 = dataframe_d['q[0]']
        Q1 = dataframe_d['q[1]']
        Q2 = dataframe_d['q[2]']
        Q3 = dataframe_d['q[3]']
        
        Gt = np.zeros((3,len(g_r_dmhe)-n_start))
        Gt_lpf = np.zeros((4,len(g_r_dmhe)-n_start))
        Gamma  = np.zeros((2,len(g_r_dmhe)-n_start))
        Gamma_dmhe = np.zeros((2,len(g_r_dmhe)-n_start)) # constant weights of DMHE
        
        i = 0
        Time = []
        time = 0
        # calculation of the ground truth data using the two steps mentioned above
        for k in range(n_start,len(g_r_dmhe),1): 
            Time += [time]
            q0, q1, q2, q3 = Q0[k], Q1[k], Q2[k], Q3[k]
            quaternion = np.array([q0, q1, q2, q3])
            R_b        = Quaternion2Rotation(quaternion)
            acc_I_ned = np.matmul(R_b,np.reshape(Acc_b_ned[:,k],(3,1)))
            f_cmdI_ned = np.matmul(R_b,(f_cmd[k]*z))
            gt        = acc_I_ned*mass+f_cmdI_ned # we do not plot the raw accelerometer data as it is too noisy
            Gt[:,i:i+1] =  np.matmul(R_enu,gt) # transfer to ENU frame
            gamma_ned = np.array([[gamma_r[k],0,gamma_q[k]]]).T
            g_dmhe_ned = np.array([[g_r_dmhe[k],0,g_q_dmhe[k]]]).T
            gamma_edu = np.matmul(R_enu,gamma_ned)
            g_dmhe_edu = np.matmul(R_enu,g_dmhe_ned)
            Gamma[:,i:i+1]  = np.reshape(gamma_edu[1:3,0],(2,1))
            Gamma_dmhe[:,i:i+1] = np.reshape(g_dmhe_edu[1:3,0],(2,1))
            time += dt
            i += 1
        
        Gt_lpf[0,:] = lfilter(taps, 1.0, Gt[0,:])
        Gt_lpf[1,:] = lfilter(taps, 1.0, Gt[1,:])
        Gt_lpf[2,:] = lfilter(taps, 1.0, Gt[2,:])
        for k in range(len(Gt_lpf[0,:])):
            Gt_lpf[3,k:k+1]=LA.norm(Gt_lpf[0:3,k])

        np.save('Gt_lpf_neuromhe',Gt_lpf)

        f_tension = dataframe_e['force']
        Tension = np.zeros((1,len(dfx_mhe)-n_start))
        print('len_tension=',len(f_tension))
        i = 0
        n_t_start = 1696 # We started the tension force recording at 11:32:22 am, sent the control command at 11:32:55
        for k in range(n_t_start,len(g_r_dmhe)-n_start+n_t_start,1):
            Tension[:,i:i+1] = f_tension[k]
            i += 1
        Tension_3d_enu = np.zeros((3,len(g_r_dmhe)-n_start))
        for k in range(len(g_r_dmhe)-n_start): # project the tension in ENU frame
            Tension_3d_enu[:,k:k+1]=-Tension[:,k]*(position_enu[0:3,k:k+1]-offset)/LA.norm(position_enu[0:3,k:k+1]-offset)
        np.save('Tension_neuromhe',Tension_3d_enu)

        rmse_px = format(mean_squared_error(ref_enu[0,:], position_enu[0,:], squared=False),'.3f')
        rmse_py = format(mean_squared_error(ref_enu[1,:], position_enu[1,:], squared=False),'.3f')
        rmse_pz = format(mean_squared_error(ref_enu[2,:], position_enu[2,:], squared=False),'.3f')
        rmse_p  = format(mean_squared_error(ref_enu[3,:], position_enu[3,:],squared=False),'.3f')
        rmse_fx = format(mean_squared_error(Gt_lpf[0,:], Df_Imhe_enu[0,n_start:len(g_r_dmhe)], squared=False),'.3f')
        rmse_fy = format(mean_squared_error(Gt_lpf[1,:], Df_Imhe_enu[1,n_start:len(g_r_dmhe)], squared=False),'.3f')
        rmse_fz = format(mean_squared_error(Gt_lpf[2,:], Df_Imhe_enu[2,n_start:len(g_r_dmhe)], squared=False),'.3f')
        rmse_f  = format(mean_squared_error(Gt_lpf[3,:], Df_Imhe_enu[3,n_start:len(g_r_dmhe)], squared=False),'.3f')
        print('rmse_px=',rmse_px,'rmse_py=',rmse_py,'rmse_pz=',rmse_pz,'rmse_p=',rmse_p)
        print('rmse_fx=',rmse_fx,'rmse_fy=',rmse_fy,'rmse_fz=',rmse_fz,'rmse_f=',rmse_f)

        # fig, (ax1,ax2,ax3) = plt.subplots(3,sharex=True,figsize=(6/cm_2_inch,6*0.8/cm_2_inch),dpi=600)
        # ax1.plot(Time,Gt_lpf[0,:], linewidth=0.5,color='black')
        # ax1.plot(Time,Df_Imhe_enu[0,n_start:len(dfx_mhe)], linewidth=0.5,color='orange')
        # ax1.plot(Time,Tension_3d_enu[0,:], linewidth=0.5,color='blue')
        # ax2.plot(Time,Gt_lpf[1,:], linewidth=0.5,color='black')
        # ax2.plot(Time,Df_Imhe_enu[1,n_start:len(dfx_mhe)], linewidth=0.5,color='orange')
        # ax2.plot(Time,Tension_3d_enu[1,:], linewidth=0.5,color='blue')
        # ax3.plot(Time,Gt_lpf[2,:], linewidth=0.5,color='black')
        # ax3.plot(Time,Df_Imhe_enu[2,n_start:len(dfx_mhe)], linewidth=0.5,color='orange')
        # ax3.plot(Time,Tension_3d_enu[2,:], linewidth=0.5,color='blue')
        # ax1.set_ylabel('$d_{x}$ [N]',labelpad=0,**font1)
        # ax2.set_ylabel('$d_{y}$ [N]',labelpad=0,**font1)
        # ax3.set_ylabel('$d_{z}$ [N]',labelpad=-0.5,**font1)
        # ax3.set_xlabel('Time [s]',labelpad=-1.5,**font1) # -4 for 8
        # ax1.tick_params(axis='y',which='major',pad=-0.25)
        # ax2.tick_params(axis='y',which='major',pad=-0.25)
        # ax3.tick_params(axis='x',which='major',pad=-1)
        # ax3.tick_params(axis='y',which='major',pad=-0.25)
        # ax3.set_xticks(np.arange(0,35,10))
        # for t in ax1.yaxis.get_major_ticks(): 
        #     t.label.set_font('Times New Roman') 
        #     t.label.set_fontsize(7)
        # for t in ax2.yaxis.get_major_ticks(): 
        #     t.label.set_font('Times New Roman') 
        #     t.label.set_fontsize(7)
        # for t in ax3.xaxis.get_major_ticks(): 
        #     t.label.set_font('Times New Roman') 
        #     t.label.set_fontsize(7)
        # for t in ax3.yaxis.get_major_ticks(): 
        #     t.label.set_font('Times New Roman') 
        #     t.label.set_fontsize(7)
        # leg=ax1.legend(['Ground truth', 'Estimation', 'Tension force'],loc='lower center',prop=font1,labelspacing=0.1)
        # leg.get_frame().set_linewidth(0.5)
        # ax1.grid(b=True,axis='both',linestyle='--', linewidth=0.5)
        # ax2.grid(b=True,axis='both',linestyle='--', linewidth=0.5)
        # ax3.grid(b=True,axis='both',linestyle='--', linewidth=0.5)
        # for axis in ['top', 'bottom', 'left', 'right']:
        #     ax1.spines[axis].set_linewidth(0.5)
        # for axis in ['top', 'bottom', 'left', 'right']:
        #     ax2.spines[axis].set_linewidth(0.5)
        # for axis in ['top', 'bottom', 'left', 'right']:
        #     ax3.spines[axis].set_linewidth(0.5)
        # plt.savefig('NeuroMHE/force_NeuroMHE_h.png')
        # plt.show()

        fig, (ax1,ax2) = plt.subplots(2,sharex=True,figsize=(8/cm_2_inch,8*0.45/cm_2_inch),dpi=600)
        ax1.plot(Time[0:len(g_r_dmhe)],Gamma[0,0:len(g_r_dmhe)],linewidth=0.5,color='navy')
        ax1.plot(Time[0:len(g_r_dmhe)],Gamma_dmhe[0,0:len(g_r_dmhe)],linewidth=0.5,color='blue')
        ax2.plot(Time[0:len(g_r_dmhe)],Gamma[1,0:len(g_r_dmhe)],linewidth=0.5,color='navy')
        ax2.plot(Time[0:len(g_r_dmhe)],Gamma_dmhe[1,0:len(g_r_dmhe)],linewidth=0.5,color='blue')
        ax1.set_ylabel('$\gamma_1$',labelpad=0.5,**font1)
        ax2.set_xlabel('Time [s]',labelpad=-4.5,**font1)
        ax2.set_ylabel('$\gamma_2$',labelpad=0.5,**font1)
        ax1.tick_params(axis='y',which='major',pad=0)
        ax2.tick_params(axis='x',which='major',pad=-0.5)
        ax2.tick_params(axis='y',which='major',pad=0)
        ax2.set_xticks(np.arange(0,35,10))
        leg=ax1.legend(['NeuroMHE', 'DMHE'],loc='lower center',prop=font1,labelspacing=0.1)
        leg.get_frame().set_linewidth(0.5)
        for t in ax1.yaxis.get_major_ticks(): 
            t.label.set_font('Times New Roman') 
            t.label.set_fontsize(7)
        for t in ax2.xaxis.get_major_ticks(): 
            t.label.set_font('Times New Roman') 
            t.label.set_fontsize(7)
        for t in ax2.yaxis.get_major_ticks(): 
            t.label.set_font('Times New Roman') 
            t.label.set_fontsize(7)
        ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax1.grid(b=True,axis='both',linestyle='--', linewidth=0.5)
        ax2.grid(b=True,axis='both',linestyle='--', linewidth=0.5)
        for axis in ['top', 'bottom', 'left', 'right']:
            ax1.spines[axis].set_linewidth(0.5)
        for axis in ['top', 'bottom', 'left', 'right']:
            ax2.spines[axis].set_linewidth(0.5)
        plt.savefig('NeuroMHE/forgetting factor_comparison.png')
        plt.show()

    elif controller == 'b':
        dataset={'a':"DMHE/log_11_2023-2-22-11-44-46_vehicle_local_position_0.csv",
             'b':"DMHE/log_11_2023-2-22-11-44-46_position_setpoint_triplet_0.csv",
             'c':"DMHE/log_11_2023-2-22-11-44-46_sensor_combined_0.csv",
             'd':"DMHE/log_11_2023-2-22-11-44-46_vehicle_attitude_0.csv"}
        tensionset = {'a':"22_Feb_2023_Tension_data/22-Feb-2023-Exp1-controller_b.csv"}
        datatension = pd.read_csv(tensionset['a'],header=None, names=['world time','dt','force'])
        local_position = pd.read_csv(dataset['a'])
        position_setpoint = pd.read_csv(dataset['b'])
        sensor         = pd.read_csv(dataset['c'])
        attitude       = pd.read_csv(dataset['d'])
        dataframe_a    = pd.DataFrame(local_position)
        dataframe_b    = pd.DataFrame(position_setpoint)
        dataframe_c    = pd.DataFrame(sensor)
        dataframe_d    = pd.DataFrame(attitude)
        dataframe_e    = pd.DataFrame(datatension)
        # load the position data in NED frame
        position_x     = dataframe_a['x']
        position_y     = dataframe_a['y']
        position_z     = dataframe_a['z']
       
        # load the forgetting factors
        n_start = 135 # During the initial interval n_start*dt, the quadrotor is still on the ground and the estimator does not begin to work
        g_r_dmhe = dataframe_b['current.vy']
        g_q_dmhe = dataframe_b['current.vz']
        position_enu   = np.zeros((4,len(g_r_dmhe)-n_start))
        i = 0
        j = 0
        Time1      = []
        time       = 0
        ref_enu        = np.zeros((4,len(g_r_dmhe)-n_start))
        # ref_enu_for_rmse_a3 = np.zeros((3,len(Ref_x)))
        t_switch = 0

        for k in range(n_start,len(g_r_dmhe),1):
            Time1 += [time]
            pos_ned      = np.array([[position_x[k],position_y[k],position_z[k]]]).T
            pos_enu      = np.matmul(R_enu,pos_ned) # transfer to ENU frame
            position_enu[0:3,j:j+1] = pos_enu
            position_enu[3,j:j+1] = LA.norm(pos_enu)
            if time <T_end:
                ref_p, ref_v, ref_a, ref_j, ref_s = uav.ref_exp1(Coeffx_exp1, Coeffy_exp1, Coeffz_exp1, time, t_switch) 
            else:
                ref_p, ref_v, ref_a, ref_j, ref_s = np.array([[1,-1,0.12]]).T, np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1))
            ref_enu[0:3,j:j+1]=ref_p
            ref_enu[3,j:j+1]=LA.norm(ref_p)
            time += dt
            j += 1
        np.save('ref_enu_dmhe',ref_enu)
        np.save('pos_enu_dmhe',position_enu)
        np.save('Time_track_dmhe',Time1)
        # Time_track = np.load('Time_track.npy')
        # ref_enu    = np.load('ref_enu.npy')
        # fig, (ax1,ax2,ax3) = plt.subplots(3, sharex=True,figsize=(8/cm_2_inch,8/cm_2_inch),dpi=600)
        # ax1.plot(Time1,position_enu[0,:],linewidth=0.5)
        # ax1.plot(Time1,ref_enu[0,:],linewidth=0.5)
        # ax2.plot(Time1,position_enu[1,:],linewidth=0.5)
        # ax2.plot(Time1,ref_enu[1,:],linewidth=0.5)
        # ax3.plot(Time1,position_enu[2,:],linewidth=0.5)
        # ax3.plot(Time1,ref_enu[2,:],linewidth=0.5)
        # # leg=ax1.legend(['Actual', 'Desired'],loc='lower left',prop=font1)
        # # leg.get_frame().set_linewidth(0.5)
        # ax1.set_ylabel('$p_{x}$ [m]',labelpad=0,**font1)
        # ax2.set_ylabel('$p_{y}$ [m]',labelpad=0,**font1)
        # ax3.set_ylabel('$p_{z}$ [m]',labelpad=0,**font1)
        # ax1.grid(b=True,axis='both',linestyle='--', linewidth=0.5)
        # ax2.grid(b=True,axis='both',linestyle='--', linewidth=0.5)
        # ax3.grid(b=True,axis='both',linestyle='--', linewidth=0.5)
        # for axis in ['top', 'bottom', 'left', 'right']:
        #     ax1.spines[axis].set_linewidth(0.5)
        # for axis in ['top', 'bottom', 'left', 'right']:
        #     ax2.spines[axis].set_linewidth(0.5)
        # for axis in ['top', 'bottom', 'left', 'right']:
        #     ax3.spines[axis].set_linewidth(0.5)
        # plt.savefig('DMHE/position-axes-DMHE.png')
        # plt.show()
        
        # plt.figure(2,figsize=(4.5/cm_2_inch,4/cm_2_inch),dpi=600)
        # ax = plt.axes(projection="3d")
        # ax.plot3D(ref_enu[0,:], ref_enu[1,:], ref_enu[2,:], linewidth=1.5, linestyle='--',color='black')
        # ax.plot3D(position_enu[0,:], position_enu[1,:], position_enu[2,:], linewidth=1,color='orange')
        # # leg=plt.legend(['Desired','Actual'],prop=font1,loc=(0.5,0.2),labelspacing = 0.1)
        # # leg.get_frame().set_linewidth(0.5)
        # ax.set_xlabel('x [m]',labelpad=-10,**font1)
        # ax.set_ylabel('y [m]',labelpad=-10,**font1)
        # ax.set_zlabel('z [m]',labelpad=-8,**font1)
        # ax.tick_params(axis='x',which='major',pad=-5)
        # ax.tick_params(axis='y',which='major',pad=-5)
        # ax.tick_params(axis='z',which='major',pad=-3)
        # for t in ax.xaxis.get_major_ticks(): 
        #     t.label.set_font('Times New Roman') 
        #     t.label.set_fontsize(7)
        # for t in ax.yaxis.get_major_ticks(): 
        #     t.label.set_font('Times New Roman') 
        #     t.label.set_fontsize(7)
        # for t in ax.zaxis.get_major_ticks(): 
        #     t.label.set_font('Times New Roman') 
        #     t.label.set_fontsize(7)
        # ax.view_init(30,60)
        # ax.xaxis._axinfo["grid"].update({"linewidth":0.5,"linestyle":'--'})
        # ax.yaxis._axinfo["grid"].update({"linewidth":0.5,"linestyle":'--'})
        # ax.zaxis._axinfo["grid"].update({"linewidth":0.5,"linestyle":'--'})
        # plt.savefig('DMHE/tracking_DMHE.png')
        # plt.show()
        
        # load the computed total force command
        f_cmd = dataframe_b['current.vx']
        
        # load the DMHE estimation data in NED frame
        dfx_mhe = dataframe_b['current.a_x']
        dfy_mhe = dataframe_b['current.a_y']
        dfz_mhe = dataframe_b['current.a_z']
        Df_Imhe_enu = np.zeros((4,len(dfx_mhe)))
        for k in range(len(dfx_mhe)):
            Dfmhe_ned = np.array([[dfx_mhe[k],dfy_mhe[k],dfz_mhe[k]]]).T
            Df_Imhe_enu[0:3,k:k+1] = np.matmul(R_enu,Dfmhe_ned)
            Df_Imhe_enu[3,k:k+1]   = LA.norm(np.matmul(R_enu,Dfmhe_ned))

        np.save('Df_Imhe_dmhe',Df_Imhe_enu[:,n_start:len(g_r_dmhe)])
        # load the accelerometer data in NED frame
        acc_x   = dataframe_c['accelerometer_m_s2[0]']
        acc_y   = dataframe_c['accelerometer_m_s2[1]']
        acc_z   = dataframe_c['accelerometer_m_s2[2]']
        Acc_b_ned   = np.zeros((3,len(acc_x)))
        for k in range(len(acc_x)):
            Acc_b_ned[:,k:k+1] = np.array([[acc_x[k],acc_y[k],acc_z[k]]]).T
        # load the quaternion
        Q0 = dataframe_d['q[0]']
        Q1 = dataframe_d['q[1]']
        Q2 = dataframe_d['q[2]']
        Q3 = dataframe_d['q[3]']
        
        Gt = np.zeros((3,len(dfx_mhe)-n_start))
        Gt_lpf = np.zeros((4,len(dfx_mhe)-n_start))
        Gamma  = np.zeros((2,len(dfx_mhe)-n_start))
        # print('len(ref_x)=',len(Ref_x),'len(acc_x)=',len(acc_x),'len(dfx_mhe)=',len(dfx_mhe),'len(position_x)=',len(position_x),'len(f_cmd)=',len(f_cmd))
        acc_I_prev = np.array([[0,0,-9.78]]).T # initial acceleration of the low-pass filter in NED frame
        Time       = []
        time       = 0
        i = 0
        # calculation of the ground truth data using the two steps mentioned above
        for k in range(n_start,len(dfx_mhe),1): 
            Time += [time]
            q0, q1, q2, q3 = Q0[k], Q1[k], Q2[k], Q3[k]
            quaternion = np.array([q0, q1, q2, q3])
            R_b        = Quaternion2Rotation(quaternion)
            acc_I_ned = np.matmul(R_b,np.reshape(Acc_b_ned[:,k],(3,1)))
            f_cmdI_ned = np.matmul(R_b,(f_cmd[k]*z))
            gt        = acc_I_ned*mass+f_cmdI_ned # we do not plot the raw accelerometer data as it is too noisy
            Gt[:,i:i+1] =  np.matmul(R_enu,gt) # transfer to ENU frame
            time += dt
            i += 1
        Gt_lpf[0,:] = lfilter(taps, 1.0, Gt[0,:])
        Gt_lpf[1,:] = lfilter(taps, 1.0, Gt[1,:])
        Gt_lpf[2,:] = lfilter(taps, 1.0, Gt[2,:])
        for k in range(len(Gt_lpf[0,:])):
            Gt_lpf[3,k:k+1]=LA.norm(Gt_lpf[0:3,k])
        
        np.save('Gt_lpf_dmhe',Gt_lpf)

        f_tension = dataframe_e['force']
        Tension = np.zeros((1,len(dfx_mhe)-n_start))
        print('len_tension=',len(f_tension))
        i = 0
        n_t_start = 860 # We started the tension force recording at 11:44:49, sent the control command at 11:45:06
        for k in range(n_t_start,len(g_r_dmhe)-n_start+n_t_start,1):
            Tension[:,i:i+1] = f_tension[k]
            i += 1

        Tension_3d_enu = np.zeros((3,len(dfx_mhe)-n_start))
        for k in range(len(dfx_mhe)-n_start): # project the tension in ENU frame
            Tension_3d_enu[:,k:k+1]=-Tension[:,k]*(position_enu[0:3,k:k+1]-offset)/LA.norm(position_enu[0:3,k:k+1]-offset)
        
        np.save('Tension_dmhe',Tension_3d_enu)

        rmse_px = format(mean_squared_error(ref_enu[0,:], position_enu[0,:], squared=False),'.3f')
        rmse_py = format(mean_squared_error(ref_enu[1,:], position_enu[1,:], squared=False),'.3f')
        rmse_pz = format(mean_squared_error(ref_enu[2,:], position_enu[2,:], squared=False),'.3f')
        rmse_p  = format(mean_squared_error(ref_enu[3,:], position_enu[3,:],squared=False),'.3f')
        rmse_fx = format(mean_squared_error(Gt_lpf[0,:], Df_Imhe_enu[0,n_start:len(g_r_dmhe)], squared=False),'.3f')
        rmse_fy = format(mean_squared_error(Gt_lpf[1,:], Df_Imhe_enu[1,n_start:len(g_r_dmhe)], squared=False),'.3f')
        rmse_fz = format(mean_squared_error(Gt_lpf[2,:], Df_Imhe_enu[2,n_start:len(g_r_dmhe)], squared=False),'.3f')
        rmse_f  = format(mean_squared_error(Gt_lpf[3,:], Df_Imhe_enu[3,n_start:len(g_r_dmhe)], squared=False),'.3f')
        print('rmse_px=',rmse_px,'rmse_py=',rmse_py,'rmse_pz=',rmse_pz,'rmse_p=',rmse_p)
        print('rmse_fx=',rmse_fx,'rmse_fy=',rmse_fy,'rmse_fz=',rmse_fz,'rmse_f=',rmse_f)
        

        # fig, (ax1,ax2,ax3) = plt.subplots(3,sharex=True,figsize=(6/cm_2_inch,6*0.8/cm_2_inch),dpi=600) # 8,8*0.5
        # ax1.plot(Time,Gt_lpf[0,:], linewidth=0.5,color='black')
        # ax1.plot(Time,Df_Imhe_enu[0,n_start:len(dfx_mhe)], linewidth=0.5,color='orange')
        # ax1.plot(Time,Tension_3d_enu[0,:], linewidth=0.5,color='blue')
        # ax2.plot(Time,Gt_lpf[1,:], linewidth=0.5,color='black')
        # ax2.plot(Time,Df_Imhe_enu[1,n_start:len(dfx_mhe)], linewidth=0.5,color='orange')
        # ax2.plot(Time,Tension_3d_enu[1,:], linewidth=0.5,color='blue')
        # ax3.plot(Time,Gt_lpf[2,:], linewidth=0.5,color='black')
        # ax3.plot(Time,Df_Imhe_enu[2,n_start:len(dfx_mhe)], linewidth=0.5,color='orange')
        # ax3.plot(Time,Tension_3d_enu[2,:], linewidth=0.5,color='blue')
        # ax1.set_ylabel('$d_{x}$ [N]',labelpad=0,**font1)
        # ax2.set_ylabel('$d_{y}$ [N]',labelpad=0,**font1)
        # ax3.set_ylabel('$d_{z}$ [N]',labelpad=-0.5,**font1)
        # ax3.set_xlabel('Time [s]',labelpad=-1.5,**font1)
        # ax1.tick_params(axis='y',which='major',pad=-0.25)
        # ax2.tick_params(axis='y',which='major',pad=-0.25)
        # ax3.tick_params(axis='x',which='major',pad=-1)
        # ax3.tick_params(axis='y',which='major',pad=-0.25)
        # ax3.set_xticks(np.arange(0,35,10))
        # for t in ax1.yaxis.get_major_ticks(): 
        #     t.label.set_font('Times New Roman') 
        #     t.label.set_fontsize(7)
        # for t in ax2.yaxis.get_major_ticks(): 
        #     t.label.set_font('Times New Roman') 
        #     t.label.set_fontsize(7)
        # for t in ax3.xaxis.get_major_ticks(): 
        #     t.label.set_font('Times New Roman') 
        #     t.label.set_fontsize(7)
        # for t in ax3.yaxis.get_major_ticks(): 
        #     t.label.set_font('Times New Roman') 
        #     t.label.set_fontsize(7)
        # # leg=ax1.legend(['Ground truth', 'Estimation', 'Tension force'],loc='center',prop=font1,labelspacing=0.1)
        # # leg.get_frame().set_linewidth(0.5)
        # ax1.grid(b=True,axis='both',linestyle='--', linewidth=0.5)
        # ax2.grid(b=True,axis='both',linestyle='--', linewidth=0.5)
        # ax3.grid(b=True,axis='both',linestyle='--', linewidth=0.5)
        # for axis in ['top', 'bottom', 'left', 'right']:
        #     ax1.spines[axis].set_linewidth(0.5)
        # for axis in ['top', 'bottom', 'left', 'right']:
        #     ax2.spines[axis].set_linewidth(0.5)
        # for axis in ['top', 'bottom', 'left', 'right']:
        #     ax3.spines[axis].set_linewidth(0.5)
        # plt.savefig('DMHE/force_DMHE_h.png')
        # plt.show()

    elif controller == 'c':
        dataset={'a':"L1-AC/log_17_2023-2-22-18-14-06_vehicle_local_position_0.csv",
             'b':"L1-AC/log_17_2023-2-22-18-14-06_position_setpoint_triplet_0.csv",
             'c':"L1-AC/log_17_2023-2-22-18-14-06_sensor_combined_0.csv",
             'd':"L1-AC/log_17_2023-2-22-18-14-06_vehicle_attitude_0.csv",
             'e':"DMHE/log_11_2023-2-22-11-44-46_position_setpoint_triplet_0.csv"}

        # dataset={'a':"L1-AC/log_12_2023-2-22-11-49-14_vehicle_local_position_0.csv",
        #      'b':"L1-AC/log_12_2023-2-22-11-49-14_position_setpoint_triplet_0.csv",
        #      'c':"L1-AC/log_12_2023-2-22-11-49-14_sensor_combined_0.csv",
        #      'd':"L1-AC/log_12_2023-2-22-11-49-14_vehicle_attitude_0.csv",
        #      'e':"DMHE/log_11_2023-2-22-11-44-46_position_setpoint_triplet_0.csv"}
        tensionset = {'a':"22_Feb_2023_Tension_data/22-Feb-2023-Exp1-controller_c_correct.csv"}
        # tensionset = {'a':"22_Feb_2023_Tension_data/22-Feb-2023-Exp1-controller_c.csv"}
        datatension = pd.read_csv(tensionset['a'],header=None, names=['world time','dt','force'])
        local_position = pd.read_csv(dataset['a'])
        position_setpoint = pd.read_csv(dataset['b'])
        sensor         = pd.read_csv(dataset['c'])
        attitude       = pd.read_csv(dataset['d'])
        data_dmhe      = pd.read_csv(dataset['e'])
        dataframe_a    = pd.DataFrame(local_position)
        dataframe_b    = pd.DataFrame(position_setpoint)
        dataframe_c    = pd.DataFrame(sensor)
        dataframe_d    = pd.DataFrame(attitude)
        dataframe_e    = pd.DataFrame(datatension)
        dataframe_dmhe = pd.DataFrame(data_dmhe)
        # load the position data in NED frame
        position_x     = dataframe_a['x']
        position_y     = dataframe_a['y']
        position_z     = dataframe_a['z']
        g_r_dmhe = dataframe_dmhe['current.vy']
        

        n_start = 143 # During the initial interval n_start*dt, the quadrotor is still on the ground and the estimator does not begin to work
        # n_start = 208 
        i = 0
        j = 0
        Time1       = []
        time       = 0
        position_enu   = np.zeros((4,len(g_r_dmhe)-n_start)) # the z positions in the last 160 data points are negative, which is due in part to inaccurate Vicon record when the quadrotor hit the ground
        ref_enu        = np.zeros((4,len(g_r_dmhe)-n_start))
        # ref_enu_for_rmse_a3 = np.zeros((3,len(Ref_x)))
        t_switch = 0

        for k in range(n_start,len(g_r_dmhe),1):
            Time1 += [time]
            pos_ned      = np.array([[position_x[k],position_y[k],position_z[k]]]).T
            pos_enu      = np.matmul(R_enu,pos_ned) # transfer to ENU frame
            position_enu[0:3,j:j+1] = pos_enu
            position_enu[3,j:j+1] = LA.norm(pos_enu)
            if time <T_end:
                ref_p, ref_v, ref_a, ref_j, ref_s = uav.ref_exp1(Coeffx_exp1, Coeffy_exp1, Coeffz_exp1, time, t_switch) 
            else:
                ref_p, ref_v, ref_a, ref_j, ref_s = np.array([[1,-1,0.12]]).T, np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1))
            ref_enu[0:3,j:j+1]=ref_p
            ref_enu[3,j:j+1]=LA.norm(ref_p)
            time += dt
            j += 1
        np.save('ref_enu_l1',ref_enu)
        np.save('pos_enu_l1',position_enu)
        np.save('Time_track_l1',Time1)

        # np.save('16-Jan-data-exp1(cable)/position_enu_c',position_enu)
        # fig, (ax1,ax2,ax3) = plt.subplots(3, sharex=True,figsize=(8/cm_2_inch,8/cm_2_inch),dpi=600)
        # ax1.plot(Time1,position_enu[0,:],linewidth=0.5)
        # ax1.plot(Time1,ref_enu[0,:],linewidth=0.5)
        # ax2.plot(Time1,position_enu[1,:],linewidth=0.5)
        # ax2.plot(Time1,ref_enu[1,:],linewidth=0.5)
        # ax3.plot(Time1,position_enu[2,:],linewidth=0.5)
        # ax3.plot(Time1,ref_enu[2,:],linewidth=0.5)
        # # leg=ax1.legend(['Actual', 'Desired'],loc='lower center',prop=font1)
        # # leg.get_frame().set_linewidth(0.5)
        # ax1.set_ylabel('$p_{x}$ [m]',labelpad=0,**font1)
        # ax2.set_ylabel('$p_{y}$ [m]',labelpad=0,**font1)
        # ax3.set_ylabel('$p_{z}$ [m]',labelpad=0,**font1)
        # ax1.grid(b=True,axis='both',linestyle='--', linewidth=0.5)
        # ax2.grid(b=True,axis='both',linestyle='--', linewidth=0.5)
        # ax3.grid(b=True,axis='both',linestyle='--', linewidth=0.5)
        # for axis in ['top', 'bottom', 'left', 'right']:
        #     ax1.spines[axis].set_linewidth(0.5)
        # for axis in ['top', 'bottom', 'left', 'right']:
        #     ax2.spines[axis].set_linewidth(0.5)
        # for axis in ['top', 'bottom', 'left', 'right']:
        #     ax3.spines[axis].set_linewidth(0.5)
        # plt.savefig('L1-AC/position-axes-L1.png')
        # plt.show()

        # plt.figure(2,figsize=(4.5/cm_2_inch,4/cm_2_inch),dpi=600)
        # ax = plt.axes(projection="3d")
        # ax.plot3D(ref_enu[0,:], ref_enu[1,:], ref_enu[2,:], linewidth=1.5, linestyle='--',color='black')
        # ax.plot3D(position_enu[0,:], position_enu[1,:], position_enu[2,:], linewidth=1,color='orange')
        # # leg=plt.legend(['Desired','Actual'],prop=font1,loc=(0.5,0.2),labelspacing = 0.1)
        # # leg.get_frame().set_linewidth(0.5)
        # ax.set_xlabel('x [m]',labelpad=-10,**font1)
        # ax.set_ylabel('y [m]',labelpad=-10,**font1)
        # ax.set_zlabel('z [m]',labelpad=-8,**font1)
        # ax.tick_params(axis='x',which='major',pad=-5)
        # ax.tick_params(axis='y',which='major',pad=-5)
        # ax.tick_params(axis='z',which='major',pad=-3)
        # for t in ax.xaxis.get_major_ticks(): 
        #     t.label.set_font('Times New Roman') 
        #     t.label.set_fontsize(7)
        # for t in ax.yaxis.get_major_ticks(): 
        #     t.label.set_font('Times New Roman') 
        #     t.label.set_fontsize(7)
        # for t in ax.zaxis.get_major_ticks(): 
        #     t.label.set_font('Times New Roman') 
        #     t.label.set_fontsize(7)
        # ax.view_init(30,60)
        # ax.xaxis._axinfo["grid"].update({"linewidth":0.5,"linestyle":'--'})
        # ax.yaxis._axinfo["grid"].update({"linewidth":0.5,"linestyle":'--'})
        # ax.zaxis._axinfo["grid"].update({"linewidth":0.5,"linestyle":'--'})
        # plt.savefig('L1-AC/tracking_L1AC.png')
        # plt.show()
        # load the computed total force command
        f_cmd = dataframe_b['current.vx']
        # load the L1 estimation data in NED frame
        dfx_mhe = dataframe_b['current.a_x']
        dfy_mhe = dataframe_b['current.a_y']
        dfz_mhe = dataframe_b['current.a_z']
        Df_Imhe_enu = np.zeros((4,len(g_r_dmhe)))
        for k in range(len(g_r_dmhe)):
            Dfmhe_ned = np.array([[dfx_mhe[k],dfy_mhe[k],dfz_mhe[k]]]).T
            Df_Imhe_enu[0:3,k:k+1] = np.matmul(R_enu,Dfmhe_ned)
            Df_Imhe_enu[3,k:k+1]   = LA.norm(np.matmul(R_enu,Dfmhe_ned))
        
        np.save('Df_Imhe_l1',Df_Imhe_enu[:,n_start:len(g_r_dmhe)])
        # load the accelerometer data in NED frame
        acc_x   = dataframe_c['accelerometer_m_s2[0]']
        acc_y   = dataframe_c['accelerometer_m_s2[1]']
        acc_z   = dataframe_c['accelerometer_m_s2[2]']
        Acc_b_ned   = np.zeros((3,len(acc_x)))
        for k in range(len(acc_x)):
            Acc_b_ned[:,k:k+1] = np.array([[acc_x[k],acc_y[k],acc_z[k]]]).T
        # load the quaternion
        Q0 = dataframe_d['q[0]']
        Q1 = dataframe_d['q[1]']
        Q2 = dataframe_d['q[2]']
        Q3 = dataframe_d['q[3]']
        
        Gt = np.zeros((3,len(g_r_dmhe)-n_start))
        Gt_lpf = np.zeros((4,len(g_r_dmhe)-n_start))
        Gamma  = np.zeros((2,len(g_r_dmhe)-n_start))
        # print('len(ref_x)=',len(Ref_x),'len(acc_x)=',len(acc_x),'len(dfx_mhe)=',len(dfx_mhe),'len(position_x)=',len(position_x),'len(f_cmd)=',len(f_cmd))
        acc_I_prev = np.array([[0,0,-9.78]]).T # initial acceleration of the low-pass filter in NED frame
        Time       = []
        time       = 0
        i = 0
        # calculation of the ground truth data using the two steps mentioned above
        for k in range(n_start,len(g_r_dmhe),1): 
            Time += [time]
            q0, q1, q2, q3 = Q0[k], Q1[k], Q2[k], Q3[k]
            quaternion = np.array([q0, q1, q2, q3])
            R_b        = Quaternion2Rotation(quaternion)
            acc_I_ned = np.matmul(R_b,np.reshape(Acc_b_ned[:,k],(3,1)))
            f_cmdI_ned = np.matmul(R_b,(f_cmd[k]*z))
            gt        = acc_I_ned*mass+f_cmdI_ned # we do not plot the raw accelerometer data as it is too noisy
            Gt[:,i:i+1] =  np.matmul(R_enu,gt) # transfer to ENU frame
            time += dt
            i += 1
        Gt_lpf[0,:] = lfilter(taps, 1.0, Gt[0,:])
        Gt_lpf[1,:] = lfilter(taps, 1.0, Gt[1,:])
        Gt_lpf[2,:] = lfilter(taps, 1.0, Gt[2,:])
        for k in range(len(Gt_lpf[0,:])):
            Gt_lpf[3,k:k+1]=LA.norm(Gt_lpf[0:3,k])
        
        np.save('Gt_lpf_l1',Gt_lpf)
        f_tension = dataframe_e['force']
        Tension = np.zeros((1,len(g_r_dmhe)-n_start))
        print('len_tension=',len(f_tension))
        i = 0
        n_t_start = 3500 # We started the tension force recording at 18:12:19, sent the control command at 18:13:29
        # n_t_start = 774
        for k in range(n_t_start,len(g_r_dmhe)-n_start+n_t_start,1):
            Tension[:,i:i+1] = f_tension[k]
            i += 1
        Tension_3d_enu = np.zeros((3,len(g_r_dmhe)-n_start))
        for k in range(len(g_r_dmhe)-n_start): # project the tension in ENU frame
            Tension_3d_enu[:,k:k+1]=-Tension[:,k]*(position_enu[0:3,k:k+1]-offset)/LA.norm(position_enu[0:3,k:k+1]-offset)
        np.save('Tension_l1',Tension_3d_enu)

        rmse_px = format(mean_squared_error(ref_enu[0,:], position_enu[0,:], squared=False),'.3f')
        rmse_py = format(mean_squared_error(ref_enu[1,:], position_enu[1,:], squared=False),'.3f')
        rmse_pz = format(mean_squared_error(ref_enu[2,:], position_enu[2,:], squared=False),'.3f')
        rmse_p  = format(mean_squared_error(ref_enu[3,:], position_enu[3,:],squared=False),'.3f')
        rmse_fx = format(mean_squared_error(Gt_lpf[0,:], Df_Imhe_enu[0,n_start:len(g_r_dmhe)], squared=False),'.3f')
        rmse_fy = format(mean_squared_error(Gt_lpf[1,:], Df_Imhe_enu[1,n_start:len(g_r_dmhe)], squared=False),'.3f')
        rmse_fz = format(mean_squared_error(Gt_lpf[2,:], Df_Imhe_enu[2,n_start:len(g_r_dmhe)], squared=False),'.3f')
        rmse_f  = format(mean_squared_error(Gt_lpf[3,:], Df_Imhe_enu[3,n_start:len(g_r_dmhe)], squared=False),'.3f')

        print('rmse_px=',rmse_px,'rmse_py=',rmse_py,'rmse_pz=',rmse_pz,'rmse_p=',rmse_p)
        print('rmse_fx=',rmse_fx,'rmse_fy=',rmse_fy,'rmse_fz=',rmse_fz,'rmse_f=',rmse_f)

        # fig, (ax1,ax2,ax3) = plt.subplots(3,sharex=True,figsize=(6/cm_2_inch,6*0.8/cm_2_inch),dpi=600)
        # ax1.plot(Time,Gt_lpf[0,:], linewidth=0.5,color='black')
        # ax1.plot(Time,Df_Imhe_enu[0,n_start:len(dfx_mhe)], linewidth=0.5,color='orange')
        # ax1.plot(Time,Tension_3d_enu[0,:], linewidth=0.5,color='blue')
        # ax2.plot(Time,Gt_lpf[1,:], linewidth=0.5,color='black')
        # ax2.plot(Time,Df_Imhe_enu[1,n_start:len(dfx_mhe)], linewidth=0.5,color='orange')
        # ax2.plot(Time,Tension_3d_enu[1,:], linewidth=0.5,color='blue')
        # ax3.plot(Time,Gt_lpf[2,:], linewidth=0.5,color='black')
        # ax3.plot(Time,Df_Imhe_enu[2,n_start:len(dfx_mhe)], linewidth=0.5,color='orange')
        # ax3.plot(Time,Tension_3d_enu[2,:], linewidth=0.5,color='blue')
        # ax1.set_ylabel('$d_{x}$ [N]',labelpad=0,**font1)
        # ax2.set_ylabel('$d_{y}$ [N]',labelpad=0,**font1)
        # ax3.set_ylabel('$d_{z}$ [N]',labelpad=-0.5,**font1)
        # ax3.set_xlabel('Time [s]',labelpad=-1.5,**font1)
        # ax1.tick_params(axis='y',which='major',pad=-0.25)
        # ax2.tick_params(axis='y',which='major',pad=-0.25)
        # ax3.tick_params(axis='x',which='major',pad=-1)
        # ax3.tick_params(axis='y',which='major',pad=-0.25)
        # ax3.set_xticks(np.arange(0,35,10))
        # for t in ax1.yaxis.get_major_ticks(): 
        #     t.label.set_font('Times New Roman') 
        #     t.label.set_fontsize(7)
        # for t in ax2.yaxis.get_major_ticks(): 
        #     t.label.set_font('Times New Roman') 
        #     t.label.set_fontsize(7)
        # for t in ax3.xaxis.get_major_ticks(): 
        #     t.label.set_font('Times New Roman') 
        #     t.label.set_fontsize(7)
        # for t in ax3.yaxis.get_major_ticks(): 
        #     t.label.set_font('Times New Roman') 
        #     t.label.set_fontsize(7)
        # # leg=ax1.legend(['Ground truth', 'Estimation', 'Tension force'],loc='center',prop=font1,labelspacing=0.1)
        # # leg.get_frame().set_linewidth(0.5)
        # ax1.grid(b=True,axis='both',linestyle='--', linewidth=0.5)
        # ax2.grid(b=True,axis='both',linestyle='--', linewidth=0.5)
        # ax3.grid(b=True,axis='both',linestyle='--', linewidth=0.5)
        # for axis in ['top', 'bottom', 'left', 'right']:
        #     ax1.spines[axis].set_linewidth(0.5)
        # for axis in ['top', 'bottom', 'left', 'right']:
        #     ax2.spines[axis].set_linewidth(0.5)
        # for axis in ['top', 'bottom', 'left', 'right']:
        #     ax3.spines[axis].set_linewidth(0.5)
        # plt.savefig('L1-AC/force-L1AC_h.png')
        # plt.show()
    
    # elif controller == 'd':
    #     dataset={'a':"UKF/log_13_2023-2-22-11-51-32_vehicle_local_position_0.csv",
    #          'b':"UKF/log_13_2023-2-22-11-51-32_position_setpoint_triplet_0.csv",
    #          'c':"UKF/log_13_2023-2-22-11-51-32_sensor_combined_0.csv",
    #          'd':"UKF/log_13_2023-2-22-11-51-32_vehicle_attitude_0.csv",
    #          'e':"DMHE/log_11_2023-2-22-11-44-46_position_setpoint_triplet_0.csv"}
    #     tensionset = {'a':"22_Feb_2023_Tension_data/22-Feb-2023-Exp1-controller_d.csv"}
    #     datatension = pd.read_csv(tensionset['a'],header=None, names=['world time','dt','force'])
    #     local_position = pd.read_csv(dataset['a'])
    #     position_setpoint = pd.read_csv(dataset['b'])
    #     sensor         = pd.read_csv(dataset['c'])
    #     attitude       = pd.read_csv(dataset['d'])
    #     data_dmhe      = pd.read_csv(dataset['e'])
    #     dataframe_a    = pd.DataFrame(local_position)
    #     dataframe_b    = pd.DataFrame(position_setpoint)
    #     dataframe_c    = pd.DataFrame(sensor)
    #     dataframe_d    = pd.DataFrame(attitude)
    #     dataframe_e    = pd.DataFrame(datatension)
    #     dataframe_dmhe = pd.DataFrame(data_dmhe)
    #     # load the position data in NED frame
    #     position_x     = dataframe_a['x']
    #     position_y     = dataframe_a['y']
    #     position_z     = dataframe_a['z']
    #     g_r_dmhe = dataframe_dmhe['current.vy']
        

    #     n_start = 143 # During the initial interval n_start*dt, the quadrotor is still on the ground and the estimator does not begin to work
    #     i = 0
    #     j = 0
    #     Time1       = []
    #     time       = 0
    #     position_enu   = np.zeros((4,len(g_r_dmhe)-n_start)) # the z positions in the last 160 data points are negative, which is due in part to inaccurate Vicon record when the quadrotor hit the ground
    #     ref_enu        = np.zeros((4,len(g_r_dmhe)-n_start))
    #     # ref_enu_for_rmse_a3 = np.zeros((3,len(Ref_x)))
    #     t_switch = 0

    #     for k in range(n_start,len(g_r_dmhe),1):
    #         Time1 += [time]
    #         pos_ned      = np.array([[position_x[k],position_y[k],position_z[k]]]).T
    #         pos_enu      = np.matmul(R_enu,pos_ned) # transfer to ENU frame
    #         position_enu[0:3,j:j+1] = pos_enu
    #         position_enu[3,j:j+1] = LA.norm(pos_enu)
    #         if time <T_end:
    #             ref_p, ref_v, ref_a, ref_j, ref_s = uav.ref_exp1(Coeffx_exp1, Coeffy_exp1, Coeffz_exp1, time, t_switch) 
    #         else:
    #             ref_p, ref_v, ref_a, ref_j, ref_s = np.array([[1,-1,0.12]]).T, np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1))
    #         ref_enu[0:3,j:j+1]=ref_p
    #         ref_enu[3,j:j+1]=LA.norm(ref_p)
    #         time += dt
    #         j += 1

    #     # np.save('16-Jan-data-exp1(cable)/position_enu_c',position_enu)
    #     fig, (ax1,ax2,ax3) = plt.subplots(3, sharex=True,figsize=(8/cm_2_inch,8/cm_2_inch),dpi=600)
    #     ax1.plot(Time1,position_enu[0,:],linewidth=0.5)
    #     ax1.plot(Time1,ref_enu[0,:],linewidth=0.5)
    #     ax2.plot(Time1,position_enu[1,:],linewidth=0.5)
    #     ax2.plot(Time1,ref_enu[1,:],linewidth=0.5)
    #     ax3.plot(Time1,position_enu[2,:],linewidth=0.5)
    #     ax3.plot(Time1,ref_enu[2,:],linewidth=0.5)
    #     # leg=ax1.legend(['Actual', 'Desired'],loc='lower center',prop=font1)
    #     # leg.get_frame().set_linewidth(0.5)
    #     ax1.set_ylabel('$p_{x}$ [m]',labelpad=0,**font1)
    #     ax2.set_ylabel('$p_{y}$ [m]',labelpad=0,**font1)
    #     ax3.set_ylabel('$p_{z}$ [m]',labelpad=0,**font1)
    #     ax1.grid(b=True,axis='both',linestyle='--', linewidth=0.5)
    #     ax2.grid(b=True,axis='both',linestyle='--', linewidth=0.5)
    #     ax3.grid(b=True,axis='both',linestyle='--', linewidth=0.5)
    #     for axis in ['top', 'bottom', 'left', 'right']:
    #         ax1.spines[axis].set_linewidth(0.5)
    #     for axis in ['top', 'bottom', 'left', 'right']:
    #         ax2.spines[axis].set_linewidth(0.5)
    #     for axis in ['top', 'bottom', 'left', 'right']:
    #         ax3.spines[axis].set_linewidth(0.5)
    #     plt.savefig('L1-AC/position-axes-L1.png')
    #     plt.show()

    #     plt.figure(2,figsize=(4.5/cm_2_inch,4/cm_2_inch),dpi=600)
    #     ax = plt.axes(projection="3d")
    #     ax.plot3D(ref_enu[0,:], ref_enu[1,:], ref_enu[2,:], linewidth=1.5, linestyle='--',color='black')
    #     ax.plot3D(position_enu[0,:], position_enu[1,:], position_enu[2,:], linewidth=1,color='orange')
    #     # leg=plt.legend(['Desired','Actual'],prop=font1,loc=(0.5,0.2),labelspacing = 0.1)
    #     # leg.get_frame().set_linewidth(0.5)
    #     ax.set_xlabel('x [m]',labelpad=-10,**font1)
    #     ax.set_ylabel('y [m]',labelpad=-10,**font1)
    #     ax.set_zlabel('z [m]',labelpad=-8,**font1)
    #     ax.tick_params(axis='x',which='major',pad=-5)
    #     ax.tick_params(axis='y',which='major',pad=-5)
    #     ax.tick_params(axis='z',which='major',pad=-3)
    #     for t in ax.xaxis.get_major_ticks(): 
    #         t.label.set_font('Times New Roman') 
    #         t.label.set_fontsize(7)
    #     for t in ax.yaxis.get_major_ticks(): 
    #         t.label.set_font('Times New Roman') 
    #         t.label.set_fontsize(7)
    #     for t in ax.zaxis.get_major_ticks(): 
    #         t.label.set_font('Times New Roman') 
    #         t.label.set_fontsize(7)
    #     ax.view_init(0,45)
    #     ax.xaxis._axinfo["grid"].update({"linewidth":0.5,"linestyle":'--'})
    #     ax.yaxis._axinfo["grid"].update({"linewidth":0.5,"linestyle":'--'})
    #     ax.zaxis._axinfo["grid"].update({"linewidth":0.5,"linestyle":'--'})
    #     plt.savefig('L1-AC/tracking_L1AC-side2.png')
    #     plt.show()
    #     # load the computed total force command
    #     f_cmd = dataframe_b['current.vx']
    #     # load the L1 estimation data in NED frame
    #     dfx_mhe = dataframe_b['current.a_x']
    #     dfy_mhe = dataframe_b['current.a_y']
    #     dfz_mhe = dataframe_b['current.a_z']
    #     Df_Imhe_enu = np.zeros((4,len(g_r_dmhe)))
    #     for k in range(len(g_r_dmhe)):
    #         Dfmhe_ned = np.array([[dfx_mhe[k],dfy_mhe[k],dfz_mhe[k]]]).T
    #         Df_Imhe_enu[0:3,k:k+1] = np.matmul(R_enu,Dfmhe_ned)
    #         Df_Imhe_enu[3,k:k+1]   = LA.norm(np.matmul(R_enu,Dfmhe_ned))
    #     # load the accelerometer data in NED frame
    #     acc_x   = dataframe_c['accelerometer_m_s2[0]']
    #     acc_y   = dataframe_c['accelerometer_m_s2[1]']
    #     acc_z   = dataframe_c['accelerometer_m_s2[2]']
    #     Acc_b_ned   = np.zeros((3,len(acc_x)))
    #     for k in range(len(acc_x)):
    #         Acc_b_ned[:,k:k+1] = np.array([[acc_x[k],acc_y[k],acc_z[k]]]).T
    #     # load the quaternion
    #     Q0 = dataframe_d['q[0]']
    #     Q1 = dataframe_d['q[1]']
    #     Q2 = dataframe_d['q[2]']
    #     Q3 = dataframe_d['q[3]']
        
    #     Gt = np.zeros((3,len(g_r_dmhe)-n_start))
    #     Gt_lpf = np.zeros((4,len(g_r_dmhe)-n_start))
    #     Gamma  = np.zeros((2,len(g_r_dmhe)-n_start))
    #     # print('len(ref_x)=',len(Ref_x),'len(acc_x)=',len(acc_x),'len(dfx_mhe)=',len(dfx_mhe),'len(position_x)=',len(position_x),'len(f_cmd)=',len(f_cmd))
    #     acc_I_prev = np.array([[0,0,-9.78]]).T # initial acceleration of the low-pass filter in NED frame
    #     Time       = []
    #     time       = 0
    #     i = 0
    #     # calculation of the ground truth data using the two steps mentioned above
    #     for k in range(n_start,len(g_r_dmhe),1): 
    #         Time += [time]
    #         q0, q1, q2, q3 = Q0[k], Q1[k], Q2[k], Q3[k]
    #         quaternion = np.array([q0, q1, q2, q3])
    #         R_b        = Quaternion2Rotation(quaternion)
    #         acc_I_ned = np.matmul(R_b,np.reshape(Acc_b_ned[:,k],(3,1)))
    #         f_cmdI_ned = np.matmul(R_b,(f_cmd[k]*z))
    #         gt        = acc_I_ned*mass+f_cmdI_ned # we do not plot the raw accelerometer data as it is too noisy
    #         Gt[:,i:i+1] =  np.matmul(R_enu,gt) # transfer to ENU frame
    #         time += dt
    #         i += 1
    #     Gt_lpf[0,:] = lfilter(taps, 1.0, Gt[0,:])
    #     Gt_lpf[1,:] = lfilter(taps, 1.0, Gt[1,:])
    #     Gt_lpf[2,:] = lfilter(taps, 1.0, Gt[2,:])
    #     for k in range(len(Gt_lpf[0,:])):
    #         Gt_lpf[3,k:k+1]=LA.norm(Gt_lpf[0:3,k])

    #     f_tension = dataframe_e['force']
    #     Tension = np.zeros((1,len(g_r_dmhe)-n_start))
    #     print('len_tension=',len(f_tension))
    #     i = 0
    #     n_t_start = 3500 # We started the tension force recording at 18:12:19, sent the control command at 18:13:29
        
    #     for k in range(n_t_start,len(g_r_dmhe)-n_start+n_t_start,1):
    #         Tension[:,i:i+1] = f_tension[k]
    #         i += 1
    #     Tension_3d_enu = np.zeros((3,len(g_r_dmhe)-n_start))
    #     for k in range(len(g_r_dmhe)-n_start): # project the tension in ENU frame
    #         Tension_3d_enu[:,k:k+1]=-Tension[:,k]*(position_enu[0:3,k:k+1]-offset)/LA.norm(position_enu[0:3,k:k+1]-offset)

    #     rmse_px = format(mean_squared_error(ref_enu[0,:], position_enu[0,:], squared=False),'.3f')
    #     rmse_py = format(mean_squared_error(ref_enu[1,:], position_enu[1,:], squared=False),'.3f')
    #     rmse_pz = format(mean_squared_error(ref_enu[2,:], position_enu[2,:], squared=False),'.3f')
    #     rmse_p  = format(mean_squared_error(ref_enu[3,:], position_enu[3,:],squared=False),'.3f')
    #     rmse_fx = format(mean_squared_error(Gt_lpf[0,:], Df_Imhe_enu[0,n_start:len(g_r_dmhe)], squared=False),'.3f')
    #     rmse_fy = format(mean_squared_error(Gt_lpf[1,:], Df_Imhe_enu[1,n_start:len(g_r_dmhe)], squared=False),'.3f')
    #     rmse_fz = format(mean_squared_error(Gt_lpf[2,:], Df_Imhe_enu[2,n_start:len(g_r_dmhe)], squared=False),'.3f')
    #     rmse_f  = format(mean_squared_error(Gt_lpf[3,:], Df_Imhe_enu[3,n_start:len(g_r_dmhe)], squared=False),'.3f')

    #     print('rmse_px=',rmse_px,'rmse_py=',rmse_py,'rmse_pz=',rmse_pz,'rmse_p=',rmse_p)
    #     print('rmse_fx=',rmse_fx,'rmse_fy=',rmse_fy,'rmse_fz=',rmse_fz,'rmse_f=',rmse_f)
    #     # np.save('16-Jan-data-exp1(cable)/Gt_lpf_c',Gt_lpf)
    #     # np.save('16-Jan-data-exp1(cable)/tension_c',Tension_3d_enu)
    #     # np.save('16-Jan-data-exp1(cable)/Df_Imhe_enu_c',Df_Imhe_enu[:,n_start:len(dfx_mhe)])
    #     # np.save('16-Jan-data-exp1(cable)/time_c',Time)

    #     fig, (ax1,ax2,ax3) = plt.subplots(3,sharex=True,figsize=(6/cm_2_inch,7/cm_2_inch),dpi=600)
    #     ax1.plot(Time,Gt_lpf[0,:], linewidth=0.5,color='black')
    #     ax1.plot(Time,Df_Imhe_enu[0,n_start:len(dfx_mhe)], linewidth=0.5,color='orange')
    #     ax1.plot(Time,Tension_3d_enu[0,:], linewidth=0.5,color='blue')
    #     ax2.plot(Time,Gt_lpf[1,:], linewidth=0.5,color='black')
    #     ax2.plot(Time,Df_Imhe_enu[1,n_start:len(dfx_mhe)], linewidth=0.5,color='orange')
    #     ax2.plot(Time,Tension_3d_enu[1,:], linewidth=0.5,color='blue')
    #     ax3.plot(Time,Gt_lpf[2,:], linewidth=0.5,color='black')
    #     ax3.plot(Time,Df_Imhe_enu[2,n_start:len(dfx_mhe)], linewidth=0.5,color='orange')
    #     ax3.plot(Time,Tension_3d_enu[2,:], linewidth=0.5,color='blue')
    #     ax1.set_ylabel('$d_{x}$ [N]',labelpad=0,**font1)
    #     ax2.set_ylabel('$d_{y}$ [N]',labelpad=0,**font1)
    #     ax3.set_ylabel('$d_{z}$ [N]',labelpad=-0.5,**font1)
    #     ax3.set_xlabel('Time [s]',labelpad=0,**font1)
    #     ax1.tick_params(axis='y',which='major',pad=-0.25)
    #     ax2.tick_params(axis='y',which='major',pad=-0.25)
    #     ax3.tick_params(axis='y',which='major',pad=-0.25)
    #     for t in ax1.yaxis.get_major_ticks(): 
    #         t.label.set_font('Times New Roman') 
    #         t.label.set_fontsize(7)
    #     for t in ax2.yaxis.get_major_ticks(): 
    #         t.label.set_font('Times New Roman') 
    #         t.label.set_fontsize(7)
    #     for t in ax3.xaxis.get_major_ticks(): 
    #         t.label.set_font('Times New Roman') 
    #         t.label.set_fontsize(7)
    #     for t in ax3.yaxis.get_major_ticks(): 
    #         t.label.set_font('Times New Roman') 
    #         t.label.set_fontsize(7)
    #     # leg=ax1.legend(['Ground truth', 'Estimation', 'Tension force'],loc='center',prop=font1,labelspacing=0.1)
    #     # leg.get_frame().set_linewidth(0.5)
    #     ax1.grid(b=True,axis='both',linestyle='--', linewidth=0.5)
    #     ax2.grid(b=True,axis='both',linestyle='--', linewidth=0.5)
    #     ax3.grid(b=True,axis='both',linestyle='--', linewidth=0.5)
    #     for axis in ['top', 'bottom', 'left', 'right']:
    #         ax1.spines[axis].set_linewidth(0.5)
    #     for axis in ['top', 'bottom', 'left', 'right']:
    #         ax2.spines[axis].set_linewidth(0.5)
    #     for axis in ['top', 'bottom', 'left', 'right']:
    #         ax3.spines[axis].set_linewidth(0.5)
    #     plt.savefig('L1-AC/force-L1AC-fir.png')
    #     plt.show()

    elif controller == 'e':
        dataset={'a':"Baseline_same_gain/log_14_2023-2-22-11-54-24_vehicle_local_position_0.csv",
             'b':"Baseline_large_gain/03_59_05_vehicle_local_position_0.csv",
             'e':"DMHE/log_11_2023-2-22-11-44-46_position_setpoint_triplet_0.csv"}
        
        local_position = pd.read_csv(dataset['a'])
        local_position_lg = pd.read_csv(dataset['b'])
        dataframe_a    = pd.DataFrame(local_position)
        dataframe_b    = pd.DataFrame(local_position_lg)
        data_dmhe      = pd.read_csv(dataset['e'])
        # load the position data in NED frame
        position_x     = dataframe_a['x']
        position_y     = dataframe_a['y']
        position_z     = dataframe_a['z']
        position_x_lg  = dataframe_b['x']
        position_y_lg  = dataframe_b['y']
        position_z_lg  = dataframe_b['z']
        g_r_dmhe       = data_dmhe['current.vy']
        n_start = 121 # During the initial interval n_start*dt, the quadrotor is still on the ground and the estimator does not begin to work
        n_start_lg = 135
        i = 0
        j = 0
        Time1       = []
        time       = 0
        position_enu   = np.zeros((4,len(position_x)-n_start)) 
        position_enu_lg= np.zeros((4,311-n_start_lg)) 
        t_switch = 0

        for k in range(n_start,len(position_x),1):
            Time1 += [time]
            pos_ned      = np.array([[position_x[k],position_y[k],position_z[k]]]).T
            pos_enu      = np.matmul(R_enu,pos_ned) # transfer to ENU frame
            position_enu[0:3,j:j+1] = pos_enu
            position_enu[3,j:j+1] = LA.norm(pos_enu)
            time += dt
            j += 1
        
        np.save('pos_enu_baseline',position_enu)

        j = 0
        for k in range(n_start_lg, 311,1):
            pos_ned_lg   = np.array([[position_x_lg[k],position_y_lg[k],position_z_lg[k]]]).T
            pos_enu_lg   = np.matmul(R_enu,pos_ned_lg)
            position_enu_lg[0:3,j:j+1] = pos_enu_lg
            position_enu_lg[3,j:j+1] = LA.norm(pos_enu_lg)
            j += 1

        ref_enu = np.load('ref_enu_neuromhe.npy')
        # plt.figure(1,figsize=(4.5/cm_2_inch,4/cm_2_inch),dpi=600)
        # ax = plt.axes(projection="3d")
        # ax.plot3D(ref_enu[0,:], ref_enu[1,:], ref_enu[2,:], linewidth=1.5, linestyle='--',color='black')
        # ax.plot3D(position_enu[0,:], position_enu[1,:], position_enu[2,:], linewidth=1,color='orange')
        # ax.plot3D(position_enu_lg[0,:], position_enu_lg[1,:], position_enu_lg[2,:], linewidth=1,color='blue')
        # leg=plt.legend(['Desired','Actual: original gain',r'Actual: $4\times$gain'],prop=font1,loc=(0.05,0.6),labelspacing = 0.1)
        # leg.get_frame().set_linewidth(0.5)
        # ax.set_xlabel('x [m]',labelpad=-10,**font1)
        # ax.set_ylabel('y [m]',labelpad=-10,**font1)
        # ax.set_zlabel('z [m]',labelpad=-8,**font1)
        # ax.tick_params(axis='x',which='major',pad=-5)
        # ax.tick_params(axis='y',which='major',pad=-5)
        # ax.tick_params(axis='z',which='major',pad=-3)
        # for t in ax.xaxis.get_major_ticks(): 
        #     t.label.set_font('Times New Roman') 
        #     t.label.set_fontsize(7)
        # for t in ax.yaxis.get_major_ticks(): 
        #     t.label.set_font('Times New Roman') 
        #     t.label.set_fontsize(7)
        # for t in ax.zaxis.get_major_ticks(): 
        #     t.label.set_font('Times New Roman') 
        #     t.label.set_fontsize(7)
        # ax.view_init(30,60)
        # ax.xaxis._axinfo["grid"].update({"linewidth":0.5,"linestyle":'--'})
        # ax.yaxis._axinfo["grid"].update({"linewidth":0.5,"linestyle":'--'})
        # ax.zaxis._axinfo["grid"].update({"linewidth":0.5,"linestyle":'--'})
        # plt.savefig('Baseline_same_gain/tracking_baseline.png')
        # plt.show()
        
        rmse_px = format(mean_squared_error(ref_enu[0,0:(len(position_x)-n_start)], position_enu[0,:], squared=False),'.3f')
        rmse_py = format(mean_squared_error(ref_enu[1,0:(len(position_x)-n_start)], position_enu[1,:], squared=False),'.3f')
        rmse_pz = format(mean_squared_error(ref_enu[2,0:(len(position_x)-n_start)], position_enu[2,:], squared=False),'.3f')
        rmse_p  = format(mean_squared_error(ref_enu[3,0:(len(position_x)-n_start)], position_enu[3,:],squared=False),'.3f')
        print('rmse_px=',rmse_px,'rmse_py=',rmse_py,'rmse_pz=',rmse_pz)
    else:
        ref_enu = np.load('ref_enu.npy')
        pos_neuromhe = np.load('pos_enu_neuromhe.npy')
        pos_dmhe     = np.load('pos_enu_dmhe.npy')
        pos_l1active = np.load('pos_enu_l1.npy')
        Time         = np.load('Time_track.npy')
        # load training mean loss and episode
        # loss and training episode
        loss_neuromhe = np.load('Loss.npy')
        loss_dmhe     = np.load('Loss_dmhe.npy')
        k_iter_neuromhe = np.load('K_iteration.npy')
        k_iter_dmhe     = np.load('K_iteration_dmhe.npy')
        # since the lengths of the above data are slightly different, we use their minimum length
        len_min      = np.min(np.array([len(pos_neuromhe[0]),len(pos_dmhe[0]),len(pos_l1active[0])]))
        fig, (ax1,ax2,ax3) = plt.subplots(3, sharex=True,figsize=(8/cm_2_inch,8*0.65/cm_2_inch),dpi=600)
        ax1.plot(Time[0:len_min],ref_enu[0,0:len_min],linewidth=0.5,color='black')
        ax1.plot(Time[0:len_min],pos_neuromhe[0,0:len_min],linewidth=1,color='orange')
        ax1.plot(Time[0:len_min],pos_dmhe[0,0:len_min],linewidth=0.5,color='green')
        ax1.plot(Time[0:len_min],pos_l1active[0,0:len_min],linewidth=0.5,color='blue')
        ax2.plot(Time[0:len_min],ref_enu[1,0:len_min],linewidth=0.5,color='black')
        ax2.plot(Time[0:len_min],pos_neuromhe[1,0:len_min],linewidth=1,color='orange')
        ax2.plot(Time[0:len_min],pos_dmhe[1,0:len_min],linewidth=0.5,color='green')
        ax2.plot(Time[0:len_min],pos_l1active[1,0:len_min],linewidth=0.5,color='blue')
        ax3.plot(Time[0:len_min],ref_enu[2,0:len_min],linewidth=0.5,color='black')
        ax3.plot(Time[0:len_min],pos_neuromhe[2,0:len_min],linewidth=1,color='orange')
        ax3.plot(Time[0:len_min],pos_dmhe[2,0:len_min],linewidth=0.5,color='green')
        ax3.plot(Time[0:len_min],pos_l1active[2,0:len_min],linewidth=0.5,color='blue')

        leg=ax1.legend(['Desired','Actual: NeuroMHE','Actual: DMHE','Actual: $\mathcal{L}_1$-AC'],loc='center left',prop=font1,labelspacing=0.1)
        leg.get_frame().set_linewidth(0.5)
        ax1.set_ylabel('$p_{x}$ [m]',labelpad=0,**font1)
        ax2.set_ylabel('$p_{y}$ [m]',labelpad=0,**font1)
        ax3.set_xlabel('Time [s]',labelpad=-3,**font1)
        ax3.set_ylabel('$p_{z}$ [m]',labelpad=5,**font1)
        ax1.grid(b=True,axis='both',linestyle='--', linewidth=0.5)
        ax2.grid(b=True,axis='both',linestyle='--', linewidth=0.5)
        ax3.grid(b=True,axis='both',linestyle='--', linewidth=0.5)
        for axis in ['top', 'bottom', 'left', 'right']:
            ax1.spines[axis].set_linewidth(0.5)
        for axis in ['top', 'bottom', 'left', 'right']:
            ax2.spines[axis].set_linewidth(0.5)
        for axis in ['top', 'bottom', 'left', 'right']:
            ax3.spines[axis].set_linewidth(0.5)
        ax1.tick_params(axis='y',which='major',pad=0)
        ax2.tick_params(axis='y',which='major',pad=0)
        ax3.tick_params(axis='x',which='major',pad=-0.5)
        ax3.tick_params(axis='y',which='major',pad=0)
        ax3.set_xticks(np.arange(0,35,10))
        ax3.set_yticks(np.arange(0,2.1,1))
        plt.savefig('NeuroMHE/comparison_position.png')
        plt.show()
        
        # comparison of mean loss in training
        # plt.figure(3, figsize=(8/cm_2_inch, 8*0.45/cm_2_inch), dpi=600)
        # ax = plt.gca()
        # plt.plot(k_iter_neuromhe, loss_neuromhe, linewidth=1, marker='o', markersize=2)
        # plt.plot(k_iter_dmhe, loss_dmhe, linewidth=1, marker='o', markersize=2)
        # plt.xlabel('Training episode', labelpad=-5, **font1)
        # plt.xticks(np.arange(0,3.1,1), **font1)
        # plt.ylabel('Mean loss',labelpad=0, **font1)
        # plt.yticks(np.arange(0,11000,2000),**font1)
        # plt.yscale("log") 
        # ax.tick_params(axis='x',which='major',pad=0)
        # ax.tick_params(axis='y',which='major',pad=0)
        # leg=ax.legend(['NeuroMHE', 'DMHE'],loc='lower left',prop=font1,labelspacing=0.15)
        # leg.get_frame().set_linewidth(0.5)
        # for axis in ['top', 'bottom', 'left', 'right']:
        #     ax.spines[axis].set_linewidth(0.5)
        # plt.grid(b=True,axis='both',linestyle='--', linewidth=0.5)
        # plt.savefig('NeuroMHE/training_meanloss_experiment.png')
        # plt.show()

else: # downwash
    n_end = 25*50 # 25s, we choose this ending time so that in plots the trajectory of the 2nd big drone will not descend 
    T_end = 24
    t_switch = 0
    if controller == 'a':
        dataset={'a':"NeuroMHE_downwash/09_12_06_vehicle_local_position_0.csv",
             'b':"NeuroMHE_downwash/09_12_06_position_setpoint_triplet_0.csv",
             'c':"NeuroMHE_downwash/09_12_06_sensor_combined_0.csv",
             'd':"NeuroMHE_downwash/09_12_06_vehicle_attitude_0.csv",
             'e':"2nd big drone/log_18_2023-2-21-14-16-50_vehicle_local_position_0.csv"}
        local_position = pd.read_csv(dataset['a'])
        position_setpoint = pd.read_csv(dataset['b'])
        sensor         = pd.read_csv(dataset['c'])
        attitude       = pd.read_csv(dataset['d'])
        bigdrone       = pd.read_csv(dataset['e'])
        dataframe_a    = pd.DataFrame(local_position)
        dataframe_b    = pd.DataFrame(position_setpoint)
        dataframe_c    = pd.DataFrame(sensor)
        dataframe_d    = pd.DataFrame(attitude)
        dataframe_e    = pd.DataFrame(bigdrone)
        # load the position data in NED frame
        position_x     = dataframe_a['x']
        position_y     = dataframe_a['y']
        position_z     = dataframe_a['z']
        pos_big_x      = dataframe_e['x']
        pos_big_y      = dataframe_e['y']
        pos_big_z      = dataframe_e['z']
        position_enu   = np.zeros((5,n_end))
        ref_enu        = np.zeros((5,n_end))
        pos_big_enu    = np.zeros((3,len(pos_big_x)))
        n_start = 1191 # During the initial interval n_start*dt, the quadrotor is still on the ground and the estimator does not begin to work
        for k in range(len(pos_big_x)):
            pos_big_ned  = np.array([[pos_big_x[k],pos_big_y[k],pos_big_z[k]]]).T
            pos_big_enu[:,k:k+1]  = np.matmul(R_enu,pos_big_ned)
        
        np.save('pos_big_dw',pos_big_enu[:,(600):(n_end+600)])
        
        i = 0
        Time       = []
        time       = 0
        for k in range(n_start,n_start+n_end,1):
            Time += [time]
            pos_ned      = np.array([[position_x[k],position_y[k],position_z[k]]]).T
            pos_enu      = np.matmul(R_enu,pos_ned)
            position_enu[0:3,i:i+1] = pos_enu # transfer to ENU frame
            position_enu[3,i:i+1]   = LA.norm(pos_enu[0:2,0])
            position_enu[4,i:i+1]   = LA.norm(pos_enu)
            if time <T_end:
                ref_p, ref_v, ref_a, ref_j, ref_s = uav.ref_exp2(Coeffx_exp2, Coeffy_exp2, Coeffz_exp2, time, t_switch) 
            else:
                ref_p, ref_v, ref_a, ref_j, ref_s = np.array([[0,0,0.12]]).T, np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1))
            ref_enu[0:3,i:i+1] = ref_p
            ref_enu[3,i:i+1]   = LA.norm(ref_p[0:2,0])
            ref_enu[4,i:i+1]   = LA.norm(ref_p)
            i += 1
            time += dt
        
        np.save('ref_enu_dw_neuromhe',ref_enu)
        np.save('pos_enu_dw_neuromhe',position_enu)
        np.save('Time_dw_neuromhe',Time)
        # load the computed total force command
        f_cmd = dataframe_b['current.vx']
        # load the forgetting factors
        gamma_r = dataframe_b['current.vy']
        gamma_p = dataframe_b['current.vz']
        # load the NeuroMHE estimation data in NED frame
        dfx_mhe = dataframe_b['current.a_x']
        dfy_mhe = dataframe_b['current.a_y']
        dfz_mhe = dataframe_b['current.a_z']
        Df_Imhe_enu = np.zeros((5,n_end))
        i = 0
        for k in range(n_start,n_start+n_end,1):
            Dfmhe_ned = np.array([[dfx_mhe[k],dfy_mhe[k],dfz_mhe[k]]]).T
            dfmhe_enu = np.matmul(R_enu,Dfmhe_ned)
            Df_Imhe_enu[0:3,i:i+1] = dfmhe_enu
            Df_Imhe_enu[3,i:i+1]   = LA.norm(dfmhe_enu[0:2,0])
            Df_Imhe_enu[4,i:i+1]   = LA.norm(dfmhe_enu)
            i += 1
        
        np.save('Df_Imhe_dw_neuromhe',Df_Imhe_enu)
        # load the accelerometer data in NED frame
        acc_x   = dataframe_c['accelerometer_m_s2[0]']
        acc_y   = dataframe_c['accelerometer_m_s2[1]']
        acc_z   = dataframe_c['accelerometer_m_s2[2]']
        Acc_b_ned   = np.zeros((3,len(acc_x)))
        for k in range(len(acc_x)):
            Acc_b_ned[:,k:k+1] = np.array([[acc_x[k],acc_y[k],acc_z[k]]]).T
        # load the quaternion
        Q0 = dataframe_d['q[0]']
        Q1 = dataframe_d['q[1]']
        Q2 = dataframe_d['q[2]']
        Q3 = dataframe_d['q[3]']
        
        Gt = np.zeros((3,n_end))
        Gt_lpf = np.zeros((5,n_end))
        Gamma  = np.zeros((2,n_end))
        # print('len(ref_x)=',len(Ref_x),'len(acc_x)=',len(acc_x),'len(dfx_mhe)=',len(dfx_mhe),'len(position_x)=',len(position_x),'len(f_cmd)=',len(f_cmd))
        acc_I_prev = np.array([[0,0,-9.78]]).T # initial acceleration of the low-pass filter in NED frame
        
        i = 0
        # calculation of the ground truth data using the two steps mentioned above
        for k in range(n_start,n_start+n_end,1): 
            q0, q1, q2, q3 = Q0[k], Q1[k], Q2[k], Q3[k]
            quaternion = np.array([q0, q1, q2, q3])
            R_b        = Quaternion2Rotation(quaternion)
            acc_I_ned = np.matmul(R_b,np.reshape(Acc_b_ned[:,k],(3,1)))
            f_cmdI_ned = np.matmul(R_b,(f_cmd[k]*z))
            gt        = acc_I_ned*mass+f_cmdI_ned # we do not plot the raw accelerometer data as it is too noisy
            Gt[:,i:i+1] =  np.matmul(R_enu,gt) # transfer to ENU frame
            gamma_ned = np.array([[gamma_r[k],0,gamma_p[k]]]).T
            gamma_edu = np.matmul(R_enu,gamma_ned)
            Gamma[:,i:i+1]  = np.reshape(gamma_edu[1:3,0],(2,1))
            
            i += 1
        # np.save('16-Jan-data-exp2(downwash)/Time',Time)
        Gt_lpf[0,:] = lfilter(taps, 1.0, Gt[0,:])
        Gt_lpf[1,:] = lfilter(taps, 1.0, Gt[1,:])
        Gt_lpf[2,:] = lfilter(taps, 1.0, Gt[2,:])
        for k in range(len(Gt_lpf[0,:])):
            Gt_lpf[3,k:k+1]=LA.norm(Gt_lpf[0:2,k])
            Gt_lpf[4,k:k+1]=LA.norm(Gt_lpf[0:3,k])
        
        np.save('Gt_lpf_dw_neuromhe',Gt_lpf)

        n_dw_start = int(8*sample_rate) # the time when the downwash takes effect
        n_dw_steady = int(12*sample_rate)
        n_dw_end   = int(17*sample_rate)
        
        rmse_pxy = format(mean_squared_error(ref_enu[3,n_dw_start:n_dw_end], position_enu[3,n_dw_start:n_dw_end], squared=False),'.3f')
        est_pz  = format(mean_squared_error(ref_enu[2,n_dw_steady:n_dw_end], position_enu[2,n_dw_steady:n_dw_end], squared=False),'.3f')
        # rmse_p   = format(mean_squared_error(ref_enu[4,n_dw_start:n_dw_end], position_enu[4,n_dw_start:n_dw_end], squared=False),'.3f')
        rmse_fxy = format(mean_squared_error(Gt_lpf[3,n_dw_start:n_dw_end], Df_Imhe_enu[3,n_dw_start:n_dw_end], squared=False),'.3f')
        rmse_fz  = format(mean_squared_error(Gt_lpf[2,n_dw_start:n_dw_end], Df_Imhe_enu[2,n_dw_start:n_dw_end], squared=False),'.3f')
        # rmse_f   = format(mean_squared_error(Gt_lpf[4,n_dw_start:n_dw_end], Df_Imhe_enu[4,n_dw_start:n_dw_end], squared=False),'.3f')
        pxy_max  = format(np.max(position_enu[3,n_dw_start:n_dw_end]),'.3f')
        ez_max   = format(1.2-np.min(position_enu[2,n_dw_start:n_dw_end]),'.3f')
        mean_steady = np.mean(position_enu[2,n_dw_steady:n_dw_end])
        ez       = (position_enu[2,n_dw_steady:n_dw_end]-mean_steady)**2
        std      = format(np.sqrt(np.mean(ez)),'.3f')
        print('rmse_pxy=',rmse_pxy,'pxy_max=',pxy_max,'estrmse_pz=',est_pz,'ez_max=',ez_max,'rmse_fxy=',rmse_fxy,'rmse_fz=',rmse_fz,'std=',std)

        # fig, (ax1,ax2) = plt.subplots(2,sharex=True,figsize=(6/cm_2_inch,6*0.6/cm_2_inch),dpi=600)
        # ax1.plot(Time,Gt_lpf[3,:], linewidth=0.5,color='black')
        # ax1.plot(Time,Df_Imhe_enu[3,:], linewidth=0.5,color='orange')
        # ax1.set_xticks(np.arange(0,25,8))
        # ax1.tick_params(axis='y',which='major',pad=0)
        # ax1.set_ylabel('$d_{xy}$ [N]',labelpad=4,**font1) 
        # ax1.add_patch(Rectangle((8, 0), 9, 1.8,
        #      facecolor = 'blue',
        #      fill=True,
        #      alpha = 0.1,
        #      lw=0.1))
        # ax1.add_patch(Rectangle((17, 0), 2, 1.8,
        #      facecolor = 'navy',
        #      fill=True,
        #      alpha = 0.2,
        #      lw=0.1))
        # leg=ax1.legend(['Ground truth', 'Estimation'],loc='upper center',prop=font1,labelspacing=0.1)
        # leg.get_frame().set_linewidth(0.5)
        # ax2.plot(Time,Gt_lpf[2,:], linewidth=0.5,color='black')
        # ax2.plot(Time,Df_Imhe_enu[2,:], linewidth=0.5,color='orange')
        # ax2.set_xticks(np.arange(0,25,8))
        # ax2.tick_params(axis='x',which='major',pad=-0.5)
        # ax2.tick_params(axis='y',which='major',pad=0)
        # ax2.set_xlabel('Time [s]',labelpad=-5.5,**font1)
        # ax2.set_ylabel('$d_{z}$ [N]',labelpad=0,**font1)
        # ax2.add_patch(Rectangle((8, -10), 9, 10,
        #      facecolor = 'blue',
        #      fill=True,
        #      alpha = 0.1,
        #      lw=0.1))
        # ax2.add_patch(Rectangle((17, -10), 2, 10,
        #      facecolor = 'navy',
        #      fill=True,
        #      alpha = 0.2,
        #      lw=0.1))
        # ax1.grid(b=True,axis='both',linestyle='--', linewidth=0.5)
        # ax2.grid(b=True,axis='both',linestyle='--', linewidth=0.5)
        # for t in ax1.yaxis.get_major_ticks(): 
        #     t.label.set_font('Times New Roman') 
        #     t.label.set_fontsize(7)
        # for t in ax2.xaxis.get_major_ticks(): 
        #     t.label.set_font('Times New Roman') 
        #     t.label.set_fontsize(7)
        # for t in ax2.yaxis.get_major_ticks(): 
        #     t.label.set_font('Times New Roman') 
        #     t.label.set_fontsize(7)
        # for axis in ['top', 'bottom', 'left', 'right']:
        #     ax1.spines[axis].set_linewidth(0.5)
        # for axis in ['top', 'bottom', 'left', 'right']:
        #     ax2.spines[axis].set_linewidth(0.5)
        # plt.savefig('NeuroMHE_downwash/force_NeuroMHE_downwash_h.png')
        # plt.show()


        # fig, (ax1,ax2) = plt.subplots(2,sharex=True,figsize=(5/cm_2_inch,5*0.65/cm_2_inch),dpi=600)
        # ax1.plot(Time,position_enu[3,:], linewidth=0.5,color='orange')
        # ax1.set_xticks(np.arange(0,25,8))
        # ax1.tick_params(axis='y',which='major',pad=0)
        # ax1.set_ylabel('$p_{xy}$ [m]',labelpad=-2,**font1)
        # ax1.yaxis.set_major_formatter(OOMFormatter(-1,"%1.1f"))
        # ax1.yaxis.get_offset_text().set_fontsize(6)
        # ax1.add_patch(Rectangle((8, 0), 9, 0.4,
        #      facecolor = 'blue',
        #      fill=True,
        #      alpha = 0.1,
        #      lw=0.1))
        # ax1.add_patch(Rectangle((17, 0), 2, 0.4,
        #      facecolor = 'navy',
        #      fill=True,
        #      alpha = 0.2,
        #      lw=0.1))
        # ax2.plot(Time,ref_enu[2,:], linewidth=0.5,color='black')
        # ax2.plot(Time,position_enu[2,:], linewidth=0.5,color='orange')
        # ax2.plot(Time,pos_big_enu[2,(600):(n_end+600)], linewidth=0.5,color='green')
        # ax2.set_xticks(np.arange(0,25,8))
        # ax2.tick_params(axis='x',which='major',pad=-0.5)
        # ax2.tick_params(axis='y',which='major',pad=0)
        # ax2.set_xlabel('Time [s]',labelpad=-6,**font1)
        # ax2.set_ylabel('$p_{z}$ [m]',labelpad=4,**font1)
        # ax2.add_patch(Rectangle((8, 0), 9, 2.2,
        #      facecolor = 'blue',
        #      fill=True,
        #      alpha = 0.1,
        #      lw=0.1))
        # ax2.add_patch(Rectangle((17, 0), 2, 2.2,
        #      facecolor = 'navy',
        #      fill=True,
        #      alpha = 0.2,
        #      lw=0.1))
        # # leg=ax2.legend(['Desired: ego quadrotor', 'Actual: ego quadrotor', 'Actual: 2nd quadrotor'],loc=(0.1,0.6),prop=font1,labelspacing=0.1)
        # # leg.get_frame().set_linewidth(0.5)
        
        # for t in ax1.yaxis.get_major_ticks(): 
        #     t.label.set_font('Times New Roman') 
        #     t.label.set_fontsize(7)
        # for t in ax2.xaxis.get_major_ticks(): 
        #     t.label.set_font('Times New Roman') 
        #     t.label.set_fontsize(7)
        # for t in ax2.yaxis.get_major_ticks(): 
        #     t.label.set_font('Times New Roman') 
        #     t.label.set_fontsize(7)
        # ax1.grid(b=True,axis='both',linestyle='--', linewidth=0.5)
        # ax2.grid(b=True,axis='both',linestyle='--', linewidth=0.5)
        # for axis in ['top', 'bottom', 'left', 'right']:
        #     ax1.spines[axis].set_linewidth(0.5)
        # for axis in ['top', 'bottom', 'left', 'right']:
        #     ax2.spines[axis].set_linewidth(0.5)
        # plt.savefig('NeuroMHE_downwash/position_NeuroMHE.png')
        # plt.show()


    elif controller == 'b':
        dataset={'a':"DMHE_downwash/09_14_48_vehicle_local_position_0.csv",
             'b':"DMHE_downwash/09_14_48_position_setpoint_triplet_0.csv",
             'c':"DMHE_downwash/09_14_48_sensor_combined_0.csv",
             'd':"DMHE_downwash/09_14_48_vehicle_attitude_0.csv",
             'e':"2nd big drone/log_18_2023-2-21-14-16-50_vehicle_local_position_0.csv"}
        local_position = pd.read_csv(dataset['a'])
        position_setpoint = pd.read_csv(dataset['b'])
        sensor         = pd.read_csv(dataset['c'])
        attitude       = pd.read_csv(dataset['d'])
        dataframe_a    = pd.DataFrame(local_position)
        dataframe_b    = pd.DataFrame(position_setpoint)
        dataframe_c    = pd.DataFrame(sensor)
        dataframe_d    = pd.DataFrame(attitude)
        bigdrone       = pd.read_csv(dataset['e'])
        dataframe_e    = pd.DataFrame(bigdrone)
        # load the position data in NED frame
        position_x     = dataframe_a['x']
        position_y     = dataframe_a['y']
        position_z     = dataframe_a['z']
        pos_big_x      = dataframe_e['x']
        pos_big_y      = dataframe_e['y']
        pos_big_z      = dataframe_e['z']
        position_enu   = np.zeros((5,n_end))
        ref_enu        = np.zeros((5,n_end))
        pos_big_enu    = np.zeros((3,len(pos_big_x)))
        n_start = 185 # During the initial interval n_start*dt, the quadrotor is still on the ground and the estimator does not begin to work
        for k in range(len(pos_big_x)):
            pos_big_ned  = np.array([[pos_big_x[k],pos_big_y[k],pos_big_z[k]]]).T
            pos_big_enu[:,k:k+1]  = np.matmul(R_enu,pos_big_ned)
        
        i = 0
        Time       = []
        time       = 0
        for k in range(n_start,n_start+n_end,1):
            Time += [time]
            pos_ned      = np.array([[position_x[k],position_y[k],position_z[k]]]).T
            pos_enu      = np.matmul(R_enu,pos_ned)
            position_enu[0:3,i:i+1] = pos_enu # transfer to ENU frame
            position_enu[3,i:i+1]   = LA.norm(pos_enu[0:2,0])
            position_enu[4,i:i+1]   = LA.norm(pos_enu)
            if time <T_end:
                ref_p, ref_v, ref_a, ref_j, ref_s = uav.ref_exp2(Coeffx_exp2, Coeffy_exp2, Coeffz_exp2, time, t_switch) 
            else:
                ref_p, ref_v, ref_a, ref_j, ref_s = np.array([[0,0,0.12]]).T, np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1))
            ref_enu[0:3,i:i+1] = ref_p
            ref_enu[3,i:i+1]   = LA.norm(ref_p[0:2,0])
            ref_enu[4,i:i+1]   = LA.norm(ref_p)
            i += 1
            time += dt
        
        np.save('ref_enu_dw_dmhe',ref_enu)
        np.save('pos_enu_dw_dmhe',position_enu)
        np.save('Time_dw_dmhe',Time)
        # load the computed total force command
        f_cmd = dataframe_b['current.vx']
        # load the forgetting factors
        gamma_r = dataframe_b['current.vy']
        gamma_p = dataframe_b['current.vz']
        # load the NeuroMHE estimation data in NED frame
        dfx_mhe = dataframe_b['current.a_x']
        dfy_mhe = dataframe_b['current.a_y']
        dfz_mhe = dataframe_b['current.a_z']
        Df_Imhe_enu = np.zeros((5,n_end))
        i = 0
        for k in range(n_start,n_start+n_end,1):
            Dfmhe_ned = np.array([[dfx_mhe[k],dfy_mhe[k],dfz_mhe[k]]]).T
            dfmhe_enu = np.matmul(R_enu,Dfmhe_ned)
            Df_Imhe_enu[0:3,i:i+1] = dfmhe_enu
            Df_Imhe_enu[3,i:i+1]   = LA.norm(dfmhe_enu[0:2,0])
            Df_Imhe_enu[4,i:i+1]   = LA.norm(dfmhe_enu)
            i += 1
        np.save('Df_Imhe_dw_dmhe',Df_Imhe_enu)
        # load the accelerometer data in NED frame
        acc_x   = dataframe_c['accelerometer_m_s2[0]']
        acc_y   = dataframe_c['accelerometer_m_s2[1]']
        acc_z   = dataframe_c['accelerometer_m_s2[2]']
        Acc_b_ned   = np.zeros((3,len(acc_x)))
        for k in range(len(acc_x)):
            Acc_b_ned[:,k:k+1] = np.array([[acc_x[k],acc_y[k],acc_z[k]]]).T
        # load the quaternion
        Q0 = dataframe_d['q[0]']
        Q1 = dataframe_d['q[1]']
        Q2 = dataframe_d['q[2]']
        Q3 = dataframe_d['q[3]']
        
        Gt = np.zeros((3,n_end))
        Gt_lpf = np.zeros((5,n_end))
        Gamma  = np.zeros((2,n_end))
        # print('len(ref_x)=',len(Ref_x),'len(acc_x)=',len(acc_x),'len(dfx_mhe)=',len(dfx_mhe),'len(position_x)=',len(position_x),'len(f_cmd)=',len(f_cmd))
        acc_I_prev = np.array([[0,0,-9.78]]).T # initial acceleration of the low-pass filter in NED frame
        
        i = 0
        # calculation of the ground truth data using the two steps mentioned above
        for k in range(n_start,n_start+n_end,1): 
            q0, q1, q2, q3 = Q0[k], Q1[k], Q2[k], Q3[k]
            quaternion = np.array([q0, q1, q2, q3])
            R_b        = Quaternion2Rotation(quaternion)
            acc_I_ned = np.matmul(R_b,np.reshape(Acc_b_ned[:,k],(3,1)))
            f_cmdI_ned = np.matmul(R_b,(f_cmd[k]*z))
            gt        = acc_I_ned*mass+f_cmdI_ned # we do not plot the raw accelerometer data as it is too noisy
            Gt[:,i:i+1] =  np.matmul(R_enu,gt) # transfer to ENU frame
            gamma_ned = np.array([[gamma_r[k],0,gamma_p[k]]]).T
            gamma_edu = np.matmul(R_enu,gamma_ned)
            Gamma[:,i:i+1]  = np.reshape(gamma_edu[1:3,0],(2,1))
            
            i += 1
        # np.save('16-Jan-data-exp2(downwash)/Time',Time)
        Gt_lpf[0,:] = lfilter(taps, 1.0, Gt[0,:])
        Gt_lpf[1,:] = lfilter(taps, 1.0, Gt[1,:])
        Gt_lpf[2,:] = lfilter(taps, 1.0, Gt[2,:])
        for k in range(len(Gt_lpf[0,:])):
            Gt_lpf[3,k:k+1]=LA.norm(Gt_lpf[0:2,k])
            Gt_lpf[4,k:k+1]=LA.norm(Gt_lpf[0:3,k])
        
        np.save('Gt_lpf_dw_dmhe',Gt_lpf)

        n_dw_start = int(8.6*sample_rate) # the time when the downwash takes effect, including the post downwash effect
        n_dw_steady = int(12.6*sample_rate)
        n_dw_end   = int(17.6*sample_rate)
        
        rmse_pxy = format(mean_squared_error(ref_enu[3,n_dw_start:n_dw_end], position_enu[3,n_dw_start:n_dw_end], squared=False),'.3f')
        est_pz  = format(mean_squared_error(ref_enu[2,n_dw_steady:n_dw_end], position_enu[2,n_dw_steady:n_dw_end], squared=False),'.3f')
        # rmse_p   = format(mean_squared_error(ref_enu[4,n_dw_start:n_dw_end], position_enu[4,n_dw_start:n_dw_end], squared=False),'.3f')
        rmse_fxy = format(mean_squared_error(Gt_lpf[3,n_dw_start:n_dw_end], Df_Imhe_enu[3,n_dw_start:n_dw_end], squared=False),'.3f')
        rmse_fz  = format(mean_squared_error(Gt_lpf[2,n_dw_start:n_dw_end], Df_Imhe_enu[2,n_dw_start:n_dw_end], squared=False),'.3f')
        # rmse_f   = format(mean_squared_error(Gt_lpf[4,n_dw_start:n_dw_end], Df_Imhe_enu[4,n_dw_start:n_dw_end], squared=False),'.3f')
        pxy_max  = format(np.max(position_enu[3,n_dw_start:n_dw_end]),'.3f')
        ez_max   = format(1.2-np.min(position_enu[2,n_dw_start:n_dw_end]),'.3f')
        mean_steady = np.mean(position_enu[2,n_dw_steady:n_dw_end])
        ez       = (position_enu[2,n_dw_steady:n_dw_end]-mean_steady)**2
        std      = format(np.sqrt(np.mean(ez)),'.3f')
        print('rmse_pxy=',rmse_pxy,'pxy_max=',pxy_max,'estrmse_pz=',est_pz,'ez_max=',ez_max,'rmse_fxy=',rmse_fxy,'rmse_fz=',rmse_fz,'std=',std)

        # fig, (ax1,ax2,ax3,ax4) = plt.subplots(4,sharex=True,figsize=(6/cm_2_inch,9/cm_2_inch),dpi=600)
        # ax1.plot(Time,Gt_lpf[3,:], linewidth=0.5,color='black')
        # ax1.plot(Time,Df_Imhe_enu[3,:], linewidth=0.5,color='orange')
        # ax1.set_xticks(np.arange(0,25,8))
        # ax1.tick_params(axis='y',which='major',pad=0)
        # ax1.set_ylabel('$d_{xy}$ [N]',labelpad=4,**font1) 
        # ax1.add_patch(Rectangle((8.5, 0), 9, 1.8,
        #      facecolor = 'blue',
        #      fill=True,
        #      alpha = 0.1,
        #      lw=0.1))
        # ax1.add_patch(Rectangle((17.5, 0), 2, 1.8,
        #      facecolor = 'navy',
        #      fill=True,
        #      alpha = 0.2,
        #      lw=0.1))
        # ax2.plot(Time,Gt_lpf[2,:], linewidth=0.5,color='black')
        # ax2.plot(Time,Df_Imhe_enu[2,:], linewidth=0.5,color='orange')
        # ax2.set_xticks(np.arange(0,25,8))
        # ax2.tick_params(axis='y',which='major',pad=0)
        # ax2.set_ylabel('$d_{z}$ [N]',labelpad=0,**font1)
        # ax2.add_patch(Rectangle((8.5, -10), 9, 10,
        #      facecolor = 'blue',
        #      fill=True,
        #      alpha = 0.1,
        #      lw=0.1))
        # ax2.add_patch(Rectangle((17.5, -10), 2, 10,
        #      facecolor = 'navy',
        #      fill=True,
        #      alpha = 0.2,
        #      lw=0.1))
        # ax3.plot(Time,position_enu[3,:], linewidth=0.5,color='orange')
        # ax3.set_xticks(np.arange(0,25,8))
        # ax3.tick_params(axis='y',which='major',pad=0)
        # ax3.set_ylabel('$p_{xy}$ [m]',labelpad=0,**font1)
        # ax3.add_patch(Rectangle((8.5, 0), 9, 0.4,
        #      facecolor = 'blue',
        #      fill=True,
        #      alpha = 0.1,
        #      lw=0.1))
        # ax3.add_patch(Rectangle((17.5, 0), 2, 0.4,
        #      facecolor = 'navy',
        #      fill=True,
        #      alpha = 0.2,
        #      lw=0.1))
        # ax4.plot(Time,ref_enu[2,:], linewidth=0.5,color='black')
        # ax4.plot(Time,position_enu[2,:], linewidth=0.5,color='orange')
        # ax4.plot(Time,pos_big_enu[2,(600):(n_end+600)], linewidth=0.5,color='green')
        # ax4.set_xticks(np.arange(0,25,8))
        # ax4.tick_params(axis='x',which='major',pad=0)
        # ax4.tick_params(axis='y',which='major',pad=0)
        # ax4.set_xlabel('Time [s]',labelpad=0,**font1)
        # ax4.set_ylabel('$p_{z}$ [m]',labelpad=4,**font1)
        # ax4.add_patch(Rectangle((8.5, 0), 9, 2.2,
        #      facecolor = 'blue',
        #      fill=True,
        #      alpha = 0.1,
        #      lw=0.1))
        # ax4.add_patch(Rectangle((17.5, 0), 2, 2.2,
        #      facecolor = 'navy',
        #      fill=True,
        #      alpha = 0.2,
        #      lw=0.1))
        # for t in ax1.yaxis.get_major_ticks(): 
        #     t.label.set_font('Times New Roman') 
        #     t.label.set_fontsize(7)
        # for t in ax2.yaxis.get_major_ticks(): 
        #     t.label.set_font('Times New Roman') 
        #     t.label.set_fontsize(7)
        # for t in ax3.yaxis.get_major_ticks(): 
        #     t.label.set_font('Times New Roman') 
        #     t.label.set_fontsize(7)
        # for t in ax4.xaxis.get_major_ticks(): 
        #     t.label.set_font('Times New Roman') 
        #     t.label.set_fontsize(7)
        # for t in ax4.yaxis.get_major_ticks(): 
        #     t.label.set_font('Times New Roman') 
        #     t.label.set_fontsize(7)
        # ax1.grid(b=True,axis='both',linestyle='--', linewidth=0.5)
        # ax2.grid(b=True,axis='both',linestyle='--', linewidth=0.5)
        # ax3.grid(b=True,axis='both',linestyle='--', linewidth=0.5)
        # ax4.grid(b=True,axis='both',linestyle='--', linewidth=0.5)
        # for axis in ['top', 'bottom', 'left', 'right']:
        #     ax1.spines[axis].set_linewidth(0.5)
        # for axis in ['top', 'bottom', 'left', 'right']:
        #     ax2.spines[axis].set_linewidth(0.5)
        # for axis in ['top', 'bottom', 'left', 'right']:
        #     ax3.spines[axis].set_linewidth(0.5)
        # for axis in ['top', 'bottom', 'left', 'right']:
        #     ax4.spines[axis].set_linewidth(0.5)
        # plt.savefig('DMHE_downwash/force_position_DMHE.png')
        # plt.show()

        # fig, (ax1,ax2) = plt.subplots(2,sharex=True,figsize=(6/cm_2_inch,6*0.6/cm_2_inch),dpi=600)
        # ax1.plot(Time,Gt_lpf[3,:], linewidth=0.5,color='black')
        # ax1.plot(Time,Df_Imhe_enu[3,:], linewidth=0.5,color='orange')
        # ax1.set_xticks(np.arange(0,25,8))
        # ax1.tick_params(axis='y',which='major',pad=0)
        # ax1.set_ylabel('$d_{xy}$ [N]',labelpad=4,**font1) 
        # ax1.add_patch(Rectangle((8.6, 0), 9, 1.8,
        #      facecolor = 'blue',
        #      fill=True,
        #      alpha = 0.1,
        #      lw=0.1))
        # ax1.add_patch(Rectangle((17.6, 0), 2, 1.8,
        #      facecolor = 'navy',
        #      fill=True,
        #      alpha = 0.2,
        #      lw=0.1))
        # ax2.plot(Time,Gt_lpf[2,:], linewidth=0.5,color='black')
        # ax2.plot(Time,Df_Imhe_enu[2,:], linewidth=0.5,color='orange')
        # ax2.set_xticks(np.arange(0,25,8))
        # ax2.tick_params(axis='x',which='major',pad=-0.5)
        # ax2.tick_params(axis='y',which='major',pad=0)
        # ax2.set_xlabel('Time [s]',labelpad=-5.5,**font1)
        # ax2.set_ylabel('$d_{z}$ [N]',labelpad=0,**font1)
        # ax2.add_patch(Rectangle((8.6, -10), 9, 10,
        #      facecolor = 'blue',
        #      fill=True,
        #      alpha = 0.1,
        #      lw=0.1))
        # ax2.add_patch(Rectangle((17.6, -10), 2, 10,
        #      facecolor = 'navy',
        #      fill=True,
        #      alpha = 0.2,
        #      lw=0.1))
        # ax1.grid(b=True,axis='both',linestyle='--', linewidth=0.5)
        # ax2.grid(b=True,axis='both',linestyle='--', linewidth=0.5)
        # for t in ax1.yaxis.get_major_ticks(): 
        #     t.label.set_font('Times New Roman') 
        #     t.label.set_fontsize(7)
        # for t in ax2.xaxis.get_major_ticks(): 
        #     t.label.set_font('Times New Roman') 
        #     t.label.set_fontsize(7)
        # for t in ax2.yaxis.get_major_ticks(): 
        #     t.label.set_font('Times New Roman') 
        #     t.label.set_fontsize(7)
        # for axis in ['top', 'bottom', 'left', 'right']:
        #     ax1.spines[axis].set_linewidth(0.5)
        # for axis in ['top', 'bottom', 'left', 'right']:
        #     ax2.spines[axis].set_linewidth(0.5)
        # plt.savefig('DMHE_downwash/force_DMHE_downwash_h.png')
        # plt.show()


        # fig, (ax1,ax2) = plt.subplots(2,sharex=True,figsize=(5/cm_2_inch,5*0.65/cm_2_inch),dpi=600)
        # ax1.plot(Time,position_enu[3,:], linewidth=0.5,color='orange')
        # ax1.set_xticks(np.arange(0,25,8))
        # ax1.tick_params(axis='y',which='major',pad=0)
        # ax1.set_ylabel('$p_{xy}$ [m]',labelpad=-2,**font1)
        # ax1.yaxis.set_major_formatter(OOMFormatter(-1,"%1.1f"))
        # ax1.yaxis.get_offset_text().set_fontsize(6)
        # ax1.add_patch(Rectangle((8.6, 0), 9, 0.4,
        #      facecolor = 'blue',
        #      fill=True,
        #      alpha = 0.1,
        #      lw=0.1))
        # ax1.add_patch(Rectangle((17.6, 0), 2, 0.4,
        #      facecolor = 'navy',
        #      fill=True,
        #      alpha = 0.2,
        #      lw=0.1))
        # ax2.plot(Time,ref_enu[2,:], linewidth=0.5,color='black')
        # ax2.plot(Time,position_enu[2,:], linewidth=0.5,color='orange')
        # ax2.plot(Time,pos_big_enu[2,(600):(n_end+600)], linewidth=0.5,color='green')
        # ax2.set_xticks(np.arange(0,25,8))
        # ax2.tick_params(axis='x',which='major',pad=-0.5)
        # ax2.tick_params(axis='y',which='major',pad=0)
        # ax2.set_xlabel('Time [s]',labelpad=-6,**font1)
        # ax2.set_ylabel('$p_{z}$ [m]',labelpad=4,**font1)
        # ax2.add_patch(Rectangle((8.6, 0), 9, 2.2,
        #      facecolor = 'blue',
        #      fill=True,
        #      alpha = 0.1,
        #      lw=0.1))
        # ax2.add_patch(Rectangle((17.6, 0), 2, 2.2,
        #      facecolor = 'navy',
        #      fill=True,
        #      alpha = 0.2,
        #      lw=0.1))
        
        # for t in ax1.yaxis.get_major_ticks(): 
        #     t.label.set_font('Times New Roman') 
        #     t.label.set_fontsize(7)
        # for t in ax2.xaxis.get_major_ticks(): 
        #     t.label.set_font('Times New Roman') 
        #     t.label.set_fontsize(7)
        # for t in ax2.yaxis.get_major_ticks(): 
        #     t.label.set_font('Times New Roman') 
        #     t.label.set_fontsize(7)
        
        # ax1.grid(b=True,axis='both',linestyle='--', linewidth=0.5)
        # ax2.grid(b=True,axis='both',linestyle='--', linewidth=0.5)
        # for axis in ['top', 'bottom', 'left', 'right']:
        #     ax1.spines[axis].set_linewidth(0.5)
        # for axis in ['top', 'bottom', 'left', 'right']:
        #     ax2.spines[axis].set_linewidth(0.5)
        
        # plt.savefig('DMHE_downwash/position_DMHE.png')
        # plt.show()


    
    elif controller == 'c':
        dataset={'a':"L1-AC_downwash/09_21_50_vehicle_local_position_0.csv",
             'b':"L1-AC_downwash/09_21_50_position_setpoint_triplet_0.csv",
             'c':"L1-AC_downwash/09_21_50_sensor_combined_0.csv",
             'd':"L1-AC_downwash/09_21_50_vehicle_attitude_0.csv",
             'e':"2nd big drone/log_18_2023-2-21-14-16-50_vehicle_local_position_0.csv"}
        local_position = pd.read_csv(dataset['a'])
        position_setpoint = pd.read_csv(dataset['b'])
        sensor         = pd.read_csv(dataset['c'])
        attitude       = pd.read_csv(dataset['d'])
        dataframe_a    = pd.DataFrame(local_position)
        dataframe_b    = pd.DataFrame(position_setpoint)
        dataframe_c    = pd.DataFrame(sensor)
        dataframe_d    = pd.DataFrame(attitude)
        bigdrone       = pd.read_csv(dataset['e'])
        dataframe_e    = pd.DataFrame(bigdrone)
        position_x     = dataframe_a['x']
        position_y     = dataframe_a['y']
        position_z     = dataframe_a['z']
        pos_big_x      = dataframe_e['x']
        pos_big_y      = dataframe_e['y']
        pos_big_z      = dataframe_e['z']
        position_enu   = np.zeros((5,n_end))
        ref_enu        = np.zeros((5,n_end))
        pos_big_enu    = np.zeros((3,len(pos_big_x)))
        n_start = 54 # During the initial interval n_start*dt, the quadrotor is still on the ground and the estimator does not begin to work
        for k in range(len(pos_big_x)):
            pos_big_ned  = np.array([[pos_big_x[k],pos_big_y[k],pos_big_z[k]]]).T
            pos_big_enu[:,k:k+1]  = np.matmul(R_enu,pos_big_ned)
        
        i = 0
        Time       = []
        time       = 0
        for k in range(n_start,n_start+n_end,1):
            Time += [time]
            pos_ned      = np.array([[position_x[k],position_y[k],position_z[k]]]).T
            pos_enu      = np.matmul(R_enu,pos_ned)
            position_enu[0:3,i:i+1] = pos_enu # transfer to ENU frame
            position_enu[3,i:i+1]   = LA.norm(pos_enu[0:2,0])
            position_enu[4,i:i+1]   = LA.norm(pos_enu)
            if time <T_end:
                ref_p, ref_v, ref_a, ref_j, ref_s = uav.ref_exp2(Coeffx_exp2, Coeffy_exp2, Coeffz_exp2, time, t_switch) 
            else:
                ref_p, ref_v, ref_a, ref_j, ref_s = np.array([[0,0,0.12]]).T, np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1))
            ref_enu[0:3,i:i+1] = ref_p
            ref_enu[3,i:i+1]   = LA.norm(ref_p[0:2,0])
            ref_enu[4,i:i+1]   = LA.norm(ref_p)
            i += 1
            time += dt
  
        np.save('ref_enu_dw_l1',ref_enu)
        np.save('pos_enu_dw_l1',position_enu)
        np.save('Time_dw_l1',Time)
        # load the computed total force command
        f_cmd = dataframe_b['current.vx']
        # load the forgetting factors
        gamma_r = dataframe_b['current.vy']
        gamma_p = dataframe_b['current.vz']
        # load the NeuroMHE estimation data in NED frame
        dfx_mhe = dataframe_b['current.a_x']
        dfy_mhe = dataframe_b['current.a_y']
        dfz_mhe = dataframe_b['current.a_z']
        Df_Imhe_enu = np.zeros((5,n_end))
        i = 0
        for k in range(n_start,n_start+n_end,1):
            Dfmhe_ned = np.array([[dfx_mhe[k],dfy_mhe[k],dfz_mhe[k]]]).T
            dfmhe_enu = np.matmul(R_enu,Dfmhe_ned)
            Df_Imhe_enu[0:3,i:i+1] = dfmhe_enu
            Df_Imhe_enu[3,i:i+1]   = LA.norm(dfmhe_enu[0:2,0])
            Df_Imhe_enu[4,i:i+1]   = LA.norm(dfmhe_enu)
            i += 1

        np.save('Df_Imhe_dw_l1',Df_Imhe_enu)
        # load the accelerometer data in NED frame
        acc_x   = dataframe_c['accelerometer_m_s2[0]']
        acc_y   = dataframe_c['accelerometer_m_s2[1]']
        acc_z   = dataframe_c['accelerometer_m_s2[2]']
        Acc_b_ned   = np.zeros((3,len(acc_x)))
        for k in range(len(acc_x)):
            Acc_b_ned[:,k:k+1] = np.array([[acc_x[k],acc_y[k],acc_z[k]]]).T
        # load the quaternion
        Q0 = dataframe_d['q[0]']
        Q1 = dataframe_d['q[1]']
        Q2 = dataframe_d['q[2]']
        Q3 = dataframe_d['q[3]']
        
        Gt = np.zeros((3,n_end))
        Gt_lpf = np.zeros((5,n_end))
        Gamma  = np.zeros((2,n_end))
        # print('len(ref_x)=',len(Ref_x),'len(acc_x)=',len(acc_x),'len(dfx_mhe)=',len(dfx_mhe),'len(position_x)=',len(position_x),'len(f_cmd)=',len(f_cmd))
        acc_I_prev = np.array([[0,0,-9.78]]).T # initial acceleration of the low-pass filter in NED frame
        
        i = 0
        # calculation of the ground truth data using the two steps mentioned above
        for k in range(n_start,n_start+n_end,1): 
            q0, q1, q2, q3 = Q0[k], Q1[k], Q2[k], Q3[k]
            quaternion = np.array([q0, q1, q2, q3])
            R_b        = Quaternion2Rotation(quaternion)
            acc_I_ned = np.matmul(R_b,np.reshape(Acc_b_ned[:,k],(3,1)))
            f_cmdI_ned = np.matmul(R_b,(f_cmd[k]*z))
            gt        = acc_I_ned*mass+f_cmdI_ned # we do not plot the raw accelerometer data as it is too noisy
            Gt[:,i:i+1] =  np.matmul(R_enu,gt) # transfer to ENU frame
            gamma_ned = np.array([[gamma_r[k],0,gamma_p[k]]]).T
            gamma_edu = np.matmul(R_enu,gamma_ned)
            Gamma[:,i:i+1]  = np.reshape(gamma_edu[1:3,0],(2,1))
            
            i += 1
        # np.save('16-Jan-data-exp2(downwash)/Time',Time)
        Gt_lpf[0,:] = lfilter(taps, 1.0, Gt[0,:])
        Gt_lpf[1,:] = lfilter(taps, 1.0, Gt[1,:])
        Gt_lpf[2,:] = lfilter(taps, 1.0, Gt[2,:])
        for k in range(len(Gt_lpf[0,:])):
            Gt_lpf[3,k:k+1]=LA.norm(Gt_lpf[0:2,k])
            Gt_lpf[4,k:k+1]=LA.norm(Gt_lpf[0:3,k])
 
        np.save('Gt_lpf_dw_l1',Gt_lpf)
        n_dw_start = int(16.25*sample_rate) # the time when the downwash takes effect, including the post downwash effect
        n_dw_steady = int(13*sample_rate)
        n_dw_end   = int(18.25*sample_rate)
        
        rmse_pxy = format(mean_squared_error(ref_enu[3,n_dw_start:n_dw_end], position_enu[3,n_dw_start:n_dw_end], squared=False),'.3f')
        est_pz  = format(mean_squared_error(ref_enu[2,n_dw_steady:n_dw_end], position_enu[2,n_dw_steady:n_dw_end], squared=False),'.3f')
        # rmse_p   = format(mean_squared_error(ref_enu[4,n_dw_start:n_dw_end], position_enu[4,n_dw_start:n_dw_end], squared=False),'.3f')
        rmse_fxy = format(mean_squared_error(Gt_lpf[3,n_dw_start:n_dw_end], Df_Imhe_enu[3,n_dw_start:n_dw_end], squared=False),'.3f')
        rmse_fz  = format(mean_squared_error(Gt_lpf[2,n_dw_start:n_dw_end], Df_Imhe_enu[2,n_dw_start:n_dw_end], squared=False),'.3f')
        # rmse_f   = format(mean_squared_error(Gt_lpf[4,n_dw_start:n_dw_end], Df_Imhe_enu[4,n_dw_start:n_dw_end], squared=False),'.3f')
        pxy_max  = format(np.max(position_enu[3,n_dw_start:n_dw_end]),'.3f')
        ez_max   = format(1.2-np.max(position_enu[2,n_dw_start:n_dw_end]),'.3f')
        mean_steady = np.mean(position_enu[2,n_dw_steady:n_dw_end])
        ez       = (position_enu[2,n_dw_steady:n_dw_end]-mean_steady)**2
        std      = format(np.sqrt(np.mean(ez)),'.3f')
        print('rmse_pxy=',rmse_pxy,'pxy_max=',pxy_max,'estrmse_pz=',est_pz,'ez_max=',ez_max,'rmse_fxy=',rmse_fxy,'rmse_fz=',rmse_fz,'std=',std)


        # fig, (ax1,ax2,ax3,ax4) = plt.subplots(4,sharex=True,figsize=(6/cm_2_inch,9/cm_2_inch),dpi=600)
        # ax1.plot(Time,Gt_lpf[3,:], linewidth=0.5,color='black')
        # ax1.plot(Time,Df_Imhe_enu[3,:], linewidth=0.5,color='orange')
        # ax1.set_xticks(np.arange(0,25,8))
        # ax1.tick_params(axis='y',which='major',pad=0)
        # ax1.set_ylabel('$d_{xy}$ [N]',labelpad=4,**font1) 
        # ax1.add_patch(Rectangle((6.75, 0), 9, 1.8,
        #      facecolor = 'blue',
        #      fill=True,
        #      alpha = 0.1,
        #      lw=0.1))
        # ax1.add_patch(Rectangle((15.75, 0), 2, 1.8,
        #      facecolor = 'navy',
        #      fill=True,
        #      alpha = 0.2,
        #      lw=0.1))
        # ax2.plot(Time,Gt_lpf[2,:], linewidth=0.5,color='black')
        # ax2.plot(Time,Df_Imhe_enu[2,:], linewidth=0.5,color='orange')
        # ax2.set_xticks(np.arange(0,25,8))
        # ax2.tick_params(axis='y',which='major',pad=0)
        # ax2.set_ylabel('$d_{z}$ [N]',labelpad=0,**font1)
        # ax2.add_patch(Rectangle((6.75, -10), 9, 10,
        #      facecolor = 'blue',
        #      fill=True,
        #      alpha = 0.1,
        #      lw=0.1))
        # ax2.add_patch(Rectangle((15.75, -10), 2, 10,
        #      facecolor = 'navy',
        #      fill=True,
        #      alpha = 0.2,
        #      lw=0.1))
        # ax3.plot(Time,position_enu[3,:], linewidth=0.5,color='orange')
        # ax3.set_xticks(np.arange(0,25,8))
        # ax3.tick_params(axis='y',which='major',pad=0)
        # ax3.set_ylabel('$p_{xy}$ [m]',labelpad=0,**font1)
        # ax3.add_patch(Rectangle((6.75, 0), 9, 0.4,
        #      facecolor = 'blue',
        #      fill=True,
        #      alpha = 0.1,
        #      lw=0.1))
        # ax3.add_patch(Rectangle((15.75, 0), 2, 0.4,
        #      facecolor = 'navy',
        #      fill=True,
        #      alpha = 0.2,
        #      lw=0.1))
        # ax4.plot(Time,ref_enu[2,:], linewidth=0.5,color='black')
        # ax4.plot(Time,position_enu[2,:], linewidth=0.5,color='orange')
        # ax4.plot(Time,pos_big_enu[2,(600):(n_end+600)], linewidth=0.5,color='green')
        # ax4.set_xticks(np.arange(0,25,8))
        # ax4.tick_params(axis='x',which='major',pad=0)
        # ax4.tick_params(axis='y',which='major',pad=0)
        # ax4.set_xlabel('Time [s]',labelpad=0,**font1)
        # ax4.set_ylabel('$p_{z}$ [m]',labelpad=4,**font1)
        # ax4.add_patch(Rectangle((6.75, 0), 9, 2.2,
        #      facecolor = 'blue',
        #      fill=True,
        #      alpha = 0.1,
        #      lw=0.1))
        # ax4.add_patch(Rectangle((15.75, 0), 2, 2.2,
        #      facecolor = 'navy',
        #      fill=True,
        #      alpha = 0.2,
        #      lw=0.1))
        # for t in ax1.yaxis.get_major_ticks(): 
        #     t.label.set_font('Times New Roman') 
        #     t.label.set_fontsize(7)
        # for t in ax2.yaxis.get_major_ticks(): 
        #     t.label.set_font('Times New Roman') 
        #     t.label.set_fontsize(7)
        # for t in ax3.yaxis.get_major_ticks(): 
        #     t.label.set_font('Times New Roman') 
        #     t.label.set_fontsize(7)
        # for t in ax4.xaxis.get_major_ticks(): 
        #     t.label.set_font('Times New Roman') 
        #     t.label.set_fontsize(7)
        # for t in ax4.yaxis.get_major_ticks(): 
        #     t.label.set_font('Times New Roman') 
        #     t.label.set_fontsize(7)
        # ax1.grid(b=True,axis='both',linestyle='--', linewidth=0.5)
        # ax2.grid(b=True,axis='both',linestyle='--', linewidth=0.5)
        # ax3.grid(b=True,axis='both',linestyle='--', linewidth=0.5)
        # ax4.grid(b=True,axis='both',linestyle='--', linewidth=0.5)
        # for axis in ['top', 'bottom', 'left', 'right']:
        #     ax1.spines[axis].set_linewidth(0.5)
        # for axis in ['top', 'bottom', 'left', 'right']:
        #     ax2.spines[axis].set_linewidth(0.5)
        # for axis in ['top', 'bottom', 'left', 'right']:
        #     ax3.spines[axis].set_linewidth(0.5)
        # for axis in ['top', 'bottom', 'left', 'right']:
        #     ax4.spines[axis].set_linewidth(0.5)
        # plt.savefig('L1-AC_downwash/force_position_L1AC_corrected.png')
        # plt.show()

        # fig, (ax1,ax2) = plt.subplots(2,sharex=True,figsize=(6/cm_2_inch,6*0.6/cm_2_inch),dpi=600) # 8, 8*0.45
        # ax1.plot(Time,Gt_lpf[3,:], linewidth=0.5,color='black')
        # ax1.plot(Time,Df_Imhe_enu[3,:], linewidth=0.5,color='orange')
        # ax1.set_xticks(np.arange(0,25,8))
        # ax1.tick_params(axis='y',which='major',pad=0)
        # ax1.set_ylabel('$d_{xy}$ [N]',labelpad=4,**font1) 
        # ax1.add_patch(Rectangle((7.25, 0), 9, 1.8,
        #      facecolor = 'blue',
        #      fill=True,
        #      alpha = 0.1,
        #      lw=0.1))
        # ax1.add_patch(Rectangle((16.25, 0), 2, 1.8,
        #      facecolor = 'navy',
        #      fill=True,
        #      alpha = 0.2,
        #      lw=0.1))
        # ax2.plot(Time,Gt_lpf[2,:], linewidth=0.5,color='black')
        # ax2.plot(Time,Df_Imhe_enu[2,:], linewidth=0.5,color='orange')
        # ax2.set_xticks(np.arange(0,25,8))
        # ax2.tick_params(axis='x',which='major',pad=-0.5)
        # ax2.tick_params(axis='y',which='major',pad=0)
        # ax2.set_xlabel('Time [s]',labelpad=-5.5,**font1)
        # ax2.set_ylabel('$d_{z}$ [N]',labelpad=0,**font1)
        # ax2.add_patch(Rectangle((7.25, -10), 9, 10,
        #      facecolor = 'blue',
        #      fill=True,
        #      alpha = 0.1,
        #      lw=0.1))
        # ax2.add_patch(Rectangle((16.25, -10), 2, 10,
        #      facecolor = 'navy',
        #      fill=True,
        #      alpha = 0.2,
        #      lw=0.1))
        # ax1.grid(b=True,axis='both',linestyle='--', linewidth=0.5)
        # ax2.grid(b=True,axis='both',linestyle='--', linewidth=0.5)
        # for t in ax1.yaxis.get_major_ticks(): 
        #     t.label.set_font('Times New Roman') 
        #     t.label.set_fontsize(7)
        # for t in ax2.xaxis.get_major_ticks(): 
        #     t.label.set_font('Times New Roman') 
        #     t.label.set_fontsize(7)
        # for t in ax2.yaxis.get_major_ticks(): 
        #     t.label.set_font('Times New Roman') 
        #     t.label.set_fontsize(7)
        # for axis in ['top', 'bottom', 'left', 'right']:
        #     ax1.spines[axis].set_linewidth(0.5)
        # for axis in ['top', 'bottom', 'left', 'right']:
        #     ax2.spines[axis].set_linewidth(0.5)
        # plt.savefig('L1-AC_downwash/force_L1AC_downwash_h.png')
        # plt.show()


        # fig, (ax1,ax2) = plt.subplots(2,sharex=True,figsize=(5/cm_2_inch,5*0.65/cm_2_inch),dpi=600)
        # ax1.plot(Time,position_enu[3,:], linewidth=0.5,color='orange')
        # ax1.set_xticks(np.arange(0,25,8))
        # ax1.tick_params(axis='y',which='major',pad=0)
        # ax1.set_ylabel('$p_{xy}$ [m]',labelpad=-2,**font1)
        # ax1.yaxis.set_major_formatter(OOMFormatter(-1,"%1.1f"))
        # ax1.yaxis.get_offset_text().set_fontsize(6)
        # ax1.add_patch(Rectangle((7.25, 0), 9, 0.4,
        #      facecolor = 'blue',
        #      fill=True,
        #      alpha = 0.1,
        #      lw=0.1))
        # ax1.add_patch(Rectangle((16.25, 0), 2, 0.4,
        #      facecolor = 'navy',
        #      fill=True,
        #      alpha = 0.2,
        #      lw=0.1))
        # ax2.plot(Time,ref_enu[2,:], linewidth=0.5,color='black')
        # ax2.plot(Time,position_enu[2,:], linewidth=0.5,color='orange')
        # ax2.plot(Time,pos_big_enu[2,(600):(n_end+600)], linewidth=0.5,color='green')
        # ax2.set_xticks(np.arange(0,25,8))
        # ax2.tick_params(axis='x',which='major',pad=-0.5)
        # ax2.tick_params(axis='y',which='major',pad=0)
        # ax2.set_xlabel('Time [s]',labelpad=-6,**font1)
        # ax2.set_ylabel('$p_{z}$ [m]',labelpad=4,**font1)
        # ax2.add_patch(Rectangle((7.25, 0), 9, 2.2,
        #      facecolor = 'blue',
        #      fill=True,
        #      alpha = 0.1,
        #      lw=0.1))
        # ax2.add_patch(Rectangle((16.25, 0), 2, 2.2,
        #      facecolor = 'navy',
        #      fill=True,
        #      alpha = 0.2,
        #      lw=0.1))
        
        # for t in ax1.yaxis.get_major_ticks(): 
        #     t.label.set_font('Times New Roman') 
        #     t.label.set_fontsize(7)
        # for t in ax2.xaxis.get_major_ticks(): 
        #     t.label.set_font('Times New Roman') 
        #     t.label.set_fontsize(7)
        # for t in ax2.yaxis.get_major_ticks(): 
        #     t.label.set_font('Times New Roman') 
        #     t.label.set_fontsize(7)
        
        # ax1.grid(b=True,axis='both',linestyle='--', linewidth=0.5)
        # ax2.grid(b=True,axis='both',linestyle='--', linewidth=0.5)
        # for axis in ['top', 'bottom', 'left', 'right']:
        #     ax1.spines[axis].set_linewidth(0.5)
        # for axis in ['top', 'bottom', 'left', 'right']:
        #     ax2.spines[axis].set_linewidth(0.5)
        
        # plt.savefig('L1-AC_downwash/position_L1AC.png')
        # plt.show()

    elif controller == 'd':
        print('nothing to do for this controller mode')
    elif controller == 'e':
        dataset={'a':"Baseline_samegain_downwash/log_7_2023-2-21-17-49-36_vehicle_local_position_0.csv",
             'b':"Baseline_largergain_downwash/log_9_2023-2-21-17-54-40_vehicle_local_position_0.csv",
             'c':"2nd big drone/log_18_2023-2-21-14-16-50_vehicle_local_position_0.csv",
             'd':"Baseline_largergain_downwash/log_9_2023-2-21-17-54-40_position_setpoint_triplet_0.csv",
             'e':"Baseline_largergain_downwash/log_9_2023-2-21-17-54-40_sensor_combined_0.csv",
             'f':"Baseline_largergain_downwash/log_9_2023-2-21-17-54-40_vehicle_attitude_0.csv"}
        local_position = pd.read_csv(dataset['a'])
        local_position_lg = pd.read_csv(dataset['b'])
        bigdrone       = pd.read_csv(dataset['c'])
        position_setpoint = pd.read_csv(dataset['d'])
        sensor         = pd.read_csv(dataset['e'])
        attitude       = pd.read_csv(dataset['f'])
        dataframe_a    = pd.DataFrame(local_position)
        dataframe_b    = pd.DataFrame(local_position_lg)
        dataframe_c    = pd.DataFrame(bigdrone)
        dataframe_d    = pd.DataFrame(position_setpoint)
        dataframe_e    = pd.DataFrame(sensor)
        dataframe_f    = pd.DataFrame(attitude)
        # load the position data in NED frame
        position_x     = dataframe_a['x']
        position_y     = dataframe_a['y']
        position_z     = dataframe_a['z']
        position_x_lg  = dataframe_b['x']
        position_y_lg  = dataframe_b['y']
        position_z_lg  = dataframe_b['z']
        pos_big_x      = dataframe_c['x']
        pos_big_y      = dataframe_c['y']
        pos_big_z      = dataframe_c['z']


        n_start = 52
        n_start_lg = 52
        position_enu   = np.zeros((5,n_end))
        position_enu_lg= np.zeros((5,n_end))
        ref_enu        = np.zeros((5,n_end))
        pos_big_enu    = np.zeros((3,len(pos_big_x)))
        n_start = 54 # During the initial interval n_start*dt, the quadrotor is still on the ground and the estimator does not begin to work
        for k in range(len(pos_big_x)):
            pos_big_ned  = np.array([[pos_big_x[k],pos_big_y[k],pos_big_z[k]]]).T
            pos_big_enu[:,k:k+1]  = np.matmul(R_enu,pos_big_ned)
        
        i = 0
        Time       = []
        time       = 0
        for k in range(n_start,n_start+n_end,1):
            Time += [time]
            pos_ned      = np.array([[position_x[k],position_y[k],position_z[k]]]).T
            pos_enu      = np.matmul(R_enu,pos_ned)
            pos_ned_lg   = np.array([[position_x_lg[k],position_y_lg[k],position_z_lg[k]]]).T
            pos_enu_lg   = np.matmul(R_enu,pos_ned_lg)
            position_enu[0:3,i:i+1] = pos_enu # transfer to ENU frame
            position_enu[3,i:i+1]   = LA.norm(pos_enu[0:2,0])
            position_enu[4,i:i+1]   = LA.norm(pos_enu)
            position_enu_lg[0:3,i:i+1] = pos_enu_lg # transfer to ENU frame
            position_enu_lg[3,i:i+1]   = LA.norm(pos_enu_lg[0:2,0])
            position_enu_lg[4,i:i+1]   = LA.norm(pos_enu_lg)
            if time <T_end:
                ref_p, ref_v, ref_a, ref_j, ref_s = uav.ref_exp2(Coeffx_exp2, Coeffy_exp2, Coeffz_exp2, time, t_switch) 
            else:
                ref_p, ref_v, ref_a, ref_j, ref_s = np.array([[0,0,0.12]]).T, np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1))
            ref_enu[0:3,i:i+1] = ref_p
            ref_enu[3,i:i+1]   = LA.norm(ref_p[0:2,0])
            ref_enu[4,i:i+1]   = LA.norm(ref_p)
            i += 1
            time += dt
        
        np.save('ref_enu_dw_baseline',ref_enu)
        np.save('pos_enu_dw_baseline_lg',position_enu_lg)
        np.save('Time_dw_baseline',Time)
        # load the computed total force command
        f_cmd = dataframe_d['current.vx']
        # load the forgetting factors
        gamma_r = dataframe_d['current.vy']
        gamma_p = dataframe_d['current.vz']
        # load the NeuroMHE estimation data in NED frame
        dfx_mhe = dataframe_d['current.a_x']
        dfy_mhe = dataframe_d['current.a_y']
        dfz_mhe = dataframe_d['current.a_z']
        Df_Imhe_enu = np.zeros((5,n_end))
        i = 0
        for k in range(n_start,n_start+n_end,1):
            Dfmhe_ned = np.array([[dfx_mhe[k],dfy_mhe[k],dfz_mhe[k]]]).T
            dfmhe_enu = np.matmul(R_enu,Dfmhe_ned)
            Df_Imhe_enu[0:3,i:i+1] = dfmhe_enu
            Df_Imhe_enu[3,i:i+1]   = LA.norm(dfmhe_enu[0:2,0])
            Df_Imhe_enu[4,i:i+1]   = LA.norm(dfmhe_enu)
            i += 1
        # load the accelerometer data in NED frame
        acc_x   = dataframe_e['accelerometer_m_s2[0]']
        acc_y   = dataframe_e['accelerometer_m_s2[1]']
        acc_z   = dataframe_e['accelerometer_m_s2[2]']
        Acc_b_ned   = np.zeros((3,len(acc_x)))
        for k in range(len(acc_x)):
            Acc_b_ned[:,k:k+1] = np.array([[acc_x[k],acc_y[k],acc_z[k]]]).T
        # load the quaternion
        Q0 = dataframe_f['q[0]']
        Q1 = dataframe_f['q[1]']
        Q2 = dataframe_f['q[2]']
        Q3 = dataframe_f['q[3]']
        
        Gt = np.zeros((3,n_end))
        Gt_lpf = np.zeros((5,n_end))
        Gamma  = np.zeros((2,n_end))
        # print('len(ref_x)=',len(Ref_x),'len(acc_x)=',len(acc_x),'len(dfx_mhe)=',len(dfx_mhe),'len(position_x)=',len(position_x),'len(f_cmd)=',len(f_cmd))
        acc_I_prev = np.array([[0,0,-9.78]]).T # initial acceleration of the low-pass filter in NED frame
        
        i = 0
        # calculation of the ground truth data using the two steps mentioned above
        for k in range(n_start,n_start+n_end,1): 
            q0, q1, q2, q3 = Q0[k], Q1[k], Q2[k], Q3[k]
            quaternion = np.array([q0, q1, q2, q3])
            R_b        = Quaternion2Rotation(quaternion)
            acc_I_ned = np.matmul(R_b,np.reshape(Acc_b_ned[:,k],(3,1)))
            f_cmdI_ned = np.matmul(R_b,(f_cmd[k]*z))
            gt        = acc_I_ned*mass+f_cmdI_ned # we do not plot the raw accelerometer data as it is too noisy
            Gt[:,i:i+1] =  np.matmul(R_enu,gt) # transfer to ENU frame
            gamma_ned = np.array([[gamma_r[k],0,gamma_p[k]]]).T
            gamma_edu = np.matmul(R_enu,gamma_ned)
            Gamma[:,i:i+1]  = np.reshape(gamma_edu[1:3,0],(2,1))
            
            i += 1
        # np.save('16-Jan-data-exp2(downwash)/Time',Time)
        Gt_lpf[0,:] = lfilter(taps, 1.0, Gt[0,:])
        Gt_lpf[1,:] = lfilter(taps, 1.0, Gt[1,:])
        Gt_lpf[2,:] = lfilter(taps, 1.0, Gt[2,:])
        for k in range(len(Gt_lpf[0,:])):
            Gt_lpf[3,k:k+1]=LA.norm(Gt_lpf[0:2,k])
            Gt_lpf[4,k:k+1]=LA.norm(Gt_lpf[0:3,k])

        n_dw_start = int(8.5*sample_rate) # the time when the downwash takes effect, including the post downwash effect
        n_dw_steady = int(12.5*sample_rate)
        n_dw_end   = int(17.5*sample_rate)
        
        rmse_pxy = format(mean_squared_error(ref_enu[3,n_dw_start:n_dw_end], position_enu[3,n_dw_start:n_dw_end], squared=False),'.3f')
        est_pz  = format(mean_squared_error(ref_enu[2,n_dw_steady:n_dw_end], position_enu[2,n_dw_steady:n_dw_end], squared=False),'.3f')
        # rmse_p   = format(mean_squared_error(ref_enu[4,n_dw_start:n_dw_end], position_enu[4,n_dw_start:n_dw_end], squared=False),'.3f')
        rmse_pxy_lg = format(mean_squared_error(ref_enu[3,n_dw_start:n_dw_end], position_enu_lg[3,n_dw_start:n_dw_end], squared=False),'.3f')
        est_pz_lg  = format(mean_squared_error(ref_enu[2,n_dw_steady:n_dw_end], position_enu_lg[2,n_dw_steady:n_dw_end], squared=False),'.3f')
        # rmse_p_lg  = format(mean_squared_error(ref_enu[4,n_dw_start:n_dw_end], position_enu_lg[4,n_dw_start:n_dw_end], squared=False),'.3f')
        pxy_max = format(np.max(position_enu[3,n_dw_start:n_dw_end]),'.3f')
        pxy_max_lg = format(np.max(position_enu_lg[3,n_dw_start:n_dw_end]),'.3f')
        ez_max   = format(1.2-np.min(position_enu[2,n_dw_start:n_dw_end]),'.3f')
        mean_steady = np.mean(position_enu[2,n_dw_steady:n_dw_end])
        ez       = (position_enu[2,n_dw_steady:n_dw_end]-mean_steady)**2
        std      = format(np.sqrt(np.mean(ez)),'.3f')
        ez_max_lg   = format(1.2-np.min(position_enu_lg[2,n_dw_start:n_dw_end]),'.3f')
        mean_steady_lg = np.mean(position_enu_lg[2,n_dw_steady:n_dw_end])
        ez_lg       = (position_enu_lg[2,n_dw_steady:n_dw_end]-mean_steady_lg)**2
        std_lg      = format(np.sqrt(np.mean(ez_lg)),'.3f')

        

        print('rmse_pxy=',rmse_pxy,'pxy_max=',pxy_max,'ez_max=',ez_max,'estrmse_pz=',est_pz,'std=',std,'rmse_pxy_lg=',rmse_pxy_lg,'pxy_max_lg=',pxy_max_lg,'ez_max_lg=',ez_max_lg,'estrmse_pz_lg=',est_pz_lg,'std_lg=',std_lg)

        # fig, (ax1,ax2) = plt.subplots(2,sharex=True,figsize=(6/cm_2_inch,6*0.6/cm_2_inch),dpi=600) # 8, 8*0.45
        # ax1.plot(Time,Gt_lpf[3,:], linewidth=0.5,color='black')
        # # ax1.plot(Time,Df_Imhe_enu[3,:], linewidth=0.5,color='orange')
        # ax1.set_xticks(np.arange(0,25,8))
        # ax1.tick_params(axis='y',which='major',pad=0)
        # ax1.set_ylabel('$d_{xy}$ [N]',labelpad=4,**font1) 
        # ax1.add_patch(Rectangle((8.5, 0), 9, 1.8,
        #      facecolor = 'blue',
        #      fill=True,
        #      alpha = 0.1,
        #      lw=0.1))
        # ax1.add_patch(Rectangle((17.5, 0), 2, 1.8,
        #      facecolor = 'navy',
        #      fill=True,
        #      alpha = 0.2,
        #      lw=0.1))
        # ax2.plot(Time,Gt_lpf[2,:], linewidth=0.5,color='black')
        # # ax2.plot(Time,Df_Imhe_enu[2,:], linewidth=0.5,color='orange')
        # ax2.set_xticks(np.arange(0,25,8))
        # ax2.tick_params(axis='x',which='major',pad=-0.5)
        # ax2.tick_params(axis='y',which='major',pad=0)
        # ax2.set_xlabel('Time [s]',labelpad=-5.5,**font1)
        # ax2.set_ylabel('$d_{z}$ [N]',labelpad=0,**font1)
        # ax2.add_patch(Rectangle((8.5, -10), 9, 10,
        #      facecolor = 'blue',
        #      fill=True,
        #      alpha = 0.1,
        #      lw=0.1))
        # ax2.add_patch(Rectangle((17.5, -10), 2, 10,
        #      facecolor = 'navy',
        #      fill=True,
        #      alpha = 0.2,
        #      lw=0.1))
        # ax1.grid(b=True,axis='both',linestyle='--', linewidth=0.5)
        # ax2.grid(b=True,axis='both',linestyle='--', linewidth=0.5)
        # for t in ax1.yaxis.get_major_ticks(): 
        #     t.label.set_font('Times New Roman') 
        #     t.label.set_fontsize(7)
        # for t in ax2.xaxis.get_major_ticks(): 
        #     t.label.set_font('Times New Roman') 
        #     t.label.set_fontsize(7)
        # for t in ax2.yaxis.get_major_ticks(): 
        #     t.label.set_font('Times New Roman') 
        #     t.label.set_fontsize(7)
        # for axis in ['top', 'bottom', 'left', 'right']:
        #     ax1.spines[axis].set_linewidth(0.5)
        # for axis in ['top', 'bottom', 'left', 'right']:
        #     ax2.spines[axis].set_linewidth(0.5)
        # plt.savefig('Baseline_samegain_downwash/force_baseline_downwash_h.png')
        # plt.show()


        # fig, (ax1,ax2) = plt.subplots(2,sharex=True,figsize=(5/cm_2_inch,5*0.65/cm_2_inch),dpi=600)
        # ax1.plot(Time,position_enu[3,:],linewidth=0.5,color='orange')
        # ax1.plot(Time,position_enu_lg[3,:],linewidth=0.5,color='blue')
        # ax1.set_xticks(np.arange(0,25,8))
        # ax1.tick_params(axis='y',which='major',pad=0)
        # ax1.set_ylabel('$p_{xy}$ [m]',labelpad=-2,**font1)
        # ax1.add_patch(Rectangle((8.5, 0), 9, 0.4,
        #      facecolor = 'blue',
        #      fill=True,
        #      alpha = 0.1,
        #      lw=0.1))
        # ax1.add_patch(Rectangle((17.5, 0), 2, 0.4,
        #      facecolor = 'navy',
        #      fill=True,
        #      alpha = 0.2,
        #      lw=0.1))
        # ax2.plot(Time,ref_enu[2,:], linewidth=0.5,color='black')
        # ax2.plot(Time,position_enu[2,:], linewidth=0.5,color='orange')
        # ax2.plot(Time,position_enu_lg[2,:],linewidth=0.5,color='blue')
        # ax2.plot(Time,pos_big_enu[2,(600):(n_end+600)], linewidth=0.5,color='green')
        # ax2.set_xticks(np.arange(0,25,8))
        # ax2.tick_params(axis='x',which='major',pad=-0.5)
        # ax2.tick_params(axis='y',which='major',pad=0)
        # ax2.set_xlabel('Time [s]',labelpad=-6,**font1)
        # ax2.set_ylabel('$p_{z}$ [m]',labelpad=4,**font1)
        # ax2.add_patch(Rectangle((8.5, 0), 9, 2.2,
        #      facecolor = 'blue',
        #      fill=True,
        #      alpha = 0.1,
        #      lw=0.1))
        # ax2.add_patch(Rectangle((17.5, 0), 2, 2.2,
        #      facecolor = 'navy',
        #      fill=True,
        #      alpha = 0.2,
        #      lw=0.1))
        # leg=ax2.legend(['Desired: ego quadrotor', 'Actual: original gain', r'Actual: $3\times$ gain', 'Actual: 2nd quadrotor'],loc=(0.09,0.6),prop=font1,labelspacing=0.1)
        # leg.get_frame().set_linewidth(0.5)
        # for t in ax1.yaxis.get_major_ticks(): 
        #     t.label.set_font('Times New Roman') 
        #     t.label.set_fontsize(7)
        # for t in ax2.xaxis.get_major_ticks(): 
        #     t.label.set_font('Times New Roman') 
        #     t.label.set_fontsize(7)
        # for t in ax2.yaxis.get_major_ticks(): 
        #     t.label.set_font('Times New Roman') 
        #     t.label.set_fontsize(7)
        # ax1.grid(b=True,axis='both',linestyle='--', linewidth=0.5)
        # ax2.grid(b=True,axis='both',linestyle='--', linewidth=0.5)
        # for axis in ['top', 'bottom', 'left', 'right']:
        #     ax1.spines[axis].set_linewidth(0.5)
        # for axis in ['top', 'bottom', 'left', 'right']:
        #     ax2.spines[axis].set_linewidth(0.5)
        # plt.savefig('Baseline_samegain_downwash/comparison_baseline_downwash.png')
        # plt.show()

    else:
        print('nothing to do for this controller mode')













"""
This is the main function that trains NeuroMHE and compares it with NeuroBEM
The training is done in a supervised learning fashion on a real flight dataset[4]
[4] Bauersfeld, L., Kaufmann, E., Foehn, P., Sun, S. and Scaramuzza, D., 2021. 
    NeuroBEM: Hybrid Aerodynamic Quadrotor Model. ROBOTICS: SCIENCE AND SYSTEM XVII.
----------------------------------------------------------------------------
WANG, Bingheng, 24 Dec. 2020, at Control & Simulation Lab, ECE Dept. NUS
Should you have any questions, please feel free to contact the author via: wangbingheng@u.nus.edu
---------------
1st version: 08,July,2021 
2nd version: 17,June,2022
"""
import UavEnv
import Robust_Flight_comparison_neurobem
from casadi import *
import time as TM
import numpy as np
import matplotlib.pyplot as plt
import uavNN
import torch
from numpy import linalg as LA
import math
import pandas as pd
from sklearn.metrics import mean_squared_error
import statistics
"""---------------------------------Learn or Evaluate?-------------------------------------"""
train = False
"""---------------------------------Load environment---------------------------------------"""
# System parameters used in the paper of 'NeuroBEM'
Sys_para = np.array([0.752, 0.00252, 0.00214, 0.00436])

# Sampling time-step, 400Hz
dt_sample = 0.0025
uav = UavEnv.quadrotor(Sys_para, dt_sample)
uav.model()
horizon = 10
# Learning rate
lr_nn   = 1e-4

"""---------------------------------Define parameterization model-----------------------------"""
D_in, D_h, D_out = 18, 100, 49
# model_QR = uavNN.Net(D_in, D_h, D_out)
r11     = np.array([[100]])

"""---------------------------------Quaternion to Rotation Matrix---------------------------"""
def Quaternion2Rotation(q):
    q0, q1, q2, q3 = q[0], q[1], q[2], q[3]
    R_h = np.array([[
        2*(q0**2+q1**2)-1, 2*q1*q2-2*q0*q3, 2*q0*q2+2*q1*q3,
        2*q0*q3+2*q1*q2, 2*(q0**2+q2**2)-1, 2*q2*q3-2*q0*q1,
        2*q1*q3-2*q0*q2, 2*q0*q1+2*q2*q3, 2*(q0**2+q3**2)-1
    ]]).T
    R = np.array([
        [2 * (q0 ** 2 + q1 ** 2) - 1, 2 * q1 * q2 - 2 * q0 * q3, 2 * q0 * q2 + 2 * q1 * q3],
        [2 * q0 * q3 + 2 * q1 * q2, 2 * (q0 ** 2 + q2 ** 2) - 1, 2 * q2 * q3 - 2 * q0 * q1],
        [2 * q1 * q3 - 2 * q0 * q2, 2 * q0 * q1 + 2 * q2 * q3, 2 * (q0 ** 2 + q3 ** 2) - 1]
    ])


    return R_h, R

"""---------------------------------Skew-symmetric Matrix-----------------------------------"""
def Skewsymmetric(v):
    v_cross = np.array([
        [0, -v[2, 0], v[1, 0]],
        [v[2, 0], 0, -v[0, 0]],
        [-v[1, 0], v[0, 0], 0]]
    )
    return v_cross

"""---------------------------------Compute Ground Truth------------------------------------"""
def GroundTruth(state, acc, mass, J_B):
    # ctrl is the 4-by-1 squared speed vector
    R_B = np.array([
        [state[6, 0], state[7, 0], state[8, 0]],
        [state[9, 0], state[10, 0], state[11, 0]],
        [state[12, 0], state[13, 0], state[14, 0]]]
    )
    P_I = np.array([[state[0, 0], state[1, 0], state[2, 0]]]).T
    v_B = np.array([[state[3, 0], state[4, 0], state[5, 0]]]).T
    gw  = np.array([[0, 0, 9.81]]).T # gravitational acceleration in inertial frame
    w_B = np.array([[state[15, 0], state[16, 0], state[17, 0]]]).T
    acc_p = np.array([[acc[0, 0], acc[1, 0], acc[2, 0]]]).T # measured in body frame
    acc_w = np.array([[acc[3, 0], acc[4, 0], acc[5, 0]]]).T # measured in body frame
    
    df    = mass*acc_p
    dt    = np.matmul(J_B, acc_w) + \
            np.cross(w_B.T, np.transpose(np.matmul(J_B, w_B))).T 
    # gt    = np.vstack((df, dt))
    return df, dt

"""---------------------------------Define MHE----------------------------------------------"""

uavMHE = Robust_Flight_comparison_neurobem.MHE(horizon, dt_sample,r11)
uavMHE.SetStateVariable(uav.xa)
uavMHE.SetOutputVariable()
uavMHE.SetNoiseVariable(uav.eta)
uavMHE.SetModelDyn(uav.dymh)
uavMHE.SetCostDyn()

"""---------------------------------Define NeuroMHE----------------------------------------------"""
uavNMHE = Robust_Flight_comparison_neurobem.KF_gradient_solver(uav.xa, r11)

"""---------------------------------Training process-----------------------------"""

# Model parameters using min-max normalization, as was done for DNN-parameterization.
def SetPara(r11, tunable_para):
    epsilon = 1e-4
    gmin, gmax = 1e-2, 1
    p_diag = np.zeros((1, 24))
    for i in range(24):
        p_diag[0, i] = epsilon + tunable_para[0, i]**2
    P0 = np.diag(p_diag[0])
    gamma_r = gmin + (gmax - gmin) * 1/(1+np.exp(-tunable_para[0, 24]))
    gamma_q = gmin + (gmax - gmin) * 1/(1+np.exp(-tunable_para[0, 42]))
    # gamma_r = gmin + tunable_para[0, 24]**2
    # gamma_q = gmin + tunable_para[0, 42]**2
    r_diag = np.zeros((1, 17))
    for i in range(17):
        r_diag[0, i] = epsilon + tunable_para[0, i+25]**2
    r      = np.hstack((r11, r_diag))
    # r      = r_diag
    r      = np.reshape(r, (1, 18))
    R_N    = np.diag(r[0])
    q_diag = np.zeros((1, 6))
    for i in range(6):
        q_diag[0, i] = epsilon + tunable_para[0, i+43]**2
    Q_N1   = np.diag(q_diag[0])
    weight_para = np.hstack((p_diag, np.reshape(gamma_r, (1,1)), r_diag, np.reshape(gamma_q,(1,1)), q_diag))
    return P0, gamma_r, gamma_q, R_N, Q_N1, weight_para

def chainRule_gradient(tunable_para):
    tunable = SX.sym('tunable', 1, 49)
    P = SX.sym('P', 1, 24)
    epsilon = 1e-4
    gmin, gmax = 1e-2, 1
    for i in range(24):
        P[0, i] = epsilon + tunable[0, i]**2
    # gamma_r = gmin + (gmax - gmin) * 1/(1+exp(-tunable_para[0, 24]))
    gamma_r = gmin + (gmax - gmin) * 1/(1+exp(-tunable[0, 24]))
    R = SX.sym('R', 1, 17)
    for i in range(17):
        R[0, i] = epsilon + tunable[0, i+25]**2
    # gamma_q = gmin + (gmax - gmin) * 1/(1+exp(-tunable_para[0, 42]))
    gamma_q = gmin + (gmax - gmin) * 1/(1+exp(-tunable[0, 42]))
    Q = SX.sym('Q', 1, 6)
    for i in range(6):
        Q[0, i] = epsilon + tunable[0, i+43]**2
    weight = horzcat(P, gamma_r, R, gamma_q, Q)
    w_jaco = jacobian(weight, tunable)
    w_jaco_fn = Function('W_fn',[tunable],[w_jaco],['tp'],['W_fnf'])
    weight_grad = w_jaco_fn(tp=tunable_para)['W_fnf'].full()
    return weight_grad

def convert(parameter):
    tunable_para = np.zeros((1,D_out))
    for i in range(D_out):
        tunable_para[0,i] = parameter[i,0]
    return tunable_para

def Train():
    # reset neural network model for pre-training a neural network model
    model_QR = uavNN.Net(D_in, D_h, D_out)
    # load the pre-trained neural network model
    # PATH1 = "pretrained_model_QR.pt"
    # model_QR = torch.load(PATH1)
    n = 1
    # Loss function
    Mean_Loss = []
    # Dictionary of the dataset for training
    train_set = {'a': "fig8_data_for_training.csv"}   # selected from "merged_2021-02-23-14-41-07_seg_3.csv"   
    
    # number of trained episode
    n_ep = 1
    Trained = []
    delta_cost = 10000
    eps  = 1
    mean_loss0 = 0
    Gamma2 = []

    # count of negative trace_sum
    count = 0

    # record loss in training
    Loss_Training = []
 
    while delta_cost >= eps:
        # Sum of loss
        sum_loss = 0.0
        # Sum of time
        sum_time = 0.0
        # index of total iteration during one episode
        
        # estimated process noise
        noise_traj = np.zeros((1,6))
        noise_list = []
        noise_list += [np.reshape(noise_traj[-1,:],(6,1))]
        Grad_time = []
        for key in train_set:
            it = 0
            Trained += [n_ep]
            # Obtain the size of the data
            dataset = pd.read_csv(train_set[key])
            dataframe = pd.DataFrame(dataset)
            time_seq = dataframe['t']
            Ntrain = int(time_seq.size/n) # sample the feedback over every 4 points
            # Estimation error matrix for one episode
            Ew_epi = np.zeros((6, Ntrain))
            Time   = np.zeros(Ntrain)
            loss_training = np.zeros((1, Ntrain))
            # Obtain the sequences of the state, acc, and motor speed from the dataframe
            angaccx_seq, angaccy_seq, angaccz_seq = dataframe['ang acc x'], dataframe['ang acc y'], dataframe[
                'ang acc z']
            angvelx_seq, angvely_seq, angvelz_seq = dataframe['ang vel x'], dataframe['ang vel y'], dataframe[
                'ang vel z']
            qx_seq, qy_seq, qz_seq, qw_seq = dataframe['quat x'], dataframe['quat y'], dataframe['quat z'], dataframe[
                'quat w']
            accx_seq, accy_seq, accz_seq = dataframe['acc x'], dataframe['acc y'], dataframe['acc z']
            velx_seq, vely_seq, velz_seq = dataframe['vel x'], dataframe['vel y'], dataframe['vel z']
            posx_seq, posy_seq, posz_seq = dataframe['pos x'], dataframe['pos y'], dataframe['pos z']
            mot1_seq, mot2_seq, mot3_seq, mot4_seq = dataframe['mot 1'], dataframe['mot 2'], dataframe['mot 3'], \
                                                     dataframe['mot 4']
            # Initial states
            r_I0 = np.array([[posx_seq[0], posy_seq[0], posz_seq[0]]]).T
            v_B0 = np.array([[velx_seq[0], vely_seq[0], velz_seq[0]]]).T
            q0 = np.array([qw_seq[0], qx_seq[0], qy_seq[0], qz_seq[0]])
            w_B0 = np.array([[angvelx_seq[0], angvely_seq[0], angvelz_seq[0]]]).T
            R_Bh0, R_B0 = Quaternion2Rotation(q0)  # 9-by-1
            v_I0 = np.matmul(R_B0, v_B0)  # +np.matmul(np.matmul(R_B0, skew_w), r_I0)
            state0 = np.vstack((r_I0, v_B0, R_Bh0, w_B0))
            acc = np.array([[accx_seq[0], accy_seq[0], accz_seq[0], angaccx_seq[0], angaccy_seq[0], angaccz_seq[0]]]).T
            control = np.array([[mot1_seq[0] ** 2, mot2_seq[0] ** 2, mot3_seq[0] ** 2, mot4_seq[0] ** 2]]).T  # Update control after the MHE is solved
            df_t, dt_t = GroundTruth(state0, acc, uav.m, uav.J)
            df_B0 = np.reshape(df_t, (3, 1))
            dt_B0 = np.reshape(dt_t, (3, 1))
            # df_B0 = np.zeros((3, 1))
            # dt_B0 = np.zeros((3, 1))
            x_hatmh = np.vstack((r_I0, v_B0, df_B0, R_Bh0, w_B0, dt_B0))
            m       = x_hatmh
            xmhe_traj = x_hatmh.T
            # Measurement list
            Y = []
            # Control list
            ctrl = []
            # Motor normalized speed list
            speed = []
            # Ground_truth list
            ground_truth = []  
            for j in range(Ntrain):
                # Solve MHE to obtain the estimated state trajectory within a horizon
                # Take measurement and motor speed
                r_I = np.array([[posx_seq[n*j], posy_seq[n*j], posz_seq[n*j]]]).T
                v_B = np.array([[velx_seq[n*j], vely_seq[n*j], velz_seq[n*j]]]).T
                q = np.array([qw_seq[n*j], qx_seq[n*j], qy_seq[n*j], qz_seq[n*j]])
                w_B = np.array([[angvelx_seq[n*j], angvely_seq[n*j], angvelz_seq[n*j]]]).T
                R_Bh, R_B = Quaternion2Rotation(q)  # 9-by-1
                v_I = np.matmul(R_B, v_B)  # +np.matmul(np.matmul(R_B, skew_w), r_I0)
                state = np.vstack((r_I, v_B, R_Bh, w_B)) 
                Y += [state]
                input_nn_QR = state # based on current measurement and speed
                tunable_para = convert(model_QR(input_nn_QR))
                P0, gamma_r, gamma_q, R_N, Q_N1, weight_para = SetPara(r11, tunable_para)
                opt_sol      = uavMHE.MHEsolver(Y, m, xmhe_traj, weight_para, j)
                xmhe_traj    = opt_sol['state_traj_opt']
                costate_traj = opt_sol['costate_traj_opt']
                noise_traj   = opt_sol['noise_traj_opt']
                if j>(horizon):
                    # update m based on xmhe_traj
                    for ix in range(len(m)):
                        m[ix,0] = xmhe_traj[1, ix]
                # Establish the auxiliary MHE system
                if it >0:
                    noise_list += [np.reshape(noise_traj[-1,:],(6,1))]
                auxSys       = uavMHE.GetAuxSys_general(xmhe_traj, costate_traj, noise_traj, weight_para, Y)
                matF, matG, matH = auxSys['matF'], auxSys['matG'], auxSys['matH']
                matddLxx, matddLxw = auxSys['matddLxx'], auxSys['matddLxw']
                matddLxe, matddLee, matddLew = auxSys['matddLxe'], auxSys['matddLee'], auxSys['matddLew']

                # Solve the auxiliary MHE system to obtain the gradient
                if j <= horizon:
                    M = np.zeros((len(m), D_out))
                else:
                    M = X_opt[1]
                
                start_time = TM.time()
                gra_opt = uavNMHE.GradientSolver_general(M, matF, matG, matddLxx, matddLxw, matddLxe, matddLee, matddLew, weight_para)
                gradtime = (TM.time() - start_time)*1000
                print("--- %s ms ---" % gradtime)
                Grad_time += [gradtime]
                X_opt = gra_opt['state_gra_traj']
          

                # Estimated disturbance
                df_Bmh = np.transpose(xmhe_traj[-1, 6:9])
                df_Bmh = np.reshape(df_Bmh, (3, 1))
                dt_Bmh = np.transpose(xmhe_traj[-1, 21:24])
                dt_Bmh = np.reshape(dt_Bmh, (3, 1))
                wrench_mhe = np.vstack((df_Bmh, dt_Bmh))

                # Input and output of the DNN
                
                print('Trained=', n_ep, 'sample=', it, 'p1=', P0[0, 0], 'gamma1=', gamma_r,'gamma2=', gamma_q,'r1=', R_N[0, 0], 'r2=', R_N[1, 1])
                print('Trained=', n_ep, 'sample=', it, 'q1=', Q_N1[0, 0], 'q2=', Q_N1[1, 1], 'q3=', Q_N1[2, 2], 'r3=', R_N[2, 2], 'r6=',R_N[5,5])
                # print('Trained=', n_ep, 'sample=', it, 'alpha1=', tunable_para[0, 24], 'alpha2=', tunable_para[0, 43])
                # Ground truth list
                acc = np.array([[accx_seq[n*j], accy_seq[n*j], accz_seq[n*j], angaccx_seq[n*j], angaccy_seq[n*j], angaccz_seq[n*j]]]).T
                df_t, dt_t = GroundTruth(Y[-1], acc, uav.m, uav.J)
                state_t       = np.vstack((df_t, dt_t))
                ground_truth += [state_t]
                print('Trained=', n_ep, 'sample=', it, 'Dist_x=', dt_t[0, 0], 'dt_Bmh_x=', dt_Bmh[0, 0], 'Dist_z=',
                        dt_t[2, 0], 'dt_Bmh_z=', dt_Bmh[2, 0])
                print('Trained=', n_ep, 'sample=', it, 'Dis_x=', df_t[0, 0], 'df_Bmh_x=', df_Bmh[0, 0], 'Dis_z=', df_t[2, 0],
                        'df_Bmh_z=', df_Bmh[2, 0])
                # Compute the gradient of loss
                dldw, loss_track = uavNMHE.ChainRule(ground_truth, xmhe_traj, X_opt)
                print('Trained=', n_ep, 'sample=', it, 'loss=', loss_track)
                # Train the tunable_para
                
                if n_ep>0:
                    weight_grad = chainRule_gradient(tunable_para)
                    dldp = np.matmul(dldw, weight_grad)
                    loss_nn_p = model_QR.myloss(model_QR(input_nn_QR), dldp)
                    optimizer_p = torch.optim.Adam(model_QR.parameters(), lr=lr_nn)
                    model_QR.zero_grad()
                    loss_nn_p.backward()
                    # Update tunable parameter and control gain
                    optimizer_p.step()
                
                
                # Store estimation error vector in the list
                gt = np.vstack((df_t, dt_t))
                Ew_epi[:, it:it + 1] = gt - wrench_mhe
                Time[it] = time_seq[n*j]
                # Sum the loss
                loss_track = np.reshape(loss_track, (1))
                loss_training[0, it:it+1]= loss_track
                sum_loss += loss_track
               
                it += 1
                np.save('cpu_time_h=60',Grad_time)
                cputime = np.zeros((len(Grad_time),1))
                for k in range(len(Grad_time)):
                    cputime[k,0] = Grad_time[k]
                print('cputime=',statistics.median(cputime))
                np.save('cpu_mediantime_h=60',statistics.median(cputime))
                
        np.save('Gamma2',Gamma2)
        np.save('Loss_Training',Loss_Training)

        mean_loss = sum_loss / it
        if n_ep == 1:
            eps = mean_loss/25
        # mean_time = sum_time / it
        Mean_Loss += [mean_loss]
        # Mean_time += [mean_time]
        if n_ep > 3:
            delta_cost = abs(mean_loss - mean_loss0)
        mean_loss0 = mean_loss
        print('learned', n_ep, 'mean_loss=', mean_loss)
        if n_ep >=10: # stop learning as it is too time-consuming
            break
        n_ep += 1

        np.save('Mean_loss', Mean_Loss)
        PATH1 = "Trained_model_n=100.pt"
        torch.save(model_QR, PATH1)
        # np.save('Mean_time', Mean_time)
        np.save('Trained', Trained)


"""---------------------------------Evaluation process-----------------------------"""
def Evaluate():
    n =1
    # Load the neural network model
    PATH1 = "Trained_model_n=100.pt"
    model_QR = torch.load(PATH1)
    # tunable_para at average
    # tunable_para = np.load('tunable_para.npy')
    # Time sequence list
    Time = []
    # Ground truth sequence list
    Gt_fx, Gt_fy, Gt_fz = [], [], []
    Gt_tx, Gt_ty, Gt_tz = [], [], []
    x, vx, wx = [], [], [] 
    x_mhe, vx_mhe, wx_mhe = [], [], []
    R11, R11_mhe = [], []
    # Estimation sequence list
    fx_mhe, fy_mhe, fz_mhe = [], [], []
    tx_mhe, ty_mhe, tz_mhe = [], [], []
    # Estimation error sequence list
    Efx, Efy, Efz, Etx, Ety, Etz = [], [], [], [], [], []
    evaluate_set = {'a':"fig8_evaluation_data_sameas_neurobem.csv"} # selected from "merged_2021-02-23-17-27-24_seg_2.csv"
    # count of step in all evaluations
    step = 0
    # Sum of loss
    sum_loss = 0.0
    # sum of time
    sum_time = 0.0
    # Measurement list
    Y = []
    # Control list
    ctrl = []
    # Motor normalized speed list
    speed = []
    # Ground_truth list for each episode
    ground_truth = []
    # Tunable parameter sequence list
    TUNable_para = []
    gamma1 = []
    gamma2 = []
    p1     = []
    p2     = []
    p3     = []
    q1     = []
    q2     = []
    q3     = []
    r1     = []
    r2     = []
    r3     = []
    alpha1 = []
    alpha2 = []
    alpha3 = []
    Trace_sum = []
    # iteration index
    it = 0
    # size of evaluation data
    size = 0
    # noise list

    noise_traj = np.zeros((1,6))
    noise_list = []
    noise_list += [np.reshape(noise_traj[-1,:],(6,1))]
    # count of negative trace_sum
    count = 0
    MHE_Cost = []
    Grad_time = []
    for key in evaluate_set:
        dataEva = pd.read_csv(evaluate_set[key])
        dataframe = pd.DataFrame(dataEva)
        time_seq = dataframe['t']
        N_ev = int(time_seq.size/n)
        size += N_ev
        # Obtain the sequences of the state, acc, and motor speed from the dataframe
        angaccx_seq, angaccy_seq, angaccz_seq = dataframe['ang acc x'], dataframe['ang acc y'], dataframe['ang acc z']
        angvelx_seq, angvely_seq, angvelz_seq = dataframe['ang vel x'], dataframe['ang vel y'], dataframe['ang vel z']
        qx_seq, qy_seq, qz_seq, qw_seq = dataframe['quat x'], dataframe['quat y'], dataframe['quat z'], dataframe['quat w']
        accx_seq, accy_seq, accz_seq = dataframe['acc x'], dataframe['acc y'], dataframe['acc z']
        velx_seq, vely_seq, velz_seq = dataframe['vel x'], dataframe['vel y'], dataframe['vel z']
        posx_seq, posy_seq, posz_seq = dataframe['pos x'], dataframe['pos y'], dataframe['pos z']
        mot1_seq, mot2_seq, mot3_seq, mot4_seq = dataframe['mot 1'], dataframe['mot 2'], dataframe['mot 3'], dataframe[
            'mot 4']
        # Initial states
        r_I0 = np.array([[posx_seq[0], posy_seq[0], posz_seq[0]]]).T
        v_B0 = np.array([[velx_seq[0], vely_seq[0], velz_seq[0]]]).T
        q0 = np.array([qw_seq[0], qx_seq[0], qy_seq[0], qz_seq[0]])
        w_B0 = np.array([[angvelx_seq[0], angvely_seq[0], angvelz_seq[0]]]).T
        R_Bh0, R_B0 = Quaternion2Rotation(q0)  # 9-by-1
        v_I0 = np.matmul(R_B0, v_B0)#+ np.matmul(np.matmul(R_B0, skew_w), r_I0)
        state0 = np.vstack((r_I0, v_B0, R_Bh0, w_B0))
        acc = np.array([[accx_seq[0], accy_seq[0], accz_seq[0], angaccx_seq[0], angaccy_seq[0], angaccz_seq[0]]]).T
        control = np.array([[mot1_seq[0] ** 2, mot2_seq[0] ** 2, mot3_seq[0] ** 2,
                              mot4_seq[0] ** 2]]).T  # Update control after the MHE is solved
        df_t, dt_t = GroundTruth(state0, acc, uav.m, uav.J)
        initial_force_noise = np.random.normal(0, 1, 3) # previously 1
        initial_force_noise = np.reshape(initial_force_noise,(3,1))
        initial_torque_noise = np.random.normal(0, 1e-2, 3)
        initial_torque_noise = np.reshape(initial_torque_noise,(3,1))
        initial_force_noise = np.array([[1.8,1.4,-1.8]]).T
        df_B0 = np.array([[0,0, Sys_para[0]*9.81]]).T
        dt_B0 = np.zeros((3,1))
        x_hatmh = np.vstack((r_I0, v_B0, df_B0, R_Bh0, w_B0, dt_B0))
        if it == 0:
            m       = x_hatmh
        xmhe_traj = x_hatmh.T
        # tunable parameter sequence for each episode
        Tunable_para = np.zeros((N_ev, 50))
        # tunable_para = np.load('mean_para_sl.npy')
        # tunable_para = np.reshape(tunable_para, (1, 50))

        for j in range(N_ev):
            # Solve MHE to obtain the estimated state trajectory within a horizon
            # Take measurement and motor speed
            Time += [time_seq[n*j]]
            r_I = np.array([[posx_seq[n*j], posy_seq[n*j], posz_seq[n*j]]]).T
            v_B = np.array([[velx_seq[n*j], vely_seq[n*j], velz_seq[n*j]]]).T
            q = np.array([qw_seq[n*j], qx_seq[n*j], qy_seq[n*j], qz_seq[n*j]])
            w_B = np.array([[angvelx_seq[n*j], angvely_seq[n*j], angvelz_seq[n*j]]]).T
            R_Bh, R_B = Quaternion2Rotation(q)  # 9-by-1
            x += [r_I[0, 0]]
            wx+= [w_B[0, 0]]
            R11 += [R_B[0, 0]]
            v_I = np.matmul(R_B, v_B)# + np.matmul(np.matmul(R_B, skew_w), r_I0)
            vx += [v_B[0, 0]]
            state = np.vstack((r_I, v_B, R_Bh, w_B)) 
            Y += [state]
            input_nn_QR = state # based on current measurement and speed
            tunable_para = convert(model_QR(input_nn_QR))
            # tunable_para = tunable_para0
            P0, gamma_r, gamma_q, R_N, Q_N1, weight_para = SetPara(r11, tunable_para)
            opt_sol      = uavMHE.MHEsolver(Y, m, xmhe_traj, weight_para, j)
            xmhe_traj    = opt_sol['state_traj_opt']
            costate_traj = opt_sol['costate_traj_opt']
            noise_traj   = opt_sol['noise_traj_opt']
            if j>(horizon):
                # update m based on xmhe_traj
                for ix in range(len(m)):
                    m[ix,0] = xmhe_traj[1, ix]

            # Establish the auxiliary MHE system
            if it >0:
                noise_list += [np.reshape(noise_traj[-1,:],(6,1))]
            auxSys       = uavMHE.GetAuxSys_general(xmhe_traj, costate_traj, noise_traj, weight_para, Y)
            matF, matG, matH = auxSys['matF'], auxSys['matG'], auxSys['matH']
            matddLxx, matddLxw = auxSys['matddLxx'], auxSys['matddLxw']
            matddLxe, matddLee, matddLew = auxSys['matddLxe'], auxSys['matddLee'], auxSys['matddLew']
            # Solve the auxiliary MHE system to obtain the gradient
            if j <= horizon:
                M = np.zeros((len(m), D_out))
            else:
                M = X_opt[1]
            start_time = TM.time()
            gra_opt = uavNMHE.GradientSolver_general(M, matF, matG, matddLxx, matddLxw, matddLxe, matddLee, matddLew, weight_para)
            gradtime = (TM.time() - start_time)*1000
            print("--- %s ms ---" % gradtime)
            Grad_time += [gradtime]
            X_opt = gra_opt['state_gra_traj']

            # Estimated disturbance
            df_Bmh = np.transpose(xmhe_traj[-1, 6:9])
            df_Bmh = np.reshape(df_Bmh, (3, 1))
            dt_Bmh = np.transpose(xmhe_traj[-1, 21:24])
            dt_Bmh = np.reshape(dt_Bmh, (3, 1))
            wrench_mhe = np.vstack((df_Bmh, dt_Bmh))
            fx_mhe += [wrench_mhe[0,0]]
            fy_mhe += [wrench_mhe[1,0]]
            fz_mhe += [wrench_mhe[2,0]]
            tx_mhe += [wrench_mhe[3,0]]
            ty_mhe += [wrench_mhe[4,0]]
            tz_mhe += [wrench_mhe[5,0]]
            x_mhe  += [xmhe_traj[-1,0]]
            vx_mhe  += [xmhe_traj[-1,3]]
            R11_mhe += [xmhe_traj[-1,9]]
            wx_mhe  += [xmhe_traj[-1,18]]
            # Input and output of the DNN
            
            print('sample=', j, 'p1=', P0[0, 0], 'gamma1=', gamma_r,'gamma2=', gamma_q, 'r1=', R_N[0, 0], 'r2=', R_N[1, 1])
            print('sample=', j, 'q1=', Q_N1[0, 0], 'q2=', Q_N1[1, 1], 'q3=', Q_N1[2, 2], 'r3=', R_N[2, 2], 'r6=', R_N[5,5])
            # print('sample=', j, 'alpha1=', tunable_para[0, 24], 'alpha2=', tunable_para[0, 43])
            P_diag = np.diag(P0)
            P_diag = np.reshape(P_diag, (1, 24))
            gamma_r= np.reshape(gamma_r, (1, 1))
            gamma_q= np.reshape(gamma_q, (1, 1))
            R_Ndiag= np.diag(R_N)
            R_Ndiag= np.reshape(R_Ndiag, (1, 18))
            Q_N1diag= np.diag(Q_N1)
            Q_N1diag= np.reshape(Q_N1diag, (1, 6))
            element_diag = np.hstack((P_diag, gamma_r, R_Ndiag, gamma_q, Q_N1diag))
            Tunable_para[j:j+1, :] = element_diag
            p1     += [P0[0, 0]]
            p2     += [P0[1, 1]]
            p3     += [P0[2, 2]]
            gamma1 += [gamma_r[0, 0]]
            gamma2 += [gamma_q[0, 0]]
            q1     += [Q_N1[0, 0]]
            q2     += [Q_N1[1, 1]]
            q3     += [Q_N1[2, 2]]
            r1     += [R_N[0, 0]]
            r2     += [R_N[1, 1]]
            r3     += [R_N[2, 2]]
            # Ground truth list
            acc = np.array([[accx_seq[n*j], accy_seq[n*j], accz_seq[n*j], angaccx_seq[n*j], angaccy_seq[n*j], angaccz_seq[n*j]]]).T
            ctrl += [np.array([[mot1_seq[n*j] ** 2, mot2_seq[n*j] ** 2, mot3_seq[n*j] ** 2,
                                mot4_seq[n*j] ** 2]]).T]  # Update control after the MHE is solved
            df_t, dt_t = GroundTruth(Y[-1], acc, uav.m, uav.J)
            state_t       = np.vstack((df_t, dt_t))
            ground_truth += [state_t]
            Gt_fx += [df_t[0,0]]
            Gt_fy += [df_t[1,0]]
            Gt_fz += [df_t[2,0]]
            Gt_tx += [dt_t[0,0]]
            Gt_ty += [dt_t[1,0]]
            Gt_tz += [dt_t[2,0]]
            print('sample=', j, 'Dist_x=', dt_t[0, 0], 'dt_Bmh_x=', dt_Bmh[0, 0], 'Dist_z=', dt_t[2, 0],
                  'dt_Bmh_z=', dt_Bmh[2, 0])
            print('sample=', j, 'Dis_x=', df_t[0, 0], 'df_Bmh_x=', df_Bmh[0, 0], 'Dis_z=', df_t[2, 0],
                  'df_Bmh_z=', df_Bmh[2, 0])

            # Compute the gradient of loss
            dp, loss_track = uavNMHE.ChainRule(ground_truth, xmhe_traj, X_opt)
            print('sample=', j, 'loss=', loss_track)
            # Store estimation error vector in the list
            gt = np.vstack((df_t, dt_t))
            E_wrench = gt - wrench_mhe
            Efx += [E_wrench[0,0]]
            Efy += [E_wrench[1,0]]
            Efz += [E_wrench[2,0]]
            Etx += [E_wrench[3,0]]
            Ety += [E_wrench[4,0]]
            Etz += [E_wrench[5,0]]
            # Sum the loss
            loss_track = np.reshape(loss_track, (1))
                
            step += 1
            sum_loss += loss_track
            np.save('Grad_cpu_time_20',Grad_time)
            # sum_para += tunable_para
        it += 1
        # mean_time = sum_time / step

    np.save('sum_loss', sum_loss)
    np.save('size_evaluation', size)
    np.save('Trace_sum',Trace_sum)
    # np.save('cpu_time_h=10',mean_time)
    mean_loss_ev = sum_loss/ size
    # mean_para    = sum_para/ size
    print('mean_loss_ev=', mean_loss_ev)
    # print('cpu_meantime=',mean_time)
    # print('mean_para=', mean_para)
    # np.save('mean_para_sl', mean_para)
    np.save('Tunable_para_evaluation', Tunable_para)
    np.save('mloss_ev', mean_loss_ev)
    np.save('Time', Time)
    np.save('Gamma1', gamma1)
    np.save('Gamma2', gamma2)
    np.save('Gt_fx', Gt_fx)
    np.save('Gt_fy', Gt_fy)
    np.save('Gt_fz', Gt_fz)
    np.save('Gt_tx', Gt_tx)
    np.save('Gt_ty', Gt_ty)
    np.save('Gt_tz', Gt_tz)
    np.save('fx_mhe', fx_mhe)
    np.save('fy_mhe', fy_mhe)
    np.save('fz_mhe', fz_mhe)
    np.save('tx_mhe', tx_mhe)
    np.save('ty_mhe', ty_mhe)
    np.save('tz_mhe', tz_mhe)
    np.save('Efx', Efx)
    np.save('Efy', Efy)
    np.save('Efz', Efz)
    np.save('Etx', Etx)
    np.save('Ety', Ety)
    np.save('Etz', Etz)
    np.save('MHE_Cost', MHE_Cost)
    """
    Plot figures
    """
    Time = np.load('Time.npy')
    Gt_fx = np.load('Gt_fx.npy')
    Gt_fy = np.load('Gt_fy.npy')
    Gt_fz = np.load('Gt_fz.npy')
    Gt_tx = np.load('Gt_tx.npy')
    Gt_ty = np.load('Gt_ty.npy')
    Gt_tz = np.load('Gt_tz.npy')
    fx_mhe = np.load('fx_mhe.npy')
    fy_mhe = np.load('fy_mhe.npy')
    fz_mhe = np.load('fz_mhe.npy')
    tx_mhe = np.load('tx_mhe.npy')
    ty_mhe = np.load('ty_mhe.npy')
    tz_mhe = np.load('tz_mhe.npy')
    Efx    = np.load('Efx.npy')
    Efy    = np.load('Efy.npy')
    Efz    = np.load('Efz.npy')
    Etx = np.load('Etx.npy')
    Ety = np.load('Ety.npy')
    Etz = np.load('Etz.npy')
    Time_fig8 = np.load('time_fig8.npy')
    Time_cpc  = np.load('time_cpc.npy')
    Time_random = np.load('time_random.npy')
    Loss_Training = np.load('Loss_Training.npy')
    # compute RMSE
    rmse_fx = mean_squared_error(fx_mhe, Gt_fx, squared=False)
    rmse_fy = mean_squared_error(fy_mhe, Gt_fy, squared=False)
    rmse_fz = mean_squared_error(fz_mhe, Gt_fz, squared=False)
    rmse_tx = mean_squared_error(tx_mhe, Gt_tx, squared=False)
    rmse_ty = mean_squared_error(ty_mhe, Gt_ty, squared=False)
    rmse_tz = mean_squared_error(tz_mhe, Gt_tz, squared=False)
    print('rmse_fx=',rmse_fx,'rmse_fy=',rmse_fy,'rmse_fz=',rmse_fz)
    print('rmse_tx=',rmse_tx,'rmse_ty=',rmse_ty,'rmse_tz=',rmse_tz)
    Fx_mhe, Fy_mhe, Fz_mhe = np.zeros((len(fx_mhe),1)), np.zeros((len(fx_mhe),1)), np.zeros((len(fx_mhe),1))
    Fxy_mhe, F_mhe = np.zeros((len(fx_mhe),1)), np.zeros((len(fx_mhe),1))
    Tx_mhe, Ty_mhe, Tz_mhe = np.zeros((len(fx_mhe),1)), np.zeros((len(fx_mhe),1)), np.zeros((len(fx_mhe),1))
    Txy_mhe, T_mhe = np.zeros((len(fx_mhe),1)), np.zeros((len(fx_mhe),1))
    GT_fx, GT_fy, GT_fz = np.zeros((len(fx_mhe),1)), np.zeros((len(fx_mhe),1)), np.zeros((len(fx_mhe),1))
    GT_fxy, GT_f = np.zeros((len(fx_mhe),1)), np.zeros((len(fx_mhe),1))
    GT_tx, GT_ty, GT_tz = np.zeros((len(fx_mhe),1)), np.zeros((len(fx_mhe),1)), np.zeros((len(fx_mhe),1))
    GT_txy, GT_t = np.zeros((len(fx_mhe),1)), np.zeros((len(fx_mhe),1))

    for i in range(len(fx_mhe)):
        Fx_mhe[i,0] = fx_mhe[i]
        Fy_mhe[i,0] = fy_mhe[i]
        Fz_mhe[i,0] = fz_mhe[i]
        Fxy_mhe[i,0]= np.sqrt(fx_mhe[i]**2 + fy_mhe[i]**2)
        F_mhe[i,0]  = np.sqrt(fx_mhe[i]**2 + fy_mhe[i]**2 + fz_mhe[i]**2)
        Tx_mhe[i,0] = tx_mhe[i]
        Ty_mhe[i,0] = ty_mhe[i]
        Tz_mhe[i,0] = tz_mhe[i]
        Txy_mhe[i,0]= np.sqrt(tx_mhe[i]**2 + ty_mhe[i]**2)
        T_mhe[i,0]  = np.sqrt(tx_mhe[i]**2 + ty_mhe[i]**2 + tz_mhe[i]**2)
        GT_fx[i,0] = Gt_fx[i]
        GT_fy[i,0] = Gt_fy[i]
        GT_fz[i,0] = Gt_fz[i]
        GT_fxy[i,0]= np.sqrt(Gt_fx[i]**2 + Gt_fy[i]**2)
        GT_f[i,0]  = np.sqrt(Gt_fx[i]**2 + Gt_fy[i]**2 + Gt_fz[i]**2)
        GT_tx[i,0] = Gt_tx[i]
        GT_ty[i,0] = Gt_ty[i]
        GT_tz[i,0] = Gt_tz[i]
        GT_txy[i,0]= np.sqrt(Gt_tx[i]**2 + Gt_ty[i]**2)
        GT_t[i,0]  = np.sqrt(Gt_tx[i]**2 + Gt_ty[i]**2 + Gt_tz[i]**2)

    rmse_fxy = mean_squared_error(Fxy_mhe, GT_fxy, squared=False)
    rmse_txy = mean_squared_error(Txy_mhe, GT_txy, squared=False)
    rmse_f = mean_squared_error(F_mhe, GT_f, squared=False)
    rmse_t = mean_squared_error(T_mhe, GT_t, squared=False)
    print('rmse_fxy=',rmse_fxy,'rmse_txy=',rmse_txy,'rmse_f=',rmse_f,'rmse_t=',rmse_t)
    np.save('rmse_fx', rmse_fx)
    np.save('rmse_fy', rmse_fy)
    np.save('rmse_fz', rmse_fz)
    np.save('rmse_tx', rmse_tx)
    np.save('rmse_ty', rmse_ty)
    np.save('rmse_tz', rmse_tz)
    np.save('rmse_fxy', rmse_fxy)
    np.save('rmse_txy', rmse_txy)
    np.save('rmse_f', rmse_f)
    np.save('rmse_t', rmse_t)
    # # Loss function
    # Mean_loss = np.load('Mean_loss.npy')
    # Dim_mean_loss = np.size(Mean_loss)
    # Trained = []
    # for i in range(Dim_mean_loss):
    #     Trained += [i]
    # plt.figure(1)
    # plt.plot(Trained, Mean_loss, linewidth=1.5,marker='o')
    # plt.xlabel('Number of episodes')
    # plt.ylabel('Mean loss')
    # plt.grid()
    # plt.savefig('./mean_loss_train.png',dpi=600)
    # plt.show()
    # Disturbance force
    plt.figure(2)
    plt.plot(Time, Gt_fx, linewidth=1, linestyle='--')
    plt.plot(Time, fx_mhe, linewidth=0.5)
    plt.xlabel('Time [s]')
    plt.ylabel('Disturbance force in x axis')
    plt.legend(['Ground truth', 'MHE estimation'])
    plt.grid()
    plt.savefig('./disturbance_x.png',dpi=600)
    plt.show()
    plt.figure(3)
    plt.plot(Time, Gt_fy, linewidth=1, linestyle='--')
    plt.plot(Time, fy_mhe, linewidth=0.5)
    plt.xlabel('Time [s]')
    plt.ylabel('Disturbance force in y axis')
    plt.legend(['Ground truth', 'MHE estimation'])
    plt.grid()
    plt.savefig('./disturbance_y.png',dpi=600)
    plt.show()
    plt.figure(4)
    plt.plot(Time, Gt_fz, linewidth=1, linestyle='--')
    plt.plot(Time, fz_mhe, linewidth=0.5)
    plt.xlabel('Time [s]')
    plt.ylabel('Disturbance force in z axis')
    plt.legend(['Ground truth', 'MHE estimation'])
    plt.grid()
    plt.savefig('./disturbance_z.png',dpi=600)
    plt.show()
    # Disturbance torque
    plt.figure(5)
    plt.plot(Time, Gt_tx, linewidth=1, linestyle='--')
    plt.plot(Time, tx_mhe, linewidth=0.5)
    plt.xlabel('Time [s]')
    plt.ylabel('Disturbance torque in x axis')
    plt.legend(['Ground truth', 'MHE estimation'])
    plt.grid()
    plt.savefig('./disturbance_torque_x.png',dpi=600)
    plt.show()
    plt.figure(6)
    plt.plot(Time, Gt_ty, linewidth=1, linestyle='--')
    plt.plot(Time, ty_mhe, linewidth=0.5)
    plt.xlabel('Time [s]')
    plt.ylabel('Disturbance torque in y axis')
    plt.legend(['Ground truth', 'MHE estimation'])
    plt.grid()
    plt.savefig('./disturbance_torque_y.png',dpi=600)
    plt.show()
    plt.figure(7)
    plt.plot(Time, Gt_tz, linewidth=1, linestyle='--')
    plt.plot(Time, tz_mhe, linewidth=0.5)
    plt.xlabel('Time [s]')
    plt.ylabel('Disturbance torque in z axis')
    plt.legend(['Ground truth', 'MHE estimation'])
    plt.grid()
    plt.savefig('./disturbance_torque_z.png',dpi=600)
    plt.show()
    # plt.figure(8)
    # plt.plot(Time,Trace_sum, linewidth=1)
    # plt.xlabel('Time [s]')
    # plt.ylabel('Summation of trace')
    # plt.grid()
    # plt.savefig('./trace_sum.png')
    # plt.show()
    # plt.figure(9)
    plt.figure(8)
    plt.plot(Time, gamma1, linewidth=1.5)
    plt.xlabel('Time [s]')
    plt.ylabel('gamma 1')
    plt.grid()
    plt.savefig('./gamma1.png',dpi=600)
    plt.show()
    plt.figure(9)
    plt.plot(Time, gamma2, linewidth=1.5)
    plt.xlabel('Time [s]')
    plt.ylabel('gamma 2')
    plt.grid()
    plt.savefig('./gamma2.png',dpi=600)
    plt.show()
    # plt.figure(10)
    # plt.plot(Time, p1, linewidth=1.5)
    # plt.xlabel('Time [s]')
    # plt.ylabel('p1')
    # plt.grid()
    # plt.savefig('./p1.png',dpi=600)
    # plt.show()
    # plt.figure(11)
    # plt.plot(Time, p2, linewidth=1.5)
    # plt.xlabel('Time [s]')
    # plt.ylabel('p2')
    # plt.grid()
    # plt.savefig('./p2.png',dpi=600)
    # plt.show()
    # plt.figure(12)
    # plt.plot(Time, p3, linewidth=1.5)
    # plt.xlabel('Time [s]')
    # plt.ylabel('p3')
    # plt.grid()
    # plt.savefig('./p3.png',dpi=600)
    # plt.show()
    # plt.figure(13)
    # plt.plot(Time, q1, linewidth=1.5)
    # plt.xlabel('Time [s]')
    # plt.ylabel('q1')
    # plt.grid()
    # plt.savefig('./q1.png',dpi=600)
    # plt.show()
    # plt.figure(14)
    # plt.plot(Time, q2, linewidth=1.5)
    # plt.xlabel('Time [s]')
    # plt.ylabel('q2')
    # plt.grid()
    # plt.savefig('./q2.png',dpi=600)
    # plt.show()
    # plt.figure(15)
    # plt.plot(Time, q3, linewidth=1.5)
    # plt.xlabel('Time [s]')
    # plt.ylabel('q3')
    # plt.grid()
    # plt.savefig('./q3.png',dpi=600)
    # plt.show()
    # plt.figure(16)
    # plt.plot(Time, r1, linewidth=1.5)
    # plt.xlabel('Time [s]')
    # plt.ylabel('r1')
    # plt.grid()
    # plt.savefig('./r1.png',dpi=600)
    # plt.show()
    # plt.figure(17)
    # plt.plot(Time, r2, linewidth=1.5)
    # plt.xlabel('Time [s]')
    # plt.ylabel('r2')
    # plt.grid()
    # plt.savefig('./r2.png',dpi=600)
    # plt.show()
    # plt.figure(18)
    # plt.plot(Time, r3, linewidth=1.5)
    # plt.xlabel('Time [s]')
    # plt.ylabel('r3')
    # plt.grid()
    # plt.savefig('./r3.png',dpi=600)
    # plt.show()
    # # Estimation error
    # plt.figure(19)
    # plt.plot(Time, Efx, linewidth=0.5)
    # plt.xlabel('Time [s]')
    # plt.ylabel('Force estimation error in x axis')
    # plt.grid()
    # plt.savefig('./efx.png',dpi=600)
    # plt.show()
    # plt.figure(20)
    # plt.plot(Time, Efy, linewidth=0.5)
    # plt.xlabel('Time [s]')
    # plt.ylabel('Force estimation error in y axis')
    # plt.grid()
    # plt.savefig('./efy.png',dpi=600)
    # plt.show()
    # plt.figure(21)
    # plt.plot(Time, Efz, linewidth=0.5)
    # plt.xlabel('Time [s]')
    # plt.ylabel('Force estimation error in z axis')
    # plt.grid()
    # plt.savefig('./efz.png',dpi=600)
    # plt.show()
    # plt.figure(22)
    # plt.plot(Time, Etx, linewidth=0.5)
    # plt.xlabel('Time [s]')
    # plt.ylabel('Torque estimation error in x axis')
    # plt.grid()
    # plt.savefig('./etx.png',dpi=600)
    # plt.show()
    # plt.figure(23)
    # plt.plot(Time, Ety, linewidth=0.5)
    # plt.xlabel('Time [s]')
    # plt.ylabel('Torque estimation error in y axis')
    # plt.grid()
    # plt.savefig('./ety.png',dpi=600)
    # plt.show()
    # plt.figure(24)
    # plt.plot(Time, Etz, linewidth=0.5)
    # plt.xlabel('Time [s]')
    # plt.ylabel('Torque estimation error in z axis')
    # plt.grid()
    # plt.savefig('./etz.png',dpi=600)
    # plt.show()
    # plt.figure(25)
    # plt.plot(Time, x, linewidth=1, linestyle='--')
    # plt.plot(Time, x_mhe, linewidth=0.5)
    # plt.legend(['ground truth','MHE estimation'])
    # plt.xlabel('Time [s]')
    # plt.ylabel('x position')
    # plt.grid()
    # plt.savefig('./x.png',dpi=600)
    # plt.show()
    # plt.figure(26)
    # plt.plot(Time, vx, linewidth=1, linestyle='--')
    # plt.plot(Time, vx_mhe, linewidth=0.5)
    # plt.legend(['ground truth','MHE estimation'])
    # plt.xlabel('Time [s]')
    # plt.ylabel('x velocity ')
    # plt.grid()
    # plt.savefig('./vx.png',dpi=600)
    # plt.show()
    # plt.figure(27)
    # plt.plot(Time, wx, linewidth=1, linestyle='--')
    # plt.plot(Time, wx_mhe, linewidth=0.5)
    # plt.legend(['ground truth','MHE estimation'])
    # plt.xlabel('Time [s]')
    # plt.ylabel('x angular velocity ')
    # plt.grid()
    # plt.savefig('./wx.png',dpi=600)
    # plt.show()
    # plt.figure(28)
    # plt.plot(Time, R11, linewidth=1, linestyle='--')
    # plt.plot(Time, R11_mhe, linewidth=0.5)
    # plt.legend(['ground truth','MHE estimation'])
    # plt.xlabel('Time [s]')
    # plt.ylabel('first entry of rotation matrix ')
    # plt.grid()
    # plt.savefig('./R11.png',dpi=600)
    # plt.show()
    # plt.figure(20)
    # for i in range(Dim_mean_loss):
    #     plt.plot(Time_fig8, Loss_Training[i], linewidth=1)
    # plt.xlabel('Time [s]')
    # plt.ylabel('Loss')
    # plt.grid()
    # plt.savefig('./loss_training.png')
    # plt.show()
    # Grad_time = np.load('Grad_cpu_time_20.npy')
    # cputime = np.zeros((len(Grad_time),1))
    # for k in range(len(Grad_time)):
    #     cputime[k,0] = Grad_time[k]
    # print('cputime=',statistics.median(cputime))
    # np.save('cpu_mediantime_h=20',statistics.median(cputime))
    # plt.figure(21)
    # df = pd.DataFrame(data=cputime,columns = ['horizon=20'])
    # df.boxplot()
    # plt.savefig('./cpu_h=20_box.png',dpi=600)
    # plt.show()
    



"""---------------------------------Main function-----------------------------"""
if train:
    Train()
    Evaluate()
else:
    Evaluate()
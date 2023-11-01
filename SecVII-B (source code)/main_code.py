"""
This is the main function that trains NeuroMHE and evaluates its performance.
----------------------------------------------------------------------------
Wang, Bingheng at Control and Simulation Lab, ECE Dept. NUS, Singapore
1st version: 10 May,2022
2nd version: 10 Oct. 2022 after receiving the reviewers' comments
Should you have any question, please feel free to contact the author via:
wangbingheng@u.nus.edu
"""
import UavEnv
import Robust_Flight
from casadi import *
import time as TM
import numpy as np
import uavNN
import torch
from numpy import linalg as LA
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os
import pandas as pd

print("=============================================")
print("Main code for training or evaluating NeuroMHE")
print("Please choose mode")
mode = input("enter 'train' or 'evaluate' without the quotation mark:")
print("=============================================")


"""---------------------------------Load environment---------------------------------------"""
uav_para = np.array([1,0.02,0.02,0.04])
wing_len = 0.5
ctrl_gain = np.array([[25,25,25, 15,15,15, 15,15,15, 1.5,1.5,1.5]]) 
if mode == 'train':
    dt_sample = 1e-2 # Sampling time-step for environment model
    dt_mhe   = 1e-2 # Sampling time-step for MHE and controllers
else:
    dt_sample = 1e-2
    dt_mhe   = 1e-2
    

ratio     = int(dt_mhe/dt_sample)
uav = UavEnv.quadrotor(uav_para, dt_mhe)
uav.model()
horizon = 10 
# Learning rate
lr_nn  = 1e-4
# First element in R_t
r11    = np.array([[100]])


"""---------------------------------Define parameterization model--------------------------"""
# Define neural network for process noise
D_in, D_h, D_out = 12, 50, 49
model_QR = uavNN.Net(D_in, D_h, D_out)

"""---------------------------------Define reference trajectory----------------------------"""
"""
The reference trajectory is generated using minisnap algorithm [2]
[2] Mellinger, D. and Kumar, V., 2011, May. 
    Minimum snap trajectory generation and control for quadrotors. 
    In 2011 IEEE international conference on robotics and automation (pp. 2520-2525). IEEE.
"""
# Load the reference trajectory polynomials
Coeffx_training,  Coeffy_training,  Coeffz_training  = np.zeros((7,8)), np.zeros((7,8)), np.zeros((7,8))
Coeffx_evaluation,  Coeffy_evaluation,  Coeffz_evaluation  = np.zeros((8,8)), np.zeros((8,8)), np.zeros((8,8))


for i in range(7): # 15s
    Coeffx_training[i,:] = np.load('Ref_synthetic_data_training_traj/coeffx'+str(i+1)+'.npy')
    Coeffy_training[i,:] = np.load('Ref_synthetic_data_training_traj/coeffy'+str(i+1)+'.npy')
    Coeffz_training[i,:] = np.load('Ref_synthetic_data_training_traj/coeffz'+str(i+1)+'.npy')
for i in range(8): # 10s
    Coeffx_evaluation[i,:] = np.load('Ref_synthetic_data_evaluation_traj/coeffx'+str(i+1)+'.npy')
    Coeffy_evaluation[i,:] = np.load('Ref_synthetic_data_evaluation_traj/coeffy'+str(i+1)+'.npy')
    Coeffz_evaluation[i,:] = np.load('Ref_synthetic_data_evaluation_traj/coeffz'+str(i+1)+'.npy')





"""---------------------------------Define controller--------------------------------------"""
GeoCtrl  = Robust_Flight.Controller(uav_para, ctrl_gain, uav.x, dt_mhe)

"""---------------------------------Define MHE---------------------------------------------"""
uavMHE = Robust_Flight.MHE(horizon, dt_mhe,r11)
uavMHE.SetStateVariable(uav.xa)
uavMHE.SetOutputVariable(uav.y)
uavMHE.SetControlVariable(uav.u)
uavMHE.SetNoiseVariable(uav.w)
uavMHE.SetModelDyn(uav.dymh)
uavMHE.SetCostDyn()

"""---------------------------------Define Kalman filter-based Gradient solver-------------"""
uavNMHE = Robust_Flight.KF_gradient_solver(GeoCtrl.ref, uav.xa, uavMHE.weight_para)

"""---------------------------------Parameterization of Neural Network Output--------------"""
epsilon0, gmin0 = 1e-4, 1e-4 # smaller the epsilon is, larger the gradient will be. This demonstrates that calculating the gradient requires the inverse of the weightings.
def SetPara(epsilon, gmin,tunable_para, r11):
    p_diag = np.zeros((1, 24))
    for i in range(24):
        p_diag[0, i] = epsilon + tunable_para[0, i]**2
    P0 = np.diag(p_diag[0])

    gamma_r = gmin + (1 - gmin) * 1/(1+np.exp(-tunable_para[0, 24]))
    gamma_q = gmin + (1 - gmin) * 1/(1+np.exp(-tunable_para[0, 42]))

    r_diag = np.zeros((1, 17))
    for i in range(17):
        r_diag[0, i] = epsilon + tunable_para[0, i+25]**2
    r      = np.hstack((r11, r_diag))
    r      = np.reshape(r, (1, 18))
    R_t    = np.diag(r[0])

    q_diag = np.zeros((1, 6))
    for i in range(6):
        q_diag[0, i] = epsilon + tunable_para[0, i+43]**2
    Q_t1   = np.diag(q_diag[0])

    weight_para = np.hstack((p_diag, np.reshape(gamma_r, (1,1)), r_diag, np.reshape(gamma_q,(1,1)), q_diag))
    return P0, gamma_r, gamma_q, R_t, Q_t1, weight_para

def convert(parameter):
    tunable_para = np.zeros((1,D_out))
    for i in range(D_out):
        tunable_para[0,i] = parameter[i,0]
    return tunable_para

def chainRule_gradient(epsilon, gmin,tunable_para):
    tunable = SX.sym('tunable', 1, D_out)
    P = SX.sym('P', 1, 24)
    for i in range(24):
        P[0, i] = epsilon + tunable[0, i]**2

    gamma_r = gmin + (1 - gmin) * 1/(1+exp(-tunable[0, 24]))
    gamma_q = gmin + (1 - gmin) * 1/(1+exp(-tunable[0, 42]))
    
    R = SX.sym('R', 1, 17)
    for i in range(17):
        R[0, i] = epsilon + tunable[0, i+25]**2

    Q = SX.sym('Q', 1, 6)
    for i in range(6):
        Q[0, i] = epsilon + tunable[0, i+43]**2

    weight = horzcat(P, gamma_r, R, gamma_q, Q)
    w_jaco = jacobian(weight, tunable)
    w_jaco_fn = Function('W_fn',[tunable],[w_jaco],['tp'],['W_fnf'])
    weight_grad = w_jaco_fn(tp=tunable_para)['W_fnf'].full()
    return weight_grad


# Initial tunable parameters for DMHE, tuned to produce the untrained mean loss similar to that of NeuroMHE 
p0, r0, q0 = 0.09, 0.1, 0.1 # 0.1,0.1,0.1
tunable_para0 = np.array([[p0,p0,p0,p0,p0,p0,p0,p0,
                                  p0,p0,p0,p0,p0,p0,p0,p0,
                                  p0,p0,p0,p0,p0,p0,p0,p0,
                                  0.08, # -0.1
                                  r0,r0,r0,r0,r0,r0,
                                  r0,r0,r0,r0,r0,r0,
                                  r0,r0,r0,r0,r0,
                                  -0.11, # 0.1
                                  q0,q0,q0,q0,q0,q0]])

alpha   = 0.6
def Train(epsilon0, gmin0, tunable_para0):
    print("===============================================")
    print("Please choose a estimator to train")
    print("'a': NeuroMHE (adaptive weights by a neural network)")
    print("'b': DMHE (locally optimal but fixed weights)")
    estimator = input("enter 'a' or 'b' without the quotation mark:")
    print("===============================================")
    # Total simulation time
    T_end   = 15
    # Total iterations
    N       = int(T_end/dt_sample)   
    if not os.path.exists("trained_data"):
        os.makedirs("trained_data")
    if estimator=='a':
        epsilon = epsilon0
        gmin    = gmin0
        # (since the neural network parameters are randomly initialized, use the saved initial model to produce training results highly similar to our paper)
        # (*It is difficult to guarantee a perfect reproduction even with the saved initial model, due to the measurement noise that affects the MHE estimation results entering the feedback controller)
        PATH0 = "trained_data/initial_nn_model.pt" 
        model_QR = torch.load(PATH0)
        # model_QR = uavNN.Net(D_in, D_h, D_out)
        # torch.save(model_QR,PATH0)

    else:
        epsilon = epsilon0
        gmin    = gmin0
        tunable_para = tunable_para0# Initial tunable parameters for DMHE

    # Load training data
 
    dis_f = np.load('training_data/Dis_f.npy') 
    dis_t = np.load('training_data/Dis_t.npy')
    np.save('trained_data/Dis_f',dis_f)
    np.save('trained_data/Dis_t',dis_t)
   
    # Lists for storing training data
    Loss        = []
    Position    = []
    Ref_P       = []
    Dis_f_mh    = []
    Dis_tau_mh  = []
    Fullstate   = []
    K_iteration = []
    # Training iteration index
    k_train     = 0
    # Initial mean loss
    mean_loss0  = 1e4
    mean_loss_untrained = mean_loss0
    # Initial change of the mean loss
    delta_loss  = 1e4
    # Threshold
    eps = 1e-1
    threshold = 3e4
    flag = 0
    flag2 = 0
    flag3 = 0

    while delta_loss >= eps:
        # Initial time
        time   = 0
        # Initial states
        x0     = np.random.normal(0,0.01)
        y0     = np.random.normal(0,0.01)
        p      = np.array([[x0,y0,0]]).T
        v      = np.zeros((3,1))
        Euler  = np.zeros((3,1))
        R_h, R_bw, R_wb = uav.dir_cosine(Euler)
        omega  = np.zeros((3,1))
        # Initial guess of state and disturbance force used in MHE
        dfg0   = np.zeros((3,1))
        dtaug0 = np.zeros((3,1))
        x_hat  = np.vstack((p,v,dfg0,R_h,omega,dtaug0))
        xmhe_traj  = x_hat
        noise_traj = np.zeros((1,6))
        # Initial reference attitude and angular velocity
        Rb_hd  = R_h
        omegad = np.zeros((3,1))
        # Lists for saving control, measurement, and reference
        ctrl   = []
        Y      = []
        Ref    = []
        # Sum of loss
        sum_loss = 0.0
        # Vectors for recoding position, reference position, and disturbance force estimate in each episode
        position = np.zeros((3, N))
        Ref_p = np.zeros((3, N))
        dis_f_mh = np.zeros((5, N))
        dis_tau_mh = np.zeros((5, N))
        State = np.zeros((18,N))
        # initial 
        x_prev = x_hat
        a_lpf_prev = np.zeros((3,1))
        j_lpf_prev = np.zeros((3,1))
        # sum of the tunable parameters for calculating the mean value as a reference, which is for tuning the initial tunable parameters of DMHE
        sum_tp = np.zeros((1,D_out))
        for k in range(N):
            # Load the reference
            t_switch = 0
            ref_p, ref_v, ref_a, ref_j, ref_s = uav.reference_training(Coeffx_training, Coeffy_training, Coeffz_training, time, t_switch) 
            # ref_p, ref_v, ref_a, ref_j, ref_s = uav.reference_traj_smooth_fig8(Coeffx_smooth_fig8, Coeffy_smooth_fig8, Coeffz_smooth_fig8, time, t_switch) 
            b1_c              = np.array([[1, 0, 0]]).T # constant desired heading direction
            ref               = np.vstack((ref_p, ref_v, Rb_hd, omegad))
            Ref_p[:, k:k+1]   = ref_p
            
            # Obtain the noisy measurement
            state             = np.vstack((p, v, R_h, omega)) # current true state
            State[:, k:k+1]   = state
            df                = dis_f[:,k] #np.reshape(np.vstack((0.2*dis_f[0,k],0.2*dis_f[1,k],dis_f[2,k])),(3,1))
            dtau              = dis_t[:,k]
            state_m           = state + np.reshape(np.random.normal(0,1e-3,18),(18,1)) 
            # state_m           = np.vstack((p+np.reshape(np.random.normal(0,1e-3,3),(3,1)), v+np.reshape(np.random.normal(0,2e-3,3),(3,1)), R_h+np.reshape(np.random.normal(0,1e-3,9),(9,1)), omega+np.reshape(np.random.normal(0,1e-3,3),(3,1))))
            R_b               = np.array([[state_m[6,0],state_m[7,0],state_m[8,0]],
                                      [state_m[9,0],state_m[10,0],state_m[11,0]],
                                      [state_m[12,0],state_m[13,0],state_m[14,0]]])
            gamma   = np.arctan(R_b[2, 1]/R_b[1, 1])
            theta   = np.arctan(R_b[0, 2]/R_b[0, 0])
            psi     = np.arcsin(-R_b[0, 1])
            Euler_m = np.array([[gamma, theta, psi]]).T
            nn_input= np.reshape(np.vstack((np.reshape(state_m[0:6,0],(6,1)),Euler_m,np.reshape(state_m[15:18,0],(3,1)))),(D_in,1))
            # nn_input= np.reshape(np.vstack((np.reshape(state_m[0:3,0],(3,1)),2*np.reshape(state_m[3:6,0],(3,1)), Euler_m,np.reshape(state_m[15:18,0],(3,1)))),(D_in,1))
            position[:,k:k+1] = p
            Y      += [state_m]
            # nn_input = Y[-1]
            Ref    += [alpha*ref+(1-alpha)*state_m]

            #----------MHE state estimation----------#
            # Generate the weighting parameters for MHE
            if estimator == 'a':
                tunable_para    = convert(model_QR(nn_input))
            P0, gamma_r, gamma_q, R_t, Q_t1, weight_para = SetPara(epsilon, gmin,tunable_para,r11)
            sum_tp += weight_para
            print('learning_iteration:', k_train, 'sample=', k, 'gamma1=', gamma_r,'gamma2=', gamma_q,'p1=',P0[0,0],'p2=',P0[1,1],'p3=',P0[2,2], 'r1=', R_t[0, 0], 'r2=', R_t[1, 1], 'r3=', R_t[2,2], 'q1=', Q_t1[0,0], 'q2=', Q_t1[1,1], 'q3=', Q_t1[2,2])
            opt_sol      = uavMHE.MHEsolver(Y, x_hat, xmhe_traj, ctrl, noise_traj, weight_para, k)
            xmhe_traj    = opt_sol['state_traj_opt']
            costate_traj = opt_sol['costate_ipopt']
            noise_traj   = opt_sol['noise_traj_opt']
            if k>horizon:
                for ix in range(len(x_hat)):# update m based on xmhe_traj
                    x_hat[ix,0] = xmhe_traj[1, ix]
            # Obtain the coefficient matricres to establish the auxiliary MHE system
            auxSys       = uavMHE.GetAuxSys_general(xmhe_traj, costate_traj, noise_traj, weight_para, Y, ctrl)
            matF, matG, matH = auxSys['matF'], auxSys['matG'], auxSys['matH']
            matddLxx, matddLxp = auxSys['matddLxx'], auxSys['matddLxp']
            matddLxw, matddLww, matddLwp = auxSys['matddLxw'], auxSys['matddLww'], auxSys['matddLwp']
            # Solve for the analytical gradient using a Kalman filter (Algorithm 1, Lemma 2)
            if k <= horizon:
                M = np.zeros((len(x_hat), D_out))
            else:
                M = X_opt[1]
            gra_opt = uavNMHE.GradientSolver_general(M, matF, matG, matddLxx, matddLxp, matddLxw, matddLww, matddLwp, P0)
            X_opt   = gra_opt['state_gra_traj']
            
            #-----------Geometric controller-------------#
            df_Imh   = np.transpose(xmhe_traj[-1, 6:9])
            df_Imh   = np.reshape(df_Imh, (3, 1)) # MHE disturbance estimate
            dtau_mh  = np.transpose(xmhe_traj[-1, 21:24])
            dtau_mh  = np.reshape(dtau_mh, (3, 1))
            dis_f_mh[0:3,k:k+1] = df_Imh
            dis_f_mh[3,k:k+1] = LA.norm(df_Imh[0:2,0])
            dis_f_mh[4,k:k+1] = LA.norm(df_Imh)
            dis_tau_mh[0:3,k:k+1] = dtau_mh
            dis_tau_mh[3,k:k+1] = LA.norm(dtau_mh[0:2,0])
            dis_tau_mh[4,k:k+1] = LA.norm(dtau_mh)
            print('learning_iteration:', k_train, 'sample=', k, 'df_Imh=', df_Imh.T, 'df_true=', df.T, 'dt_mh=',dtau_mh.T, 'dt_true=',dtau.T)
            state_mh = np.reshape(np.hstack((xmhe_traj[-1, 0:6], xmhe_traj[-1, 9:21])), (18,1))
            feedback = state_mh # here, the MHE estimates are used as the feedback signal.

            u, Rb_hd,omegad, domegad, a_lpf, j_lpf = GeoCtrl.geometric_ctrl(feedback,x_prev,a_lpf_prev,j_lpf_prev,ref_p,ref_v,ref_a,ref_j,ref_s,b1_c, df_Imh,dtau_mh)
            # u, Rb_hd, omegad, domegad  = GeoCtrl.geometric_ctrl_2(feedback,ref_p,ref_v,ref_a,ref_j,ref_s,b1_c,df_Imh, dtau_mh)  
            x_prev   = feedback
            a_lpf_prev = a_lpf
            j_lpf_prev = j_lpf
            
            ctrl    += [u]

            #-----------Policy update based on gradient descent------------#
            dldw, loss_track = uavNMHE.ChainRule(Ref, xmhe_traj,X_opt)
            weight_grad      = chainRule_gradient(epsilon, gmin,tunable_para)
            dldp             = np.matmul(dldw, weight_grad)
            norm_grad        = LA.norm(dldp)
            if k_train > 0: # train the neural network from the second learning epoch
                if norm_grad>=threshold:
                    dldp = dldp/norm_grad*threshold
                    epsilon = epsilon + 1e-3
                    gmin = gmin + 1e-6
                    if gmin >=0.005:
                        gmin = 0.005
                new_norm = LA.norm(dldp)
                if estimator == 'a':
                    loss_nn_p   = model_QR.myloss(model_QR(nn_input), dldp)
                    optimizer_p = torch.optim.Adam(model_QR.parameters(), lr=lr_nn)
                    model_QR.zero_grad()
                    loss_nn_p.backward()
                    optimizer_p.step()
                else:
                    tunable_para = tunable_para - lr_nn*dldp  
                print('sample=',k, 'loss=', loss_track, 'epsilon=', epsilon, 'gmin=', gmin, 'dldw[24]=',dldw[0,24],'dldw[42]=',dldw[0,42],'norm_dldw=', new_norm )
            print('sample=',k,'ref_p=',ref_p.T,'act_p=',p.T,'Attitude_norm=',LA.norm(57.3*Euler), 'mean_loss0=',mean_loss0)    
            # Update the system state based on the system dynamics model
            output   = uav.step(state, u, df, dtau, dt_sample)
            p        = output['p_new']
            v        = output['v_new']
            R_h      = output['R_new']
            omega    = output['omega_new']
            Euler    = output['Euler']
            # Update time
            time += dt_sample 
            loss_track = np.reshape(loss_track,(1))
            sum_loss += loss_track
            if LA.norm(57.3*Euler)>=80:
                flag = 1
                break
        mean_loss = sum_loss/N
        if k_train == 0:
            mean_tp   = sum_tp/N
        print('learning_iteration:',k_train,'mean_loss=',mean_loss, 'eps=', eps, 'mean_p=',mean_tp[0,0],'mean_gamma1=',mean_tp[0,24],'mean_r=',mean_tp[0,25],'mean_gamma2=',mean_tp[0,42],'mean_q1=',mean_tp[0,43])
        if flag == 1:
            break  
        Position    += [position]
        Dis_f_mh    += [dis_f_mh]
        Dis_tau_mh  += [dis_tau_mh]
        Ref_P       += [Ref_p]
        Fullstate   += [State]
        K_iteration += [k_train]
        if estimator == 'a':
            PATH1 = "trained_data/trained_nn_model.pt"
            torch.save(model_QR, PATH1)
        else:
            np.save('trained_data/tunable_para_trained', tunable_para)
        if k_train == 0:
            eps = mean_loss/1e2
            if eps <=1:
                eps = 1
        if k_train>2:
            delta_loss = abs(mean_loss-mean_loss0)
        
        if k_train>=1 and gamma_r<gamma_q:
            flag3 =1
        
        Loss += [mean_loss]
        if k_train >= 20 or (k_train>0 and mean_loss >= mean_loss0):
            flag2 = 1
        print('learning_iteration:',k_train,'flag=',flag,'flag2=',flag2,'flag3=',flag3)
        k_train += 1
        FULLSTATE = np.zeros((18,k_train*N))
        
        for i in range(k_train):
            FULLSTATE[:,i*N:(i+1)*N] = Fullstate[i]

        if estimator=='a':
            np.save('trained_data/Loss', Loss)
            np.save('trained_data/K_iteration',K_iteration)
            np.save('trained_data/Position_train',Position)
            np.save('trained_data/Dis_f_mh_train',Dis_f_mh)
            np.save('trained_data/Dis_tau_mh_train',Dis_tau_mh)
            np.save('trained_data/FULLSTATE_train',FULLSTATE)
            np.save('trained_data/epsilon_trained',epsilon)
            np.save('trained_data/gmin_trained',gmin) 
        else:
            np.save('trained_data/Loss_dmhe', Loss)
            np.save('trained_data/K_iteration_dmhe',K_iteration)
            np.save('trained_data/Position_train_dmhe',Position)
            np.save('trained_data/Dis_f_mh_train_dmhe',Dis_f_mh)
            np.save('trained_data/Dis_tau_mh_train_dmhe',Dis_tau_mh)
            np.save('trained_data/FULLSTATE_train_dmhe',FULLSTATE)
            np.save('trained_data/epsilon_trained_dmhe',epsilon)
            np.save('trained_data/gmin_trained_dmhe',gmin)
        
        REFP      = np.zeros((3, k_train*N))    
        for i in range(k_train):
            REFP[:,i*N:(i+1)*N] = Ref_P[i]
        np.save('trained_data/REFP',REFP)

        
        mean_loss0 = mean_loss
        if flag2 == 1 or flag3 == 1:
            break
        
    
    # iteration = np.load('K_iteration.npy')
    # loss      = np.load('Loss.npy')
    # plt.figure(1)
    # plt.plot(K_iteration, Loss, linewidth=1.5, marker='o')
    # plt.xlabel('Training episodes')
    # plt.ylabel('Mean loss')
    # plt.grid()
    # plt.savefig('./mean_loss_train.png')
    # plt.show()
    # uav.play_animation(wing_len, FULLSTATE, REFP, dt_sample)
    return gamma_r, gamma_q, flag, flag2, flag3, mean_loss


"""---------------------------------Evaluation process-----------------------------"""
def Evaluate():
    print("===============================================")
    print("Please choose a controller to evaluate")
    print("'a': NeuroMHE            + Geometric Controller")
    print("'b': DMHE                + Geometric Controller")
    print("'c': L1 Adaptive Control + Geometric Controller")
    print("'d': L1-AC (active)      + Geometric Controller")
    print("'e': Baseline Geometric Controller Alone")
    controller = input("enter 'a', or 'b',... without the quotation mark:")
    print("===============================================")
    print("Plot figures?")
    plot = input("enter 'y', or 'n',... without the quotation mark:")

    # Total simulation time
    T_end  = 10
    # Total iterations
    N      = int(T_end/dt_sample)  
    if controller == 'b': 
        epsilon = np.load('trained_data/epsilon_trained_dmhe.npy')
        gmin    = np.load('trained_data/gmin_trained_dmhe.npy')
    else:
        epsilon = np.load('trained_data/epsilon_trained.npy')
        gmin    = np.load('trained_data/gmin_trained.npy')

    # Load the trained neural network model
    PATH1 = "trained_data/trained_nn_model.pt"
    model_QR = torch.load(PATH1)
    # Load the trained weighting parameters
    tunable_para = np.load('trained_data/tunable_para_trained.npy')
    # Initial time
    time = 0

    # initial states
    x0     = np.random.normal(0,0.01)
    y0     = np.random.normal(0,0.01)
    p      = np.array([[x0,y0,0]]).T
    v      = np.zeros((3,1))
    roll   = np.random.normal(0,0.005)
    pitch  = np.random.normal(0,0.005)
    yaw    = np.random.normal(0,0.005)
    Euler  = np.array([[roll,pitch,yaw]]).T
    R_h, R_bw, R_wb = uav.dir_cosine(Euler)
    omega  = np.zeros((3,1))
    # initial guess of state and disturbance force
    pg0    = np.array([[0,0,0]]).T
    vg0    = np.zeros((3,1))
    df     = np.zeros((3,1))
    dtau   = np.zeros((3,1))
    # initial state prediction
    z_hat  = np.zeros((6,1))
    # initial reference attitude and angular velocity
    Rb_hd  = np.array([[1,0,0,0,1,0,0,0,1]]).T
    omegad = np.zeros((3,1))
    # filter priori
    x_hat  = np.vstack((pg0,vg0,df,Rb_hd,omega,dtau))
    xmhe_traj = x_hat
    noise_traj = np.zeros((1,6))
    # initial 
    x_prev = x_hat
    # control list
    ctrl   = []
    # initial control
    u      = np.array([[uav_para[0]*9.81,0,0,0]]).T
    # measurement list (position and velocity)
    Y      = []
    # reference list
    Ref    = []
    # sum of loss
    sum_loss = 0.0
    # record the position tracking performance  
    position = np.zeros((5, int(N/ratio)))
    velocity = np.zeros((3, int(N/ratio)))
    Omega    = np.zeros((3, int(N/ratio)))
    Omegad   = np.zeros((3, int(N/ratio)))
    dOmegad  = np.zeros((3, int(N/ratio)))
    # record the disturbance  
    dis_f =  np.zeros((5, int(N/ratio)))
    dis_t =  np.zeros((5, int(N/ratio)))
    # Dis_f = np.zeros((3, N))
    # Dis_t = np.zeros((3, N))
    Dis_f = np.load('evaluation_data/Dis_f.npy') # use the saved disturbance data generated offline on the unseen race-track trajectory.
    Dis_t = np.load('evaluation_data/Dis_t.npy')
    # Dis_f = np.load('training_data/Dis_f.npy') 
    # Dis_t = np.load('training_data/Dis_t.npy')
    # record the disturbance estimates 
    df_MH = np.zeros((5, int(N/ratio)))
    dtau_MH = np.zeros((5, int(N/ratio)))
    # record L1 estimation without filtering
    df_L1   = np.zeros((3, int(N/ratio)))
    dtau_L1 = np.zeros((3, int(N/ratio)))
    # record the covariance inverse of the process noise 
    cov_f = np.zeros((3, N))
    cov_t = np.zeros((3, N))
    # record the weighting matrix 
    tp    = np.zeros((D_out, N))
    Q_k   = np.zeros((3, N))
    Q_k2   = np.zeros((3, N))
    # Record attitude
    EULER = np.zeros((4, int(N/ratio)))
    # Record velocity norm
    vel_norm = np.zeros(N)
    # record the reference position trajectory
    Ref_p = np.zeros((5, int(N/ratio)))
    # record the reference velocity trajectory
    Ref_v = np.zeros((3, int(N/ratio)))
    # record the reference Euler trajectory
    Ref_euler = np.zeros((3, int(N/ratio)))
    # record the reference acceleration trajectory
    Ref_a = np.zeros((3, N))
    # record the reference jerk trajectory
    Ref_j = np.zeros((3, N))
    # record the reference snap trajectory
    Ref_s = np.zeros((3, N))
    # record the state estimates
    p_MH  = np.zeros((3, int(N/ratio)))
    v_MH  = np.zeros((3, int(N/ratio)))
    att_MH= np.zeros((3, int(N/ratio)))
    w_MH  = np.zeros((3, int(N/ratio)))
    # record the measurements
    p_m   = np.zeros((3, int(N/ratio)))
    v_m   = np.zeros((3, int(N/ratio)))
    att_m = np.zeros((3, int(N/ratio)))
    w_m   = np.zeros((3, int(N/ratio)))
    # record time
    Time  = np.zeros(N)
    Timemhe = np.zeros(int(N/ratio))
    # initial varibales in the low-pass filters
    a_lpf_prev = np.zeros((3,1))
    j_lpf_prev = np.zeros((3,1))
    sig_f_prev = 0
    sig_fu_prev = np.zeros((2,1))
    sig_t1_prev = np.zeros((3,1))
    sig_t2_prev = np.zeros((3,1))
    # time index in MHE
    kmhe = 0
    kf   = 0
    flag = 0
    flag2 = 0
    for k in range(N):
        Time[k] = time
        
        # get reference
        t_switch = 0
        dpara = np.array([2,1,1, 2,1,1, 2,1,1, 0.1,0.1,0.1, 0.1,0.1,0.1, 0.1,0.1,0.1]) 
        # ref_p, ref_v, ref_a, ref_j, ref_s = uav.reference_training(Coeffx_training, Coeffy_training, Coeffz_training, time, t_switch) 
        ref_p, ref_v, ref_a, ref_j, ref_s = uav.reference_evaluation(Coeffx_evaluation, Coeffy_evaluation, Coeffz_evaluation, time, t_switch) 
        b1_c  = np.array([[1, 0, 0]]).T # constant desired heading direction
        ref   = np.vstack((ref_p, ref_v, Rb_hd, omegad)) 
        
        # Obtain the noisy measurement (position and velocity)
        state             = np.vstack((p, v, R_h, omega)) # current true state
        # state_m           = state + np.reshape(np.random.normal(0,1e-3,18),(18,1)) 
        state_m           = np.vstack((p+np.reshape(np.random.normal(0,1e-3,3),(3,1)), v+np.reshape(np.random.normal(0,5e-3,3),(3,1)), R_h+np.reshape(np.random.normal(0,1e-3,9),(9,1)), omega+np.reshape(np.random.normal(0,1e-3,3),(3,1))))
        R_bm              = np.array([[state_m[6,0],state_m[7,0],state_m[8,0]],
                                      [state_m[9,0],state_m[10,0],state_m[11,0]],
                                      [state_m[12,0],state_m[13,0],state_m[14,0]]])
        gamma   = np.arctan(R_bm[2, 1]/R_bm[1, 1])
        theta   = np.arctan(R_bm[0, 2]/R_bm[0, 0])
        psi     = np.arcsin(-R_bm[0, 1])
        Euler_m = np.array([[gamma, theta, psi]]).T
        omega_m = np.reshape(state_m[15:18,0],(3,1))
        vel_norm[k] = LA.norm(v)
        nn_input= np.reshape(np.vstack((np.reshape(state_m[0:6,0],(6,1)),Euler_m,np.reshape(state_m[15:18,0],(3,1)))),(D_in,1))
        # nn_input= np.reshape(np.vstack((np.reshape(state_m[0:3,0],(3,1)),2*np.reshape(state_m[3:6,0],(3,1)), Euler_m,np.reshape(state_m[15:18,0],(3,1)))),(D_in,1))

        # State estimation and control 100Hz
        if (k%ratio)==0: 
            Y            += [state_m]
            # nn_input = Y[-1]
            Timemhe[kmhe] = time
            position[0:3,kmhe:kmhe+1] = p
            position[3,kmhe:kmhe+1] = LA.norm(p[0:2,0])
            position[4,kmhe:kmhe+1] = LA.norm(p)
            velocity[:,kmhe:kmhe+1] = v
            EULER[:,kmhe:kmhe+1] = np.reshape(np.vstack((Euler,LA.norm(57.3*Euler))),(4,1))
            p_m[:,kmhe:kmhe+1] = np.vstack((state_m[0,0],state_m[1,0],state_m[2,0]))
            v_m[:,kmhe:kmhe+1] = np.vstack((state_m[3,0],state_m[4,0],state_m[5,0]))
            att_m[:,kmhe:kmhe+1] = Euler_m
            w_m[:,kmhe:kmhe+1]   = omega_m
            Omega[:,kmhe:kmhe+1] = omega
            Ref_p[0:3,kmhe:kmhe+1] = ref_p
            Ref_p[3,kmhe:kmhe+1] = LA.norm(ref_p[0:2,0])
            Ref_p[4,kmhe:kmhe+1] = LA.norm(ref_p)
            Ref_v[:,kmhe:kmhe+1] = ref_v
            Ref    += [alpha*ref+(1-alpha)*state_m]
            df = np.reshape(Dis_f[0:3,kmhe],(3,1))
            dtau = np.reshape(Dis_t[0:3,kmhe],(3,1))
            dis_f[0:3,kmhe:kmhe+1] = df
            dis_f[3,kmhe:kmhe+1] = LA.norm(df[0:2,0])
            dis_f[4,kmhe:kmhe+1] = LA.norm(df)
            dis_t[0:3,kmhe:kmhe+1] = dtau
            dis_t[3,kmhe:kmhe+1] = LA.norm(dtau[0:2,0])
            dis_t[4,kmhe:kmhe+1] = LA.norm(dtau)
            
            #----------MHE state estimation----------#
            if controller == 'a' or controller == 'b':
                if controller != 'b':
                    tunable_para    = convert(model_QR(nn_input))
                P0, gamma_r, gamma_q, R_t, Q_t1, weight_para = SetPara(epsilon, gmin,tunable_para,r11)
                opt_sol      = uavMHE.MHEsolver(Y, x_hat, xmhe_traj, ctrl, noise_traj, weight_para, kmhe)
                xmhe_traj    = opt_sol['state_traj_opt']
                noise_traj   = opt_sol['noise_traj_opt']
                p_MH[:,kmhe:kmhe+1] = np.reshape(xmhe_traj[-1,0:3],(3,1))
                v_MH[:,kmhe:kmhe+1] = np.reshape(xmhe_traj[-1,3:6],(3,1))
                R_b_mh              = np.array([[xmhe_traj[-1,9],xmhe_traj[-1,10],xmhe_traj[-1,11]],
                                            [xmhe_traj[-1,12],xmhe_traj[-1,13],xmhe_traj[-1,14]],
                                            [xmhe_traj[-1,15],xmhe_traj[-1,16],xmhe_traj[-1,17]]])
                gamma_mh   = np.arctan(R_b_mh[2, 1]/R_b_mh[1, 1])
                theta_mh   = np.arctan(R_b_mh[0, 2]/R_b_mh[0, 0])
                psi_mh     = np.arcsin(-R_b_mh[0, 1])
                Euler_mh = np.array([[gamma_mh, theta_mh, psi_mh]]).T
                att_MH[:,kmhe:kmhe+1]=Euler_mh
                w_MH[:,kmhe:kmhe+1]=np.reshape(xmhe_traj[-1,18:21],(3,1))
                if kmhe>horizon:
                # update m based on xmhe_traj
                    for ix in range(len(x_hat)):
                        x_hat[ix,0] = xmhe_traj[1, ix]

                loss_track = uavNMHE.loss_horizon(xmhe_traj,Ref,horizon,kmhe)
                loss_track = np.reshape(loss_track,(1))
                sum_loss += loss_track
                df_Imh   = xmhe_traj[-1, 6:9]
                df_Imh   = np.reshape(df_Imh, (3, 1)) # MHE disturbance estimate
                dtau_mh  = xmhe_traj[-1, 21:24]
                dtau_mh  = np.reshape(dtau_mh, (3, 1))
            
            
            # state_mh = np.reshape(np.hstack((xmhe_traj[-1, 0:6], xmhe_traj[-1, 9:21])), (18,1))
            feedback = state_m

            #-----------L1-Adaptive controller-----------#
            R_bw     = np.array([[feedback[6,0],feedback[7,0],feedback[8,0]],
                                 [feedback[9,0],feedback[10,0],feedback[11,0]],
                                 [feedback[12,0],feedback[13,0],feedback[14,0]]])
            if controller == 'c' or controller == 'd':
                # Piecewise-constant adaptation law
                sig_hat_m, sig_hat_um, As = GeoCtrl.L1_adaptive_law(feedback,z_hat)
                # Update state prediction
                rmhe  = feedback[6:15,0]
                vmhe  = feedback[3:6,0]
                omegamhe = feedback[15:18,0]
                # sig_hat_m, sig_hat_um = np.zeros((4,1)), np.zeros((2,1))
                z_hat = uav.predictor_L1(z_hat,rmhe,vmhe,omegamhe,u,sig_hat_m,sig_hat_um,As,dt_mhe)
                # Low-pass filter
                f_l1  = sig_hat_m[0,0]
                t_l1  = np.reshape(sig_hat_m[1:4,0],(3,1))
                wf_coff, wt1_coff, wt2_coff = 8, 4, 6 
                time_constf, time_constt1, time_constt2 = 1/wf_coff, 1/wt1_coff, 1/wt2_coff
                f_l1_lpf = GeoCtrl.lowpass_filter(time_constf,f_l1,sig_f_prev) 
                sig_f_prev = f_l1_lpf
                fu_l1_lpf = GeoCtrl.lowpass_filter(time_constf,sig_hat_um,sig_fu_prev)
                sig_fu_prev = fu_l1_lpf
                # Two cascaded low-pass filters for the torque
                t_l1_lpf1 = GeoCtrl.lowpass_filter(time_constt1,t_l1,sig_t1_prev)   
                sig_t1_prev = t_l1_lpf1 
                t_l1_lpf2 = GeoCtrl.lowpass_filter(time_constt2,t_l1_lpf1,sig_t2_prev)   
                sig_t2_prev = t_l1_lpf2  
                u_ad = -np.vstack((f_l1_lpf,t_l1_lpf2))
            
            #-----------Geometric controller-------------#
            if controller == 'a' or controller == 'b':
                df_I_hat, dtau_hat = df_Imh, dtau_mh
                df_I_ctrl, dtau_ctrl = df_Imh, dtau_mh
            elif controller == 'c':
                df_I_ctrl, dtau_ctrl = np.zeros((3,1)), np.zeros((3,1))
                df_l1_lpf  = np.reshape(np.vstack((fu_l1_lpf,f_l1_lpf)),(3,1))
                df_I_hat   = np.matmul(R_bw,df_l1_lpf)
                dtau_hat   = t_l1_lpf2
                df_B_l1    = np.reshape(np.vstack((sig_hat_um,f_l1)),(3,1))
                df_l1      = np.matmul(R_bw,df_B_l1)
                dtau_l1    = t_l1
            elif controller == 'd':
                df_l1_lpf  = np.reshape(np.vstack((fu_l1_lpf,f_l1_lpf)),(3,1))
                df_I_hat, dtau_hat = np.matmul(R_bw,df_l1_lpf),t_l1_lpf2
                df_I_ctrl, dtau_ctrl = np.matmul(R_bw,df_l1_lpf),t_l1_lpf2
            else:
                df_I_hat, dtau_hat = np.zeros((3,1)), np.zeros((3,1))
                df_I_ctrl, dtau_ctrl = np.zeros((3,1)), np.zeros((3,1))
            df_MH[0:3,kmhe:kmhe+1] = df_I_hat
            df_MH[3,kmhe:kmhe+1] = LA.norm(df_I_hat[0:2,0])
            df_MH[4,kmhe:kmhe+1] = LA.norm(df_I_hat)
            dtau_MH[0:3,kmhe:kmhe+1] = dtau_hat 
            dtau_MH[3,kmhe:kmhe+1] = LA.norm(dtau_hat[0:2,0])
            dtau_MH[4,kmhe:kmhe+1] = LA.norm(dtau_hat)
           
            u, Rb_hd,omegad, domegad, a_lpf, j_lpf = GeoCtrl.geometric_ctrl(feedback,x_prev,a_lpf_prev,j_lpf_prev,ref_p,ref_v,ref_a,ref_j,ref_s,b1_c, df_I_ctrl,dtau_ctrl)
            # u, Rb_hd, omegad, domegad  = GeoCtrl.geometric_ctrl_2(feedback,ref_p,ref_v,ref_a,ref_j,ref_s,b1_c,df_Imh, dtau_mh)  
            gamma_d   = np.arctan(Rb_hd[6, 0]/Rb_hd[4, 0])
            theta_d   = np.arctan(Rb_hd[2, 0]/Rb_hd[0, 0])
            psi_d     = np.arcsin(-Rb_hd[1, 0])
            Euler_d   = np.array([[gamma_d, theta_d, psi_d]]).T
            Ref_euler[:,kmhe:kmhe+1] = 57.3*Euler_d
            
            x_prev   = feedback
            a_lpf_prev = a_lpf
            j_lpf_prev = j_lpf

            if controller == 'c':
                u = u + u_ad

            ctrl    += [u]
            kmhe += 1
            w, vf_inv,vt_inv   = uav.dis_noise(state,Euler, dpara)

        
        # print('sample=', k, 'v_hat=',z_hat[0:3,0].T,'v_real=',v.T)
        Omegad[:,k:k+1] = omegad 
        dOmegad[:,k:k+1] = domegad
        print('sample=', k, 'df_I_hat=', df_I_hat.T, 'df_true=', df.T, 'dtau_hat=', dtau_hat.T, 'dtau_true=',dtau.T)
        if controller == 'a' or controller =='b':
            tp[:,k:k+1]  = np.reshape(weight_para,(D_out,1))
            Q_k[:,k:k+1] = np.reshape(gamma_q*np.vstack((Q_t1[0,0],Q_t1[1,1],Q_t1[2,2])),(3,1))
            Q_k2[:,k:k+1] = np.reshape(gamma_q**2*np.vstack((Q_t1[0,0],Q_t1[1,1],Q_t1[2,2])),(3,1))

            print('sample=', k, 'gamma1=', gamma_r,'gamma2=', gamma_q,'epsilon=', epsilon, 'gmin=', gmin,'r1=', R_t[0, 0], 'r2=', R_t[1, 1], 'r3=', R_t[2,2], 'q1=', Q_t1[0,0], 'q2=', Q_t1[1,1], 'q3=', Q_t1[2,2])
        # update the system state based on the system dynamics model
        output   = uav.step(state, u, df, dtau, dt_sample)
        p        = output['p_new']
        v        = output['v_new']
        R_h      = output['R_new']
        omega    = output['omega_new']
        Euler    = output['Euler']
           
        # df,dtau = uav.dis(w,df,dtau,dt_sample)
        euler_max = max(EULER[3,:])
        # if k == 400:
        #     df_z = df[2,0]
        #     if df_z > -10:
        #         flag = 1 
        if euler_max > 120 and k>400:
            flag2 = 1
        print('sample=',k,'ref_p=',ref_p.T,'act_p=',p.T,'norm of p=',LA.norm(p),'Attitude=',Euler.T,'Angular rate=',omega.T,'Angle_max=',euler_max)
        fx_max0 = max(dis_f[0,:])
        fy_max0 = max(dis_f[1,:])
        fx_min0 = min(dis_f[0,:])
        fy_min0 = min(dis_f[1,:])
        dfz_min  = min(dis_f[2,:])
        dfz_last = dis_f[2,-1]
        # fx_mean = abs(np.mean(dis_f[0,:]))
        # fy_mean = abs(np.mean(dis_f[1,:]))
        fx_max = max(np.array([fx_max0,abs(fx_min0)]))
        fy_max = max(np.array([fy_max0,abs(fy_min0)]))
        if flag == 1:
            break
        if flag2 == 1:
            break
        if LA.norm(p) > 1e2 or LA.norm(omega)>1e2:
            break

        cov_f[:,k:k+1] = vf_inv
        cov_t[:,k:k+1] = vt_inv
        #update time
        time += dt_sample
    
    mean_loss    = sum_loss/(N/ratio)
    if not os.path.exists("evaluation_results"):
        os.makedirs("evaluation_results")
    # print('mean_loss=',mean_loss)
    # np.save('evaluation_results/Time_evaluation',Time)
    
    # np.save('evaluation_results/cov_f_evaluation',cov_f)
    # np.save('evaluation_results/p_m',p_m)
    # np.save('evaluation_results/v_m',v_m)
    # np.save('evaluation_results/att_m',att_m)
    # np.save('evaluation_results/w_m',w_m)
    # np.save('evaluation_results/Reference_position',Ref_p)
    if controller == 'a':
        np.save('evaluation_results/p_MH',p_MH)
        np.save('evaluation_results/v_MH',v_MH)
        np.save('evaluation_results/att_MH',att_MH)
        np.save('evaluation_results/w_MH',w_MH)
        np.save('evaluation_results/p_m',p_m)
        np.save('evaluation_results/v_m',v_m)
        np.save('evaluation_results/att_m',att_m)
        np.save('evaluation_results/w_m',w_m)
        np.save('evaluation_results/df_MH',df_MH)
        np.save('evaluation_results/dtau_MH',dtau_MH)
        np.save('evaluation_results/Tunable_para_NeuroMHE',tp)
        np.save('evaluation_results/Q_k',Q_k)
        np.save('evaluation_results/p_gt_neuromhe',position)
        np.save('evaluation_results/v_gt_neuromhe',velocity)
        np.save('evaluation_results/att_gt_neuromhe',EULER)
        np.save('evaluation_results/w_gt_neuromhe',Omega)
    elif controller == 'b':
        np.save('evaluation_results/p_MH_dmhe',p_MH)
        np.save('evaluation_results/v_MH_dmhe',v_MH)
        np.save('evaluation_results/att_MH_dmhe',att_MH)
        np.save('evaluation_results/w_MH_dmhe',w_MH)
        np.save('evaluation_results/df_MH_dmhe',df_MH)
        np.save('evaluation_results/dtau_MH_dmhe',dtau_MH)
        np.save('evaluation_results/Tunable_para_dmhe',tp)
        np.save('evaluation_results/Q_k_dmhe',Q_k)
        np.save('evaluation_results/p_gt_dmhe',position)
        np.save('evaluation_results/v_gt_dmhe',velocity)
        np.save('evaluation_results/att_gt_dmhe',EULER)
        np.save('evaluation_results/w_gt_dmhe',Omega)
    elif controller == 'c':
        np.save('evaluation_results/p_gt_l1',position)
        np.save('evaluation_results/v_gt_l1',velocity)
        np.save('evaluation_results/att_gt_l1',EULER)
        np.save('evaluation_results/w_gt_l1',Omega)
        np.save('evaluation_results/df_MH_l1',df_MH)
        np.save('evaluation_results/dtau_MH_l1',dtau_MH)
    elif controller == 'd':
        np.save('evaluation_results/p_gt_l1_active',position)
        np.save('evaluation_results/v_gt_l1_active',velocity)
        np.save('evaluation_results/att_gt_l1_active',EULER)
        np.save('evaluation_results/w_gt_l1_active',Omega)
        np.save('evaluation_results/df_MH_l1_active',df_MH)
        np.save('evaluation_results/dtau_MH_l1_active',dtau_MH)
    else:
        np.save('evaluation_results/p_gt_baseline',position)
        np.save('evaluation_results/v_gt_baseline',velocity)
        np.save('evaluation_results/att_gt_baseline',EULER)
        np.save('evaluation_results/w_gt_baseline',Omega)
   

    

    # if not os.path.exists("evaluation_data"):
    #     os.makedirs("evaluation_data")
    # np.save('evaluation_data/Dis_f',dis_f)
    # np.save('evaluation_data/Dis_t',dis_t)

    # if not os.path.exists("training_data"):
    #     os.makedirs("training_data")
    # np.save('training_data/Dis_f',dis_f)
    # np.save('training_data/Dis_t',dis_t)
    
    
    # compute RMSE of estimaton error and tracking error
    rmse_fx = format(mean_squared_error(df_MH[0,:], dis_f[0,:], squared=False),'.4f')
    rmse_fy = format(mean_squared_error(df_MH[1,:], dis_f[1,:], squared=False),'.4f')
    rmse_fz = format(mean_squared_error(df_MH[2,:], dis_f[2,:], squared=False),'.4f')
    rmse_fxy = format(mean_squared_error(df_MH[3,:], dis_f[3,:], squared=False),'.4f')
    rmse_f  = format(mean_squared_error(df_MH[4,:], dis_f[4,:], squared=False),'.4f')
    rmse_tx = format(mean_squared_error(dtau_MH[0,:], dis_t[0,:], squared=False),'.4f')
    rmse_ty = format(mean_squared_error(dtau_MH[1,:], dis_t[1,:], squared=False),'.4f')
    rmse_tz = format(mean_squared_error(dtau_MH[2,:], dis_t[2,:], squared=False),'.4f')
    rmse_txy = format(mean_squared_error(dtau_MH[3,:], dis_t[3,:], squared=False),'.4f')
    rmse_t  = format(mean_squared_error(dtau_MH[4,:], dis_t[4,:], squared=False),'.4f')
    rmse_px = format(mean_squared_error(position[0,:], Ref_p[0,:], squared=False),'.4f')
    rmse_py = format(mean_squared_error(position[1,:], Ref_p[1,:], squared=False),'.4f')
    rmse_pz = format(mean_squared_error(position[2,:], Ref_p[2,:], squared=False),'.4f')
    rmse_pxy = format(mean_squared_error(position[3,:], Ref_p[3,:], squared=False),'.4f')
    rmse_p = format(mean_squared_error(position[4,:], Ref_p[4,:], squared=False),'.4f')
    rmse_mx = format(mean_squared_error(position[0,:], p_m[0,:], squared=False),'.4f')
    rmse_my = format(mean_squared_error(position[1,:], p_m[1,:], squared=False),'.4f')
    rmse_mz = format(mean_squared_error(position[2,:], p_m[2,:], squared=False),'.4f')
    rmse_mhx = format(mean_squared_error(position[0,:], p_MH[0,:], squared=False),'.4f')
    rmse_mhy = format(mean_squared_error(position[1,:], p_MH[1,:], squared=False),'.4f')
    rmse_mhz = format(mean_squared_error(position[2,:], p_MH[2,:], squared=False),'.4f')
    rmse_mvx = format(mean_squared_error(velocity[0,:], v_m[0,:], squared=False),'.4f')
    rmse_mvy = format(mean_squared_error(velocity[1,:], v_m[1,:], squared=False),'.4f')
    rmse_mvz = format(mean_squared_error(velocity[2,:], v_m[2,:], squared=False),'.4f')
    rmse_mhvx = format(mean_squared_error(velocity[0,:], v_MH[0,:], squared=False),'.4f')
    rmse_mhvy = format(mean_squared_error(velocity[1,:], v_MH[1,:], squared=False),'.4f')
    rmse_mhvz = format(mean_squared_error(velocity[2,:], v_MH[2,:], squared=False),'.4f')
    rmse_mroll= format(mean_squared_error(EULER[0,:], att_m[0,:], squared=False),'.4f')
    rmse_mpit = format(mean_squared_error(EULER[1,:], att_m[1,:], squared=False),'.4f')
    rmse_myaw = format(mean_squared_error(EULER[2,:], att_m[2,:], squared=False),'.4f')
    rmse_mhroll= format(mean_squared_error(EULER[0,:], att_MH[0,:], squared=False),'.4f')
    rmse_mhpit = format(mean_squared_error(EULER[1,:], att_MH[1,:], squared=False),'.4f')
    rmse_mhyaw = format(mean_squared_error(EULER[2,:], att_MH[2,:], squared=False),'.4f')
    rmse_mwx  = format(mean_squared_error(Omega[0,:], w_m[0,:], squared=False),'.4f')
    rmse_mwy  = format(mean_squared_error(Omega[1,:], w_m[1,:], squared=False),'.4f')
    rmse_mwz  = format(mean_squared_error(Omega[2,:], w_m[2,:], squared=False),'.4f')
    rmse_mhwx = format(mean_squared_error(Omega[0,:], w_MH[0,:], squared=False),'.4f')
    rmse_mhwy = format(mean_squared_error(Omega[1,:], w_MH[1,:], squared=False),'.4f')
    rmse_mhwz = format(mean_squared_error(Omega[2,:], w_MH[2,:], squared=False),'.4f')

    rmse    = np.vstack((rmse_fx,rmse_fy,rmse_fz,rmse_fxy,rmse_f,rmse_tx,rmse_ty,rmse_tz,rmse_txy,rmse_t, rmse_px,rmse_py,rmse_pz,rmse_pxy,rmse_p))
    # np.save('RMSE_evaluation',rmse)
    # print('rmse_fx=',rmse_fx,'rmse_fy=',rmse_fy,'rmse_fz=',rmse_fz,'rmse_fxy=',rmse_fxy,'rmse_f=',rmse_f)
    # print('rmse_tx=',rmse_tx,'rmse_ty=',rmse_ty,'rmse_tz=',rmse_tz,'rmse_txy=',rmse_txy,'rmse_t=',rmse_t)
    # print('rmse_px=',rmse_px,'rmse_py=',rmse_py,'rmse_pz=',rmse_pz,'rmse_pxy=',rmse_pxy,'rmse_p=',rmse_p)
    # print('rmse_mx=',rmse_mx,'rmse_my=',rmse_my,'rmse_mz=',rmse_mz)
    # print('rmse_mhx=',rmse_mhx,'rmse_mhy=',rmse_mhy,'rmse_mhz=',rmse_mhz)
    # print('rmse_mvx=',rmse_mvx,'rmse_mvy=',rmse_mvy,'rmse_mvz=',rmse_mvz)
    # print('rmse_mhvx=',rmse_mhvx,'rmse_mhvy=',rmse_mhvy,'rmse_mhvz=',rmse_mhvz)
    # print('rmse_mroll=',rmse_mroll,'rmse_mpit=',rmse_mpit,'rmse_myaw=',rmse_myaw)
    # print('rmse_mhroll=',rmse_mhroll,'rmse_mhpit=',rmse_mhpit,'rmse_mhyaw=',rmse_mhyaw)
    # print('rmse_mwx=',rmse_mwx,'rmse_mwy=',rmse_mwy,'rmse_mwz=',rmse_mwz)
    # print('rmse_mhwx=',rmse_mhwx,'rmse_mhwy=',rmse_mhwy,'rmse_mhwz=',rmse_mhwz)
    rmse_mhvz = float(rmse_mhvz)
    rmse_mhwx = float(rmse_mhwx)
    rmse_mhwy = float(rmse_mhwy)
    """
    Plot figures
    """
    if plot == 'y':
        if controller == 'a':
            K_iteration = np.load('trained_data/K_iteration.npy')
            Loss        = np.load('trained_data/Loss.npy')
            plt.figure(1)
            plt.plot(K_iteration, Loss, linewidth=1.5, marker='o')
            plt.xlabel('Training episodes')
            plt.ylabel('Mean loss')
            plt.grid()
            plt.savefig('evaluation_results/mean_loss_train.png')
            plt.show()
        elif controller == 'b':
            K_iteration = np.load('trained_data/K_iteration_dmhe.npy')
            Loss        = np.load('trained_data/Loss_dmhe.npy')
            plt.figure(1)
            plt.plot(K_iteration, Loss, linewidth=1.5, marker='o')
            plt.xlabel('Training episodes')
            plt.ylabel('Mean loss (DMHE)')
            plt.grid()
            plt.savefig('evaluation_results/mean_loss_train_dmhe.png')
            plt.show()

        plt.figure(2)
        plt.plot(Timemhe, dis_f[0,:], linewidth=1, linestyle='--')
        plt.plot(Timemhe, df_MH[0,:], linewidth=1)
        plt.xlabel('Time [s]')
        plt.ylabel('Disturbance force in x axis')
        if controller == 'a':
            plt.legend(['Ground truth', 'NeuroMHE'])
            plt.grid()
            plt.savefig('evaluation_results/dfx_evaluation_NeuroMHE.png')
        elif controller == 'b':
            plt.legend(['Ground truth', 'DMHE'])
            plt.grid()
            plt.savefig('evaluation_results/dfx_evaluation_DMHE.png')
        elif controller == 'c':
            plt.legend(['Ground truth', 'L1-AC'])
            plt.grid()
            plt.savefig('evaluation_results/dfx_evaluation_L1AC.png')
        elif controller == 'd':
            plt.legend(['Ground truth', 'L1-AC'])
            plt.grid()
            plt.savefig('evaluation_results/dfx_evaluation_L1AC_active.png')
        else:
            plt.legend(['Ground truth', 'Baseline'])
            plt.grid()
            plt.savefig('evaluation_results/dfx_evaluation_baseline.png')
        plt.show()

        plt.figure(3)
        plt.plot(Timemhe, dis_f[1,:], linewidth=1, linestyle='--')
        plt.plot(Timemhe, df_MH[1,:], linewidth=1)
        plt.xlabel('Time [s]')
        plt.ylabel('Disturbance force in y axis')
        if controller == 'a':
            plt.legend(['Ground truth', 'NeuroMHE'])
            plt.grid()
            plt.savefig('evaluation_results/dfy_evaluation_NeuroMHE.png')
        elif controller == 'b':
            plt.legend(['Ground truth', 'DMHE'])
            plt.grid()
            plt.savefig('evaluation_results/dfy_evaluation_DMHE.png')
        elif controller == 'c':
            plt.legend(['Ground truth', 'L1-AC'])
            plt.grid()
            plt.savefig('evaluation_results/dfy_evaluation_L1AC.png')
        elif controller == 'd':
            plt.legend(['Ground truth', 'L1-AC'])
            plt.grid()
            plt.savefig('evaluation_results/dfy_evaluation_L1AC_active.png')
        else:
            plt.legend(['Ground truth', 'Baseline'])
            plt.grid()
            plt.savefig('evaluation_results/dfy_evaluation_baseline.png')
        plt.show()

        plt.figure(4)
        plt.plot(Timemhe, dis_f[2,:], linewidth=1, linestyle='--')
        plt.plot(Timemhe, df_MH[2,:], linewidth=1)
        plt.xlabel('Time [s]')
        plt.ylabel('Disturbance force in z axis')
        if controller == 'a':
            plt.legend(['Ground truth', 'NeuroMHE'])
            plt.grid()
            plt.savefig('evaluation_results/dfz_evaluation_NeuroMHE.png')
        elif controller == 'b':
            plt.legend(['Ground truth', 'DMHE'])
            plt.grid()
            plt.savefig('evaluation_results/dfz_evaluation_DMHE.png')
        elif controller == 'c':
            plt.legend(['Ground truth', 'L1-AC'])
            plt.grid()
            plt.savefig('evaluation_results/dfz_evaluation_L1AC.png')
        elif controller == 'd':
            plt.legend(['Ground truth', 'L1-AC'])
            plt.grid()
            plt.savefig('evaluation_results/dfz_evaluation_L1AC_active.png')
        else:
            plt.legend(['Ground truth', 'Baseline'])
            plt.grid()
            plt.savefig('evaluation_results/dfz_evaluation_baseline.png')
        plt.show()

        plt.figure(5)
        plt.plot(Timemhe, dis_t[0,:], linewidth=1, linestyle='--')
        plt.plot(Timemhe, dtau_MH[0,:], linewidth=1)
        plt.xlabel('Time [s]')
        plt.ylabel('Disturbance torque in x axis')
        if controller == 'a':
            plt.legend(['Ground truth', 'NeuroMHE'])
            plt.grid()
            plt.savefig('evaluation_results/dtx_evaluation_NeuroMHE.png')
        elif controller == 'b':
            plt.legend(['Ground truth', 'DMHE'])
            plt.grid()
            plt.savefig('evaluation_results/dtx_evaluation_DMHE.png')
        elif controller == 'c':
            plt.legend(['Ground truth', 'L1-AC'])
            plt.grid()
            plt.savefig('evaluation_results/dtx_evaluation_L1AC.png')
        elif controller == 'd':
            plt.legend(['Ground truth', 'L1-AC'])
            plt.grid()
            plt.savefig('evaluation_results/dtx_evaluation_L1AC_active.png')
        else:
            plt.legend(['Ground truth', 'Baseline'])
            plt.grid()
            plt.savefig('evaluation_results/dtx_evaluation_baseline.png')
        plt.show()

        plt.figure(6)
        plt.plot(Timemhe, dis_t[1,:], linewidth=1, linestyle='--')
        plt.plot(Timemhe, dtau_MH[1,:], linewidth=1)
        plt.xlabel('Time [s]')
        plt.ylabel('Disturbance torque in y axis')
        if controller == 'a':
            plt.legend(['Ground truth', 'NeuroMHE'])
            plt.grid()
            plt.savefig('evaluation_results/dty_evaluation_NeuroMHE.png')
        elif controller == 'b':
            plt.legend(['Ground truth', 'DMHE'])
            plt.grid()
            plt.savefig('evaluation_results/dty_evaluation_DMHE.png')
        elif controller == 'c':
            plt.legend(['Ground truth', 'L1-AC'])
            plt.grid()
            plt.savefig('evaluation_results/dty_evaluation_L1AC.png')
        elif controller == 'd':
            plt.legend(['Ground truth', 'L1-AC'])
            plt.grid()
            plt.savefig('evaluation_results/dty_evaluation_L1AC_active.png')
        else:
            plt.legend(['Ground truth', 'Baseline'])
            plt.grid()
            plt.savefig('evaluation_results/dty_evaluation_baseline.png')
        plt.show()

        plt.figure(7)
        plt.plot(Timemhe, dis_t[2,:], linewidth=1, linestyle='--')
        plt.plot(Timemhe, dtau_MH[2,:], linewidth=1)
        plt.xlabel('Time [s]')
        plt.ylabel('Disturbance torque in z axis')
        if controller == 'a':
            plt.legend(['Ground truth', 'NeuroMHE'])
            plt.grid()
            plt.savefig('evaluation_results/dtz_evaluation_NeuroMHE.png')
        elif controller == 'b':
            plt.legend(['Ground truth', 'DMHE'])
            plt.grid()
            plt.savefig('evaluation_results/dtz_evaluation_DMHE.png')
        elif controller == 'c':
            plt.legend(['Ground truth', 'L1-AC'])
            plt.grid()
            plt.savefig('evaluation_results/dtz_evaluation_L1AC.png')
        elif controller == 'd':
            plt.legend(['Ground truth', 'L1-AC'])
            plt.grid()
            plt.savefig('evaluation_results/dtz_evaluation_L1AC_active.png')
        else:
            plt.legend(['Ground truth', 'Baseline'])
            plt.grid()
            plt.savefig('evaluation_results/dtz_evaluation_baseline.png')
        plt.show()

        plt.figure(14)
        plt.plot(Time, cov_f[0,:], linewidth=1)
        plt.plot(Time, cov_f[1,:], linewidth=1)
        plt.plot(Time, cov_f[2,:], linewidth=1)
        plt.xlabel('Time [s]')
        plt.ylabel('Covariance inverse')
        plt.legend(['x direction','y direction','z direction'])
        plt.grid()
        plt.savefig('evaluation_results/covariance_inverse.png')
        plt.show()

        plt.figure(15)
        plt.plot(Time, cov_t[0,:], linewidth=1)
        plt.plot(Time, cov_t[1,:], linewidth=1)
        plt.plot(Time, cov_t[2,:], linewidth=1)
        plt.xlabel('Time [s]')
        plt.ylabel('Covariance inverse torque')
        plt.legend(['x direction','y direction','z direction'])
        plt.grid()
        plt.savefig('evaluation_results/covariance_inverse_torque.png')
        plt.show()


        if controller == 'a' or controller == 'b':
            fig, (ax1, ax2)= plt.subplots(2, sharex=True)
            ax1.plot(Time, tp[24,:], linewidth=1)
            ax2.plot(Time, tp[42,:], linewidth=1)
            ax1.set_ylabel('$\gamma_1$')
            ax2.set_xlabel('Time [s]')
            ax2.set_ylabel('$\gamma_2$')
            ax1.grid()
            ax2.grid()
            if controller == 'a':
                plt.savefig('evaluation_results/forgetting_factors.png')
            elif controller == 'b':
                plt.savefig('evaluation_results/forgetting_factors_dmhe.png')
            plt.show()

            plt.figure(16)
            plt.plot(Time, Q_k[0,:], linewidth=1)
            plt.plot(Time, Q_k[1,:], linewidth=1)
            plt.plot(Time, Q_k[2,:], linewidth=1)
            plt.xlabel('Time [s]')
            plt.ylabel('Q elements')
            plt.legend(['$\gamma_2*q_1$','$\gamma_2*q_2$','$\gamma_2*q_3$'])
            plt.grid()
            if controller == 'a':
                plt.savefig('evaluation_results/Q.png')
            elif controller == 'b':
                plt.savefig('evaluation_results/Q_dmhe.png')    
            plt.show()

            plt.figure(17)
            plt.plot(Time, Q_k2[0,:], linewidth=1)
            plt.plot(Time, Q_k2[1,:], linewidth=1)
            plt.plot(Time, Q_k2[2,:], linewidth=1)
            plt.xlabel('Time [s]')
            plt.ylabel('Q elements')
            plt.legend(['$\gamma_2^2*q_1$','$\gamma_2^2*q_2$','$\gamma_2^2*q_3$'])
            plt.grid()
            if controller == 'a':
                plt.savefig('evaluation_results/Q_square.png')
            elif controller == 'b':
                plt.savefig('evaluation_results/Q_square_dmhe.png')    
            plt.show()

        plt.figure(18)
        ax = plt.axes(projection="3d")
        ax.plot3D(position[0,:], position[1,:], position[2,:], linewidth=1.5)
        ax.plot3D(Ref_p[0,:], Ref_p[1,:], Ref_p[2,:], linewidth=1, linestyle='--')
        plt.legend(['Actual', 'Desired'])
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.grid()
        if controller == 'a':
            plt.savefig('evaluation_results/tracking_3D_neuromhe.png')
        elif controller == 'b':
            plt.savefig('evaluation_results/tracking_3D_dmhe.png')
        elif controller == 'c':
            plt.savefig('evaluation_results/tracking_3D_L1.png')
        elif controller == 'd':
            plt.savefig('evaluation_results/tracking_3D_L1active.png')
        else:
            plt.savefig('evaluation_results/tracking_3D_baseline.png')
        plt.show()
    
        plt.figure(19)
        # plt.plot(Timemhe, Ref_p[0,:], linewidth=1)
        plt.plot(Timemhe, p_m[0,:], linewidth=1)
        plt.plot(Timemhe, p_MH[0,:], linewidth=1)
        plt.plot(Timemhe, position[0,:], linewidth=1)

        plt.xlabel('Time [s]')
        plt.ylabel('x [m]')
        if controller == 'a':
            plt.legend(['measured','estimated','real'])
            plt.grid()
            plt.savefig('evaluation_results/x_neuromhe.png')
        elif controller == 'b':
            plt.legend(['desired','real-DMHE'])
            plt.grid()
            plt.savefig('evaluation_results/x_dmhe.png')
        elif controller == 'c':
            plt.legend(['desired','real-L1'])
            plt.grid()
            plt.savefig('evaluation_results/x_L1.png')
        elif controller == 'd':
            plt.legend(['desired','real-L1-active'])
            plt.grid()
            plt.savefig('evaluation_results/x_L1active.png')
        else:
            plt.legend(['desired','real-Baseline'])
            plt.grid()
            plt.savefig('evaluation_results/x_baseline.png')
        plt.show()

        plt.figure(20)
        # plt.plot(Timemhe, Ref_p[0,:], linewidth=1)
        plt.plot(Timemhe, p_m[1,:], linewidth=1)
        plt.plot(Timemhe, p_MH[1,:], linewidth=1)
        plt.plot(Timemhe, position[1,:], linewidth=1)
        plt.xlabel('Time [s]')
        plt.ylabel('y [m]')
        if controller == 'a':
            plt.legend(['measured','estimated','real'])
            plt.grid()
            plt.savefig('evaluation_results/y_neuromhe.png')
        elif controller == 'b':
            plt.legend(['desired','real-DMHE'])
            plt.grid()
            plt.savefig('evaluation_results/y_dmhe.png')
        elif controller == 'c':
            plt.legend(['desired','real-L1'])
            plt.grid()
            plt.savefig('evaluation_results/x_L1.png')
        elif controller == 'd':
            plt.legend(['desired','real-L1-active'])
            plt.grid()
            plt.savefig('evaluation_results/y_L1active.png')
        else:
            plt.legend(['desired','real-Baseline'])
            plt.grid()
            plt.savefig('evaluation_results/y_baseline.png')
        plt.show()
    
        plt.figure(21)
        # plt.plot(Timemhe, Ref_p[0,:], linewidth=1)
        plt.plot(Timemhe, p_m[2,:], linewidth=1)
        plt.plot(Timemhe, p_MH[2,:], linewidth=1)
        plt.plot(Timemhe, position[2,:], linewidth=1)
        plt.xlabel('Time [s]')
        plt.ylabel('z [m]')
        if controller == 'a':
            plt.legend(['measured','estimated','real'])
            plt.grid()
            plt.savefig('evaluation_results/z_neuromhe.png')
        elif controller == 'b':
            plt.legend(['desired','real-DMHE'])
            plt.grid()
            plt.savefig('evaluation_results/z_dmhe.png')
        elif controller == 'c':
            plt.legend(['desired','real-L1'])
            plt.grid()
            plt.savefig('evaluation_results/z_L1.png')
        elif controller == 'd':
            plt.legend(['desired','real-L1-active'])
            plt.grid()
            plt.savefig('evaluation_results/z_L1active.png')
        else:
            plt.legend(['desired','real-Baseline'])
            plt.grid()
            plt.savefig('evaluation_results/z_baseline.png')
        plt.show()

        plt.figure(22)
        # plt.plot(Timemhe, Ref_v[0,:], linewidth=1)
        plt.plot(Timemhe, v_m[0,:], linewidth=1)
        plt.plot(Timemhe, v_MH[0,:], linewidth=1)
        plt.plot(Timemhe, velocity[0,:], linewidth=1)
        plt.xlabel('Time [s]')
        plt.ylabel('vx [m/s]')
        plt.legend(['measured','estimated','real'])
        plt.grid()
        plt.savefig('evaluation_results/velocity_x.png')
        plt.show()

        plt.figure(23)
        # plt.plot(Timemhe, Ref_v[1,:], linewidth=1)
        plt.plot(Timemhe, v_m[1,:], linewidth=1)
        plt.plot(Timemhe, v_MH[1,:], linewidth=1)
        plt.plot(Timemhe, velocity[1,:], linewidth=1)
        plt.xlabel('Time [s]')
        plt.ylabel('vy [m/s]')
        plt.legend(['measured','estimated','real'])
        plt.grid()
        plt.savefig('evaluation_results/velocity_y.png')
        plt.show()
    
        plt.figure(24)
        # plt.plot(Timemhe, Ref_v[2,:], linewidth=1)
        plt.plot(Timemhe, v_m[2,:], linewidth=1)
        plt.plot(Timemhe, v_MH[2,:], linewidth=1)
        plt.plot(Timemhe, velocity[2,:], linewidth=1)
        plt.xlabel('Time [s]')
        plt.ylabel('vz [m/s]')
        plt.legend(['measured','estimated','real'])
        plt.grid()
        plt.savefig('evaluation_results/velocity_z.png')
        plt.show()

        # plt.figure(24)
        # plt.plot(Timemhe, EULER[0,:], linewidth=1)
        # plt.plot(Timemhe, att_MH[0,:], linewidth=1)
        # plt.xlabel('Time [s]')
        # plt.ylabel('roll [rad]')
        # plt.legend(['real', 'NeuroMHE'])
        # plt.grid()
        # plt.savefig('./Euler roll.png')
        # plt.show()

        # plt.figure(25)
        # plt.plot(Timemhe, EULER[1,:], linewidth=1)
        # plt.plot(Timemhe, att_MH[1,:], linewidth=1)
        # plt.xlabel('Time [s]')
        # plt.ylabel('pitch [rad]')
        # plt.legend(['real', 'NeuroMHE'])
        # plt.grid()
        # plt.savefig('./Euler pitch.png')
        # plt.show()

        # plt.figure(26)
        # plt.plot(Timemhe, EULER[2,:], linewidth=1)
        # plt.plot(Timemhe, att_MH[2,:], linewidth=1)
        # plt.xlabel('Time [s]')
        # plt.ylabel('yaw [rad]')
        # plt.legend(['real', 'NeuroMHE'])
        # plt.grid()
        # plt.savefig('./Euler yaw.png')
        # plt.show()

        if controller == 'a':
            plt.figure(27)
            plt.plot(Timemhe,  w_m[0,:], linewidth=1)
            plt.plot(Timemhe,  w_MH[0,:], linewidth=1)
            plt.plot(Timemhe,  Omega[0,:], linewidth=1)
            plt.xlabel('Time [s]')
            plt.ylabel('angular rate x [rad/s]')
            plt.legend(['measured','estimated','real'])
            plt.grid()
            plt.savefig('evaluation_results/angular rate x_neuromhe.png')
            plt.show()

            plt.figure(28)
            plt.plot(Timemhe,  w_m[1,:], linewidth=1)
            plt.plot(Timemhe,  w_MH[1,:], linewidth=1)
            plt.plot(Timemhe,  Omega[1,:], linewidth=1)
            plt.xlabel('Time [s]')
            plt.ylabel('angular rate y [rad/s]')
            plt.legend(['measured','estimated','real'])
            plt.grid()
            plt.savefig('evaluation_results/angular rate y_neuromhe.png')
            plt.show()

            plt.figure(29)
            plt.plot(Timemhe,  w_m[2,:], linewidth=1)
            plt.plot(Timemhe,  w_MH[2,:], linewidth=1)
            plt.plot(Timemhe,  Omega[2,:], linewidth=1)
            plt.xlabel('Time [s]')
            plt.ylabel('angular rate z [rad/s]')
            plt.legend(['measured','estimated','real'])
            plt.grid()
            plt.savefig('evaluation_results/angular rate z_neuromhe.png')
            plt.show()

        # plt.figure(30)
        # plt.plot(Timemhe, att_m[0,:], linewidth=1)
        # plt.plot(Timemhe, att_MH[0,:], linewidth=1)
        # plt.plot(Timemhe, EULER[0,:], linewidth=1)
        # # plt.plot(Timemhe, Ref_euler[0,:], linewidth=1)
        # plt.xlabel('Time [s]')
        # plt.ylabel('Roll angle [deg]')
        # plt.legend(['measured','estimated','real'])
        # plt.grid()
        # plt.savefig('./roll.png')
        # plt.show()

        # plt.figure(31)
        # plt.plot(Timemhe, att_m[1,:], linewidth=1)
        # plt.plot(Timemhe, att_MH[1,:], linewidth=1)
        # plt.plot(Timemhe, EULER[1,:], linewidth=1)
        # # plt.plot(Timemhe, Ref_euler[1,:], linewidth=1)
        # plt.xlabel('Time [s]')
        # plt.ylabel('Pitch angle [deg]')
        # plt.legend(['measured','estimated','real'])
        # plt.grid()
        # plt.savefig('./pitch.png')
        # plt.show()

        # plt.figure(32)
        # plt.plot(Timemhe, att_m[2,:], linewidth=1)
        # plt.plot(Timemhe, att_MH[2,:], linewidth=1)
        # plt.plot(Timemhe, EULER[2,:], linewidth=1)
        # # plt.plot(Timemhe, Ref_euler[2,:], linewidth=1)
        # plt.xlabel('Time [s]')
        # plt.ylabel('Yaw angle [deg]')
        # plt.legend(['measured','estimated','real'])
        # plt.grid()
        # plt.savefig('./yaw.png')
        # plt.show()

        # plt.figure(33)
        # plt.plot(Timemhe, EULER[3,:], linewidth=1)
        # plt.xlabel('Time [s]')
        # plt.ylabel('Euler angle norm [deg]')
        # # plt.legend(['real','desired'])
        # plt.grid()
        # plt.savefig('./angle norm.png')
        # plt.show()

        # plt.figure(34)
        # plt.plot(Time, vel_norm, linewidth=1)
        # plt.xlabel('Time [s]')
        # plt.ylabel('velocity norm [m/s]')
        # plt.grid()
        # plt.savefig('evaluation_results/vel_norm.png')
        # plt.show()

   
    # return df_z, dfz_min, dfz_last, euler_max, k, fx_max, fy_max
    return rmse

"""---------------------------------Main function-----------------------------"""
if mode =="train":
    Train(epsilon0, gmin0, tunable_para0)
    Evaluate()
else:
    Evaluate()
    # evaluate over 100 episodes
    # Rmse_fx, Rmse_fy, Rmse_fz, Rmse_fxy, Rmse_f = [], [], [], [], []
    # Rmse_tx, Rmse_ty, Rmse_tz, Rmse_txy, Rmse_t = [], [], [], [], []
    # Rmse_px, Rmse_py, Rmse_pz, Rmse_pxy, Rmse_p = [], [], [], [], []
    # for i in range(100):
    #     rmse = Evaluate()
    #     print('No.',i)
    #     Rmse_fx += [rmse[0,0]]
    #     Rmse_fy += [rmse[1,0]]
    #     Rmse_fz += [rmse[2,0]]
    #     Rmse_fxy+= [rmse[3,0]]
    #     Rmse_f  += [rmse[4,0]]
    #     Rmse_tx += [rmse[5,0]]
    #     Rmse_ty += [rmse[6,0]]
    #     Rmse_tz += [rmse[7,0]]
    #     Rmse_txy+= [rmse[8,0]]
    #     Rmse_t  += [rmse[9,0]]
    #     Rmse_px += [rmse[10,0]]
    #     Rmse_py += [rmse[11,0]]
    #     Rmse_pz += [rmse[12,0]]
    #     Rmse_pxy+= [rmse[13,0]]
    #     Rmse_p  += [rmse[14,0]]
    #     np.save('RMSE_boxplot_data/Rmse_fx_l1active',Rmse_fx)
    #     np.save('RMSE_boxplot_data/Rmse_fy_l1active',Rmse_fy)
    #     np.save('RMSE_boxplot_data/Rmse_fz_l1active',Rmse_fz)
    #     np.save('RMSE_boxplot_data/Rmse_fxy_l1active',Rmse_fxy)
    #     np.save('RMSE_boxplot_data/Rmse_f_l1active',Rmse_f)
    #     np.save('RMSE_boxplot_data/Rmse_tx_l1active',Rmse_tx)
    #     np.save('RMSE_boxplot_data/Rmse_ty_l1active',Rmse_ty)
    #     np.save('RMSE_boxplot_data/Rmse_tz_l1active',Rmse_tz)
    #     np.save('RMSE_boxplot_data/Rmse_txy_l1active',Rmse_txy)
    #     np.save('RMSE_boxplot_data/Rmse_t_l1active',Rmse_t)
    #     np.save('RMSE_boxplot_data/Rmse_px_l1active',Rmse_px)
    #     np.save('RMSE_boxplot_data/Rmse_py_l1active',Rmse_py)
    #     np.save('RMSE_boxplot_data/Rmse_pz_l1active',Rmse_pz)
    #     np.save('RMSE_boxplot_data/Rmse_pxy_l1active',Rmse_pxy)
    #     np.save('RMSE_boxplot_data/Rmse_p_l1active',Rmse_p)

    # RMSE_fx = np.zeros((len(Rmse_fx),1))
    # RMSE_fy = np.zeros((len(Rmse_fy),1))
    # RMSE_fz = np.zeros((len(Rmse_fz),1))
    # RMSE_fxy = np.zeros((len(Rmse_fxy),1))
    # RMSE_f  = np.zeros((len(Rmse_f),1))
    # RMSE_tx = np.zeros((len(Rmse_tx),1))
    # RMSE_ty = np.zeros((len(Rmse_ty),1))
    # RMSE_tz = np.zeros((len(Rmse_tz),1))
    # RMSE_txy = np.zeros((len(Rmse_txy),1))
    # RMSE_t = np.zeros((len(Rmse_t),1))
    # RMSE_px = np.zeros((len(Rmse_px),1))
    # RMSE_py = np.zeros((len(Rmse_py),1))
    # RMSE_pz = np.zeros((len(Rmse_pz),1))
    # RMSE_pxy = np.zeros((len(Rmse_pxy),1))
    # RMSE_p = np.zeros((len(Rmse_p),1))
    # print('len of rmse=',len(Rmse_fx))
    # for i in range(len(Rmse_fx)):
    #     RMSE_fx[i,0] = Rmse_fx[i]
    #     RMSE_fy[i,0] = Rmse_fy[i]
    #     RMSE_fz[i,0] = Rmse_fz[i]
    #     RMSE_fxy[i,0]= Rmse_fxy[i]
    #     RMSE_f[i,0]  = Rmse_f[i]
    #     RMSE_tx[i,0] = Rmse_tx[i]
    #     RMSE_ty[i,0] = Rmse_ty[i]
    #     RMSE_tz[i,0] = Rmse_tz[i]
    #     RMSE_txy[i,0]= Rmse_txy[i]
    #     RMSE_t[i,0]  = Rmse_t[i]
    #     RMSE_px[i,0] = Rmse_px[i]
    #     RMSE_py[i,0] = Rmse_py[i]
    #     RMSE_pz[i,0] = Rmse_pz[i]
    #     RMSE_pxy[i,0]= Rmse_pxy[i]
    #     RMSE_p[i,0]  = Rmse_p[i]
    
    # plt.figure(1)
    # box_data = np.hstack((RMSE_fx,RMSE_fy,RMSE_fz,RMSE_fxy,RMSE_f))
    # df = pd.DataFrame(data=box_data,columns = ['fx','fy','fz','fxy','f'])
    # df.boxplot()
    # plt.savefig('RMSE_boxplot_data/rmse_force_box_l1active.png',dpi=600)
    # plt.show()  
    # plt.figure(2)
    # box_data = np.hstack((RMSE_tx,RMSE_ty,RMSE_tz,RMSE_txy,RMSE_t))
    # df = pd.DataFrame(data=box_data,columns = ['tx','ty','tz','txy','t'])
    # df.boxplot()
    # plt.savefig('RMSE_boxplot_data/rmse_torque_box_l1active.png',dpi=600)
    # plt.show()
    # plt.figure(3)
    # box_data = np.hstack((RMSE_px,RMSE_py,RMSE_pz,RMSE_pxy,RMSE_p))
    # df = pd.DataFrame(data=box_data,columns = ['px','py','pz','pxy','p'])
    # df.boxplot()
    # plt.savefig('RMSE_boxplot_data/rmse_position_box_l1active.png',dpi=600)
    # plt.show()

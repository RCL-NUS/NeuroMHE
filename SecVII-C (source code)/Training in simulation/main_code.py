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
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.kalman import UnscentedKalmanFilter as UKF

print("=============================================")
print("Main code for training or evaluating NeuroMHE")
print("Please choose mode")
mode = input("enter 'train' or 'evaluate' without the quotation mark:")
print("=============================================")


"""---------------------------------Load environment---------------------------------------"""
uav_para = np.array([1,0.02,0.02,0.04])
wing_len = 0.5

if mode == 'train':
    dt_sample = 1e-2
    dt_mhe   = 2e-2
    dt_att   = 1e-2
else:
    dt_sample = 1e-2
    dt_mhe   = 1e-2
    dt_att   = 1e-2 # Sampling time-step for attitude controller

ctrl_gain = np.array([[40,40,40, 12,12,12, 16.6,16.6,16.6, 1,1,1]]) # this control gain is tuned for training purpose in simulation.
ratio     = int(dt_mhe/dt_sample)
ratio_att = int(dt_att/dt_sample)
uav = UavEnv.quadrotor(uav_para, dt_mhe)
uav.model()
horizon = 10 
# Learning rate
lr_nn  = 1e-4
# First element in R_t
r11    = np.array([[100]])


"""---------------------------------Define parameterization model--------------------------"""
# Define neural network for process noise
D_in, D_h, D_out = 3, 20, 14
model_QR = uavNN.Net(D_in, D_h, D_out)

"""---------------------------------Define reference trajectory----------------------------"""
"""
The reference trajectory is generated using minisnap algorithm [2]
[2] Mellinger, D. and Kumar, V., 2011, May. 
    Minimum snap trajectory generation and control for quadrotors. 
    In 2011 IEEE international conference on robotics and automation (pp. 2520-2525). IEEE.
"""
# Load the reference trajectory polynomials
Coeffx_gazebo, Coeffy_gazebo, Coeffz_gazebo = np.zeros((5,8)), np.zeros((5,8)), np.zeros((5,8))
Coeffx_exp1, Coeffy_exp1, Coeffz_exp1 = np.zeros((12,8)), np.zeros((12,8)), np.zeros((12,8))
Coeffx_exp1_lfast, Coeffy_exp1_lfast, Coeffz_exp1_lfast = np.zeros((12,8)), np.zeros((12,8)), np.zeros((12,8))
Coeffx_exp2, Coeffy_exp2, Coeffz_exp2 = np.zeros((3,8)), np.zeros((3,8)), np.zeros((3,8))
Coeffx_exp2_lo, Coeffy_exp2_lo, Coeffz_exp2_lo = np.zeros((9,8)), np.zeros((9,8)), np.zeros((9,8))

for i in range(3): # 18s
    Coeffx_exp2[i,:] = np.load('Ref_exp2/coeffx'+str(i+1)+'.npy')
    Coeffy_exp2[i,:] = np.load('Ref_exp2/coeffy'+str(i+1)+'.npy')
    Coeffz_exp2[i,:] = np.load('Ref_exp2/coeffz'+str(i+1)+'.npy')

for i in range(12): # 33s
    Coeffx_exp1[i,:] = np.load('Ref_exp1/coeffx'+str(i+1)+'.npy')
    Coeffy_exp1[i,:] = np.load('Ref_exp1/coeffy'+str(i+1)+'.npy')
    Coeffz_exp1[i,:] = np.load('Ref_exp1/coeffz'+str(i+1)+'.npy')

for i in range(12): # 33s
    Coeffx_exp1_lfast[i,:] = np.load('Ref_exp1_littlefast/coeffx'+str(i+1)+'.npy')
    Coeffy_exp1_lfast[i,:] = np.load('Ref_exp1_littlefast/coeffy'+str(i+1)+'.npy')
    Coeffz_exp1_lfast[i,:] = np.load('Ref_exp1_littlefast/coeffz'+str(i+1)+'.npy')
for i in range(9):#28
    Coeffx_exp2_lo[i,:] = np.load('Ref_exp2_linearoscillation/coeffx'+str(i+1)+'.npy')
    Coeffy_exp2_lo[i,:] = np.load('Ref_exp2_linearoscillation/coeffy'+str(i+1)+'.npy')
    Coeffz_exp2_lo[i,:] = np.load('Ref_exp2_linearoscillation/coeffz'+str(i+1)+'.npy')

"""---------------------------------Define controller--------------------------------------"""
GeoCtrl  = Robust_Flight.Controller(uav_para, ctrl_gain, dt_att)

"""---------------------------------Define MHE---------------------------------------------"""
uavMHE = Robust_Flight.MHE(horizon, dt_mhe,r11)
uavMHE.SetStateVariable(uav.xp)
uavMHE.SetOutputVariable(uav.y)
uavMHE.SetControlVariable(uav.f)
uavMHE.SetNoiseVariable(uav.wf)
uavMHE.SetRotationVariable(uav.R_B)
uavMHE.SetModelDyn(uav.dymh,uav.dyukf)
uavMHE.SetCostDyn()

"""---------------------------------Define Kalman filter-based Gradient solver-------------"""
uavNMHE = Robust_Flight.KF_gradient_solver(uav.xp, uavMHE.weight_para)

"""---------------------------------Parameterization of Neural Network Output--------------"""
epsilon0, gmin0 = 1e-4, 1e-4
def SetPara(epsilon, gmin,tunable_para, r11):
    p_diag = np.zeros((1, 6))
    for i in range(6):
        p_diag[0, i] = epsilon + tunable_para[0, i]**2
    P0 = np.diag(p_diag[0])

    gamma_r = gmin + (1 - gmin) * 1/(1+np.exp(-tunable_para[0, 6]))
    gamma_q = gmin + (1 - gmin) * 1/(1+np.exp(-tunable_para[0, 10]))

    r_diag = np.zeros((1, 3))
    for i in range(3):
        r_diag[0, i] = epsilon + tunable_para[0, i+7]**2
    R_t    = np.diag(r_diag[0])

    q_diag = np.zeros((1, 3))
    for i in range(3):
        q_diag[0, i] = epsilon + tunable_para[0, i+11]**2
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
    P = SX.sym('P', 1, 6)
    for i in range(6):
        P[0, i] = epsilon + tunable[0, i]**2

    gamma_r = gmin + (1 - gmin) * 1/(1+exp(-tunable[0, 6]))
    gamma_q = gmin + (1 - gmin) * 1/(1+exp(-tunable[0, 10]))
    
    R = SX.sym('R', 1, 3)
    for i in range(3):
        R[0, i] = epsilon + tunable[0, i+7]**2

    Q = SX.sym('Q', 1, 3)
    for i in range(3):
        Q[0, i] = epsilon + tunable[0, i+11]**2

    weight = horzcat(P, gamma_r, R, gamma_q, Q)
    w_jaco = jacobian(weight, tunable)
    w_jaco_fn = Function('W_fn',[tunable],[w_jaco],['tp'],['W_fnf'])
    weight_grad = w_jaco_fn(tp=tunable_para)['W_fnf'].full()
    return weight_grad
"""---------------------------------Define model for UKF---------------------------"""
def DynUKF(x, dt, U):
    u = U[0]
    Rb = np.array([[U[1],U[2],U[3]],
                   [U[4],U[5],U[6]],
                   [U[7],U[8],U[9]]])
    x_next = uavMHE.MDyn_ukf_fn(s=x, c=u, R=Rb)['MDyn_ukff']
    x_next = np.reshape(x_next, (6))
    return x_next

def OutputUKF(x):
    uavMHE.diffKKT_general()
    H = uavMHE.H_fn(x0=x)['Hf']
    output = np.matmul(H, x)
    y = np.reshape(output, (3))
    return y

# UKF settings
sigmas = MerweScaledSigmaPoints(6, alpha=.1, beta=2., kappa=1)
ukf    = UKF(dim_x=6, dim_z=3, fx=DynUKF, hx=OutputUKF, dt=dt_mhe, points=sigmas)

# Covariance matrices for UKF, tuned using the training trajectory
wr, wqd, wqf = 1e-2,1e-2,0.2
ukf.R = np.diag([wr, wr, wr]) # measurement noise
ukf.Q = np.diag([wqd, wqd, wqd, wqf,wqf, wqf]) # process noise

# Initial tunable parameters for DMHE, tuned to produce a similar initial mean_loss with NeuroMHE
p0, r0, q0 = 0.45, 0.1, 0.5
tunable_para0 = np.array([[p0,p0,p0,p0,p0,p0,
                           0.1,
                           r0,r0,r0,
                           0.1,
                           q0,q0,q0]])
def normalization(v):
    gain = 7
    vn   = np.zeros((3,1))
    for i in range(3):
        vn[i:i+1,0] = gain*np.tanh(v[i,0])
    return vn 
ep   = 0

alpha    = np.array([[0.825,0.825,0.825]])
def Train(epsilon0, gmin0, tunable_para0):
    print("===============================================")
    print("Please choose which estimator to train")
    print("'a': NeuroMHE (adaptive weights by a neural network)")
    print("'b': DMHE (locally optimal but fixed weights)")
    estimator =  input("enter 'a' or 'b' without the quotation mark:")
    print("===============================================")
    # Total simulation time
    T_end   = 33
    # Total iterations
    N       = int(T_end/dt_sample)   
    if estimator=='a':
        epsilon = epsilon0
        gmin    = gmin0
        # (since the neural network parameters are randomly initialized, use the saved initial model to replicate the same training results in the paper)
        PATH0 = "trained_data/initial_nn_model.pt"
        model_QR = torch.load(PATH0)
        # model_QR = uavNN.Net(D_in, D_h, D_out)
        # torch.save(model_QR,PATH0)

    else:
        epsilon = epsilon0
        gmin    = gmin0
        tunable_para = tunable_para0# Initial tunable parameters for DMHE
   
    # Lists for storing training data
    Loss        = []
    Position    = []
    Ref_P       = []
    Dis_f_mh    = []
    K_iteration = []
    # Training iteration index
    k_train     = 0
    # Initial mean loss
    mean_loss0  = 0
    # Initial change of the mean loss
    delta_loss  = 1e7
    # Load training data
    dis_f = np.load('training_data/Dis_f.npy') 
    np.save('trained_data/Dis_f',dis_f)
    # Threshold
    eps = 1e-1
    threshold = 1e5
    phase = 1
    L0  = 1.5
    meanloss_untrained = 0

    while delta_loss >= eps:
        flag   = 0
        flag2  = 0
        # Initial time
        time   = 0
        # Initial states
        x0     = np.random.normal(1,0.01)
        y0     = np.random.normal(1,0.01)
        p      = np.array([[x0,y0,0.12]]).T
        v      = np.zeros((3,1))
        Euler  = np.zeros((3,1))
        R_h, R_bw, R_wb = uav.dir_cosine(Euler)
        omega  = np.zeros((3,1))
        # Initial guess of state and disturbance force used in MHE
        df     = np.zeros((3,1))
        df_est = np.zeros((3,1))
        dtau   = np.zeros((3,1))
        x_hat  = np.vstack((v,df))
        xmhe_traj  = x_hat
        noise_traj = np.zeros((1,3))
        # Lists for storing control, measurement, rotation matrix, ground truth, and reference
        ctrl   = []
        Y      = []
        R_seq  = []
        Gt     = []
        Ref    = []
        # Sum of loss
        sum_loss = 0.0
        # Vectors for recoding position, reference position, and disturbance force estimate in each episode
        position = np.zeros((3, N))
        Ref_p = np.zeros((3, N))
        dis_f_mh = np.zeros((3, N))
        # Initial states in the low-pass filters 
        v_prev = v
        a_lpf_prev = np.zeros((3,1))
        j_lpf_prev = np.zeros((3,1))
        # Natural length of the elastic band
        kmhe = 0
        # initial guess of control signal in the low-pass filter
        u_prev = np.array([[uav_para[0]*9.8,0,0,0]]).T
        for k in range(N):
            # Load the reference
            t_switch = 0
            ref_p, ref_v, ref_a, ref_j, ref_s = uav.ref_exp1_training(Coeffx_exp1_lfast, Coeffy_exp1_lfast, Coeffz_exp1_lfast, time, t_switch)
            # ref_p, ref_v, ref_a, ref_j, ref_s = uav.ref_traj_training_for_experiment(Coeffx_training2, Coeffy_training2, Coeffz_training2, time, t_switch) 
            b1_c              = np.array([[1, 0, 0]]).T # constant desired heading direction
            # load disturbance data
            
            state             = np.vstack((p, v, R_h, omega)) # current true state
            if (k%ratio)==0:
                df                = dis_f[:,kmhe] 
                # Obtain the noisy measurement
                xp_m          = np.vstack((p,v)) + np.reshape(np.random.normal(0,1e-3,6),(6,1)) # set the standard deviation of measurement noise to be 1e-3 since the accuracy of OptiTrack can be 0.2 mm
                p_m           = np.reshape(xp_m[0:3,0],(3,1))
                v_m           = np.reshape(xp_m[3:6,0],(3,1))
                nn_input      = v_m 
                Y      += [v_m]
                R_B     = np.array([[R_h[0,0],R_h[1,0],R_h[2,0]],
                                [R_h[3,0],R_h[4,0],R_h[5,0]],
                                [R_h[6,0],R_h[7,0],R_h[8,0]]])
                R_seq  += [R_B]

                #----------MHE state estimation----------#
                # Generate the weighting parameters for MHE
                if estimator == 'a':
                    tunable_para    = convert(model_QR(nn_input))
                P0, gamma_r, gamma_q, R_t, Q_t1, weight_para = SetPara(epsilon, gmin,tunable_para,r11)
                print('learning_iteration:', k_train, 'sample=', k, 'gamma1=', gamma_r,'gamma2=', gamma_q,'r1=', R_t[0, 0], 'r2=', R_t[1, 1], 'r3=', R_t[2,2], 'q1=', Q_t1[0,0], 'q2=', Q_t1[1,1], 'q3=', Q_t1[2,2])
                opt_sol      = uavMHE.MHEsolver(Y, R_seq, x_hat, xmhe_traj, ctrl, noise_traj, weight_para, kmhe)
                xmhe_traj    = opt_sol['state_traj_opt']
                costate_traj = opt_sol['costate_ipopt']
                noise_traj   = opt_sol['noise_traj_opt']
                if kmhe>horizon:
                    for ix in range(len(x_hat)):# update m based on xmhe_traj
                        x_hat[ix,0] = xmhe_traj[1, ix]
                # Obtain the coefficient matricres to establish the auxiliary MHE system
                auxSys       = uavMHE.GetAuxSys_general(xmhe_traj, costate_traj, noise_traj, weight_para, Y, ctrl, R_seq)
                matF, matG, matH = auxSys['matF'], auxSys['matG'], auxSys['matH']
                matddLxx, matddLxp = auxSys['matddLxx'], auxSys['matddLxp']
                matddLxw, matddLww, matddLwp = auxSys['matddLxw'], auxSys['matddLww'], auxSys['matddLwp']
                # Solve for the analytical gradient using a Kalman filter (Algorithm 1, Lemma 2)
                if kmhe <= horizon:
                    M = np.zeros((len(x_hat), D_out))
                else:
                    M = X_opt[1]
                gra_opt = uavNMHE.GradientSolver_general(M, matF, matG, matddLxx, matddLxp, matddLxw, matddLww, matddLwp, P0)
                X_opt   = gra_opt['state_gra_traj']
            
                #-----------Geometric controller-------------#
                v_mhe    = np.reshape(xmhe_traj[-1, 0:3], (3, 1))
                df_Imh   = np.reshape(xmhe_traj[-1, 3:6], (3, 1)) # MHE disturbance estimate
                df_est   = df_Imh
                dtau_mh  = np.zeros((3,1))
                dis_f_mh[:,k:k+1] = df_Imh
                print('learning_iteration:', k_train, 'sample=', k, 'df_Imh=', df_Imh.T, 'df_true=', df.T)
                #-----------Policy update based on gradient descent------------#
                # gt       = np.vstack((p,v))
                 
                Alpha    = np.diag(alpha[0])
                softref  = np.matmul(Alpha,ref_v)+np.matmul((np.identity(3)-Alpha),v_m)
                # Gt      += [np.reshape(np.vstack((softref, df)),(9,1))]
                Gt      += [softref]
                # Gt      += [df]
                dldw, loss_track = uavNMHE.ChainRule(Gt, xmhe_traj,X_opt)
                weight_grad      = chainRule_gradient(epsilon, gmin,tunable_para)
                dldp             = np.matmul(dldw, weight_grad)
                norm_grad        = LA.norm(dldp)
                if k_train > 0: # train the neural network from the second learning epoch
                    if norm_grad>=threshold:
                        dldp = dldp/norm_grad*threshold
                        epsilon = epsilon + 1e-3
                        gmin = gmin + 1e-5
                        if gmin >=0.1: # a very small gmin can reduce the estimation error and delay, but will also lead to instablity in high-fidelity simulation like Gazebo and real world experiments
                            gmin = 0.1
                    new_norm = LA.norm(dldp)
                    if estimator == 'a':
                        loss_nn_p   = model_QR.myloss(model_QR(nn_input), dldp)
                        optimizer_p = torch.optim.Adam(model_QR.parameters(), lr=lr_nn)
                        model_QR.zero_grad()
                        loss_nn_p.backward()
                        optimizer_p.step()
                    else:
                        tunable_para = tunable_para - lr_nn*dldp  
                    print('sample=',k, 'loss=', loss_track, 'phase=',phase, 'epsilon=', epsilon, 'gmin=', gmin, 'dldw[6]=',dldw[0,6],'dldw[10]=',dldw[0,10],'norm_dldw=', new_norm )
                print('sample=',k,'ref_p=',ref_p.T,'act_p=',p.T,'Attitude_norm=',LA.norm(57.3*Euler), 'mean_loss0=',mean_loss0)    
                # xp_m = np.vstack((np.reshape(xp_m[0:3,0],(3,1)),v_m))
                Fd = GeoCtrl.position_ctrl(xp_m,ref_p,ref_v,ref_a,df_Imh)
                kmhe += 1
            
            if (k%ratio_att)==0:
                feedback = np.vstack((xp_m,R_h,omega))
                u, R_B_dh,omegad,domegad, a_lpf, j_lpf = GeoCtrl.attitude_ctrl(feedback,Fd,v_prev,a_lpf_prev,j_lpf_prev,ref_p,ref_v,ref_a,ref_j,ref_s,b1_c,dtau_mh)
                v_prev   = v_m
                a_lpf_prev = a_lpf
                j_lpf_prev = j_lpf
                wf_coff = 60 # 40, 20, 30
                time_constf = 1/wf_coff
                u_lpf = GeoCtrl.lowpass_filter(time_constf,u,u_prev) 
                u_prev = u_lpf
                f = u[0,0]
        
            if (k%ratio)==0: 
                ctrl    += [f]

            # Update the system state based on the system dynamics model
            # to mimic the real scenario where a certain delay may occur in the lower-level controller, we apply a 1st-order low-pass filter to the control signal.
            output   = uav.step(state, u_lpf, df, dtau, dt_sample)
            p        = output['p_new']
            v        = output['v_new']
            R_h      = output['R_new']
            omega    = output['omega_new']
            Euler    = output['Euler']
            # Update disturbance
            # df, length, v_lpf = GeoCtrl.dis(p,L_prev,v_lpf_prev,L0)
            # L_prev     = length
            # v_lpf_prev = v_lpf
            # Update time
            time += dt_sample 
            loss_track = np.reshape(loss_track,(1))
            sum_loss += loss_track
            if LA.norm(57.3*Euler)>=120:
                flag = 1
                break
        mean_loss = sum_loss/N
        Dis_f_mh    += [dis_f_mh]
        K_iteration += [k_train]
        if estimator == 'a':
            PATH1 = "trained_data/trained_nn_model.pt"
            torch.save(model_QR, PATH1)
        else:
            np.save('trained_data/tunable_para_trained', tunable_para)
        if k_train == 0:
            meanloss_untrained = mean_loss
            eps = mean_loss/5e2
            if eps <=1:
                eps = 1
        if k_train>2:
            delta_loss = abs(mean_loss-mean_loss0)
        
        Loss += [mean_loss]

        if k_train >= 20 or (k_train>0 and mean_loss >= mean_loss0):
            flag2 = 1
        k_train += 1
        mean_loss0 = mean_loss
        print('learning_iteration:',k_train,'mean_loss=',mean_loss, 'eps=', eps, 'flag=',flag,'flag2=',flag2)

        if flag == 1 or flag2 == 1:
            break
        
        if not os.path.exists("trained_data"):
            os.makedirs("trained_data")
        
        if estimator=='a':
            np.save('trained_data/Loss', Loss)
            np.save('trained_data/K_iteration',K_iteration)
            np.save('trained_data/Position_train',Position)
            np.save('trained_data/Dis_f_mh_train',Dis_f_mh)
            np.save('trained_data/epsilon_trained',epsilon)
            np.save('trained_data/gmin_trained',gmin) 
        else:
            np.save('trained_data/Loss_dmhe', Loss)
            np.save('trained_data/K_iteration_dmhe',K_iteration)
            np.save('trained_data/Position_train_dmhe',Position)
            np.save('trained_data/Dis_f_mh_train_dmhe',Dis_f_mh)
            np.save('trained_data/epsilon_trained_dmhe',epsilon)
            np.save('trained_data/gmin_trained_dmhe',gmin)  
        
        # if meanloss_untrained <=3100 or meanloss_untrained>=3500:
        #     break
        

    # plt.figure(1)
    # plt.plot(K_iteration, Loss, linewidth=1.5, marker='o')
    # plt.xlabel('Training episodes')
    # plt.ylabel('Mean loss')
    # plt.grid()
    # plt.savefig('./mean_loss_train.png')
    # plt.show()
    # uav.play_animation(wing_len, FULLSTATE, REFP, dt_sample)
    return gamma_r, gamma_q, gmin, mean_loss, flag, flag2


"""---------------------------------Evaluation process-----------------------------"""
def Evaluate():
    print("===============================================")
    print("Please choose which controller to evaluate")
    print("'a': NeuroMHE            + Geometric Controller")
    print("'b': DMHE                + Geometric Controller")
    print("'c': L1 Adaptive Control + Geometric Controller")
    print("'d': UKF                 + Geometric Controller")
    controller = input("enter 'a', or 'b',... without the quotation mark:")
    print("===============================================")

    # Total simulation time
    T_end  = 33
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

    # Initial states
    x0     = np.random.normal(1,0.01)
    y0     = np.random.normal(1,0.01)
    p      = np.array([[x0,y0,0.1]]).T
    v      = np.zeros((3,1))
    roll   = np.random.normal(0,0.01)
    pitch  = np.random.normal(0,0.01)
    yaw    = np.random.normal(0,0.01)
    Euler  = np.array([[roll,pitch,yaw]]).T
    R_h, R_bw, R_wb = uav.dir_cosine(Euler)
    omega  = np.zeros((3,1))
    # Initial guess of state and disturbance force
    vg0    = np.zeros((3,1))
    df     = np.zeros((3,1))
    df_est = np.zeros((3,1))
    dtau   = np.zeros((3,1))
    # Initial state prediction
    z_hat  = np.zeros((3,1))
    # Filter priori
    x_hat  = np.vstack((vg0,df))
    xmhe_traj = x_hat
    noise_traj = np.zeros((1,3))
    # Initial control
    f      = uav_para[0]*9.78
    # Lists for saving measurement, control, reference, ground truth, and rotation matrix
    Y      = []
    ctrl   = []
    Ref    = []
    Gt     = []
    R_seq  = []
    # sum of loss
    sum_loss = 0.0
    # record the position tracking performance  
    position = np.zeros((3, int(N/ratio)))
    velocity = np.zeros((3, int(N/ratio)))
    Omega    = np.zeros((3, N))
    Omegad   = np.zeros((3, N))
    dOmegad  = np.zeros((3, N))
    # record the disturbance  
    dis_f =  np.zeros((3, int(N/ratio)))
    df_total =  np.zeros((1, int(N/ratio)))
    dis_t =  np.zeros((3, int(N/ratio)))
    Dis_f = np.zeros((3, N))
    # Dis_f_gt = np.load('training_data/Dis_f.npy')
    Dis_f_dot = np.zeros((3, N))
    inv_cov_df = np.zeros((3, N))
    
    # record the disturbance estimates 
    df_MH = np.zeros((3, int(N/ratio)))
    df_total_MH = np.zeros((1, int(N/ratio)))
    # record L1 estimation without filtering
    df_L1   = np.zeros((3, int(N/ratio)))
    dtau_L1 = np.zeros((3, int(N/ratio)))
    # record the weighting matrix 
    tp    = np.zeros((D_out, N))
    # record the reference position trajectory
    Ref_p = np.zeros((3, int(N/ratio)))
    norm_input = np.zeros(int(N/ratio))
    # record the reference velocity trajectory
    Ref_v = np.zeros((3, int(N/ratio)))
    # Record the position estimates
    p_MH  = np.zeros((3, int(N/ratio)))
    v_MH  = np.zeros((3, int(N/ratio)))
    # Record the position measurements
    P_m   = np.zeros((3, int(N/ratio)))
    V_m   = np.zeros((3, int(N/ratio)))
    # Record acceleration
    acc   = np.zeros((3, int(N/ratio)))
    # Record time
    Time  = np.zeros(N)
    Timemhe = np.zeros(int(N/ratio))
    # Record attitude
    EULER = np.zeros((4, N))
    # initial varibales in the low-pass filters
    v_prev = vg0
    a_lpf_prev = np.zeros((3,1))
    j_lpf_prev = np.zeros((3,1))
    sig_f_prev = 0
    sig_fu_prev = np.zeros((2,1))
    sig_t1_prev = np.zeros((3,1))
    sig_t2_prev = np.zeros((3,1))
    u_prev = np.array([[uav_para[0]*9.8,0,0,0]]).T
    # time index in MHE
    kmhe = 0
    kf   = 0
    # Natural length of the elastic band
    L0     = 1.5 
    # Previous dis_f
    dis_f_prev = np.zeros((3,1))
    dis_f_lpf_prev = np.zeros((3,1))
    for k in range(N):
        Time[k] = time
        EULER[:,k:k+1] = np.reshape(np.vstack((57.3*Euler,LA.norm(57.3*Euler))),(4,1))
        # get reference
        t_switch = 0
        # ref_p, ref_v, ref_a, ref_j, ref_s = uav.ref_exp2_linearoscillation(Coeffx_exp2_lo, Coeffy_exp2_lo, Coeffz_exp2_lo, time, t_switch) 
        ref_p, ref_v, ref_a, ref_j, ref_s = uav.ref_exp1_training(Coeffx_exp1, Coeffy_exp1, Coeffz_exp1, time, t_switch) 
        # ref_p, ref_v, ref_a, ref_j, ref_s = uav.ref_traj_training_for_experiment(Coeffx_training2, Coeffy_training2, Coeffz_training2, time, t_switch) 
        # ref_p, ref_v, ref_a, ref_j, ref_s = uav.ref_gazebo(Coeffx_gazebo, Coeffy_gazebo, Coeffz_gazebo, time, t_switch) 
        # ref_p, ref_v, ref_a, ref_j, ref_s = uav.ref_exp2(Coeffx_exp2, Coeffy_exp2, Coeffz_exp2, time, t_switch) 
        b1_c  = np.array([[1, 0, 0]]).T # constant desired heading direction
        ref   = np.vstack((ref_p, ref_v))
        Omega[:,k:k+1] = omega
        Dis_f[:,k:k+1] = df
        df_dot = (df-dis_f_prev)/dt_sample
        wf_coff = 5 # 40, 20, 30
        time_constf = 1/wf_coff
        df_l1_lpf = GeoCtrl.lowpass_filter(time_constf,df_dot,dis_f_lpf_prev) 
        dis_f_lpf_prev = df_l1_lpf
        dis_f_prev = df
        Dis_f_dot[:,k:k+1] = df_l1_lpf
        inv_cov_df[:,k:k+1] = 1/(abs(df_l1_lpf)+1)
        # Obtain the noisy measurement (position and velocity)
        state          = np.vstack((p, v, R_h, omega)) # current true state
        xp             = np.vstack((p, v))# current positional state
        xp_m           = xp + np.reshape(np.random.normal(0,1e-3,6),(6,1)) # set the standard deviation of measurement noise to be 1e-3 since the accuracy of OptiTrack can be 0.2 mm
        p_m            = np.reshape(xp_m[0:3,0],(3,1))
        v_m            = np.reshape(xp_m[3:6,0],(3,1))
        # State estimation and control 50Hz
        if (k%ratio)==0: 
            # df                = np.reshape(Dis_f_gt[:,kmhe],(3,1))
            P_m[:,kmhe:kmhe+1] = np.vstack((xp_m[0,0],xp_m[1,0],xp_m[2,0]))
            V_m[:,kmhe:kmhe+1] = np.vstack((xp_m[3,0],xp_m[4,0],xp_m[5,0]))
            nn_input      = v_m
            Y            += [v_m]
            Timemhe[kmhe] = time
            norm_input[kmhe] = LA.norm(nn_input)
            position[:,kmhe:kmhe+1] = p
            velocity[:,kmhe:kmhe+1] = v
            Ref_p[:,kmhe:kmhe+1] = ref_p
            Ref_v[:,kmhe:kmhe+1] = ref_v
            Ref                 += [ref]
            R_B     = np.array([[R_h[0,0],R_h[1,0],R_h[2,0]],
                                [R_h[3,0],R_h[4,0],R_h[5,0]],
                                [R_h[6,0],R_h[7,0],R_h[8,0]]])
            R_seq  += [R_B] 
            
            #----------MHE state estimation----------#
            if controller != 'b':
                tunable_para    = convert(model_QR(nn_input))
                # tunable_p_eshape(tunable_para_large[0,42],(1,1))
            # Q_mhe   = np.reshape(tunable_para_large[0,43:46],(1,3))
            # tunable_para = np.hstack((P_mhe,gamma_r,R_mhe,gamma_q,Q_mhe))

            P0, gamma_r, gamma_q, R_t, Q_t1, weight_para = SetPara(epsilon, gmin,tunable_para,r11)
            opt_sol      = uavMHE.MHEsolver(Y, R_seq, x_hat, xmhe_traj, ctrl, noise_traj, weight_para, kmhe)
            xmhe_traj    = opt_sol['state_traj_opt']
            noise_traj   = opt_sol['noise_traj_opt']
            
            v_MH[:,kmhe:kmhe+1] = np.reshape(xmhe_traj[-1,0:3],(3,1))
            if kmhe>horizon:
            # update m based on xmhe_traj
                for ix in range(len(x_hat)):
                    x_hat[ix,0] = xmhe_traj[1, ix]
            
            Alpha    = np.diag(alpha[0])
            softref  = np.matmul(Alpha,ref_v)+np.matmul((np.identity(3)-Alpha),v_m)
            # Gt      += [np.reshape(np.vstack((softref, df)),(9,1))]
            Gt      += [softref]
            # Gt      += [df]
            loss_track = uavNMHE.loss_horizon(xmhe_traj,Gt,horizon,kmhe)
            loss_track = np.reshape(loss_track,(1))
            sum_loss += loss_track
            df_Imh   = xmhe_traj[-1, 3:6]
            df_Imh   = np.reshape(df_Imh, (3, 1)) # MHE disturbance estimate
            df_est   = df_Imh 
            dtau_mh  = np.zeros((3,1))
            dis_f[:,kmhe:kmhe+1] = df
            df_total[:,kmhe:kmhe+1] = LA.norm(df)
            
            #-----------UKF------------------------------#
            if controller == 'd':
                u_ukf = np.vstack((np.reshape(f,(1,1)),R_h))
                u_ukf = np.reshape(u_ukf,(10))
                ukf.predict(U=u_ukf)
                y     = np.reshape(v_m,(3))
                ukf.update(z=y)
                Xukf = ukf.x.copy()
                df_Imh = np.reshape(Xukf[3:6],(3,1))
            #-----------L1-Adaptive controller-----------#
            if controller == 'c':
                feedback = np.vstack((xp_m,R_h,omega))
                # Piecewise-constant adaptation law
                sig_hat_m, sig_hat_um, As = GeoCtrl.L1_adaptive_law(feedback,R_B,z_hat)
                # Update state prediction
                vmhe  = feedback[3:6,0]
                # sig_hat_m, sig_hat_um = np.zeros((4,1)), np.zeros((2,1))
                z_hat = uav.predictor_L1(z_hat,R_B,vmhe,f,sig_hat_m,sig_hat_um,As,dt_mhe)
                # Low-pass filter
                wf_coff = 5 # 40, 20, 30
                time_constf = 1/wf_coff
                f_l1_lpf = GeoCtrl.lowpass_filter(time_constf,sig_hat_m,sig_f_prev) 
                sig_f_prev = f_l1_lpf
                fu_l1_lpf = GeoCtrl.lowpass_filter(time_constf,sig_hat_um,sig_fu_prev)
                sig_fu_prev = fu_l1_lpf
                u_ad = -f_l1_lpf
            
            #-----------Geometric controller-------------#
            if controller == 'c':
                # df_I_ctrl, dtau_ctrl = np.zeros((3,1)), np.zeros((3,1))
                df_l1_lpf  = np.reshape(np.vstack((fu_l1_lpf,f_l1_lpf)),(3,1))
                df_I_hat   = np.matmul(R_B,df_l1_lpf)
                df_I_ctrl, dtau_ctrl = df_I_hat, np.zeros((3,1))
                df_B_l1    = np.reshape(np.vstack((sig_hat_um,sig_hat_m)),(3,1))
                df_l1      = np.matmul(R_B,df_B_l1)
                
            else:
                # df_I_ctrl, dtau_ctrl = df_Imh, dtau_mh
                df_I_hat, dtau_hat = df_Imh, dtau_mh
                df_I_ctrl, dtau_ctrl = df_Imh, dtau_mh
            df_MH[:,kmhe:kmhe+1] = df_I_hat
            df_total_MH[:,kmhe:kmhe+1] = LA.norm(df_I_hat)
            # dtau_MH[:,kmhe:kmhe+1] = dtau_hat 
            # df_L1[:,kmhe:kmhe+1] = df_l1
            # dtau_L1[:,kmhe:kmhe+1] = dtau_l1  
            Fd = GeoCtrl.position_ctrl(xp_m,ref_p,ref_v,ref_a,df_I_ctrl)
            kmhe += 1
            
        if (k%ratio_att)==0:
            feedback = np.vstack((xp_m,R_h,omega))
            u, R_B_dh,omegad,domegad, a_lpf, j_lpf = GeoCtrl.attitude_ctrl(feedback,Fd,v_prev,a_lpf_prev,j_lpf_prev,ref_p,ref_v,ref_a,ref_j,ref_s,b1_c,dtau_mh)
            v_prev   = v_m
            a_lpf_prev = a_lpf
            j_lpf_prev = j_lpf
            wf_coff = 60 # 40, 20, 30
            time_constf = 1/wf_coff
            u_lpf = GeoCtrl.lowpass_filter(time_constf,u,u_prev) 
            u_prev = u_lpf
            f = u[0,0]
        
        if (k%ratio)==0: 
            ctrl    += [f]
            
        print('sample=', k, 'gamma1=', gamma_r,'gamma2=', gamma_q,'epsilon=', epsilon, 'gmin=', gmin, 'r1=', R_t[0, 0], 'r2=', R_t[1, 1], 'r3=', R_t[2,2], 'q1=', Q_t1[0,0], 'q2=', Q_t1[1,1], 'q3=', Q_t1[2,2])
        # print('sample=', k, 'v_hat=',z_hat[0:3,0].T,'v_real=',v.T)
        Omegad[:,k:k+1] = omegad 
        dOmegad[:,k:k+1] = domegad
        # df       = dis_f[:,k]
        # dtau     = dis_t[:,k]
        print('sample=', k, 'df_I_hat=', df_I_hat.T, 'df_true=', df.T)
        tp[:,k:k+1]  = np.reshape(weight_para,(D_out,1))
        # update the system state based on the system dynamics model
        output   = uav.step(state, u_lpf, df, dtau, dt_sample)
        p        = output['p_new']
        v        = output['v_new']
        R_h      = output['R_new']
        omega    = output['omega_new']
        Euler    = output['Euler']
        print('sample=',k,'ref_p=',ref_p.T,'act_p=',p.T,'Attitude=',Euler.T,'Angular rate=',omega.T) 
        # Update disturbance
        df = GeoCtrl.dis(p,L0)
        #update time
        time += dt_sample
    mean_loss    = sum_loss/(N/ratio)
    print('mean_loss=',mean_loss)
    np.save('Time_evaluation',Timemhe)
    # np.save('Position_evaluation',position)
    # np.save('Dist_f_evaluation',dis_f)
    np.save('Dist_f_NeuroMHE_evaluation',df_MH)
    # np.save('Tunable_para_evaluation_NeuroMHE',tp)
    np.save('position',position)
    if not os.path.exists("evaluation_data"):
        os.makedirs("evaluation_data")
    # np.save('evaluation_data/Dis_f',Dis_f)
    # np.save('evaluation_data/Dis_t',Dis_t)
    if not os.path.exists("training_data"):
        os.makedirs("training_data")
    # np.save('training_data/Dis_f',dis_f)
    # np.save('training_data/Dis_t',Dis_t)
    # np.save('Reference_position',Ref_p)
    # np.save('Dis_f_evaluation',dis_f)
    # np.save('Dis_t_for_training',dis_t)

    # compute RMSE of estimaton error and tracking error
    rmse_fx = mean_squared_error(df_MH[0,:], dis_f[0,:], squared=False)
    rmse_fy = mean_squared_error(df_MH[1,:], dis_f[1,:], squared=False)
    rmse_fz = mean_squared_error(df_MH[2,:], dis_f[2,:], squared=False)
    rmse_f  = mean_squared_error(df_total_MH, df_total, squared=False)
    # rmse_tx = mean_squared_error(dtau_MH[0,:], dis_t[0,:], squared=False)
    # rmse_ty = mean_squared_error(dtau_MH[1,:], dis_t[1,:], squared=False)
    # rmse_tz = mean_squared_error(dtau_MH[2,:], dis_t[2,:], squared=False)
    rmse_px = mean_squared_error(position[0,:], Ref_p[0,:], squared=False)
    rmse_py = mean_squared_error(position[1,:], Ref_p[1,:], squared=False)
    rmse_pz = mean_squared_error(position[2,:], Ref_p[2,:], squared=False)
    # rmse_mx = mean_squared_error(position[0,:], p_m[0,:], squared=False)
    # rmse_my = mean_squared_error(position[1,:], p_m[1,:], squared=False)
    # rmse_mz = mean_squared_error(position[2,:], p_m[2,:], squared=False)
    # rmse_mhx = mean_squared_error(position[0,:], p_MH[0,:], squared=False)
    # rmse_mhy = mean_squared_error(position[1,:], p_MH[1,:], squared=False)
    # rmse_mhz = mean_squared_error(position[2,:], p_MH[2,:], squared=False)
    rmse_mvx = mean_squared_error(velocity[0,:], V_m[0,:], squared=False)
    rmse_mvy = mean_squared_error(velocity[1,:], V_m[1,:], squared=False)
    rmse_mvz = mean_squared_error(velocity[2,:], V_m[2,:], squared=False)
    rmse_mhvx = mean_squared_error(velocity[0,:], v_MH[0,:], squared=False)
    rmse_mhvy = mean_squared_error(velocity[1,:], v_MH[1,:], squared=False)
    rmse_mhvz = mean_squared_error(velocity[2,:], v_MH[2,:], squared=False)
    rmse    = np.vstack((rmse_fx,rmse_fy,rmse_fz, rmse_px,rmse_py,rmse_pz))
    np.save('RMSE_evaluation',rmse)
    print('rmse_fx=',rmse_fx,'rmse_fy=',rmse_fy,'rmse_fz=',rmse_fz,'rmse_f=',rmse_f)
    # print('rmse_tx=',rmse_tx,'rmse_ty=',rmse_ty,'rmse_tz=',rmse_tz)
    print('rmse_px=',rmse_px,'rmse_py=',rmse_py,'rmse_pz=',rmse_pz)
    # print('rmse_mx=',rmse_mx,'rmse_my=',rmse_my,'rmse_mz=',rmse_mz)
    # print('rmse_mhx=',rmse_mhx,'rmse_mhy=',rmse_mhy,'rmse_mhz=',rmse_mhz)
    print('rmse_mvx=',rmse_mvx,'rmse_mvy=',rmse_mvy,'rmse_mvz=',rmse_mvz)
    print('rmse_mhvx=',rmse_mhvx,'rmse_mhvy=',rmse_mhvy,'rmse_mhvz=',rmse_mhvz)
    """
    Plot figures
    """
    if controller == 'a':
        K_iteration = np.load('trained_data/K_iteration.npy')
        Loss        = np.load('trained_data/Loss.npy')
        plt.figure(1)
        plt.plot(K_iteration, Loss, linewidth=1.5, marker='o')
        plt.xlabel('Training episodes')
        plt.ylabel('Mean loss')
        plt.grid()
        plt.savefig('./mean_loss_train.png')
        plt.show()
    elif controller =='b':
        K_iteration = np.load('trained_data/K_iteration_dmhe.npy')
        Loss        = np.load('trained_data/Loss_dmhe.npy')
        plt.figure(1)
        plt.plot(K_iteration, Loss, linewidth=1.5, marker='o')
        plt.xlabel('Training episodes')
        plt.ylabel('Mean loss')
        plt.grid()
        plt.savefig('./mean_loss_train_dmhe.png')
        plt.show()
    
    
    # disturbance
    plt.figure(2)
    plt.plot(Timemhe, dis_f[0,:], linewidth=1, linestyle='--')
    plt.plot(Timemhe, df_MH[0,:], linewidth=1)
    plt.xlabel('Time [s]')
    plt.ylabel('Disturbance force in x axis')
    plt.legend(['Ground truth', 'NeuroMHE'])
    plt.grid()
    plt.savefig('./dfx_NeuroMHE.png')
    plt.show()

    plt.figure(3)
    plt.plot(Timemhe, dis_f[1,:], linewidth=1, linestyle='--')
    plt.plot(Timemhe, df_MH[1,:], linewidth=1)
    plt.xlabel('Time [s]')
    plt.ylabel('Disturbance force in y axis')
    plt.legend(['Ground truth', 'NeuroMHE'])
    plt.grid()
    plt.savefig('./dfy_NeuroMHE.png')
    plt.show()

    plt.figure(4)
    plt.plot(Timemhe, dis_f[2,:], linewidth=1, linestyle='--')
    plt.plot(Timemhe, df_MH[2,:], linewidth=1)
    plt.xlabel('Time [s]')
    plt.ylabel('Disturbance force in z axis')
    plt.legend(['Ground truth', 'NeuroMHE'])
    plt.grid()
    plt.savefig('./dfz_NeuroMHE.png')
    plt.show()

    plt.figure(5)
    plt.plot(Timemhe, np.reshape(df_total,(np.size(Timemhe))), linewidth=1, linestyle='--')
    plt.plot(Timemhe, np.reshape(df_total_MH,(np.size(Timemhe))), linewidth=1)
    plt.xlabel('Time [s]')
    plt.ylabel('Total disturbance force')
    plt.legend(['Ground truth', 'NeuroMHE'])
    plt.grid()
    plt.savefig('./df_NeuroMHE.png')
    plt.show()

    fig, (ax1, ax2)= plt.subplots(2, sharex=True)
    ax1.plot(Time, tp[6,:], linewidth=1)
    ax2.plot(Time, tp[10,:], linewidth=1)
    ax1.set_ylabel('$\gamma_1$')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('$\gamma_2$')
    ax1.grid()
    ax2.grid()
    if controller == 'a':
        plt.savefig('./forgetting_factors.png')
    elif controller == 'b':
        plt.savefig('./forgetting_factors_dmhe.png')
    plt.show()

    plt.figure(7)
    plt.plot(Time, tp[7,:], linewidth=1)
    plt.plot(Time, tp[8,:], linewidth=1)
    plt.plot(Time, tp[9,:], linewidth=1)
    plt.xlabel('Time [s]')
    plt.ylabel('R elements')
    plt.legend(['$r_1$','$r_2$','$r_3$'])
    plt.grid()
    if controller == 'a':
        plt.savefig('./R.png')
    elif controller == 'b':
        plt.savefig('./R_dmhe.png')    
    plt.show()
    
    plt.figure(8)
    plt.plot(Time, tp[11,:], linewidth=1)
    plt.plot(Time, tp[12,:], linewidth=1)
    plt.plot(Time, tp[13,:], linewidth=1)
    plt.xlabel('Time [s]')
    plt.ylabel('Q elements')
    plt.legend(['$q_1$','$q_2$','$q_3$'])
    plt.grid()
    if controller == 'a':
        plt.savefig('./Q.png')
    elif controller == 'b':
        plt.savefig('./Q_dmhe.png')    
    plt.show()
    
    # Trajectory
    plt.figure(9)
    ax = plt.axes(projection="3d")
    ax.plot3D(position[0,:], position[1,:], position[2,:], linewidth=1.5)
    ax.plot3D(Ref_p[0,:], Ref_p[1,:], Ref_p[2,:], linewidth=1, linestyle='--')
    plt.legend(['Actual', 'Desired'])
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    # plt.zlabel('z [m]')
    plt.grid()
    plt.savefig('./tracking_3D.png')
    plt.show()
    
    plt.figure(10)
    plt.plot(Timemhe, P_m[0,:], linewidth=1)
    # plt.plot(Timemhe, p_MH[0,:], linewidth=1)
    plt.plot(Timemhe, position[0,:], linewidth=1)
    plt.plot(Timemhe, Ref_p[0,:], linewidth=1)
    plt.xlabel('Time [s]')
    plt.ylabel('x')
    plt.legend(['measured','real','desired'])
    plt.grid()
    plt.savefig('./position_x.png')
    plt.show()

    plt.figure(11)
    plt.plot(Timemhe, P_m[1,:], linewidth=1)
    # plt.plot(Timemhe, p_MH[1,:], linewidth=1)
    plt.plot(Timemhe, position[1,:], linewidth=1)
    plt.plot(Timemhe, Ref_p[1,:], linewidth=1)
    plt.xlabel('Time [s]')
    plt.ylabel('y')
    plt.legend(['measured', 'real','desired'])
    plt.grid()
    plt.savefig('./position_y.png')
    plt.show()
    
    plt.figure(12)
    plt.plot(Timemhe, P_m[2,:], linewidth=1)
    # plt.plot(Timemhe, p_MH[2,:], linewidth=1)
    plt.plot(Timemhe, position[2,:], linewidth=1)
    plt.plot(Timemhe, Ref_p[2,:], linewidth=1)
    plt.xlabel('Time [s]')
    plt.ylabel('z')
    plt.legend(['measured', 'real','desired'])
    plt.grid()
    plt.savefig('./position_z.png')
    plt.show()

    plt.figure(13)
    plt.plot(Timemhe, V_m[0,:], linewidth=1)
    plt.plot(Timemhe, v_MH[0,:], linewidth=1)
    plt.plot(Timemhe, velocity[0,:], linewidth=1)
    plt.plot(Timemhe, Ref_v[0,:], linewidth=1)
    plt.xlabel('Time [s]')
    plt.ylabel('vx')
    plt.legend(['measured', 'estimated','real','desired'])
    plt.grid()
    plt.savefig('./velocity_x.png')
    plt.show()

    plt.figure(14)
    plt.plot(Timemhe, V_m[1,:], linewidth=1)
    plt.plot(Timemhe, v_MH[1,:], linewidth=1)
    plt.plot(Timemhe, velocity[1,:], linewidth=1)
    plt.plot(Timemhe, Ref_v[1,:], linewidth=1)
    plt.xlabel('Time [s]')
    plt.ylabel('vy')
    plt.legend(['measured', 'estimated','real','desired'])
    plt.grid()
    plt.savefig('./velocity_y.png')
    plt.show()
    
    plt.figure(15)
    plt.plot(Timemhe, V_m[2,:], linewidth=1)
    plt.plot(Timemhe, v_MH[2,:], linewidth=1)
    plt.plot(Timemhe, velocity[2,:], linewidth=1)
    plt.plot(Timemhe, Ref_v[2,:], linewidth=1)
    plt.xlabel('Time [s]')
    plt.ylabel('vz')
    plt.legend(['measured', 'estimated','real','desired'])
    plt.grid()
    plt.savefig('./velocity_z.png')
    plt.show()

    # plt.figure(15)
    # plt.plot(Time,  Omega[0,:], linewidth=1)
    # plt.plot(Time, Omegad[0,:], linewidth=1)
    # plt.xlabel('Time [s]')
    # plt.ylabel('angular rate x')
    # plt.legend(['omega_x', 'omegax_ref'])
    # plt.grid()
    # plt.savefig('./angular rate x.png')
    # plt.show()

    # plt.figure(16)
    # plt.plot(Time,  Omega[1,:], linewidth=1)
    # plt.plot(Time, Omegad[1,:], linewidth=1)
    # plt.xlabel('Time [s]')
    # plt.ylabel('angular rate y')
    # plt.legend(['omega_y', 'omegay_ref'])
    # plt.grid()
    # plt.savefig('./angular rate y.png')
    # plt.show()

    # plt.figure(17)
    # plt.plot(Time,  Omega[2,:], linewidth=1)
    # plt.plot(Time, Omegad[2,:], linewidth=1)
    # plt.xlabel('Time [s]')
    # plt.ylabel('angular rate z')
    # plt.legend(['omega_z', 'omegaz_ref'])
    # plt.grid()
    # plt.savefig('./angular rate z.png')
    # plt.show()

    plt.figure(18)
    plt.plot(Time, Dis_f_dot[0,:], linewidth=1)
    plt.plot(Time, Dis_f_dot[1,:], linewidth=1)
    plt.plot(Time, Dis_f_dot[2,:], linewidth=1)
    plt.xlabel('Time [s]')
    plt.ylabel('df_dot [N/s]')
    plt.legend(['dot_fx', 'dot_fy','dot_fz'])
    plt.grid()
    plt.savefig('./df dot.png')
    plt.show()
    
    plt.figure(19)
    plt.plot(Time, inv_cov_df[0,:], linewidth=1)
    plt.plot(Time, inv_cov_df[1,:], linewidth=1)
    plt.plot(Time, inv_cov_df[2,:], linewidth=1)
    plt.xlabel('Time [s]')
    plt.ylabel('df_dot [N/s]')
    plt.legend(['inv_cov_dfx', 'inv_cov_dfx','inv_cov_dfx'])
    plt.grid()
    plt.savefig('./inv_cov_df.png')
    plt.show()

    plt.figure(20)
    plt.plot(Timemhe,norm_input, linewidth=1)
    plt.xlabel('Time [s]')
    plt.ylabel('norm of nn_input [m/s]')
    plt.grid()
    plt.savefig('./norm_input.png')
    plt.show()

    # plt.figure(18)
    # plt.plot(Time, EULER[0,:], linewidth=1)
    # plt.plot(Time, EULER[1,:], linewidth=1)
    # plt.plot(Time, EULER[2,:], linewidth=1)
    # plt.plot(Time, EULER[3,:], linewidth=1)
    # plt.xlabel('Time [s]')
    # plt.ylabel('Euler angle [deg]')
    # plt.legend(['roll','pitch','yaw','norm'])
    # plt.grid()
    # plt.savefig('./euler angle.png')
    # plt.show()
    

"""---------------------------------Main function-----------------------------"""
if mode =="train":
    Train(epsilon0, gmin0, tunable_para0)
else:
    Evaluate()
    

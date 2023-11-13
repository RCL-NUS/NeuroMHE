"""
This file plots figures showing wrench estimation and trajectory-tracking performances of NeuroMHE
Evaluation data-set is synthetic disturbance data which is state-dependent
==============================================================================================
Wang Bingheng, at Control and Simulation Lab, NUS, Singapore
first version: 24 Dec. 2021
second version: 27 May. 2022
third version: 22 Feb 2022 after receiving the reviewers' comments
wangbingheng@u.nus.edu
"""
import numpy as np
# import matplotlib as mpl
import matplotlib.pyplot as plt
# from matplotlib.pyplot import MultipleLocator
import matplotlib.ticker
from UavEnv import *
from sklearn.metrics import mean_squared_error
import pandas as pd
from statistics import mean 
from statistics import median
from matplotlib.patches import Patch

uav_para = np.array([1,0.02,0.02,0.04])
wing_len = 1
# Sampling time-step for MHE
dt_sample = 1e-2
uav = quadrotor(uav_para,dt_sample)

if not os.path.exists("plots_in_paper"):
    os.makedirs("plots_in_paper")
"""--------------Load data----------------"""
# position history during training
Position_training = np.load('trained_data/Position_train.npy')
act_x_0 = Position_training[0][0,:]
act_y_0 = Position_training[0][1,:]
act_z_0 = Position_training[0][2,:]
act_x_1 = Position_training[1][0,:]
act_y_1 = Position_training[1][1,:]
act_z_1 = Position_training[1][2,:]
act_x_2 = Position_training[2][0,:]
act_y_2 = Position_training[2][1,:]
act_z_2 = Position_training[2][2,:]
# position history in evaluation
# position_eval_NeuroMHE = np.load('Position_NeuroMHE_evaluation.npy')
# position_eval_DMHE     = np.load('Position_DMHE_evaluation.npy')
# reference position 
Ref_p = np.load('trained_data/REFP.npy')
Ref_p_e = np.load('evaluation_results/Reference_position.npy')
ref_x = Ref_p[0,:]
ref_y = Ref_p[1,:]
ref_z = Ref_p[2,:]
ref_x_e = Ref_p_e[0,:]
ref_y_e = Ref_p_e[1,:]
ref_z_e = Ref_p_e[2,:]
# time
Time = np.load('trained_data/Time_training.npy')
Time_e = np.load('evaluation_results/Time_evaluation.npy')
# ground truth force in training
Gt = np.load('trained_data/Dis_f.npy')
Gt_tau = np.load('trained_data/Dis_t.npy')
# ground truth force (inertial frame) in evaluation
Gt_e = np.load('evaluation_data/Dis_f.npy')
Gt_tau_e = np.load('evaluation_data/Dis_t.npy')
# force estimate history of NeuroMHE in training
Df_neuromhe_train = np.load('trained_data/Dis_f_mh_train.npy')
Dtau_neuromhe_train = np.load('trained_data/Dis_tau_mh_train.npy')
dfx_train_0  = Df_neuromhe_train[0][0,:]
dfy_train_0  = Df_neuromhe_train[0][1,:]
dfz_train_0  = Df_neuromhe_train[0][2,:]
dfxy_train_0 = Df_neuromhe_train[0][3,:]
dfx_train_1  = Df_neuromhe_train[1][0,:]
dfy_train_1  = Df_neuromhe_train[1][1,:]
dfz_train_1  = Df_neuromhe_train[1][2,:]
dfxy_train_1 = Df_neuromhe_train[1][3,:]
dfx_train_2  = Df_neuromhe_train[2][0,:]
dfy_train_2  = Df_neuromhe_train[2][1,:]
dfz_train_2  = Df_neuromhe_train[2][2,:]
dfxy_train_2 = Df_neuromhe_train[2][3,:]
dtxy_train_0 = Dtau_neuromhe_train[0][3,:]
dtz_train_0  = Dtau_neuromhe_train[0][2,:]
dtxy_train_1 = Dtau_neuromhe_train[1][3,:]
dtz_train_1  = Dtau_neuromhe_train[1][2,:]
dtxy_train_2 = Dtau_neuromhe_train[2][3,:]
dtz_train_2  = Dtau_neuromhe_train[2][2,:]

Disf_xy = np.zeros(len(Gt[0]))
Dist_xy = np.zeros(len(Gt[0]))
Groundt = np.zeros((4,len(Gt[0])))
for i in range(len(Gt[0])):
    Disf_xy[i] = LA.norm(Gt[0:2,i])
    Dist_xy[i] = LA.norm(Gt_tau[0:2,i])
    gt         = np.vstack((np.reshape(Gt[:,i],(3,1)),LA.norm(Gt[0:2,i])))
    Groundt[:,i:i+1] = gt
               
# force and torque estimation in evaluation
df_MH = np.load('evaluation_results/df_MH.npy')
df_MH_dmhe = np.load('evaluation_results/df_MH_dmhe.npy')
df_MH_l1    = np.load('evaluation_results/df_MH_l1.npy')
df_MH_l1active = np.load('evaluation_results/df_MH_l1_active.npy')
dtau_MH = np.load('evaluation_results/dtau_MH.npy')
dtau_MH_dmhe = np.load('evaluation_results/dtau_MH_dmhe.npy')
dtau_MH_l1   = np.load('evaluation_results/dtau_MH_l1.npy')
dtau_MH_l1active = np.load('evaluation_results/dtau_MH_l1_active.npy')
# full state history in training
FULLSTATE = np.load('trained_data/FULLSTATE_train.npy')
REFP      = np.load('trained_data/REFP.npy')

# time-varying weight matrices in evaluation
weight_para = np.load('trained_data/Tunable_para_NeuroMHE.npy')
Q_k         = np.load('trained_data/Q_k.npy')
weight_para_e = np.load('evaluation_results/Tunable_para_NeuroMHE.npy')
Q_k_e       = np.load('evaluation_results/Q_k.npy')

# rmse of force estimation error and trajectory-tracking error
p_gt = np.load('trained_data/p_gt_neuromhe.npy')
p_m  = np.load('trained_data/p_m.npy')
p_MH = np.load('trained_data/p_MH.npy')
v_gt = np.load('trained_data/v_gt_neuromhe.npy')
v_m  = np.load('trained_data/v_m.npy')
v_MH = np.load('trained_data/v_MH.npy')
att_gt = np.load('trained_data/att_gt_neuromhe.npy')
att_m  = np.load('trained_data/att_m.npy')
att_MH = np.load('trained_data/att_MH.npy')
w_gt = np.load('trained_data/w_gt_neuromhe.npy')
w_m  = np.load('trained_data/w_m.npy')
w_MH = np.load('trained_data/w_MH.npy')
p_norm_gt = np.zeros(np.size(p_gt,1))
p_norm_m  = np.zeros(np.size(p_gt,1))
p_norm_MH = np.zeros(np.size(p_gt,1))
v_norm_gt = np.zeros(np.size(p_gt,1))
v_norm_m  = np.zeros(np.size(p_gt,1))
v_norm_MH = np.zeros(np.size(p_gt,1))
att_norm_gt = np.zeros(np.size(p_gt,1))
att_norm_m  = np.zeros(np.size(p_gt,1))
att_norm_MH = np.zeros(np.size(p_gt,1))
w_norm_gt = np.zeros(np.size(p_gt,1))
w_norm_m  = np.zeros(np.size(p_gt,1))
w_norm_MH = np.zeros(np.size(p_gt,1))

# for i in range(np.size(p_gt,1)):
#     p_norm_gt[i]=LA.norm(p_gt[0:3,i])
#     p_norm_m[i] =LA.norm(p_m[:,i])
#     p_norm_MH[i]=LA.norm(p_MH[0:3,i])
#     v_norm_gt[i]=LA.norm(v_gt[0:3,i])
#     v_norm_m[i] =LA.norm(v_m[:,i])
#     v_norm_MH[i]=LA.norm(v_MH[0:3,i])
#     att_norm_gt[i]=LA.norm(att_gt[0:3,i])
#     att_norm_m[i] =LA.norm(att_m[:,i])
#     att_norm_MH[i]=LA.norm(att_MH[0:3,i])
#     w_norm_gt[i]=LA.norm(w_gt[0:3,i])
#     w_norm_m[i] =LA.norm(w_m[:,i])
#     w_norm_MH[i]=LA.norm(w_MH[0:3,i])
# rmse_p = format(mean_squared_error(p_norm_MH, p_norm_gt, squared=False),'.4f')
# rmse_v = format(mean_squared_error(v_norm_MH, v_norm_gt, squared=False),'.4f')
# rmse_att = format(mean_squared_error(att_norm_MH, att_norm_gt, squared=False),'.4f')
# rmse_w = format(mean_squared_error(w_norm_MH, w_norm_gt, squared=False),'.4f')
# rmse_pm = format(mean_squared_error(p_norm_m, p_norm_gt, squared=False),'.4f')
# rmse_vm = format(mean_squared_error(v_norm_m, v_norm_gt, squared=False),'.4f')
# rmse_attm = format(mean_squared_error(att_norm_m, att_norm_gt, squared=False),'.4f')
# rmse_wm = format(mean_squared_error(w_norm_m, w_norm_gt, squared=False),'.4f')


rmse_mhx = format(mean_squared_error(p_gt[0,:], p_MH[0,:], squared=False),'.4f')
rmse_mhy = format(mean_squared_error(p_gt[1,:], p_MH[1,:], squared=False),'.4f')
rmse_mhz = format(mean_squared_error(p_gt[2,:], p_MH[2,:], squared=False),'.4f')
rmse_mhvx = format(mean_squared_error(v_gt[0,:], v_MH[0,:], squared=False),'.4f')
rmse_mhvy = format(mean_squared_error(v_gt[1,:], v_MH[1,:], squared=False),'.4f')
rmse_mhvz = format(mean_squared_error(v_gt[2,:], v_MH[2,:], squared=False),'.4f')
rmse_mhroll= format(mean_squared_error(att_gt[0,:], att_MH[0,:], squared=False),'.4f')
rmse_mhpit = format(mean_squared_error(att_gt[1,:], att_MH[1,:], squared=False),'.4f')
rmse_mhyaw = format(mean_squared_error(att_gt[2,:], att_MH[2,:], squared=False),'.4f')
rmse_mhwx = format(mean_squared_error(w_gt[0,:], w_MH[0,:], squared=False),'.4f')
rmse_mhwy = format(mean_squared_error(w_gt[1,:], w_MH[1,:], squared=False),'.4f')
rmse_mhwz = format(mean_squared_error(w_gt[2,:], w_MH[2,:], squared=False),'.4f')
# print('rmse_p=',rmse_p,'rmse_v=',rmse_v,'rmse_att=',rmse_att,'rmse_w=',rmse_w)
# print('rmse_pm=',rmse_pm,'rmse_vm=',rmse_vm,'rmse_attm=',rmse_attm,'rmse_wm=',rmse_wm)
# print('rmse_mhx=',rmse_mhx,'rmse_mhy=',rmse_mhy,'rmse_mhz=',rmse_mhz)
# print('rmse_mhvx=',rmse_mhvx,'rmse_mhvy=',rmse_mhvy,'rmse_mhvz=',rmse_mhvz)
# print('rmse_mhroll=',rmse_mhroll,'rmse_mhpit=',rmse_mhpit,'rmse_mhyaw=',rmse_mhyaw)
# print('rmse_mhwx=',rmse_mhwx,'rmse_mhwy=',rmse_mhwy,'rmse_mhwz=',rmse_mhwz)

# position in evaluation
p_gt_neuromhe = np.load('evaluation_results/p_gt_neuromhe.npy')
p_gt_dmhe     = np.load('evaluation_results/p_gt_dmhe.npy')
p_gt_l1       = np.load('evaluation_results/p_gt_l1.npy')
p_gt_l1_active = np.load('evaluation_results/p_gt_l1_active.npy')
p_gt_baseline = np.load('evaluation_results/p_gt_baseline.npy')

# load Rmse data over 100 episodes
rmse_fxy_neuromhe = np.load('RMSE_boxplot_data/Rmse_fxy_a.npy')
rmse_fz_neuromhe  = np.load('RMSE_boxplot_data/Rmse_fz_a.npy')
rmse_txy_neuromhe = np.load('RMSE_boxplot_data/Rmse_txy_a.npy')
rmse_tz_neuromhe  = np.load('RMSE_boxplot_data/Rmse_tz_a.npy')
rmse_pxy_neuromhe = np.load('RMSE_boxplot_data/Rmse_pxy_a.npy')
rmse_pz_neuromhe  = np.load('RMSE_boxplot_data/Rmse_pz_a.npy')
rmse_fxy_dmhe = np.load('RMSE_boxplot_data/Rmse_fxy_b.npy')
rmse_fz_dmhe  = np.load('RMSE_boxplot_data/Rmse_fz_b.npy')
rmse_txy_dmhe = np.load('RMSE_boxplot_data/Rmse_txy_b.npy')
rmse_tz_dmhe  = np.load('RMSE_boxplot_data/Rmse_tz_b.npy')
rmse_pxy_dmhe = np.load('RMSE_boxplot_data/Rmse_pxy_b.npy')
rmse_pz_dmhe  = np.load('RMSE_boxplot_data/Rmse_pz_b.npy')
rmse_fxy_l1 = np.load('RMSE_boxplot_data/Rmse_fxy_c.npy')
rmse_fz_l1  = np.load('RMSE_boxplot_data/Rmse_fz_c.npy')
rmse_txy_l1 = np.load('RMSE_boxplot_data/Rmse_txy_c.npy')
rmse_tz_l1  = np.load('RMSE_boxplot_data/Rmse_tz_c.npy')
rmse_pxy_l1 = np.load('RMSE_boxplot_data/Rmse_pxy_c.npy')
rmse_pz_l1  = np.load('RMSE_boxplot_data/Rmse_pz_c.npy')
rmse_fxy_l1active = np.load('RMSE_boxplot_data/Rmse_fxy_d.npy')
rmse_fz_l1active  = np.load('RMSE_boxplot_data/Rmse_fz_d.npy')
rmse_txy_l1active = np.load('RMSE_boxplot_data/Rmse_txy_d.npy')
rmse_tz_l1active  = np.load('RMSE_boxplot_data/Rmse_tz_d.npy')
rmse_pxy_l1active = np.load('RMSE_boxplot_data/Rmse_pxy_d.npy')
rmse_pz_l1active  = np.load('RMSE_boxplot_data/Rmse_pz_d.npy')
Rmse_fxy_neuromhe = np.zeros(len(rmse_fxy_neuromhe))
Rmse_fz_neuromhe  = np.zeros(len(rmse_fxy_neuromhe))
Rmse_txy_neuromhe = np.zeros(len(rmse_fxy_neuromhe))
Rmse_tz_neuromhe  = np.zeros(len(rmse_fxy_neuromhe))
Rmse_pxy_neuromhe = np.zeros(len(rmse_pxy_neuromhe))
Rmse_pz_neuromhe  = np.zeros(len(rmse_pz_neuromhe))
Rmse_fxy_dmhe = np.zeros(len(rmse_fxy_neuromhe))
Rmse_fz_dmhe  = np.zeros(len(rmse_fxy_neuromhe))
Rmse_txy_dmhe = np.zeros(len(rmse_fxy_neuromhe))
Rmse_tz_dmhe  = np.zeros(len(rmse_fxy_neuromhe))
Rmse_pxy_dmhe = np.zeros(len(rmse_pxy_neuromhe))
Rmse_pz_dmhe  = np.zeros(len(rmse_pz_neuromhe))
Rmse_fxy_l1 = np.zeros(len(rmse_fxy_neuromhe))
Rmse_fz_l1  = np.zeros(len(rmse_fxy_neuromhe))
Rmse_txy_l1 = np.zeros(len(rmse_fxy_neuromhe))
Rmse_tz_l1  = np.zeros(len(rmse_fxy_neuromhe))
Rmse_pxy_l1 = np.zeros(len(rmse_pxy_neuromhe))
Rmse_pz_l1  = np.zeros(len(rmse_pz_neuromhe))
Rmse_fxy_l1active = np.zeros(len(rmse_fxy_neuromhe))
Rmse_fz_l1active  = np.zeros(len(rmse_fxy_neuromhe))
Rmse_txy_l1active = np.zeros(len(rmse_fxy_neuromhe))
Rmse_tz_l1active  = np.zeros(len(rmse_fxy_neuromhe))
Rmse_pxy_l1active = np.zeros(len(rmse_pxy_neuromhe))
Rmse_pz_l1active  = np.zeros(len(rmse_pz_neuromhe))
for i in range(len(rmse_fxy_neuromhe)):
    Rmse_fxy_neuromhe[i] = rmse_fxy_neuromhe[i]
    Rmse_fz_neuromhe[i]  = rmse_fz_neuromhe[i]
    Rmse_txy_neuromhe[i] = rmse_txy_neuromhe[i]
    Rmse_tz_neuromhe[i]  = rmse_tz_neuromhe[i]
    Rmse_pxy_neuromhe[i] = rmse_pxy_neuromhe[i]
    Rmse_pz_neuromhe[i]  = rmse_pz_neuromhe[i]
    Rmse_fxy_dmhe[i] = rmse_fxy_dmhe[i]
    Rmse_fz_dmhe[i]  = rmse_fz_dmhe[i]
    Rmse_txy_dmhe[i] = rmse_txy_dmhe[i]
    Rmse_tz_dmhe[i]  = rmse_tz_dmhe[i]
    Rmse_pxy_dmhe[i] = rmse_pxy_dmhe[i]
    Rmse_pz_dmhe[i]  = rmse_pz_dmhe[i]
    Rmse_fxy_l1[i] = rmse_fxy_l1[i]
    Rmse_fz_l1[i]  = rmse_fz_l1[i]
    Rmse_txy_l1[i] = rmse_txy_l1[i]
    Rmse_tz_l1[i]  = rmse_tz_l1[i]
    Rmse_pxy_l1[i] = rmse_pxy_l1[i]
    Rmse_pz_l1[i]  = rmse_pz_l1[i]
    Rmse_fxy_l1active[i] = rmse_fxy_l1active[i]
    Rmse_fz_l1active[i]  = rmse_fz_l1active[i]
    Rmse_txy_l1active[i] = rmse_txy_l1active[i]
    Rmse_tz_l1active[i]  = rmse_tz_l1active[i]
    Rmse_pxy_l1active[i] = rmse_pxy_l1active[i]
    Rmse_pz_l1active[i]  = rmse_pz_l1active[i]


mean_fxy_neuromhe = format(mean(Rmse_fxy_neuromhe),'.3f')
mean_fz_neuromhe  = format(mean(Rmse_fz_neuromhe),'.3f')
mean_txy_neuromhe = format(mean(Rmse_txy_neuromhe),'.3f')
mean_tz_neuromhe  = format(mean(Rmse_tz_neuromhe),'.3f')
mean_pxy_neuromhe = format(mean(Rmse_pxy_neuromhe),'.3f')
mean_pz_neuromhe  = format(mean(Rmse_pz_neuromhe),'.3f')
mean_fxy_dmhe = format(mean(Rmse_fxy_dmhe),'.3f')
mean_fz_dmhe  = format(mean(Rmse_fz_dmhe),'.3f')
mean_txy_dmhe = format(mean(Rmse_txy_dmhe),'.3f')
mean_tz_dmhe  = format(mean(Rmse_tz_dmhe),'.3f')
mean_pxy_dmhe = format(mean(Rmse_pxy_dmhe),'.3f')
mean_pz_dmhe  = format(mean(Rmse_pz_dmhe),'.3f')
mean_fxy_l1 = format(mean(Rmse_fxy_l1),'.3f')
mean_fz_l1  = format(mean(Rmse_fz_l1),'.3f')
mean_txy_l1 = format(mean(Rmse_txy_l1),'.3f')
mean_tz_l1  = format(mean(Rmse_tz_l1),'.3f')
mean_pxy_l1 = format(mean(Rmse_pxy_l1),'.3f')
mean_pz_l1  = format(mean(Rmse_pz_l1),'.3f')
mean_fxy_l1active = format(mean(Rmse_fxy_l1active),'.3f')
mean_fz_l1active  = format(mean(Rmse_fz_l1active),'.3f')
mean_txy_l1active = format(mean(Rmse_txy_l1active),'.3f')
mean_tz_l1active  = format(mean(Rmse_tz_l1active),'.3f')
mean_pxy_l1active = format(mean(Rmse_pxy_l1active),'.3f')
mean_pz_l1active  = format(mean(Rmse_pz_l1active),'.3f')


# print('mean_fxy_neuromhe=',mean_fxy_neuromhe,'mean_fz_neuromhe=',mean_fz_neuromhe,'mean_txy_neuromhe=',mean_txy_neuromhe,'mean_tz_neuromhe=',mean_tz_neuromhe,'mean_pxy_neuromhe=',mean_pxy_neuromhe,'mean_pz_neuromhe=',mean_pz_neuromhe)
# print('mean_fxy_dmhe=',mean_fxy_dmhe,'mean_fz_dmhe=',mean_fz_dmhe,'mean_txy_dmhe=',mean_txy_dmhe,'mean_tz_dmhe=',mean_tz_dmhe,'mean_pxy_dmhe=',mean_pxy_dmhe,'mean_pz_dmhe=',mean_pz_dmhe)
# print('mean_fxy_l1active=',mean_fxy_l1active,'mean_fz_l1active=',mean_fz_l1active,'mean_txy_l1active=',mean_txy_l1active,'mean_tz_l1active=',mean_tz_l1active,'mean_pxy_l1active=',mean_pxy_l1active,'mean_pz_l1active=',mean_pz_l1active)
# print('mean_fxy_l1=',mean_fxy_l1,'mean_fz_l1=',mean_fz_l1,'mean_txy_l1=',mean_txy_l1,'mean_tz_l1=',mean_tz_l1,'mean_pxy_l1=',mean_pxy_l1,'mean_pz_l1=',mean_pz_l1)

median_fxy_neuromhe = format(median(Rmse_fxy_neuromhe),'.3f')
median_fz_neuromhe  = format(median(Rmse_fz_neuromhe),'.3f')
median_txy_neuromhe = format(median(Rmse_txy_neuromhe),'.3f')
median_tz_neuromhe  = format(median(Rmse_tz_neuromhe),'.3f')
median_pxy_neuromhe = format(median(Rmse_pxy_neuromhe),'.3f')
median_pz_neuromhe  = format(median(Rmse_pz_neuromhe),'.3f')
median_fxy_dmhe = format(median(Rmse_fxy_dmhe),'.3f')
median_fz_dmhe  = format(median(Rmse_fz_dmhe),'.3f')
median_txy_dmhe = format(median(Rmse_txy_dmhe),'.3f')
median_tz_dmhe  = format(median(Rmse_tz_dmhe),'.3f')
median_pxy_dmhe = format(median(Rmse_pxy_dmhe),'.3f')
median_pz_dmhe  = format(median(Rmse_pz_dmhe),'.3f')
median_fxy_l1 = format(median(Rmse_fxy_l1),'.3f')
median_fz_l1  = format(median(Rmse_fz_l1),'.3f')
median_txy_l1 = format(median(Rmse_txy_l1),'.3f')
median_tz_l1  = format(median(Rmse_tz_l1),'.3f')
median_pxy_l1 = format(median(Rmse_pxy_l1),'.3f')
median_pz_l1  = format(median(Rmse_pz_l1),'.3f')
median_fxy_l1active = format(median(Rmse_fxy_l1active),'.3f')
median_fz_l1active  = format(median(Rmse_fz_l1active),'.3f')
median_txy_l1active = format(median(Rmse_txy_l1active),'.3f')
median_tz_l1active  = format(median(Rmse_tz_l1active),'.3f')
median_pxy_l1active = format(median(Rmse_pxy_l1active),'.3f')
median_pz_l1active  = format(median(Rmse_pz_l1active),'.3f')


# print('median_fxy_neuromhe=',median_fxy_neuromhe,'median_fz_neuromhe=',median_fz_neuromhe,'median_txy_neuromhe=',median_txy_neuromhe,'median_tz_neuromhe=',median_tz_neuromhe,'median_pxy_neuromhe=',median_pxy_neuromhe,'median_pz_neuromhe=',median_pz_neuromhe)
# print('median_fxy_dmhe=',median_fxy_dmhe,'median_fz_dmhe=',median_fz_dmhe,'median_txy_dmhe=',median_txy_dmhe,'median_tz_dmhe=',median_tz_dmhe,'median_pxy_dmhe=',median_pxy_dmhe,'median_pz_dmhe=',median_pz_dmhe)
# print('median_fxy_l1active=',median_fxy_l1active,'median_fz_l1active=',median_fz_l1active,'median_txy_l1active=',median_txy_l1active,'median_tz_l1active=',median_tz_l1active,'median_pxy_l1active=',median_pxy_l1active,'median_pz_l1active=',median_pz_l1active)
# print('median_fxy_l1=',median_fxy_l1,'median_fz_l1=',median_fz_l1,'median_txy_l1=',median_txy_l1,'median_tz_l1=',median_tz_l1,'median_pxy_l1=',median_pxy_l1,'median_pz_l1=',median_pz_l1)


# inverse of noise covariance
Cov_inv_f = np.load('trained_data/cov_f_training.npy')
Cov_inv_f_e = np.load('evaluation_results/cov_f_evaluation.npy')
inverse_fx = Cov_inv_f[0,:]
inverse_fy = Cov_inv_f[1,:]
inverse_fz = Cov_inv_f[2,:]
inverse_fx_e = Cov_inv_f_e[0,:]
inverse_fy_e = Cov_inv_f_e[1,:]
inverse_fz_e = Cov_inv_f_e[2,:]

# loss and training episode
loss_neuromhe = np.load('trained_data/Loss.npy')
loss_dmhe     = np.load('trained_data/Loss_dmhe.npy')
k_iter_neuromhe = np.load('trained_data/K_iteration.npy')
k_iter_dmhe     = np.load('trained_data/K_iteration_dmhe.npy')
"""------------Plot figures---------------"""
font1 = {'family':'arial',
         'weight':'normal',
         'style':'normal', 'size':6}
font2 = {'family':'arial',
         'weight':'normal',
         'style':'normal', 'size':6}
cm_2_inch = 2.54
# class ScalarFormatterClass(ScalarFormatter):
#     def _set_format(self):
#         self.format = "%1.2f"
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


# 3D trajectory tracking figure
plt.figure(1, figsize=(10/cm_2_inch,10*0.7/cm_2_inch),dpi=600)
ax = plt.axes(projection="3d")
ax.xaxis._axinfo["grid"].update({"linewidth":0.5})
ax.yaxis._axinfo["grid"].update({"linewidth":0.5})
ax.zaxis._axinfo["grid"].update({"linewidth":0.5})
ax.xaxis._axinfo["grid"].update({"linestyle":'--'})
ax.yaxis._axinfo["grid"].update({"linestyle":'--'})
ax.zaxis._axinfo["grid"].update({"linestyle":'--'})
state_ut = FULLSTATE[:,0*1500:1*1500]
ax.plot3D(ref_x, ref_y, ref_z, linewidth=1.5, color='black')
ax.plot3D(act_x_0, act_y_0, act_z_0, linewidth=0.5,color='red')
ax.plot3D(act_x_1, act_y_1, act_z_1, linewidth=0.5,color='blue')
ax.plot3D(act_x_2, act_y_2, act_z_2, linewidth=0.5,color='orange')
leg=plt.legend(['Reference', 'Untrained','Trained: 1st episode','Trained: 2nd episode'],loc='upper center',prop=font2, bbox_to_anchor=(0.5, 0.98),labelspacing=0.15,ncol = 2,columnspacing=0.5,borderpad=0.3,handletextpad=0.4,handlelength=1.5)
leg.get_frame().set_linewidth(0.5)
position = uav.get_quadrotor_position(wing_len,state_ut)
for i in range(4):
    c_x, c_y, c_z = position[0:3,i*250+400]
    r1_x, r1_y, r1_z = position[3:6,i*250+400]
    r2_x, r2_y, r2_z = position[6:9,i*250+400]
    r3_x, r3_y, r3_z = position[9:12,i*250+400]
    r4_x, r4_y, r4_z = position[12:15,i*250+400]
    ax.plot([c_x, r1_x], [c_y, r1_y], [c_z, r1_z], linewidth=0.5, color='black', marker='o', markersize=1)
    ax.plot([c_x, r2_x], [c_y, r2_y], [c_z, r2_z], linewidth=0.5, color='black', marker='o', markersize=1)
    ax.plot([c_x, r3_x], [c_y, r3_y], [c_z, r3_z], linewidth=0.5, color='black', marker='o', markersize=1)
    ax.plot([c_x, r4_x], [c_y, r4_y], [c_z, r4_z], linewidth=0.5, color='black', marker='o', markersize=1)


ax.tick_params(axis='x',which='major',pad=-5, length=1)
ax.set_xlabel('x [m]', labelpad=-9, **font1)
ax.set_xlim(-3,3)
ax.set_xticks([-4,-2,0,2,4])
ax.tick_params(axis='y',which='major',pad=-5, length=1)
ax.set_ylabel('y [m]', labelpad=-10, **font1)
ax.set_ylim(-6,6)
ax.set_yticks([-6,-3,0,3,6])
ax.tick_params(axis='z',which='major',pad=-5, length=1)
ax.set_zlabel('z [m]', labelpad=-12, **font1)
ax.set_zlim(0,7)
ax.set_zticks([0,2,4,6])
for t in ax.xaxis.get_major_ticks(): 
    t.label.set_font('arial') 
    t.label.set_fontsize(6)
for t in ax.yaxis.get_major_ticks(): 
    t.label.set_font('arial')
    t.label.set_fontsize(6)
for t in ax.zaxis.get_major_ticks(): 
    t.label.set_font('arial')
    t.label.set_fontsize(6)
for axis in [ax.w_xaxis, ax.w_yaxis, ax.w_zaxis]:
    axis.line.set_linewidth(0.5)
ax.view_init(20,-160)
plt.savefig('plots_in_paper/3d_trajectory.png',bbox_inches="tight", pad_inches=0,dpi=600)
plt.show()




# comparison of mean loss in training
plt.figure(3, figsize=(8/cm_2_inch, 8*0.4/cm_2_inch), dpi=600)
ax = plt.gca()
plt.plot(k_iter_neuromhe, loss_neuromhe, linewidth=1, marker='o', markersize=2)
plt.plot(k_iter_dmhe, loss_dmhe, linewidth=1, marker='o', markersize=2)
plt.xlabel('Training episode', labelpad=0, **font1)
plt.xticks(np.arange(0,3.1,1), **font1)
plt.ylabel('Mean loss',labelpad=0, **font1)
plt.yticks(np.arange(0,710,100),**font1)
plt.yscale("log") 
ax.tick_params(axis='x',which='major',pad=0, length=1)
ax.tick_params(axis='y',which='major',pad=0, length=1)
leg=ax.legend(['NeuroMHE', 'DMHE'],loc='lower left',prop=font1)
leg.get_frame().set_linewidth(0.5)
for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(0.5)
for t in ax.xaxis.get_major_ticks(): 
    t.label.set_font('arial') 
    t.label.set_fontsize(6)
for t in ax.yaxis.get_major_ticks(): 
    t.label.set_font('arial')
    t.label.set_fontsize(6)
plt.grid(b=True,axis='both',linestyle='--', linewidth=0.5)
plt.savefig('plots_in_paper/training_meanloss.png',bbox_inches="tight", pad_inches=0.01,dpi=600)
plt.show()

#------------force and torque estimation in training---------------#
plt.figure(4, figsize=(4.25/cm_2_inch,4.25*0.7/cm_2_inch), dpi=600)
ax = plt.gca()
plt.plot(Time, Disf_xy, linewidth=1.5,color='black')
plt.plot(Time, dfxy_train_0, linewidth=0.5,color='red')
plt.plot(Time, dfxy_train_1, linewidth=0.5,color='blue')
plt.plot(Time, dfxy_train_2, linewidth=0.5,color='orange')
plt.xlabel('Time [s]', labelpad=0, **font1)
plt.xticks(np.arange(0,16,5),**font1)
plt.ylabel('$d_{f_{xy}}$ [N]', labelpad=0, **font1)
plt.yticks(np.arange(0,16,5),**font1)
ax.set_ylim(0,18)
ax.tick_params(axis='x',which='major',pad=0,length=1)
ax.tick_params(axis='y',which='major',pad=0,length=1)
for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(0.5)
# ax.yaxis.set_major_formatter(OOMFormatter(-2,"%1.1f"))
for t in ax.xaxis.get_major_ticks(): 
    t.label.set_font('arial') 
    t.label.set_fontsize(6)
for t in ax.yaxis.get_major_ticks(): 
    t.label.set_font('arial')
    t.label.set_fontsize(6)
leg=plt.legend(['Ground truth', 'Untrained','Trained: 1st episode','Trained: 2nd episode'],loc='upper center',prop=font2, bbox_to_anchor=(0.42, 1.25),labelspacing=0.15,ncol = 2,columnspacing=0.3,borderpad=0.3,handletextpad=0.3,handlelength=0.8)
leg.get_frame().set_linewidth(0.5)
plt.grid(b=True,axis='both',linestyle='--', linewidth=0.5)
plt.savefig('plots_in_paper/force_xy_training_neuromhe.png',bbox_inches="tight", pad_inches=0.01,dpi=600)
plt.show()

plt.figure(5, figsize=(4.25/cm_2_inch,4.25*0.7/cm_2_inch), dpi=600)
ax = plt.gca()
plt.plot(Time, Gt[2,:], linewidth=1.5,color='black')
plt.plot(Time, dfz_train_0, linewidth=0.5,color='red')
plt.plot(Time, dfz_train_1, linewidth=0.5,color='blue')
plt.plot(Time, dfz_train_2, linewidth=0.5,color='orange')
plt.xlabel('Time [s]', labelpad=0, **font1)
plt.xticks(np.arange(0,16,5),**font1)
plt.ylabel('$d_{f_{z}}$ [N]', labelpad=0, **font1)
plt.yticks(np.arange(-24,1,8),**font1)
ax.tick_params(axis='x',which='major',pad=0,length=1)
ax.tick_params(axis='y',which='major',pad=0,length=1)
for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(0.5)
for t in ax.xaxis.get_major_ticks(): 
    t.label.set_font('arial') 
    t.label.set_fontsize(6)
for t in ax.yaxis.get_major_ticks(): 
    t.label.set_font('arial')
    t.label.set_fontsize(6)
plt.grid(b=True,axis='both',linestyle='--', linewidth=0.5)
plt.savefig('plots_in_paper/force_z_training_neuromhe.png',bbox_inches="tight", pad_inches=0.01,dpi=600)
plt.show()

plt.figure(6, figsize=(4.25/cm_2_inch,4.25*0.7/cm_2_inch), dpi=600)
ax = plt.gca()
plt.plot(Time, Dist_xy, linewidth=1.5,color='black')
plt.plot(Time, dtxy_train_0, linewidth=0.5,color='red')
plt.plot(Time, dtxy_train_1, linewidth=0.5,color='blue')
plt.plot(Time, dtxy_train_2, linewidth=0.5,color='orange')
plt.xlabel('Time [s]', labelpad=0, **font1)
plt.xticks(np.arange(0,16,5),**font1)
plt.ylabel('$d_{\\tau_{xy}}$ [Nm]', labelpad=0, **font1)
plt.yticks(np.arange(0,0.1,0.03),**font1)
ax.set_ylim(0,0.093)
ax.tick_params(axis='x',which='major',pad=0,length=1)
ax.tick_params(axis='y',which='major',pad=0,length=1)
for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(0.5)
for t in ax.xaxis.get_major_ticks(): 
    t.label.set_font('arial') 
    t.label.set_fontsize(6)
for t in ax.yaxis.get_major_ticks(): 
    t.label.set_font('arial')
    t.label.set_fontsize(6)
ax.yaxis.set_major_formatter(OOMFormatter(-1,"%1.1f"))
ax.yaxis.get_offset_text().set_fontsize(5)
plt.grid(b=True,axis='both',linestyle='--', linewidth=0.5)
plt.savefig('plots_in_paper/torque_xy_training_neuromhe.png',bbox_inches="tight", pad_inches=0.01,dpi=600)
plt.show()

plt.figure(7, figsize=(4.25/cm_2_inch,4.25*0.7/cm_2_inch), dpi=600)
ax = plt.gca()
plt.plot(Time, Gt_tau[2,:], linewidth=1.5,color='black')
plt.plot(Time, dtz_train_0, linewidth=0.5,color='red')
plt.plot(Time, dtz_train_1, linewidth=0.5,color='blue')
plt.plot(Time, dtz_train_2, linewidth=0.5,color='orange')
plt.xlabel('Time [s]', labelpad=0, **font1)
plt.xticks(np.arange(0,16,5),**font1)
plt.ylabel('$d_{\\tau_{z}}$ [Nm]', labelpad=0, **font1)
plt.yticks(np.arange(-0.12,0.01,0.04),**font1)
ax.tick_params(axis='x',which='major',pad=0,length=1)
ax.tick_params(axis='y',which='major',pad=0,length=1)
for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(0.5)
for t in ax.xaxis.get_major_ticks(): 
    t.label.set_font('arial') 
    t.label.set_fontsize(6)
for t in ax.yaxis.get_major_ticks(): 
    t.label.set_font('arial')
    t.label.set_fontsize(6)
ax.yaxis.set_major_formatter(OOMFormatter(-1,"%1.1f"))
ax.yaxis.get_offset_text().set_fontsize(5)
plt.grid(b=True,axis='both',linestyle='--', linewidth=0.5)
plt.savefig('plots_in_paper/torque_z_training_neuromhe.png',bbox_inches="tight", pad_inches=0.01,dpi=600)
plt.show()


# inverse of noise covariance
plt.figure(8, figsize=(4.25/cm_2_inch,4.25*0.7/cm_2_inch), dpi=600)
ax = plt.gca()
plt.plot(Time_e, inverse_fx_e**2, linewidth=0.5)
plt.plot(Time_e, inverse_fy_e**2, linewidth=0.5)
plt.plot(Time_e, inverse_fz_e**2, linewidth=0.5)
plt.xlabel('Time [s]', labelpad=0, **font1)
plt.xticks(np.arange(0,11,3.5),**font1)
plt.ylabel('Inverse of covariance',labelpad=0, **font1)
plt.yticks(np.arange(0,1.1,0.5),**font1)
leg=ax.legend(['$\sigma^{-2}_{f_x}$', '$\sigma^{-2}_{f_y}$','$\sigma^{-2}_{f_z}$'],loc='upper center',prop=font1,labelspacing=0.15,borderpad=0.5,handletextpad=0.5,handlelength=1.5)
leg.get_frame().set_linewidth(0.5)
ax.tick_params(axis='x',which='major',pad=0,length=1)
ax.tick_params(axis='y',which='major',pad=0,length=1)
for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(0.5)
for t in ax.yaxis.get_major_ticks(): 
    t.label.set_font('arial') 
    t.label.set_fontsize(6)
for t in ax.yaxis.get_major_ticks(): 
    t.label.set_font('arial') 
    t.label.set_fontsize(6)
# ax.yaxis.set_major_formatter(OOMFormatter(-2,"%1.1f"))
plt.grid(b=True,axis='both',linestyle='--', linewidth=0.5)
plt.savefig('plots_in_paper/inverse_force_evaluation.png',bbox_inches="tight", pad_inches=0.01,dpi=600)
plt.show()

# forgetting factor

plt.figure(9, figsize=(4.25/cm_2_inch,4.25*0.7/cm_2_inch), dpi=600)
ax = plt.gca()
plt.plot(Time_e, weight_para_e[24,:], linewidth=0.5)
plt.xlabel('Time [s]', labelpad=0, **font1)
plt.xticks(np.arange(0,11,3.5),**font1)
plt.ylabel('Forgetting factor $\gamma_1$',labelpad=0, **font1)
plt.yticks(np.arange(0.8,1.02,0.1),**font1)
# leg = ax.legend(fontsize=6)
ax.tick_params(axis='x',which='major',pad=0,length=1)
ax.tick_params(axis='y',which='major',pad=0,length=1)
for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(0.5)
plt.grid(b=True,axis='both',linestyle='--', linewidth=0.5)
for t in ax.yaxis.get_major_ticks(): 
    t.label.set_font('arial') 
    t.label.set_fontsize(6)
for t in ax.yaxis.get_major_ticks(): 
    t.label.set_font('arial') 
    t.label.set_fontsize(6)
# ax.yaxis.set_major_formatter(OOMFormatter(-2,"%1.1f"))
# ax.yaxis.get_offset_text().set_fontsize(6)
plt.savefig('plots_in_paper/gamma1_fig_evaluation.png',bbox_inches="tight", pad_inches=0.01,dpi=600)
plt.show()


plt.figure(10, figsize=(4.25/cm_2_inch,4.25*0.7/cm_2_inch), dpi=600)
ax = plt.gca()
plt.plot(Time_e, weight_para_e[42,:], linewidth=0.5)
plt.xlabel('Time [s]', labelpad=0, **font1)
plt.xticks(np.arange(0,11,3.5),**font1)
plt.ylabel('Forgetting factor $\gamma_2$',labelpad=0, **font1)
plt.yticks(np.arange(0,0.25,0.1),**font1)
# leg = ax.legend(fontsize=6)
ax.tick_params(axis='x',which='major',pad=0,length=1)
ax.tick_params(axis='y',which='major',pad=0,length=1)
for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(0.5)
plt.grid(b=True,axis='both',linestyle='--', linewidth=0.5)
for t in ax.yaxis.get_major_ticks(): 
    t.label.set_font('arial') 
    t.label.set_fontsize(6)
for t in ax.yaxis.get_major_ticks(): 
    t.label.set_font('arial') 
    t.label.set_fontsize(6)
# ax.yaxis.set_major_formatter(OOMFormatter(-2,"%1.1f"))
# ax.yaxis.get_offset_text().set_fontsize(6)
plt.savefig('plots_in_paper/gamma2_fig_evaluation.png',bbox_inches="tight", pad_inches=0.01,dpi=600)
plt.show()

plt.figure(11, figsize=(4.25/cm_2_inch,4.25*0.7/cm_2_inch), dpi=600)
ax = plt.gca()
plt.plot(Time_e, Q_k_e[0,:], linewidth=0.5)
plt.plot(Time_e, Q_k_e[1,:], linewidth=0.5)
plt.plot(Time_e, Q_k_e[2,:], linewidth=0.5)
plt.xlabel('Time [s]', labelpad=0, **font1)
plt.xticks(np.arange(0,11,3.5),**font1)
plt.ylabel('Discounted $Q_{t-1}$',labelpad=0, **font1)
plt.yticks(np.arange(0,0.12,0.05),**font1)
leg=ax.legend(['$\gamma_2 q_1$', '$\gamma_2 q_2$','$\gamma_2 q_3$'],loc='upper center',prop=font1,labelspacing=0.15,borderpad=0.5,handletextpad=0.5,handlelength=1.5)
leg.get_frame().set_linewidth(0.5)
ax.tick_params(axis='x',which='major',pad=0,length=1)
ax.tick_params(axis='y',which='major',pad=0,length=1)
for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(0.5)
plt.grid(b=True,axis='both',linestyle='--', linewidth=0.5)
ax.yaxis.set_major_formatter(OOMFormatter(-1,"%1.1f"))
ax.yaxis.get_offset_text().set_fontsize(5)
for t in ax.yaxis.get_major_ticks(): 
    t.label.set_font('arial') 
    t.label.set_fontsize(6)
for t in ax.yaxis.get_major_ticks(): 
    t.label.set_font('arial') 
    t.label.set_fontsize(6)
plt.savefig('plots_in_paper/discountedQ_force_evaluation.png',bbox_inches="tight", pad_inches=0.01,dpi=600)
plt.show()

plt.figure(12, figsize=(7.9/cm_2_inch,5/cm_2_inch), dpi=600)
ax = plt.gca()
plt.plot(ref_y_e,-ref_x_e,  linewidth=3,color='black')
plt.plot(p_gt_neuromhe[1,:], -p_gt_neuromhe[0,:], linewidth=1.5,color='orange')
plt.plot(p_gt_dmhe[1,:], -p_gt_dmhe[0,:], linewidth=0.75,color='blue')
plt.plot(p_gt_l1[1,:], -p_gt_l1[0,:], linewidth=0.5,linestyle='--',color='#F97306')
plt.plot(p_gt_l1_active[1,:], -p_gt_l1_active[0,:], linewidth=0.5,linestyle='--',color='green')
plt.plot(p_gt_baseline[1,:], -p_gt_baseline[0,:], linewidth=0.5,color='red')
plt.xlabel('y [m]', labelpad=0, **font1)
plt.ylabel('x [m]', labelpad=0, **font1)
plt.xticks(np.arange(-4,6,2),**font1)
plt.yticks(np.arange(-4,6,2),**font1)
leg=ax.legend(['Reference', 'NeuroMHE','DMHE','$\mathcal{L}_1$-AC','Active $\mathcal{L}_1$-AC','Baseline'],loc='upper center',prop=font2, bbox_to_anchor=(0.5, 1.2),labelspacing=0.15,ncol = 3,columnspacing=0.3,borderpad=0.5,handletextpad=0.5,handlelength=1.5)
# leg=ax.legend(['Desired', 'DMHE_old','DMHE_new'],loc='upper left',prop=font1)
leg.get_frame().set_linewidth(0.5)
ax.tick_params(axis='x',which='major',pad=0,length=1)
ax.tick_params(axis='y',which='major',pad=0,length=1)
for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(0.5)
for t in ax.yaxis.get_major_ticks(): 
    t.label.set_font('arial') 
    t.label.set_fontsize(6)
for t in ax.yaxis.get_major_ticks(): 
    t.label.set_font('arial') 
    t.label.set_fontsize(6)
plt.grid(b=True,axis='both',linestyle='--', linewidth=0.5)
plt.savefig('plots_in_paper/planar_evaluation.png',bbox_inches="tight", pad_inches=0.01,dpi=600)
plt.show()

plt.figure(13, figsize=(8.1/cm_2_inch,2.5/cm_2_inch), dpi=600)
ax = plt.gca()
plt.plot(Time_e, p_gt_neuromhe[2,:]-ref_z_e, linewidth=1.5,color='orange')
plt.plot(Time_e, p_gt_dmhe[2,:]-ref_z_e, linewidth=0.75,color='blue')
plt.plot(Time_e, p_gt_l1[2,:]-ref_z_e, linewidth=1,linestyle='--',color='#F97306')
plt.plot(Time_e, p_gt_l1_active[2,:]-ref_z_e, linewidth=1,linestyle='--',color='green')
plt.plot(Time_e, p_gt_baseline[2,:]-ref_z_e, linewidth=1,color='red')
plt.xlabel('Time [s]', labelpad=0, **font1)
plt.ylabel('Tracking error $e_z$ [m]', labelpad=0, **font1)
plt.xticks(np.arange(0,11,3.5),**font1)
plt.yticks(np.arange(-0.4,0.21,0.2),**font1)
# leg=ax.legend(['Desired', 'NeuroMHE','DMHE','L1-AC','Active L1-AC','Baseline'],loc='upper left',prop=font1)
# leg.get_frame().set_linewidth(0.5)
ax.tick_params(axis='x',which='major',pad=0,length=1)
ax.tick_params(axis='y',which='major',pad=0,length=1)
for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(0.5)
for t in ax.yaxis.get_major_ticks(): 
    t.label.set_font('arial') 
    t.label.set_fontsize(6)
for t in ax.yaxis.get_major_ticks(): 
    t.label.set_font('arial') 
    t.label.set_fontsize(6)
# ax.yaxis.set_major_formatter(OOMFormatter(-1,"%1.1f"))
# ax.yaxis.get_offset_text().set_fontsize(6)
plt.grid(b=True,axis='both',linestyle='--', linewidth=0.5)
plt.savefig('plots_in_paper/z_evaluation.png',bbox_inches="tight", pad_inches=0.01,dpi=600)
plt.show()

# plot the quadrotor state estimation against the ground truth state
fig, (ax1,ax2,ax3) = plt.subplots(3,sharex=True,figsize=(4.25/cm_2_inch,5/cm_2_inch),dpi=600)
ax1.plot(Time,p_gt[0,:], linewidth=1.5,color='black')
ax1.plot(Time,p_MH[0,:], linewidth=0.5,color='orange')
ax1.tick_params(axis='x',which='major',pad=0,length=1)
ax1.tick_params(axis='y',which='major',pad=0,length=1)
ax1.set_yticks(np.arange(-3,3.1,3))
ax2.plot(Time,p_gt[1,:], linewidth=1.5,color='black')
ax2.plot(Time,p_MH[1,:], linewidth=0.5,color='orange')
ax2.tick_params(axis='x',which='major',pad=0,length=1)
ax2.tick_params(axis='y',which='major',pad=0,length=1)
ax3.plot(Time,p_gt[2,:], linewidth=1.5,color='black')
ax3.plot(Time,p_MH[2,:], linewidth=0.5,color='orange')
ax3.tick_params(axis='x',which='major',pad=0,length=1)
ax3.tick_params(axis='y',which='major',pad=0,length=1)
ax3.set_xticks(np.arange(0,16,5))
ax1.set_ylabel('$p_{x}$ [m]',labelpad=0,**font1)
ax2.set_ylabel('$p_{y}$ [m]',labelpad=0,**font1)
ax3.set_ylabel('$p_{z}$ [m]',labelpad=0,**font1)
ax3.set_xlabel('Time [s]',labelpad=0,**font1)
for t in ax1.yaxis.get_major_ticks(): 
    t.label.set_font('arial') 
    t.label.set_fontsize(6)
for t in ax2.yaxis.get_major_ticks(): 
    t.label.set_font('arial') 
    t.label.set_fontsize(6)
for t in ax3.xaxis.get_major_ticks(): 
    t.label.set_font('arial') 
    t.label.set_fontsize(6)
for t in ax3.yaxis.get_major_ticks(): 
    t.label.set_font('arial') 
    t.label.set_fontsize(6)
leg=ax1.legend(['Ground truth', 'Estimation'],loc='upper center',prop=font2, bbox_to_anchor=(0.5, 1.4),labelspacing=0.15,ncol = 2,columnspacing=0.3,borderpad=0.3,handletextpad=0.3,handlelength=1.25)
leg.get_frame().set_linewidth(0.5)
ax1.grid(b=True,axis='both',linestyle='--', linewidth=0.5)
ax2.grid(b=True,axis='both',linestyle='--', linewidth=0.5)
ax3.grid(b=True,axis='both',linestyle='--', linewidth=0.5)
for axis in ['top', 'bottom', 'left', 'right']:
    ax1.spines[axis].set_linewidth(0.5)
for axis in ['top', 'bottom', 'left', 'right']:
    ax2.spines[axis].set_linewidth(0.5)
for axis in ['top', 'bottom', 'left', 'right']:
    ax3.spines[axis].set_linewidth(0.5)
plt.savefig('plots_in_paper/position_neuromhe.png',bbox_inches="tight", pad_inches=0.01,dpi=600)
plt.show()

fig, (ax1,ax2,ax3) = plt.subplots(3,sharex=True,figsize=(4.25/cm_2_inch,5/cm_2_inch),dpi=600)
ax1.plot(Time,v_gt[0,:], linewidth=1.5,color='black')
ax1.plot(Time,v_MH[0,:], linewidth=0.5,color='orange')
ax1.tick_params(axis='x',which='major',pad=0,length=1)
ax1.tick_params(axis='y',which='major',pad=0,length=1)
ax1.set_yticks(np.arange(-3,3.1,3))
ax2.plot(Time,v_gt[1,:], linewidth=1.5,color='black')
ax2.plot(Time,v_MH[1,:], linewidth=0.5,color='orange')
ax2.tick_params(axis='x',which='major',pad=0,length=1)
ax2.tick_params(axis='y',which='major',pad=0,length=1)
ax2.set_yticks(np.arange(-3,3.1,3))
ax3.plot(Time,v_gt[2,:], linewidth=1.5,color='black')
ax3.plot(Time,v_MH[2,:], linewidth=0.5,color='orange')
ax3.tick_params(axis='x',which='major',pad=0,length=1)
ax3.tick_params(axis='y',which='major',pad=0,length=1)
ax3.set_xticks(np.arange(0,16,5))
ax3.set_yticks(np.arange(-3,3.1,3))
ax1.set_ylabel('$v_{x}$ [m/s]',labelpad=0,**font1)
ax2.set_ylabel('$v_{y}$ [m/s]',labelpad=0,**font1)
ax3.set_ylabel('$v_{z}$ [m/s]',labelpad=0,**font1)
ax3.set_xlabel('Time [s]',labelpad=0,**font1)
for t in ax1.yaxis.get_major_ticks(): 
    t.label.set_font('arial') 
    t.label.set_fontsize(6)
for t in ax2.yaxis.get_major_ticks(): 
    t.label.set_font('arial') 
    t.label.set_fontsize(6)
for t in ax3.xaxis.get_major_ticks(): 
    t.label.set_font('arial') 
    t.label.set_fontsize(6)
for t in ax3.yaxis.get_major_ticks(): 
    t.label.set_font('arial') 
    t.label.set_fontsize(6)
# leg=ax2.legend(['Ground truth', 'Estimated'],loc='lower left',prop=font1,labelspacing=0.1)
# leg.get_frame().set_linewidth(0.5)
ax1.grid(b=True,axis='both',linestyle='--', linewidth=0.5)
ax2.grid(b=True,axis='both',linestyle='--', linewidth=0.5)
ax3.grid(b=True,axis='both',linestyle='--', linewidth=0.5)
for axis in ['top', 'bottom', 'left', 'right']:
    ax1.spines[axis].set_linewidth(0.5)
for axis in ['top', 'bottom', 'left', 'right']:
    ax2.spines[axis].set_linewidth(0.5)
for axis in ['top', 'bottom', 'left', 'right']:
    ax3.spines[axis].set_linewidth(0.5)
plt.savefig('plots_in_paper/velocity_neuromhe.png',bbox_inches="tight", pad_inches=0.01,dpi=600)
plt.show()

fig, (ax1,ax2,ax3) = plt.subplots(3,sharex=True,figsize=(4.25/cm_2_inch,5/cm_2_inch),dpi=600)
ax1.plot(Time,att_gt[0,:], linewidth=1.5,color='black')
ax1.plot(Time,att_MH[0,:], linewidth=0.5,color='orange')
ax1.tick_params(axis='x',which='major',pad=0,length=1)
ax1.tick_params(axis='y',which='major',pad=0,length=1)
ax1.set_yticks(np.arange(-0.6,0.1,0.6))
ax2.plot(Time,att_gt[1,:], linewidth=1.5,color='black')
ax2.plot(Time,att_MH[1,:], linewidth=0.5,color='orange')
ax2.tick_params(axis='x',which='major',pad=0,length=1)
ax2.tick_params(axis='y',which='major',pad=0,length=1)
ax3.plot(Time,att_gt[2,:], linewidth=1.5,color='black')
ax3.plot(Time,att_MH[2,:], linewidth=0.5,color='orange')
ax3.tick_params(axis='x',which='major',pad=0,length=1)
ax3.tick_params(axis='y',which='major',pad=0,length=1)
ax3.yaxis.set_major_formatter(OOMFormatter(-2,"%1.1f"))
ax3.yaxis.get_offset_text().set_fontsize(5)
ax3.yaxis.get_offset_text().set_verticalalignment('center')
ax3.set_xticks(np.arange(0,16,5))
ax1.set_ylabel('Roll [rad]',labelpad=0,**font1)
ax2.set_ylabel('Pitch [rad]',labelpad=0,**font1)
ax3.set_ylabel('Yaw [rad]',labelpad=0,**font1)
ax3.set_xlabel('Time [s]',labelpad=0,**font1)
for t in ax1.yaxis.get_major_ticks(): 
    t.label.set_font('arial') 
    t.label.set_fontsize(6)
for t in ax2.yaxis.get_major_ticks(): 
    t.label.set_font('arial') 
    t.label.set_fontsize(6)
for t in ax3.xaxis.get_major_ticks(): 
    t.label.set_font('arial') 
    t.label.set_fontsize(6)
for t in ax3.yaxis.get_major_ticks(): 
    t.label.set_font('arial') 
    t.label.set_fontsize(6)
ax1.grid(b=True,axis='both',linestyle='--', linewidth=0.5)
ax2.grid(b=True,axis='both',linestyle='--', linewidth=0.5)
ax3.grid(b=True,axis='both',linestyle='--', linewidth=0.5)
for axis in ['top', 'bottom', 'left', 'right']:
    ax1.spines[axis].set_linewidth(0.5)
for axis in ['top', 'bottom', 'left', 'right']:
    ax2.spines[axis].set_linewidth(0.5)
for axis in ['top', 'bottom', 'left', 'right']:
    ax3.spines[axis].set_linewidth(0.5)
plt.savefig('plots_in_paper/euler_neuromhe.png',bbox_inches="tight", pad_inches=0.01,dpi=600)
plt.show()

fig, (ax1,ax2,ax3) = plt.subplots(3,sharex=True,figsize=(4.25/cm_2_inch,5/cm_2_inch),dpi=600)
ax1.plot(Time,w_gt[0,:], linewidth=1.5,color='black')
ax1.plot(Time,w_MH[0,:], linewidth=0.5,color='orange')
ax1.tick_params(axis='x',which='major',pad=0,length=1)
ax1.tick_params(axis='y',which='major',pad=0,length=1)
ax1.set_yticks(np.arange(-2,2.1,2))
ax2.plot(Time,w_gt[1,:], linewidth=1.5,color='black')
ax2.plot(Time,w_MH[1,:], linewidth=0.5,color='orange')
ax2.tick_params(axis='x',which='major',pad=0,length=1)
ax2.tick_params(axis='y',which='major',pad=0,length=1)
ax2.set_yticks(np.arange(-2,2.1,2))
ax3.plot(Time,w_gt[2,:], linewidth=1.5,color='black')
ax3.plot(Time,w_MH[2,:], linewidth=0.5,color='orange')
ax3.tick_params(axis='x',which='major',pad=0,length=1)
ax3.tick_params(axis='y',which='major',pad=0,length=1)
ax3.set_yticks(np.arange(-0.7,0.61,0.7))
ax3.set_xticks(np.arange(0,16,5))
ax1.set_ylabel('$\omega_{x}$ [rad/s]',labelpad=0,**font1)
ax2.set_ylabel('$\omega_{y}$ [rad/s]',labelpad=0,**font1)
ax3.set_ylabel('$\omega_{z}$ [rad/s]',labelpad=0,**font1)
ax3.set_xlabel('Time [s]',labelpad=0,**font1)
for t in ax1.yaxis.get_major_ticks(): 
    t.label.set_font('arial') 
    t.label.set_fontsize(6)
for t in ax2.yaxis.get_major_ticks(): 
    t.label.set_font('arial') 
    t.label.set_fontsize(6)
for t in ax3.xaxis.get_major_ticks(): 
    t.label.set_font('arial') 
    t.label.set_fontsize(6)
for t in ax3.yaxis.get_major_ticks(): 
    t.label.set_font('arial') 
    t.label.set_fontsize(6)
# leg=ax2.legend(['Ground truth', 'Estimated'],loc='lower left',prop=font1,labelspacing=0.1)
# leg.get_frame().set_linewidth(0.5)
ax1.grid(b=True,axis='both',linestyle='--', linewidth=0.5)
ax2.grid(b=True,axis='both',linestyle='--', linewidth=0.5)
ax3.grid(b=True,axis='both',linestyle='--', linewidth=0.5)
for axis in ['top', 'bottom', 'left', 'right']:
    ax1.spines[axis].set_linewidth(0.5)
for axis in ['top', 'bottom', 'left', 'right']:
    ax2.spines[axis].set_linewidth(0.5)
for axis in ['top', 'bottom', 'left', 'right']:
    ax3.spines[axis].set_linewidth(0.5)
plt.savefig('plots_in_paper/angular_rate_neuromhe.png',bbox_inches="tight", pad_inches=0.01,dpi=600)
plt.show()

plt.figure(14, figsize=(4.25/cm_2_inch,4.25*0.7/cm_2_inch), dpi=600)
ax = plt.gca()
plt.plot(Time_e, Gt_e[3,:], linewidth=1,color='black')
plt.plot(Time_e, df_MH[3,:], linewidth=0.5,color='orange')
plt.plot(Time_e, df_MH_dmhe[3,:], linewidth=0.5,color='blue')
plt.plot(Time_e, df_MH_l1[3,:],linewidth=0.5,linestyle='--', color='#F97306') # #F97306
plt.plot(Time_e, df_MH_l1active[3,:], linewidth=0.5,linestyle='--',color='green')
plt.xlabel('Time [s]', labelpad=0, **font1)
plt.xticks(np.arange(0,11,3.5),**font1)
plt.ylabel('$d_{f_{xy}}$ [N]',labelpad=0, **font1)
plt.yticks(np.arange(0,16,5),**font1)
leg=ax.legend(['Ground truth', 'NeuroMHE','DMHE','$\mathcal{L}_1$-AC','Active $\mathcal{L}_1$-AC'],loc='upper center',prop=font2, bbox_to_anchor=(0.5, 1.375),labelspacing=0.15,ncol = 2,columnspacing=0.3,borderpad=0.3,handletextpad=0.3,handlelength=1.2)
# leg=ax.legend(['Ground truth', 'NeuroMHE','DMHE_old','DMHE_new'],loc='lower center',prop=font1,labelspacing=0.1)
leg.get_frame().set_linewidth(0.5)
ax.tick_params(axis='x',which='major',pad=0,length=1)
ax.tick_params(axis='y',which='major',pad=0,length=1)
for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(0.5)
plt.grid(b=True,axis='both',linestyle='--', linewidth=0.5)
for t in ax.yaxis.get_major_ticks(): 
    t.label.set_font('arial') 
    t.label.set_fontsize(6)
for t in ax.yaxis.get_major_ticks(): 
    t.label.set_font('arial') 
    t.label.set_fontsize(6)
# ax.yaxis.set_major_formatter(OOMFormatter(-2,"%1.1f"))
# ax.yaxis.get_offset_text().set_fontsize(6)
plt.savefig('plots_in_paper/force_xy_evaluation.png',bbox_inches="tight", pad_inches=0.01,dpi=600)
plt.show()

plt.figure(15, figsize=(4.25/cm_2_inch,4.25*0.7/cm_2_inch), dpi=600)
ax = plt.gca()
plt.plot(Time_e, Gt_e[2,:], linewidth=1,color='black')
plt.plot(Time_e, df_MH[2,:], linewidth=0.5,color='orange')
plt.plot(Time_e, df_MH_dmhe[2,:], linewidth=0.5,color='blue')
plt.plot(Time_e, df_MH_l1[2,:],linewidth=0.5,linestyle='--',color='#F97306')
plt.plot(Time_e, df_MH_l1active[2,:], linewidth=0.5,linestyle='--',color='green')
plt.xlabel('Time [s]', labelpad=0, **font1)
plt.xticks(np.arange(0,11,3.5),**font1)
plt.ylabel('$d_{f_{z}}$ [N]',labelpad=0, **font1)
plt.yticks(np.arange(-15,0.1,5),**font1)
# leg=ax.legend(['Ground truth', 'NeuroMHE','DMHE_old','DMHE_new'],loc='upper center',prop=font1,labelspacing=0.1)
# leg.get_frame().set_linewidth(0.5)
ax.tick_params(axis='x',which='major',pad=0,length=1)
ax.tick_params(axis='y',which='major',pad=0,length=1)
for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(0.5)
plt.grid(b=True,axis='both',linestyle='--', linewidth=0.5)
# ax.yaxis.set_major_formatter(OOMFormatter(-2,"%1.1f"))
# ax.yaxis.get_offset_text().set_fontsize(6)
for t in ax.yaxis.get_major_ticks(): 
    t.label.set_font('arial') 
    t.label.set_fontsize(6)
for t in ax.yaxis.get_major_ticks(): 
    t.label.set_font('arial') 
    t.label.set_fontsize(6)
plt.savefig('plots_in_paper/force_z_evaluation.png',bbox_inches="tight", pad_inches=0.01,dpi=600)
plt.show()

plt.figure(16, figsize=(4.25/cm_2_inch,4.25*0.7/cm_2_inch), dpi=600)
ax = plt.gca()
plt.plot(Time_e, Gt_tau_e[3,:], linewidth=1,color='black')
plt.plot(Time_e, dtau_MH[3,:], linewidth=0.5,color='orange')
plt.plot(Time_e, dtau_MH_dmhe[3,:], linewidth=0.5,color='blue')
plt.plot(Time_e, dtau_MH_l1[3,:],linewidth=0.5,linestyle='--',color='#F97306')
plt.plot(Time_e, dtau_MH_l1active[3,:], linewidth=0.5,linestyle='--',color='green')
plt.xlabel('Time [s]', labelpad=0, **font1)
plt.xticks(np.arange(0,11,3.5),**font1)
plt.ylabel('$d_{\\tau_{xy}}$ [Nm]',labelpad=0, **font1)
plt.yticks(np.arange(0,0.16,0.05),**font1)
# leg=ax.legend(['Ground truth', 'NeuroMHE','DMHE','Active L1-AC'],loc='lower center',prop=font1,labelspacing=0.1)
# leg.get_frame().set_linewidth(0.5)
ax.tick_params(axis='x',which='major',pad=0,length=1)
ax.tick_params(axis='y',which='major',pad=0,length=1)
for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(0.5)
plt.grid(b=True,axis='both',linestyle='--', linewidth=0.5)
ax.yaxis.set_major_formatter(OOMFormatter(-1,"%1.1f"))
ax.yaxis.get_offset_text().set_fontsize(5)
for t in ax.yaxis.get_major_ticks(): 
    t.label.set_font('arial') 
    t.label.set_fontsize(6)
for t in ax.yaxis.get_major_ticks(): 
    t.label.set_font('arial') 
    t.label.set_fontsize(6)
plt.savefig('plots_in_paper/torque_xy_evaluation.png',bbox_inches="tight", pad_inches=0.01,dpi=600)
plt.show()


plt.figure(17, figsize=(4.25/cm_2_inch,4.25*0.7/cm_2_inch), dpi=600)
ax = plt.gca()
plt.plot(Time_e, Gt_tau_e[2,:], linewidth=1,color='black')
plt.plot(Time_e, dtau_MH[2,:], linewidth=0.5,color='orange')
plt.plot(Time_e, dtau_MH_dmhe[2,:], linewidth=0.5,color='blue')
plt.plot(Time_e, dtau_MH_l1[2,:],linewidth=0.5,linestyle='--',color='#F97306')
plt.plot(Time_e, dtau_MH_l1active[2,:], linewidth=0.5,linestyle='--',color='green')
plt.xlabel('Time [s]', labelpad=0, **font1)
plt.xticks(np.arange(0,11,3.5),**font1)
plt.ylabel('$d_{\\tau_{z}}$ [Nm]',labelpad=0, **font1)
plt.yticks(np.arange(-0.08,0.06,0.04),**font1)
# leg=ax.legend(['Ground truth', 'NeuroMHE','DMHE','Active L1-AC'],loc='lower center',prop=font1,labelspacing=0.1)
# leg.get_frame().set_linewidth(0.5)
ax.tick_params(axis='x',which='major',pad=0,length=1)
ax.tick_params(axis='y',which='major',pad=0,length=1)
for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(0.5)
plt.grid(b=True,axis='both',linestyle='--', linewidth=0.5)
for t in ax.yaxis.get_major_ticks(): 
    t.label.set_font('arial') 
    t.label.set_fontsize(6)
for t in ax.yaxis.get_major_ticks(): 
    t.label.set_font('arial') 
    t.label.set_fontsize(6)
ax.yaxis.set_major_formatter(OOMFormatter(-1,"%1.1f"))
ax.yaxis.get_offset_text().set_fontsize(5)
plt.savefig('plots_in_paper/torque_z_evaluation.png',bbox_inches="tight", pad_inches=0.01,dpi=600)
plt.show()


# -------------boxplot over 100 episode----------------#

# dataset
neuromhe_f = pd.DataFrame(
    {'$d_{f_{xy}}$':Rmse_fxy_neuromhe,
     '$d_{f_{z}}$':Rmse_fz_neuromhe}
)
dmhe_f = pd.DataFrame(
    {'$d_{f_{xy}}$':Rmse_fxy_dmhe,
     '$d_{f_{z}}$': Rmse_fz_dmhe}
)
l1_f   = pd.DataFrame(
    {'$d_{f_{xy}}$': Rmse_fxy_l1,
     '$d_{f_{z}}$': Rmse_fz_l1}
)
l1active_f = pd.DataFrame(
    {'$d_{f_{xy}}$': Rmse_fxy_l1active,
     '$d_{f_{z}}$': Rmse_fz_l1active}
)
neuromhe_t = pd.DataFrame(
    {'$d_{\\tau_{xy}}$':Rmse_txy_neuromhe,
     '$d_{\\tau_{z}}$':Rmse_tz_neuromhe}
)
dmhe_t = pd.DataFrame(
    {'$d_{\\tau_{xy}}$':Rmse_txy_dmhe,
     '$d_{\\tau_{z}}$': Rmse_tz_dmhe}
)
l1_t = pd.DataFrame(
    {'$d_{\\tau_{xy}}$': Rmse_txy_l1,
     '$d_{\\tau_{z}}$': Rmse_tz_l1}
)
l1active_t = pd.DataFrame(
    {'$d_{\\tau_{xy}}$': Rmse_txy_l1active,
     '$d_{\\tau_{z}}$': Rmse_tz_l1active}
)
neuromhe_p = pd.DataFrame(
    {'$p_{xy}$':Rmse_pxy_neuromhe,
     '$p_{z}$':Rmse_pz_neuromhe}
)
dmhe_p = pd.DataFrame(
    {'$p_{xy}$':Rmse_pxy_dmhe,
     '$p_{z}$': Rmse_pz_dmhe}
)
l1_p = pd.DataFrame(
    {'$p_{xy}$': Rmse_pxy_l1,
     '$p_{z}$': Rmse_pz_l1}
)
l1active_p = pd.DataFrame(
    {'$p_{xy}$': Rmse_pxy_l1active,
     '$p_{z}$': Rmse_pz_l1active}
)


datasets_f = [neuromhe_f,dmhe_f,l1active_f,l1_f]
datasets_t = [neuromhe_t,dmhe_t,l1active_t,l1_t]
datasets_p = [neuromhe_p,dmhe_p,l1active_p,l1_p]

plt.figure(18, figsize=(8/cm_2_inch,2.25/cm_2_inch), dpi=600)
ax = plt.gca()
# Get the max of the dataset
all_maximums = [d.max(axis=1).values for d in datasets_f]
dataset_maximums = [max(m) for m in all_maximums]
y_max = max(dataset_maximums)
# Get the min of the dataset
all_minimums = [d.min(axis=1).values for d in datasets_f]
dataset_minimums = [min(m) for m in all_minimums]
y_min = min(dataset_minimums)
# Calculate the y-axis range
y_range = y_max - y_min
# Set x-positions for boxes
x_pos_range = np.arange(len(datasets_f)) / (len(datasets_f) - 1)
x_pos = (x_pos_range * 0.685) + 0.655
colours = ['orange','blue', 'green','#F97306']
for i, data in enumerate(datasets_f):
    positions = [x_pos[i] + j * 1 for j in range(len(data.T))]
    k = i % len(colours)
    flierprops = dict(marker='o', markersize=0.25, markerfacecolor=colours[k], markeredgecolor=colours[k],
                  linestyle='none')
    bp = plt.boxplot(
        np.array(data), widths=0.6 / len(datasets_f),
        labels=list(datasets_f[0]), patch_artist=True,
        positions=[x_pos[i] + j * 1 for j in range(len(data.T))],
        showfliers=True, flierprops=flierprops
       
    ) # whis=[5, 95]
    # Fill the boxes with colours (requires patch_artist=True)
    k = i % len(colours)
    for box in bp['boxes']:
        box.set(facecolor=colours[k])
    for element in ['boxes', 'fliers', 'means', 'medians']:
        plt.setp(bp[element], color=colours[k])
    for element in ['whiskers', 'caps']:
        plt.setp(bp[element], color=colours[k])
        plt.setp(bp[element], color=colours[k])
    plt.setp(bp['medians'], color='black')
    # Get the samples' medians
    medians = [bp['medians'][j].get_ydata()[0] for j in range(len(data.T))]
    medians = [str(round(s, 3)) for s in medians]
    # Increase the height of the plot by 5% to fit the labels
    plt.ylim([y_min - 0.05 * y_range, y_max - 0.05 * y_range])
    # Set the y-positions for the labels
    y_pos = y_min - 0.02 * y_range
    for tick, label in zip(range(len(data.T)), plt.xticks()):
        # k = tick % 2
        plt.text(
            positions[tick], y_pos, fr'{medians[tick]}',
            horizontalalignment='center', fontsize = 5
        ) #'$\tilde{d_f} =' + fr'

# Axis ticks and labels
plt.xticks(np.arange(len(list(datasets_f[0]))) + 1,**font1)
plt.yticks(**font1)
plt.ylabel('RMSE [N]',labelpad=0,**font1)
plt.yticks(np.arange(0.4,2.5,0.6),**font1)
ax.xaxis.set_minor_locator(matplotlib.ticker.FixedLocator(
    np.array(range(len(list(datasets_f[0])) + 1)) + 0.5))
ax.tick_params(axis='x', which='major', pad=0,length=1)
ax.tick_params(axis='y', which='major', pad=0,length=1)
# Legend
legend_elements = []
groups = ['NeuroMHE','DMHE','Active $\mathcal{L}_1$-AC','$\mathcal{L}_1$-AC']
for i in range(len(datasets_f)):
    j = i % len(groups)
    k = i % len(colours)
    legend_elements.append(Patch(facecolor=colours[k], label=groups[j]))
leg=plt.legend(handles=legend_elements, loc='upper center',prop=font2, bbox_to_anchor=(0.5, 1.31),labelspacing=0.15,ncol = 4,columnspacing=1,borderpad=0.5,handletextpad=0.3,handlelength=1.5)
leg.get_frame().set_linewidth(0.5)
for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(0.5)
for t in ax.yaxis.get_major_ticks(): 
    t.label.set_font('arial') 
    t.label.set_fontsize(6)
for t in ax.yaxis.get_major_ticks(): 
    t.label.set_font('arial') 
    t.label.set_fontsize(6)
plt.grid(b=True,axis='both',linestyle='--', linewidth=0.5)
plt.savefig('plots_in_paper/force_rmse_comparsion.png',bbox_inches="tight", pad_inches=0.01,dpi=600)
plt.show()


plt.figure(19, figsize=(8/cm_2_inch,2.25/cm_2_inch), dpi=600)
ax = plt.gca()
# Get the max of the dataset
all_maximums = [d.max(axis=1).values for d in datasets_t]
dataset_maximums = [max(m) for m in all_maximums]
y_max = max(dataset_maximums)
# Get the min of the dataset
all_minimums = [d.min(axis=1).values for d in datasets_t]
dataset_minimums = [min(m) for m in all_minimums]
y_min = min(dataset_minimums)
# Calculate the y-axis range
y_range = y_max - y_min
# Set x-positions for boxes
x_pos_range = np.arange(len(datasets_t)) / (len(datasets_t) - 1)
x_pos = (x_pos_range * 0.685) + 0.655
colours = ['orange','blue', 'green','#F97306']
for i, data in enumerate(datasets_t):
    positions = [x_pos[i] + j * 1 for j in range(len(data.T))]
    k = i % len(colours)
    flierprops = dict(marker='o', markersize=0.25, markerfacecolor=colours[k], markeredgecolor=colours[k],
                  linestyle='none')
    bp = plt.boxplot(
        np.array(data), widths=0.6 / len(datasets_t),
        labels=list(datasets_t[0]), patch_artist=True,
        positions=[x_pos[i] + j * 1 for j in range(len(data.T))],
        showfliers=True, flierprops=flierprops
    )
    # Fill the boxes with colours (requires patch_artist=True)
    k = i % len(colours)
    for box in bp['boxes']:
        box.set(facecolor=colours[k])
    for element in ['boxes', 'fliers', 'means', 'medians']:
        plt.setp(bp[element], color=colours[k])
    for element in ['whiskers', 'caps']:
        plt.setp(bp[element], color=colours[k])
        plt.setp(bp[element], color=colours[k])
    plt.setp(bp['medians'], color='black')
    # Get the samples' medians
    medians = [bp['medians'][j].get_ydata()[0] for j in range(len(data.T))]
    medians = [str(round(s, 3)) for s in medians]
    # Increase the height of the plot by 5% to fit the labels
    plt.ylim([y_min - 0.15 * y_range, y_max + 0 * y_range])
    # Set the y-positions for the labels
    y_pos = y_min - 0.12 * y_range
    for tick, label in zip(range(len(data.T)), plt.xticks()):
        # k = tick % 2
        plt.text(
            positions[tick], y_pos, fr'{medians[tick]}',
            horizontalalignment='center', fontsize = 5
        ) #'$\tilde{d_f} =' + fr'

# Axis ticks and labels
plt.xticks(np.arange(len(list(datasets_t[0]))) + 1,**font1)
plt.yticks(**font1)
plt.ylabel('RMSE [Nm]',labelpad=0,**font1)
plt.yticks(np.arange(0,0.07,0.02),**font1)
ax.xaxis.set_minor_locator(matplotlib.ticker.FixedLocator(
    np.array(range(len(list(datasets_t[0])) + 1)) + 0.5)
)
ax.tick_params(axis='x', which='major', pad=0,length=1)
ax.tick_params(axis='y', which='major', pad=0,length=1)
# legend_elements = []
# groups = ['NeuroMHE','DMHE','Active $\mathcal{L}_1$-AC','$\mathcal{L}_1$-AC']
# for i in range(len(datasets_f)):
#     j = i % len(groups)
#     k = i % len(colours)
#     legend_elements.append(Patch(facecolor=colours[k], label=groups[j]))
# leg=plt.legend(handles=legend_elements, prop=font1)
# leg.get_frame().set_linewidth(0.5)
for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(0.5)
plt.grid(b=True,axis='both',linestyle='--', linewidth=0.5)
ax.yaxis.set_major_formatter(OOMFormatter(-1,"%1.1f"))
ax.yaxis.get_offset_text().set_fontsize(5)
for t in ax.yaxis.get_major_ticks(): 
    t.label.set_font('arial') 
    t.label.set_fontsize(6)
for t in ax.yaxis.get_major_ticks(): 
    t.label.set_font('arial') 
    t.label.set_fontsize(6)
plt.savefig('plots_in_paper/torque_rmse_comparsion.png',bbox_inches="tight", pad_inches=0.01,dpi=600)
plt.show()

plt.figure(20, figsize=(8/cm_2_inch,2.25/cm_2_inch), dpi=600)
ax = plt.gca()
# Get the max of the dataset
all_maximums = [d.max(axis=1).values for d in datasets_p]
dataset_maximums = [max(m) for m in all_maximums]
y_max = max(dataset_maximums)
# Get the min of the dataset
all_minimums = [d.min(axis=1).values for d in datasets_p]
dataset_minimums = [min(m) for m in all_minimums]
y_min = min(dataset_minimums)
# Calculate the y-axis range
y_range = y_max - y_min
# Set x-positions for boxes
x_pos_range = np.arange(len(datasets_p)) / (len(datasets_p) - 1)
x_pos = (x_pos_range * 0.685) + 0.655
colours = ['orange','blue', 'green','#F97306']
for i, data in enumerate(datasets_p):
    positions = [x_pos[i] + j * 1 for j in range(len(data.T))]
    k = i % len(colours)
    flierprops = dict(marker='o', markersize=0.25, markerfacecolor=colours[k], markeredgecolor=colours[k],
                  linestyle='none')
    bp = plt.boxplot(
        np.array(data), whis=[5, 95], widths=0.6 / len(datasets_p),
        labels=list(datasets_p[0]), patch_artist=True,
        positions=[x_pos[i] + j * 1 for j in range(len(data.T))],
        showfliers=True, flierprops=flierprops
    )
    # Fill the boxes with colours (requires patch_artist=True)
    k = i % len(colours)
    for box in bp['boxes']:
        box.set(facecolor=colours[k])
    for element in ['boxes', 'fliers', 'means', 'medians']:
        plt.setp(bp[element], color=colours[k])
    for element in ['whiskers', 'caps']:
        plt.setp(bp[element], color=colours[k])
        plt.setp(bp[element], color=colours[k])
    plt.setp(bp['medians'], color='black')
    # Get the samples' medians
    medians = [bp['medians'][j].get_ydata()[0] for j in range(len(data.T))]
    medians = [str(round(s, 3)) for s in medians]
    # Increase the height of the plot by 5% to fit the labels
    plt.ylim([y_min - 0.1 * y_range, y_max - 0.05* y_range])
    # Set the y-positions for the labels
    y_pos = y_min - 0.08 * y_range
    for tick, label in zip(range(len(data.T)), plt.xticks()):
        # k = tick % 2
        plt.text(
            positions[tick], y_pos, fr'{medians[tick]}',
            horizontalalignment='center', fontsize = 5
        ) #'$\tilde{d_f} =' + fr'

# Axis ticks and labels
plt.xticks(np.arange(len(list(datasets_p[0]))) + 1,**font1)
plt.yticks(**font1)
plt.ylabel('RMSE [m]',labelpad=0,**font1)
plt.yticks(np.arange(0,0.25,0.08),**font1)
ax.xaxis.set_minor_locator(matplotlib.ticker.FixedLocator(
    np.array(range(len(list(datasets_p[0])) + 1)) + 0.5)
)
ax.tick_params(axis='x', which='major', pad=0,length=1)
ax.tick_params(axis='y', which='major', pad=0,length=1)
# groups = ['NeuroMHE','DMHE','Active $\mathcal{L}_1$-AC']
# for i in range(len(datasets_f)):
#     j = i % len(groups)
#     k = i % len(colours)
#     legend_elements.append(Patch(facecolor=colours[k], label=groups[j]))
# leg=plt.legend(handles=legend_elements, prop=font1)
# leg.get_frame().set_linewidth(0.5)
for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(0.5)
plt.grid(b=True,axis='both',linestyle='--', linewidth=0.5)
for t in ax.yaxis.get_major_ticks(): 
    t.label.set_font('arial') 
    t.label.set_fontsize(6)
for t in ax.yaxis.get_major_ticks(): 
    t.label.set_font('arial') 
    t.label.set_fontsize(6)
ax.yaxis.set_major_formatter(OOMFormatter(-1,"%1.1f"))
ax.yaxis.get_offset_text().set_fontsize(5)
plt.savefig('plots_in_paper/position_rmse_comparsion.png',bbox_inches="tight", pad_inches=0.01,dpi=600)
plt.show()

# play animation
# k=2
# uav.play_animation(wing_len, FULLSTATE,k, REFP, dt_sample)
# uav.play_animation_force(Time,Df_neuromhe_train, k, Groundt, dt_sample)

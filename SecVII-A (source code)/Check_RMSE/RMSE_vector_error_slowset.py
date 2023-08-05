"""
In response to the reviwers' concerns regarding the correctness of the results in Tables II and V,
this python file aspires to provide an explanation of how the results were obtained and a comparison of 
RMSEs in the total disturbance between two different computation methods (i.e., scalar error vs vector error).
=============================================================================================================
Wang, Bingheng, (wangbingheng@u.nus.edu) July 4th 2023 at NUS, Singapore
"""

import numpy as np
from sklearn.metrics import mean_squared_error


print("==========================================================================================================")
print("Please choose which trajectory to evaluate")
print("'a': 3d circle 1,        Vmean=4.6169, Vmax=9.6930, Tmean=0.0070, Tmax=0.2281, Fmean=10.3181, Fmax=18.5641")
print("'b': linear oscillation, Vmean=5.1243, Vmax=15.0917,Tmean=0.0236, Tmax=0.4653, Fmean=12.0337, Fmax=34.5543")
print("'c': lemniscate 1,       Vmean=2.5747, Vmax=6.5158, Tmean=0.0059, Tmax=0.0357, Fmean=8.1929,  Fmax=10.8099")
print("'d': race track 1,       Vmean=5.8450, Vmax=11.2642,Tmean=0.0217, Tmax=0.4807, Fmean=11.0304, Fmax=16.4014")
print("'e': race track 2,       Vmean=6.9813, Vmax=14.2610,Tmean=0.0365, Tmax=0.4669, Fmean=14.6680, Fmax=30.0297")
print("'f': 3d circle 2,        Vmean=5.9991, Vmax=11.8102,Tmean=0.0087, Tmax=0.1109, Fmean=13.7672, Fmax=28.6176")
print("'g': lemniscate 2,       Vmean=1.6838, Vmax=3.4546, Tmean=0.0043, Tmax=0.0138, Fmean=7.6071,  Fmax=7.8332")
print("'h': melon 1,            Vmean=3.4488, Vmax=6.8019, Tmean=0.0074, Tmax=0.0883, Fmean=8.8032,  Fmax=14.4970")
print("'i': lemniscate 3,       Vmean=6.7462, Vmax=13.6089,Tmean=0.0290, Tmax=0.4830, Fmean=13.9946, Fmax=24.2318")
print("'j': lemniscate 4,       Vmean=9.1848, Vmax=17.7242,Tmean=0.0483, Tmax=0.5224, Fmean=18.5576, Fmax=36.2017")
print("'k': melon 2,            Vmean=7.0817, Vmax=12.1497,Tmean=0.0124, Tmax=0.1967, Fmean=17.2943, Fmax=34.4185")
print("'l': random point,       Vmean=2.5488, Vmax=8.8238, Tmean=0.0292, Tmax=0.6211, Fmean=9.1584,  Fmax=29.0121")
print("'m': ellipse,            Vmean=9.4713, Vmax=16.5371,Tmean=0.0117, Tmax=0.0962, Fmean=19.4827, Fmax=35.0123")
key = input("enter 'a', or 'b',... without the quotation mark:")
print("==========================================================================================================")

#---load the ground truth data---#
Gt_fx = np.load('evaluation_vector_error_slowset/Gt_fx_'+str(key)+'.npy')
Gt_fy = np.load('evaluation_vector_error_slowset/Gt_fy_'+str(key)+'.npy')
Gt_fz = np.load('evaluation_vector_error_slowset/Gt_fz_'+str(key)+'.npy')
Gt_tx = np.load('evaluation_vector_error_slowset/Gt_tx_'+str(key)+'.npy')
Gt_ty = np.load('evaluation_vector_error_slowset/Gt_ty_'+str(key)+'.npy')
Gt_tz = np.load('evaluation_vector_error_slowset/Gt_tz_'+str(key)+'.npy')
Gt_fxy = np.load('evaluation_vector_error_slowset/Gt_fxy_'+str(key)+'.npy') # 2-norm of [Gt_fx, Gt_fy] for scalar error
Gt_f   = np.load('evaluation_vector_error_slowset/Gt_f_'+str(key)+'.npy') # 2-norm of [Gt_fx, Gt_fy, Gt_fz] for scalar error
Gt_txy = np.load('evaluation_vector_error_slowset/Gt_txy_'+str(key)+'.npy') # 2-norm of [Gt_tx, Gt_ty] for scalar error
Gt_t   = np.load('evaluation_vector_error_slowset/Gt_t_'+str(key)+'.npy') # 2-norm of [Gt_tx, Gt_ty, Gt_tz] for scalar error

#---load the NeuroBEM prediction data (copied from the open-source dataset)---#
Bemnn_fx  = np.load('evaluation_vector_error_slowset/Bemnn_fx_'+str(key)+'.npy')
Bemnn_fy  = np.load('evaluation_vector_error_slowset/Bemnn_fy_'+str(key)+'.npy')
Bemnn_fz  = np.load('evaluation_vector_error_slowset/Bemnn_fz_'+str(key)+'.npy')
Bemnn_tx  = np.load('evaluation_vector_error_slowset/Bemnn_tx_'+str(key)+'.npy')
Bemnn_ty  = np.load('evaluation_vector_error_slowset/Bemnn_ty_'+str(key)+'.npy')
Bemnn_tz  = np.load('evaluation_vector_error_slowset/Bemnn_tz_'+str(key)+'.npy')
Bemnn_fxy = np.load('evaluation_vector_error_slowset/Bemnn_fxy_'+str(key)+'.npy') # 2-norm of [Bemnn_fx, Bemnn_fy] for scalar error
Bemnn_f   = np.load('evaluation_vector_error_slowset/Bemnn_f_'+str(key)+'.npy') # 2-norm of [Bemnn_fx, Bemnn_fy, Bemnn_fz] for scalar error
Bemnn_txy = np.load('evaluation_vector_error_slowset/Bemnn_txy_'+str(key)+'.npy') # 2-norm of [Bemnn_tx, Bemnn_ty] for scalar error
Bemnn_t   = np.load('evaluation_vector_error_slowset/Bemnn_t_'+str(key)+'.npy') # 2-norm of [Bemnn_tx, Bemnn_ty, Bemnn_tz] for scalar error
#---load the NeuroBEM residual data of f_xy, f, tau_xy, and tau (vector error)---#
E_BEM_fxy = np.load('evaluation_vector_error_slowset/error_fxy_bem_v_'+str(key)+'.npy')
E_BEM_f   = np.load('evaluation_vector_error_slowset/error_f_bem_v_'+str(key)+'.npy')
E_BEM_txy = np.load('evaluation_vector_error_slowset/error_txy_bem_v_'+str(key)+'.npy')
E_BEM_t   = np.load('evaluation_vector_error_slowset/error_t_bem_v_'+str(key)+'.npy')

#---load the NeuroMHE prediction data---#
# a slightly better model retrained by adjusting the weight in the loss function and allowing for more training episodes
fx_mhe  = np.load('evaluation_vector_error_slowset/fx_mhe_'+str(key)+'.npy') 
fy_mhe  = np.load('evaluation_vector_error_slowset/fy_mhe_'+str(key)+'.npy')
fz_mhe  = np.load('evaluation_vector_error_slowset/fz_mhe_'+str(key)+'.npy')
tx_mhe  = np.load('evaluation_vector_error_slowset/tx_mhe_'+str(key)+'.npy')
ty_mhe  = np.load('evaluation_vector_error_slowset/ty_mhe_'+str(key)+'.npy')
tz_mhe  = np.load('evaluation_vector_error_slowset/tz_mhe_'+str(key)+'.npy')
fxy_mhe = np.load('evaluation_vector_error_slowset/fxy_mhe_'+str(key)+'.npy') # 2-norm of [fx_mhe, fy_mhe] for scalar error
f_mhe   = np.load('evaluation_vector_error_slowset/f_mhe_'+str(key)+'.npy') # 2-norm of [fx_mhe, fy_mhe, fz_mhe] for scalar error
txy_mhe = np.load('evaluation_vector_error_slowset/txy_mhe_'+str(key)+'.npy') # 2-norm of [tx_mhe, ty_mhe] for scalar error
t_mhe   = np.load('evaluation_vector_error_slowset/t_mhe_'+str(key)+'.npy') # 2-norm of [tx_mhe, ty_mhe, tz_mhe] for scalar error
#---load the NeuroMHE residual data of f_xy, f, tau_xy, and tau (vector error)---#
E_MHE_fxy = np.load('evaluation_vector_error_slowset/error_fxy_mhe_v_'+str(key)+'.npy')
E_MHE_f   = np.load('evaluation_vector_error_slowset/error_f_mhe_v_'+str(key)+'.npy')
E_MHE_txy = np.load('evaluation_vector_error_slowset/error_txy_mhe_v_'+str(key)+'.npy')
E_MHE_t   = np.load('evaluation_vector_error_slowset/error_t_mhe_v_'+str(key)+'.npy')

#--------------------------------------------------------------------------------#
#-------Two Computation Methods for the RMSEs of f_xy, f, tau_xy, and tau--------#
#--------------------------------------------------------------------------------#

#----------Scalar Error-based RMSE of f_xy, f, tau_xy, and tau----------#
# (as used in the manuscript T-RO 23-0314 v1)
# Error defined as the difference between two norms 
# i.e., error = norm of [fx_mhe, fy_mhe, fz_mhe] - norm of [Gt_fx, Gt_fy, Gt_fz]
#-----------------------------------------------------------------------#
# RMSEs of NeuroMHE
rmse_fxy   = format(mean_squared_error(fxy_mhe, Gt_fxy, squared=False),'.3f')
rmse_txy   = format(mean_squared_error(txy_mhe, Gt_txy, squared=False),'.3f')
rmse_f     = format(mean_squared_error(f_mhe, Gt_f, squared=False),'.3f')
rmse_t     = format(mean_squared_error(t_mhe, Gt_t, squared=False),'.3f')
# RMSEs of NeuroBEM
rmse_fxy_bemnn = format(mean_squared_error(Bemnn_fxy, Gt_fxy, squared=False),'.3f')
rmse_txy_bemnn = format(mean_squared_error(Bemnn_txy, Gt_txy, squared=False),'.3f')
rmse_f_bemnn   = format(mean_squared_error(Bemnn_f, Gt_f, squared=False),'.3f')
rmse_t_bemnn   = format(mean_squared_error(Bemnn_t, Gt_t, squared=False),'.3f')


#----------Vector Error-based RMSE of f_xy, f, tau_xy, and tau----------#
# (as suggested by the reviewers)
# Error defined as the 2-norm of the difference between two vectors 
# i.e., error = norm of ([fx_mhe, fy_mhe, fz_mhe] - [Gt_fx, Gt_fy, Gt_fz])
#       RMSE  = sqrt(sum of error^2/N)      
#-----------------------------------------------------------------------#
N   = len(Gt_fx)
# RMSEs of NeuroMHE
sum_fxy, sum_txy, sum_f, sum_t = 0, 0, 0, 0
for k in range(N):
    errorf_x = fx_mhe[k] - Gt_fx[k]
    errorf_y = fy_mhe[k] - Gt_fy[k]
    errorf_z = fz_mhe[k] - Gt_fz[k]
    errort_x = tx_mhe[k] - Gt_tx[k]
    errort_y = ty_mhe[k] - Gt_ty[k]
    errort_z = tz_mhe[k] - Gt_tz[k]
    sum_fxy += (errorf_x**2 + errorf_y**2)
    sum_f   += (errorf_x**2 + errorf_y**2 + errorf_z**2)
    sum_txy += (errort_x**2 + errort_y**2)
    sum_t   += (errort_x**2 + errort_y**2 + errort_z**2)
rmse_fxy_vector = format(np.sqrt(sum_fxy/N),'.3f')
rmse_f_vector   = format(np.sqrt(sum_f/N),'.3f')
rmse_txy_vector = format(np.sqrt(sum_txy/N),'.3f')
rmse_t_vector   = format(np.sqrt(sum_t/N),'.3f')
# or using the following code, the results should be the same
ground_truth = np.zeros(N) # the ground truth of the error (in the ideal case, the error should be zero)
# rmse_fxy_vector = format(mean_squared_error(E_MHE_fxy,ground_truth,squared=False),'.3f')
# rmse_f_vector   = format(mean_squared_error(E_MHE_f,ground_truth,squared=False),'.3f')
# rmse_txy_vector = format(mean_squared_error(E_MHE_txy,ground_truth,squared=False),'.3f')
# rmse_t_vector   = format(mean_squared_error(E_MHE_t,ground_truth,squared=False),'.3f')

# RMSEs of NeuroBEM
sum_fxy_bem, sum_txy_bem, sum_f_bem, sum_t_bem = 0, 0, 0, 0
for k in range(N):
    errorf_x = Bemnn_fx[k] - Gt_fx[k] 
    errorf_y = Bemnn_fy[k] - Gt_fy[k]
    errorf_z = Bemnn_fz[k] - Gt_fz[k]
    errort_x = Bemnn_tx[k] - Gt_tx[k]
    errort_y = Bemnn_ty[k] - Gt_ty[k]
    errort_z = Bemnn_tz[k] - Gt_tz[k]
    sum_fxy_bem += (errorf_x**2 + errorf_y**2)
    sum_f_bem   += (errorf_x**2 + errorf_y**2 + errorf_z**2)
    sum_txy_bem += (errort_x**2 + errort_y**2)
    sum_t_bem   += (errort_x**2 + errort_y**2 + errort_z**2)
rmse_fxy_bemnn_vector = format(np.sqrt(sum_fxy_bem/N),'.3f')
rmse_f_bemnn_vector   = format(np.sqrt(sum_f_bem/N),'.3f')
rmse_txy_bemnn_vector = format(np.sqrt(sum_txy_bem/N),'.3f')
rmse_t_bemnn_vector   = format(np.sqrt(sum_t_bem/N),'.3f')
# or using the following code, the results should be the same
# rmse_fxy_bemnn_vector = format(mean_squared_error(E_BEM_fxy,ground_truth,squared=False),'.3f')
# rmse_f_bemnn_vector   = format(mean_squared_error(E_BEM_f,ground_truth,squared=False),'.3f')
# rmse_txy_bemnn_vector = format(mean_squared_error(E_BEM_txy,ground_truth,squared=False),'.3f')
# rmse_t_bemnn_vector   = format(mean_squared_error(E_BEM_t,ground_truth,squared=False),'.3f')

#----------------------------------------------------------------------------#
#-------RMSEs of f_x, f_y, f_z, tau_x, tau_y, and tau_z in body frame--------#
#----------------------------------------------------------------------------#
# RMSEs of NeuroMHE
rmse_fx = format(mean_squared_error(fx_mhe, Gt_fx, squared=False),'.3f')
rmse_fy = format(mean_squared_error(fy_mhe, Gt_fy, squared=False),'.3f')
rmse_fz = format(mean_squared_error(fz_mhe, Gt_fz, squared=False),'.3f')
rmse_tx = format(mean_squared_error(tx_mhe, Gt_tx, squared=False),'.3f')
rmse_ty = format(mean_squared_error(ty_mhe, Gt_ty, squared=False),'.3f')
rmse_tz = format(mean_squared_error(tz_mhe, Gt_tz, squared=False),'.3f')
# RMSEs of NeuroBEM
rmse_fx_bemnn  = format(mean_squared_error(Bemnn_fx, Gt_fx, squared=False),'.3f')
rmse_fy_bemnn  = format(mean_squared_error(Bemnn_fy, Gt_fy, squared=False),'.3f')
rmse_fz_bemnn  = format(mean_squared_error(Bemnn_fz, Gt_fz, squared=False),'.3f')
rmse_tx_bemnn  = format(mean_squared_error(Bemnn_tx, Gt_tx, squared=False),'.3f')
rmse_ty_bemnn  = format(mean_squared_error(Bemnn_ty, Gt_ty, squared=False),'.3f')
rmse_tz_bemnn  = format(mean_squared_error(Bemnn_tz, Gt_tz, squared=False),'.3f')

# print('---------------------------------RMSEs with scalar error (body frame and m=0.772kg)-----------------------------')
# print('----------------------------------------------------------------------------------------------------------------')
# print('Traj. | Method |   f_x   |   f_y   |   f_z   |   t_x   |   t_y   |   t_z   |   f_xy   |   t_xy   |   f   |   t')
# print('----------------------------------------------------------------------------------------------------------------')
# print(key, ' | NeuroBEM | ', rmse_fx_bemnn, ' | ', rmse_fy_bemnn,' | ',rmse_fz_bemnn, ' | ', rmse_tx_bemnn, ' | ', rmse_ty_bemnn, ' | ', rmse_tz_bemnn, ' | ', rmse_fxy_bemnn, ' | ', rmse_txy_bemnn, ' | ', rmse_f_bemnn, ' | ', rmse_t_bemnn)
# print(key, ' | NeuroMHE | ', rmse_fx, ' | ', rmse_fy, ' | ', rmse_fz, ' | ', rmse_tx, ' | ', rmse_ty, ' | ', rmse_tz, ' | ', rmse_fxy, ' | ', rmse_txy, ' | ', rmse_f, ' | ', rmse_t)
# print('----------------------------------------------------------------------------------------------------------------')
# print("================================================================================================================")
print('---------------------------------RMSEs with vector error (body frame and m=0.772kg)-----------------------------')
print('----------------------------------------------------------------------------------------------------------------')
print('Traj. | Method |   f_x   |   f_y   |   f_z   |   t_x   |   t_y   |   t_z   |   f_xy   |   t_xy   |   f   |   t')
print('----------------------------------------------------------------------------------------------------------------')
print(key, ' | NeuroBEM | ', rmse_fx_bemnn, ' | ', rmse_fy_bemnn,' | ',rmse_fz_bemnn, ' | ', rmse_tx_bemnn, ' | ', rmse_ty_bemnn, ' | ', rmse_tz_bemnn, ' | ', rmse_fxy_bemnn_vector, ' | ', rmse_txy_bemnn_vector, ' | ', rmse_f_bemnn_vector, ' | ', rmse_t_bemnn_vector)
print(key, ' | NeuroMHE | ', rmse_fx, ' | ', rmse_fy, ' | ', rmse_fz, ' | ', rmse_tx, ' | ', rmse_ty, ' | ', rmse_tz, ' | ', rmse_fxy_vector, ' | ', rmse_txy_vector, ' | ', rmse_f_vector, ' | ', rmse_t_vector)
print('----------------------------------------------------------------------------------------------------------------')
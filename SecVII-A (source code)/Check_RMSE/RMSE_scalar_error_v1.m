%MATLAB code for re-producing the RMSE results presented in our previous manuscript (T-RO 23-0314 v1)
clear

disp('==========================================================================================================')
disp('Please select the trajectory for evaluation')
disp("'a': 3d circle 1,        Vmean=4.6169, Vmax=9.6930, Tmean=0.0070, Tmax=0.2281, Fmean=10.3181, Fmax=18.5641 ")
disp("'b': linear oscillation, Vmean=5.1243, Vmax=15.0917,Tmean=0.0236, Tmax=0.4653, Fmean=12.0337, Fmax=34.5543")
disp("'c': lemniscate 1,       Vmean=2.5747, Vmax=6.5158, Tmean=0.0059, Tmax=0.0357, Fmean=8.1929,  Fmax=10.8099")
disp("'d': race track 1,       Vmean=5.8450, Vmax=11.2642,Tmean=0.0217, Tmax=0.4807, Fmean=11.0304, Fmax=16.4014")
disp("'e': race track 2,       Vmean=6.9813, Vmax=14.2610,Tmean=0.0365, Tmax=0.4669, Fmean=14.6680, Fmax=30.0297")
disp("'f': 3d circle 2,        Vmean=5.9991, Vmax=11.8102,Tmean=0.0087, Tmax=0.1109, Fmean=13.7672, Fmax=28.6176")
disp("'g': lemniscate 2,       Vmean=1.6838, Vmax=3.4546, Tmean=0.0043, Tmax=0.0138, Fmean=7.6071,  Fmax=7.8332")
disp("'h': melon 1,            Vmean=3.4488, Vmax=6.8019, Tmean=0.0074, Tmax=0.0883, Fmean=8.8032,  Fmax=14.4970")
disp("'i': lemniscate 3,       Vmean=6.7462, Vmax=13.6089,Tmean=0.0290, Tmax=0.4830, Fmean=13.9946, Fmax=24.2318")
disp("'j': lemniscate 4,       Vmean=9.1848, Vmax=17.7242,Tmean=0.0483, Tmax=0.5224, Fmean=18.5576, Fmax=36.2017")
disp("'k': melon 2,            Vmean=7.0817, Vmax=12.1497,Tmean=0.0124, Tmax=0.1967, Fmean=17.2943, Fmax=34.4185")
disp("'l': random point,       Vmean=2.5488, Vmax=8.8238, Tmean=0.0292, Tmax=0.6211, Fmean=9.1584,  Fmax=29.0121")
disp("'m': ellipse,            Vmean=9.4713, Vmax=16.5371,Tmean=0.0117, Tmax=0.0962, Fmean=19.4827, Fmax=35.0123")
key    = input("enter 'a', or 'b',... without the quotation mark:",'s');
disp('==========================================================================================================')
%---load the NeuroMHE estimation data (the same model used in T-RO 23-0314 v1)---%
filename=['evaluation_scalar_error_v1/disest_mhe_',key,'.csv'];
disest_mhe = readmatrix(filename);
%---load the NeuroBEM prediction data---%
if key=='a'
    bemnn = importdata('MATLAB_code_for_mass_verification/bemnn20210218134423seg2.mat'); % the .mat file is imported from the corresponding .csv file from the open-sourced dataset
elseif key=='b'
    bemnn = importdata('MATLAB_code_for_mass_verification/bemnn20210218165335seg2.mat');
elseif key=='c'
    bemnn = importdata('MATLAB_code_for_mass_verification/bemnn20210218170320seg2.mat');
elseif key=='d'
    bemnn = importdata('MATLAB_code_for_mass_verification/bemnn20210218171908seg2.mat');
elseif key=='e'
    bemnn = importdata('MATLAB_code_for_mass_verification/bemnn20210218172600seg1.mat');
elseif key=='f'
    bemnn = importdata('MATLAB_code_for_mass_verification/bemnn20210218180845seg1.mat');
elseif key=='g'
    bemnn = importdata('MATLAB_code_for_mass_verification/bemnn20210223104803seg2.mat');
elseif key=='h'
    bemnn = importdata('MATLAB_code_for_mass_verification/bemnn20210223114138seg3.mat');
elseif key=='i'
    bemnn = importdata('MATLAB_code_for_mass_verification/bemnn20210223142148seg3.mat');
elseif key=='j'
    bemnn = importdata('MATLAB_code_for_mass_verification/bemnn20210223172724seg2.mat');
elseif key=='k'
    bemnn = importdata('MATLAB_code_for_mass_verification/bemnn20210223194506seg2.mat');
elseif key=='l'
    bemnn = importdata('MATLAB_code_for_mass_verification/bemnn20210223222625seg2.mat');
else
    bemnn = importdata('MATLAB_code_for_mass_verification/bemnn20210223225417seg1.mat');
end

%------------------------------------------------------------------------%
%RMSE Computation (Metrics: f in world frame, scalar-error formula, m=0.772kg, as used in T-RO 23-0314 v1)
%------------------------------------------------------------------------%

m_updated        = 0.772; %as updated in the NeuroBEM's website https://rpg.ifi.uzh.ch/neuro_bem/Readme.html
J                = diag([0.0025,0.0021,0.0043]);
[r,c]            = size(bemnn);
ground_truth_f   = zeros(r,5); % f_x,f_y,f_z,f_xy,f_total for scalar-error formula
ground_truth_t   = zeros(r,5); % t_x,t_y,t_z,t_xy,t_total for scalar-error formula
bemnn_f_I        = zeros(r,5); % f_bemnnx,f_bemnny,f_bemnnz,f_bemnnxy,f_bemnntotal for scalar-error formula
bemnn_t_B        = zeros(r,5); % t_bemnnx,t_bemnny,t_bemnnz,t_bemnnxy,t_bemnntotal for scalar-error formula
mhe_est_fscalar  = zeros(r,2); % f_mhe_xy, f_mhe_total
mhe_est_tscalar  = zeros(r,2); % t_mhe_xy, t_mhe_total
for i=1:1:r
    acc_p                 = bemnn(i,12:14).';
    acc_w                 = bemnn(i,2:4).';
    w                     = bemnn(i,5:7).';
    [f_updated,tau]       = ground_truth(w,acc_p,acc_w,m_updated,J);
    q                     = bemnn(i,8:11);
    R                     = q_2_rotation_unit(q);
    f_updated_I           = R*f_updated; %no need of this transformation in other three files where f is expressed in body frame
    ground_truth_f(i,1:3) = f_updated_I.';
    ground_truth_f(i,4)   = norm(f_updated_I(1:2,1)); % planar force sqrt(f_x^2+f_y^2) for scalar-error
    ground_truth_f(i,5)   = norm(f_updated_I(1:3,1));
    ground_truth_t(i,1:3) = tau.';
    ground_truth_t(i,4)   = norm(tau(1:2,1));
    ground_truth_t(i,5)   = norm(tau(1:3,1));
    f_bemnn_B             = bemnn(i,30:32);
    f_bemnn_I             = R*f_bemnn_B.';
    bemnn_f_I(i,1:3)      = f_bemnn_I.';
    bemnn_f_I(i,4)        = norm(f_bemnn_I(1:2,1));
    bemnn_f_I(i,5)        = norm(f_bemnn_I(1:3,1));
    bemnn_t_B(i,1:3)      = bemnn(i,33:35);
    bemnn_t_B(i,4)        = norm(bemnn(i,33:34));
    bemnn_t_B(i,5)        = norm(bemnn(i,33:35));
    mhe_est_fscalar(i,1)  = norm(disest_mhe(i,1:2));
    mhe_est_fscalar(i,2)  = norm(disest_mhe(i,1:3));
    mhe_est_tscalar(i,1)  = norm(disest_mhe(i,4:5));
    mhe_est_tscalar(i,2)  = norm(disest_mhe(i,4:6));
end

%RMSE of NeuroBEM
rmse_fx_bemnn  = round(sqrt(mean((ground_truth_f(:,1)-bemnn_f_I(:,1)).^2)),3);
rmse_fy_bemnn  = round(sqrt(mean((ground_truth_f(:,2)-bemnn_f_I(:,2)).^2)),3);
rmse_fz_bemnn  = round(sqrt(mean((ground_truth_f(:,3)-bemnn_f_I(:,3)).^2)),3);
rmse_fxy_bemnn = round(sqrt(mean((ground_truth_f(:,4)-bemnn_f_I(:,4)).^2)),3); % scalar-error for f_xy
rmse_f_bemnn   = round(sqrt(mean((ground_truth_f(:,5)-bemnn_f_I(:,5)).^2)),3); % scalar-error for f_total
rmse_tx_bemnn  = round(sqrt(mean((ground_truth_t(:,1)-bemnn_t_B(:,1)).^2)),3);
rmse_ty_bemnn  = round(sqrt(mean((ground_truth_t(:,2)-bemnn_t_B(:,2)).^2)),3);
rmse_tz_bemnn  = round(sqrt(mean((ground_truth_t(:,3)-bemnn_t_B(:,3)).^2)),3);
rmse_txy_bemnn = round(sqrt(mean((ground_truth_t(:,4)-bemnn_t_B(:,4)).^2)),3); % scalar-error for t_xy
rmse_t_bemnn   = round(sqrt(mean((ground_truth_t(:,5)-bemnn_t_B(:,5)).^2)),3); % scalar-error for t_total

%RMSE of NeuroMHE
rmse_fx        = round(sqrt(mean((ground_truth_f(:,1)-disest_mhe(:,1)).^2)),3);
rmse_fy        = round(sqrt(mean((ground_truth_f(:,2)-disest_mhe(:,2)).^2)),3);
rmse_fz        = round(sqrt(mean((ground_truth_f(:,3)-disest_mhe(:,3)).^2)),3);
rmse_fxy       = round(sqrt(mean((ground_truth_f(:,4)-mhe_est_fscalar(:,1)).^2)),3);
rmse_f         = round(sqrt(mean((ground_truth_f(:,5)-mhe_est_fscalar(:,2)).^2)),3);
rmse_tx        = round(sqrt(mean((ground_truth_t(:,1)-disest_mhe(:,4)).^2)),3);
rmse_ty        = round(sqrt(mean((ground_truth_t(:,2)-disest_mhe(:,5)).^2)),3);
rmse_tz        = round(sqrt(mean((ground_truth_t(:,3)-disest_mhe(:,6)).^2)),3);
rmse_txy       = round(sqrt(mean((ground_truth_t(:,4)-mhe_est_tscalar(:,1)).^2)),3);
rmse_t         = round(sqrt(mean((ground_truth_t(:,5)-mhe_est_tscalar(:,2)).^2)),3);

disp('-------------RMSEs with scalar error (f in world frame and m=0.772kg, as used in T-RO 23-0314 v1)---------------')
disp('----------------------------------------------------------------------------------------------------------------')
disp('The selected trajectory:')
disp(key)
disp('------------------------------------------Below are the RMSE comparisons----------------------------------------')
%Display a table
T = table([rmse_fx_bemnn;rmse_fx],[rmse_fy_bemnn;rmse_fy],[rmse_fz_bemnn;rmse_fz],[rmse_tx_bemnn;rmse_tx],[rmse_ty_bemnn;rmse_ty],[rmse_tz_bemnn;rmse_tz],[rmse_fxy_bemnn;rmse_fxy],[rmse_txy_bemnn;rmse_txy],[rmse_f_bemnn;rmse_f],[rmse_t_bemnn;rmse_t],'VariableNames',{'f_x','f_y','f_z','t_x','t_y','t_z','f_xy','t_xy','f','t'},'RowNames',{'NeuroBEM','NeuroMHE'});
disp(T)




%Matlab function of computing the ground truth force and torque from the NeuroBEM's open-source dataset
%Wang, Bingheng, 9 Aug. 2023
%---------------------------------------------------------------------------------%
% w    : measured augular velocity, expressed in body frame
% acc_p: measured linear acceleration including the gravitational acceleration g, expressed in body frame
% acc_w: measured angular acceleration, expressed in body frame
% m    : mass (originally reported as 0.752 kg in the paper, but updated to be 0.772 kg in the NeuroBEM's website, the updated value should be used)
% J    : moment of inertia

function [f,tau]=ground_truth(w,acc_p,acc_w,m,J)
   f   = m*acc_p; 
   tau = J*acc_w + skew_symmetric(w)*(J*w);
end
function w=skew_symmetric(v)
   w=[ 0,-v(3),v(2);
    v(3),0,-v(1);
   -v(2),v(1),0];
end

function R=q_2_rotation_unit(q)
q=q/norm(q);
q0=q(4); q1=q(1);q2=q(2);q3=q(3);
R_bi = [2*(q0^2+q1^2)-1, 2*q1*q2-2*q0*q3, 2*q0*q2+2*q1*q3;
     2*q0*q3+2*q1*q2, 2*(q0^2+q2^2)-1, 2*q2*q3-2*q0*q1;
     2*q1*q3-2*q0*q2, 2*q0*q1+2*q2*q3, 2*(q0^2+q3^2)-1];
R    = R_bi;
end

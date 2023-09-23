%MATLAB code for re-producing the RMSE results (the slightly improved network model trained on the slow training set with modified weightings in the loss function)
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
%---load the NeuroMHE estimation data---%
filename=['evaluation_vector_error_slowtrainingset/disest_mhe_',key,'.csv'];
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
%RMSE Computation (Appropriate Metrics: f in body frame, vector-error formula, m=0.772kg)
%------------------------------------------------------------------------%

m_updated        = 0.772; %as updated in the NeuroBEM's website https://rpg.ifi.uzh.ch/neuro_bem/Readme.html
J                = diag([0.0025,0.0021,0.0043]);
[r,c]            = size(bemnn);
ground_truth_f   = zeros(r,3); % f_x,f_y,f_z
ground_truth_t   = zeros(r,3); % t_x,t_y,t_z

for i=1:1:r
    acc_p                 = bemnn(i,12:14).';
    acc_w                 = bemnn(i,2:4).';
    w                     = bemnn(i,5:7).';
    [f_updated,tau]       = ground_truth(w,acc_p,acc_w,m_updated,J);
    ground_truth_f(i,1:3) = f_updated.';
    ground_truth_t(i,1:3) = tau.';
end

%RMSE of NeuroBEM
rmse_fx_bemnn  = round(sqrt(mean((ground_truth_f(:,1)-bemnn(:,30)).^2)),3);
rmse_fy_bemnn  = round(sqrt(mean((ground_truth_f(:,2)-bemnn(:,31)).^2)),3);
rmse_fz_bemnn  = round(sqrt(mean((ground_truth_f(:,3)-bemnn(:,32)).^2)),3);
error_fxy_bemnn= ground_truth_f(:,1:2)-bemnn(:,30:31);
rmse_fxy_bemnn = round(sqrt(mean(vecnorm(error_fxy_bemnn,2,2).^2)),3); % vector-error for f_xy
error_f_bemnn  = ground_truth_f(:,1:3)-bemnn(:,30:32);
rmse_f_bemnn   = round(sqrt(mean(vecnorm(error_f_bemnn,2,2).^2)),3); % vector-error for f_total
rmse_tx_bemnn  = round(sqrt(mean((ground_truth_t(:,1)-bemnn(:,33)).^2)),3);
rmse_ty_bemnn  = round(sqrt(mean((ground_truth_t(:,2)-bemnn(:,34)).^2)),3);
rmse_tz_bemnn  = round(sqrt(mean((ground_truth_t(:,3)-bemnn(:,35)).^2)),3);
error_txy_bemnn= ground_truth_t(:,1:2)-bemnn(:,33:34);
rmse_txy_bemnn = round(sqrt(mean(vecnorm(error_txy_bemnn,2,2).^2)),3); % vector-error for t_xy
error_t_bemnn  = ground_truth_t(:,1:3)-bemnn(:,33:35);
rmse_t_bemnn   = round(sqrt(mean(vecnorm(error_t_bemnn,2,2).^2)),3); % vector-error for t_total

%RMSE of NeuroMHE
rmse_fx        = round(sqrt(mean((ground_truth_f(:,1)-disest_mhe(:,1)).^2)),3);
rmse_fy        = round(sqrt(mean((ground_truth_f(:,2)-disest_mhe(:,2)).^2)),3);
rmse_fz        = round(sqrt(mean((ground_truth_f(:,3)-disest_mhe(:,3)).^2)),3);
error_fxy_mhe  = ground_truth_f(:,1:2)-disest_mhe(:,1:2);
rmse_fxy       = round(sqrt(mean(vecnorm(error_fxy_mhe,2,2).^2)),3);
error_f_mhe    = ground_truth_f(:,1:3)-disest_mhe(:,1:3);
rmse_f         = round(sqrt(mean(vecnorm(error_f_mhe,2,2).^2)),3);
rmse_tx        = round(sqrt(mean((ground_truth_t(:,1)-disest_mhe(:,4)).^2)),3);
rmse_ty        = round(sqrt(mean((ground_truth_t(:,2)-disest_mhe(:,5)).^2)),3);
rmse_tz        = round(sqrt(mean((ground_truth_t(:,3)-disest_mhe(:,6)).^2)),3);
error_txy_mhe  = ground_truth_t(:,1:2)-disest_mhe(:,4:5);
rmse_txy       = round(sqrt(mean(vecnorm(error_txy_mhe,2,2).^2)),3);
error_t_mhe    = ground_truth_t(:,1:3)-disest_mhe(:,4:6);
rmse_t         = round(sqrt(mean(vecnorm(error_t_mhe,2,2).^2)),3);

disp('------------------------------RMSEs with vector error (body frame and m=0.772kg)--------------------------------')
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


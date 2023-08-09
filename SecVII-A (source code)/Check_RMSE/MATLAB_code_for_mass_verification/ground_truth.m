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

function w=skew_symmetric(v)
w=[ 0,-v(3),v(2);
    v(3),0,-v(1);
   -v(2),v(1),0];